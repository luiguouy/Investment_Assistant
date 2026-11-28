"""Capital-aware swing SL/TP/Trailing-SL level calculator for OKX perpetual swaps."""

import argparse
import json
import sys
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from okx_agent.universe import DEFAULT_UNIVERSE_PATH, get_instrument_policy


def _to_dec(val: float | str | Decimal) -> Decimal:
    return Decimal(str(val))


def _floor_step(val: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return val
    return (val // step) * step


def _round_tick(val: float, tick_sz: float) -> float:
    if tick_sz <= 0:
        return round(val, 6)
    t_dec = _to_dec(tick_sz).normalize()
    q = (_to_dec(val) / t_dec).quantize(Decimal("1"), rounding=ROUND_DOWN) * t_dec
    return float(q)


def recommend_capital_profile(equity_usd: float, perp_type: str = "crypto") -> dict[str, Any]:
    """Select capital-tier sizing, split weights, and leverage caps based on available USDT."""
    is_stock = str(perp_type).strip().lower() == "stock_token"
    if equity_usd < 500.0:
        return {
            "tier": "micro",
            "risk_pct": 0.02,
            "w1": 0.5,
            "w2": 0.5,
            "leverage": 4.0 if is_stock else 6.0,
            "max_concurrent_positions": 1,
            "min_stop_pct": 0.006,
            "max_stop_pct": 0.045,
            "max_hold_hours": 48.0,
        }
    return {
        "tier": "standard",
        "risk_pct": 0.01,
        "w1": 0.5,
        "w2": 0.5,
        "leverage": 4.0 if is_stock else 8.0,
        "max_concurrent_positions": 3,
        "min_stop_pct": 0.006,
        "max_stop_pct": 0.045,
        "max_hold_hours": 48.0,
    }


def compute_chandelier_trailing_sl(
    side: str,
    current_sl: float,
    atr: float,
    peak_or_trough_px: float,
    confirmed_swing_px: float | None = None,
    tick_sz: float = 0.01,
    swing_atr_k: float = 0.5,
    chandelier_atr_k: float = 2.0,
) -> float:
    """Compute hybrid structural + Chandelier ATR trailing SL (monotonic: never widens)."""
    side_l = side.strip().lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}'; expected 'buy' or 'sell'")

    if side_l == "buy":
        cands = [current_sl, peak_or_trough_px - chandelier_atr_k * atr]
        if confirmed_swing_px is not None:
            cands.append(confirmed_swing_px - swing_atr_k * atr)
        return _round_tick(max(cands), tick_sz)

    cands = [current_sl, peak_or_trough_px + chandelier_atr_k * atr]
    if confirmed_swing_px is not None:
        cands.append(confirmed_swing_px + swing_atr_k * atr)
    return _round_tick(min(cands), tick_sz)


def compute_swing_levels(
    inst_id: str,
    side: str,
    entry: float,
    swing_anchor: float,
    atr: float,
    equity_usd: float,
    ct_val: float,
    lot_sz: float,
    min_sz: float,
    tick_sz: float = 0.01,
    tp1_structure_px: float | None = None,
    tp2_structure_px: float | None = None,
    is_2b_sweep: bool = False,
    taker_fee_rate: float = 0.0005,
    slippage_bps: float = 2.0,
    funding_cost_rate: float = 0.0,
    inst_category: str | None = None,
    universe_path: str | Path = DEFAULT_UNIVERSE_PATH,
) -> dict[str, Any]:
    """Compute capital-aware SL, TP1, TP2, BE buffer, and trailing milestones for OKX SWAP."""
    sym = inst_id.strip().upper()
    side_l = side.strip().lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}'; expected 'buy' or 'sell'")

    policy = get_instrument_policy(sym, universe_path=universe_path, inst_category=inst_category)
    perp_type = policy["perp_type"]
    profile = recommend_capital_profile(equity_usd, perp_type=perp_type)
    sign = 1.0 if side_l == "buy" else -1.0

    # 1. Structural + ATR buffered initial SL
    if perp_type == "stock_token":
        buf_k = 1.0 if is_2b_sweep else 0.8
    else:
        buf_k = 1.5 if is_2b_sweep else 1.2

    sl_raw = swing_anchor - sign * buf_k * atr
    sl = _round_tick(sl_raw, tick_sz)
    r_dist = abs(entry - sl)
    stop_pct = (r_dist / entry) if entry > 0 else 0.0

    warnings: list[str] = []
    executable = True
    if (side_l == "buy" and sl >= entry) or (side_l == "sell" and sl <= entry) or r_dist <= 0:
        return {
            "executable": False,
            "reason": "INVALID_SL_GEOMETRY",
            "inst_id": sym,
            "side": side_l,
            "entry": entry,
            "sl": sl,
        }

    if stop_pct < profile["min_stop_pct"] or stop_pct > profile["max_stop_pct"]:
        executable = False
        warnings.append(
            f"STOP_WIDTH_OUT_OF_CORRIDOR: stop_pct={stop_pct:.4f} outside "
            f"[{profile['min_stop_pct']:.4f}, {profile['max_stop_pct']:.4f}]"
        )

    # 2. Friction ratio (both TP and SL sides)
    total_friction_rate = 2.0 * abs(taker_fee_rate) + (slippage_bps / 10000.0) + max(0.0, funding_cost_rate)
    friction_r = (entry * total_friction_rate) / r_dist

    # 3. Sizing & automatic single-tranche degradation for micro accounts
    risk_usd = equity_usd * profile["risk_pct"]
    lot_dec = _to_dec(lot_sz)
    min_dec = _to_dec(min_sz)
    raw_sz_dec = _to_dec(risk_usd) / (_to_dec(r_dist) * _to_dec(ct_val))
    total_sz_dec = _floor_step(raw_sz_dec, lot_dec)
    total_sz = float(total_sz_dec)

    if total_sz_dec < min_dec:
        executable = False
        warnings.append(f"SIZE_BELOW_MIN: total_sz={total_sz} < min_sz={min_sz}")

    can_split = total_sz_dec >= (min_dec * Decimal("2"))
    if can_split:
        split_mode = "split_two_tranche"
        w1, w2 = profile["w1"], profile["w2"]
        tp1_default = entry + sign * 1.8 * r_dist
        tp1 = _round_tick(tp1_default, tick_sz)
        if tp1_structure_px is not None:
            struct_rr = abs(tp1_structure_px - entry) / r_dist
            struct_net_rr = max(0.0, struct_rr - friction_r) / (1.0 + friction_r)
            closer = (tp1_structure_px < tp1) if side_l == "buy" else (tp1_structure_px > tp1)
            valid_dir = (tp1_structure_px > entry) if side_l == "buy" else (tp1_structure_px < entry)
            if valid_dir and closer and struct_net_rr >= 1.2:
                tp1 = _round_tick(tp1_structure_px, tick_sz)

        # Solve TP2 so weighted net RR >= 2.55 (leaving slight safety margin above 2.50)
        tp1_gross_rr = abs(tp1 - entry) / r_dist
        target_gross_weighted = 2.55 * (1.0 + friction_r) + friction_r
        needed_tp2_rr = max(3.5, (target_gross_weighted - w1 * tp1_gross_rr) / max(w2, 1e-6))
        tp2_default = entry + sign * needed_tp2_rr * r_dist
        tp2 = _round_tick(tp2_default, tick_sz)
        if tp2_structure_px is not None:
            valid_tp2 = (tp2_structure_px > tp1) if side_l == "buy" else (tp2_structure_px < tp1)
            if valid_tp2:
                s_tp2_rr = abs(tp2_structure_px - entry) / r_dist
                s_gross_w = w1 * tp1_gross_rr + w2 * s_tp2_rr
                s_net_w = max(0.0, s_gross_w - friction_r) / (1.0 + friction_r)
                if s_net_w >= 2.5:
                    tp2 = _round_tick(tp2_structure_px, tick_sz)

        be_trigger_px = tp1
        gross_weighted_rr = w1 * (abs(tp1 - entry) / r_dist) + w2 * (abs(tp2 - entry) / r_dist)
    else:
        split_mode = "single_with_trailing"
        w1, w2 = 1.0, 0.0
        target_single_rr = max(3.0, 2.55 * (1.0 + friction_r) + friction_r)
        tp1 = _round_tick(entry + sign * target_single_rr * r_dist, tick_sz)
        tp2 = None
        be_trigger_px = _round_tick(entry + sign * 1.5 * r_dist, tick_sz)
        gross_weighted_rr = abs(tp1 - entry) / r_dist

    net_weighted_rr = max(0.0, gross_weighted_rr - friction_r) / (1.0 + friction_r)
    be_buffer_rate = 2.0 * abs(taker_fee_rate) + 0.0005
    breakeven_px = _round_tick(entry * (1.0 + sign * be_buffer_rate), tick_sz)

    return {
        "executable": executable,
        "warnings": warnings,
        "inst_id": sym,
        "perp_type": perp_type,
        "side": side_l,
        "capital_tier": profile["tier"],
        "split_mode": split_mode,
        "risk_pct": profile["risk_pct"],
        "leverage": profile["leverage"],
        "entry": _round_tick(entry, tick_sz),
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "w1": w1,
        "w2": w2,
        "one_r_px": round(r_dist, 6),
        "stop_pct": round(stop_pct, 6),
        "total_sz": total_sz,
        "risk_usd": round(risk_usd, 2),
        "net_weighted_rr": round(net_weighted_rr, 3),
        "be_trigger_px": be_trigger_px,
        "breakeven_px": breakeven_px,
        "trail_milestones": [
            _round_tick(entry + sign * 2.2 * r_dist, tick_sz),
            _round_tick(entry + sign * 3.0 * r_dist, tick_sz),
        ],
        "max_hold_hours": profile["max_hold_hours"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Capital-Aware Swing SL/TP Level Calculator")
    parser.add_argument("--json", required=True, help="JSON payload with inst_id, side, entry, swing_anchor, atr, equity_usd, ct_val, lot_sz, min_sz")
    args = parser.parse_args()
    payload = json.loads(args.json)
    result = compute_swing_levels(**payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("executable") else 1


if __name__ == "__main__":
    sys.exit(main())
