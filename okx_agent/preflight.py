"""Deterministic pre-order risk-reward, sizing, liquidity, and ledger gate for OKX SWAP."""

import argparse
import json
import sys
import time
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from okx_agent.universe import DEFAULT_UNIVERSE_PATH, get_instrument_policy
from risk_ledger import DEFAULT_LEDGER_PATH, check_pre_trade_risk


def _to_decimal(val: Any) -> Decimal:
    return Decimal(str(val))


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value // step) * step


def _format_decimal(value: Decimal, step: Decimal | None = None) -> str:
    if step is not None and step > 0:
        quant = step.normalize()
        exp = quant.as_tuple().exponent
        if isinstance(exp, int) and exp < 0:
            value = value.quantize(Decimal(10) ** exp, rounding=ROUND_DOWN)
        else:
            value = value.quantize(Decimal("1"), rounding=ROUND_DOWN)
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _add_rejection(rejections: list[dict], code: str, detail: str) -> None:
    rejections.append({"code": code, "detail": detail})


def _build_cl_ord_id(inst_id: str, tranche_idx: int) -> str:
    prefix = "".join(ch for ch in inst_id.upper() if ch.isalnum())[:12]
    ts_ms = int(time.time() * 1000) % 10_000_000_000
    return f"okx{prefix}{ts_ms}t{tranche_idx}"[:32]


def evaluate_preflight(payload: dict) -> dict:
    """Evaluate all AGENTS.md Section 1 hard gates and emit atomic split-TP order payloads."""
    rejections: list[dict] = []

    inst_id = str(payload.get("inst_id", "")).strip().upper()
    side = str(payload.get("side", "")).strip().lower()
    ord_type = str(payload.get("ord_type", "limit")).strip().lower()
    allow_unlisted = bool(payload.get("allow_unlisted", False))

    universe_path = Path(payload["universe_path"]) if payload.get("universe_path") else DEFAULT_UNIVERSE_PATH
    ledger_path = Path(payload["ledger_path"]) if payload.get("ledger_path") else DEFAULT_LEDGER_PATH

    policy = get_instrument_policy(
        inst_id,
        universe_path=universe_path,
        inst_category=payload.get("inst_category"),
    )
    defaults = policy["defaults"]
    perp_rules = policy.get("perp_type_rules", {})

    min_weighted_rr = float(defaults.get("min_weighted_rr", 2.5))
    min_tp1_rr = float(payload.get("min_tp1_rr", defaults.get("min_tp1_rr", 1.2)))
    min_stop_pct = float(payload.get("min_stop_pct", defaults.get("min_stop_pct", 0.003)))
    max_stop_pct = float(payload.get("max_stop_pct", defaults.get("max_stop_pct", 0.08)))
    max_leverage = float(payload.get("max_leverage", defaults.get("max_leverage", 20.0)))
    max_spread_bps = float(defaults.get("max_spread_bps", 5.0))
    min_top5_depth_ratio = float(defaults.get("min_top5_depth_ratio", 3.0))
    min_liq_distance_r = float(policy["min_liq_distance_r"])

    if not allow_unlisted and not (policy["in_universe"] and policy["enabled"]):
        _add_rejection(
            rejections,
            "REJECTED_INSTRUMENT_NOT_IN_UNIVERSE",
            f"{inst_id} is not enabled in {universe_path.name}",
        )

    market_verdict = payload.get("market_verdict")
    if market_verdict is not None and str(market_verdict).strip().upper() != "ALLOW":
        _add_rejection(
            rejections,
            "REJECTED_MARKET_VERDICT_MISMATCH",
            f"market_verdict={market_verdict} is not ALLOW",
        )
    market_dir = payload.get("market_direction")
    if market_dir is not None:
        expected_dir = "LONG" if side == "buy" else "SHORT"
        if str(market_dir).strip().upper() != expected_dir:
            _add_rejection(
                rejections,
                "REJECTED_MARKET_VERDICT_MISMATCH",
                f"market_direction={market_dir} opposes order side={side}",
            )

    if perp_rules.get("avoid_earnings_gap") and bool(payload.get("earnings_within_48h", False)):
        _add_rejection(
            rejections,
            "REJECTED_EARNINGS_GAP_WINDOW",
            f"{inst_id} is within 48h earnings gap window",
        )

    if perp_rules.get("long_only") and side == "sell" and not bool(payload.get("allow_stock_token_short", False)):
        _add_rejection(
            rejections,
            "REJECTED_STOCK_TOKEN_LONG_ONLY",
            f"{inst_id} ({policy['perp_type']}) enforces V2.1 long-only rule; set allow_stock_token_short=true to override",
        )

    td_mode = str(payload.get("td_mode", defaults.get("default_td_mode", "isolated"))).strip().lower()
    pos_side = str(payload.get("pos_side", "net")).strip().lower()
    simulated_trading = bool(payload.get("simulated_trading", defaults.get("default_simulated_trading", True)))
    sl_trigger_px_type = str(defaults.get("default_sl_trigger_px_type", "mark"))
    tp_trigger_px_type = str(defaults.get("default_tp_trigger_px_type", "last"))

    entry = float(payload["entry"])
    sl = float(payload["sl"])
    tp1 = float(payload["tp1"])
    tp2 = float(payload["tp2"]) if payload.get("tp2") is not None else None

    if tp2 is None:
        w1 = float(payload.get("w1", 1.0))
        w2 = float(payload.get("w2", 0.0))
    else:
        w1 = float(payload.get("w1", 0.5))
        w2 = float(payload.get("w2", 0.5))

    risk_pct = float(payload["risk_pct"])
    equity_usd = float(payload["equity_usd"])
    avail_bal_usd = float(payload["avail_bal_usd"]) if payload.get("avail_bal_usd") is not None else None
    leverage = float(payload["leverage"])
    if leverage > max_leverage:
        _add_rejection(
            rejections,
            "REJECTED_LEVERAGE_EXCEEDED",
            f"leverage={leverage:.1f}x > max_leverage={max_leverage:.1f}x",
        )
    if perp_rules.get("avoid_weekend_full_size") and bool(payload.get("is_weekend_or_off_hours", False)):
        if risk_pct > 0.005 and not bool(payload.get("allow_weekend_reduced", False)):
            _add_rejection(
                rejections,
                "REJECTED_WEEKEND_STOCK_TOKEN_RISK",
                f"stock_token weekend/off-hours risk_pct={risk_pct:.4f} exceeds 0.0050 reduced cap",
            )
    expected_hold_funding_periods = float(
        payload.get("expected_hold_funding_periods", defaults.get("default_expected_hold_funding_periods", 3.0))
    )
    maint_margin_rate = float(payload.get("maint_margin_rate", 0.005))

    acct_lv = int(payload["acct_lv"])
    ct_val = float(payload["ct_val"])
    lot_sz_dec = _to_decimal(payload["lot_sz"])
    min_sz_dec = _to_decimal(payload["min_sz"])
    tick_sz_dec = _to_decimal(payload.get("tick_sz", "0.01"))

    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    funding_rate = float(payload.get("funding_rate", 0.0))
    # OKX API returns taker fee rate as negative (e.g. "-0.0005"); normalize to positive cost.
    taker_fee_rate = abs(float(payload.get("taker_fee_rate", float(defaults.get("default_taker_fee_bps", 5.0)) / 10000.0)))
    slippage_bps = float(payload.get("slippage_bps", defaults.get("default_slippage_bps", 2.0)))

    # 1. Directional price geometry & weights
    geometry_ok = True
    if side == "buy":
        if not (sl < entry < tp1 and (tp2 is None or tp2 >= tp1)):
            geometry_ok = False
    elif side == "sell":
        if not (sl > entry > tp1 and (tp2 is None or tp2 <= tp1)):
            geometry_ok = False
    else:
        geometry_ok = False

    if not geometry_ok:
        _add_rejection(
            rejections,
            "REJECTED_INVALID_PRICE_GEOMETRY",
            f"Invalid {side} geometry: entry={entry}, sl={sl}, tp1={tp1}, tp2={tp2}",
        )

    weights_ok = w1 > 0 and w2 >= 0 and abs((w1 + w2) - 1.0) <= 1e-6 and (tp2 is not None or abs(w2) <= 1e-9)
    if not weights_ok:
        _add_rejection(
            rejections,
            "REJECTED_INVALID_WEIGHTS",
            f"Invalid split weights: w1={w1}, w2={w2}, tp2={tp2}",
        )

    # 2. Account level check
    if acct_lv < 2:
        _add_rejection(
            rejections,
            "REJECTED_ACCOUNT_LEVEL",
            f"acctLv={acct_lv} < 2; switch OKX account to Single-currency Margin mode",
        )

    # 3. Orderbook spread & depth metrics
    spread_bps = None
    bid_top5_usd = 0.0
    ask_top5_usd = 0.0
    min_top5_usd = 0.0
    if not bids or not asks:
        _add_rejection(rejections, "REJECTED_DEPTH_INSUFFICIENT", "Orderbook bids or asks are empty")
    else:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_px = (best_bid + best_ask) / 2.0
        if mid_px <= 0 or best_ask < best_bid:
            _add_rejection(rejections, "REJECTED_SPREAD_EXCEEDED", f"Invalid top of book: bid={best_bid}, ask={best_ask}")
        else:
            spread_bps = ((best_ask - best_bid) / mid_px) * 10000.0
            if spread_bps > max_spread_bps:
                _add_rejection(
                    rejections,
                    "REJECTED_SPREAD_EXCEEDED",
                    f"spread_bps={spread_bps:.2f} > max {max_spread_bps:.2f}",
                )
        bid_top5_usd = sum(float(lvl[0]) * float(lvl[1]) * ct_val for lvl in bids[:5])
        ask_top5_usd = sum(float(lvl[0]) * float(lvl[1]) * ct_val for lvl in asks[:5])
        min_top5_usd = min(bid_top5_usd, ask_top5_usd)

    # 4. Sizing & Risk
    r_dist = abs(entry - sl)
    if r_dist <= 0 or ct_val <= 0 or equity_usd <= 0 or leverage <= 0 or lot_sz_dec <= 0:
        _add_rejection(
            rejections,
            "REJECTED_INVALID_INPUTS",
            f"Non-positive parameter detected: r_dist={r_dist}, ct_val={ct_val}, equity={equity_usd}, lev={leverage}",
        )
        return {
            "passed": False,
            "status": "REJECTED",
            "inst_id": inst_id,
            "perp_type": policy["perp_type"],
            "bucket": policy["bucket"],
            "rejections": rejections,
            "metrics": {},
            "orders": [],
        }

    risk_usd = equity_usd * risk_pct
    raw_sz_dec = _to_decimal(risk_usd) / (_to_decimal(r_dist) * _to_decimal(ct_val))
    total_sz_dec = _floor_to_step(raw_sz_dec, lot_sz_dec)
    total_sz = float(total_sz_dec)

    if total_sz_dec < min_sz_dec:
        _add_rejection(
            rejections,
            "REJECTED_SIZE_BELOW_MIN",
            f"total_sz={total_sz} < min_sz={float(min_sz_dec)}",
        )

    tranche1_sz_dec = total_sz_dec
    tranche2_sz_dec = Decimal("0")
    if tp2 is not None and total_sz_dec >= min_sz_dec:
        tranche1_sz_dec = _floor_to_step(total_sz_dec * _to_decimal(w1), lot_sz_dec)
        tranche2_sz_dec = total_sz_dec - tranche1_sz_dec
        if tranche1_sz_dec < min_sz_dec or tranche2_sz_dec < min_sz_dec:
            _add_rejection(
                rejections,
                "REJECTED_SPLIT_SIZE_BELOW_MIN",
                f"Split sizes ({float(tranche1_sz_dec)}, {float(tranche2_sz_dec)}) below min_sz={float(min_sz_dec)}",
            )

    stop_pct = r_dist / entry if entry > 0 else 0.0
    if stop_pct < min_stop_pct or stop_pct > max_stop_pct:
        _add_rejection(
            rejections,
            "REJECTED_STOP_WIDTH_OUT_OF_RANGE",
            f"stop_pct={stop_pct:.4f} outside [{min_stop_pct:.4f}, {max_stop_pct:.4f}]",
        )

    notional_usd = (total_sz if total_sz > 0 else float(raw_sz_dec)) * entry * ct_val
    required_margin_usd = notional_usd / leverage
    if avail_bal_usd is not None and required_margin_usd > avail_bal_usd:
        _add_rejection(
            rejections,
            "REJECTED_INSUFFICIENT_AVAILABLE_MARGIN",
            f"required_margin_usd={required_margin_usd:.2f} > avail_bal_usd={avail_bal_usd:.2f}",
        )

    if bids and asks and min_top5_usd < min_top5_depth_ratio * notional_usd:
        _add_rejection(
            rejections,
            "REJECTED_DEPTH_INSUFFICIENT",
            f"min_top5_usd={min_top5_usd:.2f} < {min_top5_depth_ratio:.1f}x notional ({notional_usd:.2f})",
        )

    # 5. Friction & Net Weighted R:R (friction applied to both TP reward and SL loss)
    fee_round_trip_rate = 2.0 * taker_fee_rate
    half_spread_rate = ((spread_bps or 0.0) / 10000.0) / 2.0
    entry_spread_rate = half_spread_rate if ord_type == "market" else 0.0
    exit_spread_rate = half_spread_rate
    slippage_rate = slippage_bps / 10000.0
    signed_funding = funding_rate if side == "buy" else -funding_rate
    funding_cost_rate = max(0.0, signed_funding * expected_hold_funding_periods)

    total_friction_rate = fee_round_trip_rate + entry_spread_rate + exit_spread_rate + slippage_rate + funding_cost_rate
    friction_px = entry * total_friction_rate
    friction_r = friction_px / r_dist

    tp1_gross_rr = abs(tp1 - entry) / r_dist
    tp1_net_rr = max(0.0, tp1_gross_rr - friction_r) / (1.0 + friction_r)

    planned_gross_rr = (w1 * abs(tp1 - entry) + (w2 * abs(tp2 - entry) if tp2 is not None else 0.0)) / r_dist
    planned_net_rr = max(0.0, planned_gross_rr - friction_r) / (1.0 + friction_r)

    if tp2 is not None and total_sz_dec > 0 and tranche1_sz_dec > 0 and tranche2_sz_dec > 0:
        eff_w1 = float(tranche1_sz_dec / total_sz_dec)
        eff_w2 = float(tranche2_sz_dec / total_sz_dec)
    else:
        eff_w1, eff_w2 = w1, w2

    effective_gross_rr = (eff_w1 * abs(tp1 - entry) + (eff_w2 * abs(tp2 - entry) if tp2 is not None else 0.0)) / r_dist
    effective_net_rr = max(0.0, effective_gross_rr - friction_r) / (1.0 + friction_r)

    if tp2 is not None and tp1_net_rr < min_tp1_rr:
        _add_rejection(
            rejections,
            "REJECTED_TP1_RR_BELOW_MIN",
            f"tp1_net_rr={tp1_net_rr:.3f} < min_tp1_rr={min_tp1_rr:.2f} (gross={tp1_gross_rr:.3f}, friction_r={friction_r:.3f})",
        )

    if planned_net_rr < min_weighted_rr or effective_net_rr < min_weighted_rr:
        _add_rejection(
            rejections,
            "REJECTED_RR_BELOW_MIN",
            f"net_weighted_rr={min(planned_net_rr, effective_net_rr):.3f} < min {min_weighted_rr:.2f} "
            f"(gross={effective_gross_rr:.3f}, friction_r={friction_r:.3f})",
        )

    # 6. Liquidation guard
    if payload.get("liq_px") is not None:
        liq_px = float(payload["liq_px"])
    elif side == "buy":
        liq_px = entry * (1.0 - 1.0 / leverage + maint_margin_rate)
    else:
        liq_px = entry * (1.0 + 1.0 / leverage - maint_margin_rate)

    liq_outside_sl = (liq_px < sl) if side == "buy" else (liq_px > sl)
    if not liq_outside_sl:
        _add_rejection(
            rejections,
            "REJECTED_LIQ_INSIDE_SL",
            f"liq_px={liq_px:.4f} is inside/at SL={sl:.4f} for {side}",
        )

    liq_dist_r = abs(liq_px - entry) / r_dist
    if liq_dist_r < min_liq_distance_r:
        _add_rejection(
            rejections,
            "REJECTED_LIQ_DISTANCE_TOO_CLOSE",
            f"|Liq_Px - Entry|={liq_dist_r:.2f}R < required {min_liq_distance_r:.2f}R ({policy['perp_type']})",
        )

    # 7. Persistent Risk Ledger gate
    actual_risk_usd = total_sz * r_dist * ct_val
    actual_risk_pct = (actual_risk_usd / equity_usd) if total_sz > 0 else risk_pct
    ledger_gate = check_pre_trade_risk(inst_id, actual_risk_pct, side=side, ledger_path=ledger_path)
    if not ledger_gate["passed"]:
        _add_rejection(
            rejections,
            f"LEDGER_{ledger_gate['reason']}",
            ledger_gate["detail"],
        )

    passed = len(rejections) == 0
    orders: list[dict] = []
    if passed:
        px_str = _format_decimal(_to_decimal(entry), tick_sz_dec)
        sl_str = _format_decimal(_to_decimal(sl), tick_sz_dec)
        tp1_str = _format_decimal(_to_decimal(tp1), tick_sz_dec)

        tranches = [(1, tranche1_sz_dec, tp1_str)]
        if tp2 is not None:
            tp2_str = _format_decimal(_to_decimal(tp2), tick_sz_dec)
            tranches.append((2, tranche2_sz_dec, tp2_str))

        for idx, sz_dec, tp_str in tranches:
            order = {
                "instId": inst_id,
                "tdMode": td_mode,
                "side": side,
                "posSide": pos_side,
                "ordType": ord_type,
                "sz": _format_decimal(sz_dec, lot_sz_dec),
                "tpTriggerPx": tp_str,
                "tpOrdPx": "-1",
                "tpTriggerPxType": tp_trigger_px_type,
                "slTriggerPx": sl_str,
                "slOrdPx": "-1",
                "slTriggerPxType": sl_trigger_px_type,
                "simulatedTrading": simulated_trading,
                "clOrdId": _build_cl_ord_id(inst_id, idx),
            }
            if ord_type != "market":
                order["px"] = px_str
            orders.append(order)

    status = "PASS" if passed else ("REJECTED (< 2.5R)" if any(r["code"] == "REJECTED_RR_BELOW_MIN" for r in rejections) else "REJECTED")

    return {
        "passed": passed,
        "status": status,
        "inst_id": inst_id,
        "perp_type": policy["perp_type"],
        "bucket": policy["bucket"],
        "rejections": rejections,
        "metrics": {
            "one_r_px": round(r_dist, 6),
            "stop_pct": round(stop_pct, 6),
            "risk_usd_target": round(risk_usd, 4),
            "risk_usd_actual": round(actual_risk_usd, 4),
            "actual_risk_pct": round(actual_risk_pct, 6),
            "total_sz": total_sz,
            "tranche1_sz": float(tranche1_sz_dec),
            "tranche2_sz": float(tranche2_sz_dec),
            "notional_usd": round(notional_usd, 2),
            "required_margin_usd": round(required_margin_usd, 2),
            "spread_bps": round(spread_bps, 4) if spread_bps is not None else None,
            "bid_top5_usd": round(bid_top5_usd, 2),
            "ask_top5_usd": round(ask_top5_usd, 2),
            "min_top5_usd": round(min_top5_usd, 2),
            "taker_fee_rate": taker_fee_rate,
            "funding_cost_rate": round(funding_cost_rate, 6),
            "friction_r": round(friction_r, 4),
            "tp1_net_rr": round(tp1_net_rr, 4),
            "gross_weighted_rr": round(effective_gross_rr, 4),
            "net_weighted_rr": round(effective_net_rr, 4),
            "liq_px": round(liq_px, 4),
            "liq_dist_r": round(liq_dist_r, 3),
            "min_liq_distance_r": min_liq_distance_r,
        },
        "orders": orders,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX SWAP Pre-Order Hard Gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to JSON input file")
    group.add_argument("--json", help="Inline JSON payload string")
    args = parser.parse_args()

    if args.input:
        with Path(args.input).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(args.json)

    result = evaluate_preflight(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
