"""Live OKX exchange state vs. risk_ledger reconciler and failure-mode playbooks."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from risk_ledger import DEFAULT_LEDGER_PATH, load_ledger


def _format_sz(val: float) -> str:
    s = f"{val:.8f}".rstrip("0").rstrip(".")
    return s or "0"


def _infer_position_side(pos_row: dict) -> tuple[str, str]:
    pos_side = str(pos_row.get("posSide", "net")).strip().lower()
    pos_qty = float(pos_row.get("pos", 0))
    if pos_side in ("long", "short"):
        return ("buy" if pos_side == "long" else "sell"), pos_side
    return ("buy" if pos_qty > 0 else "sell"), "net"


def reconcile_state(
    positions: list[dict],
    open_orders: list[dict],
    algo_orders: list[dict],
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    simulated_trading: bool = True,
    default_sl_by_inst: dict[str, float] | None = None,
) -> dict:
    """Reconcile live OKX positions/orders/algos against risk_ledger.json (AGENTS.md Section 4)."""
    actions: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    ledger = load_ledger(ledger_path)
    ledger_open = ledger.get("open_positions", [])
    ledger_by_inst = {str(p.get("symbol", "")).strip().upper(): p for p in ledger_open}
    sl_overrides = {k.strip().upper(): float(v) for k, v in (default_sl_by_inst or {}).items()}

    live_positions = [p for p in positions if abs(float(p.get("pos", 0) or 0)) > 0]
    live_inst_ids: set[str] = set()

    for p in live_positions:
        inst_id = str(p.get("instId", "")).strip().upper()
        live_inst_ids.add(inst_id)
        pos_qty = abs(float(p["pos"]))
        side, pos_side = _infer_position_side(p)
        mgn_mode = str(p.get("mgnMode", "isolated")).strip().lower()

        covered_sl_sz = 0.0
        for ao in algo_orders:
            if str(ao.get("instId", "")).strip().upper() != inst_id:
                continue
            sl_px = str(ao.get("slTriggerPx", "") or "").strip()
            if not sl_px:
                continue
            if str(ao.get("slTriggerPxType", "last")).strip().lower() != "mark":
                warnings.append({
                    "code": "WARNING_SL_NOT_MARK_TRIGGERED",
                    "inst_id": inst_id,
                    "algo_id": ao.get("algoId"),
                    "detail": f"slTriggerPxType is '{ao.get('slTriggerPxType')}' instead of 'mark'",
                })
            covered_sl_sz += float(ao.get("sz", 0) or 0)

        uncovered_sz = pos_qty - covered_sl_sz
        if uncovered_sz > 1e-9:
            sl_px = sl_overrides.get(inst_id)
            if sl_px is None and inst_id in ledger_by_inst:
                sl_px = float(ledger_by_inst[inst_id]["sl"])

            close_side = "sell" if side == "buy" else "buy"
            params: dict[str, Any] = {
                "instId": inst_id,
                "tdMode": mgn_mode,
                "side": close_side,
                "ordType": "conditional",
                "sz": _format_sz(uncovered_sz),
                "slOrdPx": "-1",
                "slTriggerPxType": "mark",
                "reduceOnly": True,
                "simulatedTrading": simulated_trading,
            }
            if sl_px is not None:
                params["slTriggerPx"] = str(sl_px)
            if pos_side in ("long", "short"):
                params["posSide"] = pos_side

            actions.append({
                "action": "ACTION_PLACE_MISSING_SL",
                "inst_id": inst_id,
                "uncovered_sz": round(uncovered_sz, 8),
                "tool": "swap_place_algo_order",
                "params": params,
                "detail": f"Live position {pos_qty} contracts has only {covered_sl_sz} covered by SL algo orders",
            })

        if inst_id not in ledger_by_inst:
            actions.append({
                "action": "LEDGER_MISSING_LIVE_POSITION",
                "inst_id": inst_id,
                "detail": f"Live OKX position for {inst_id} is not recorded in risk_ledger.json",
            })

    for inst_id in sorted(ledger_by_inst):
        if inst_id not in live_inst_ids:
            has_open_entry = any(
                str(o.get("instId", "")).strip().upper() == inst_id for o in open_orders
            )
            if not has_open_entry:
                actions.append({
                    "action": "LEDGER_STALE_OPEN_POSITION",
                    "inst_id": inst_id,
                    "detail": f"{inst_id} is open in risk_ledger.json but has no live position or open order on OKX",
                })

    return {
        "status": "OK" if not actions else "ACTION_REQUIRED",
        "live_positions_count": len(live_positions),
        "actions": actions,
        "warnings": warnings,
    }


def plan_breakeven_amend(
    inst_id: str,
    side: str,
    entry: float,
    mark_px: float,
    algo_id: str,
    mgn_mode: str = "isolated",
    pos_side: str = "net",
    simulated_trading: bool = True,
    fee_buffer_rate: float = 0.0,
    taker_fee_rate: float = 0.0,
    funding_cost_rate: float = 0.0,
) -> dict:
    """Validate markPx vs fee-buffered breakeven price before issuing atomic BE amend (Playbook 2)."""
    sym = inst_id.strip().upper()
    side_l = side.strip().lower()
    total_buffer = max(0.0, float(fee_buffer_rate)) + 2.0 * abs(float(taker_fee_rate)) + max(0.0, float(funding_cost_rate))
    if total_buffer > 0.0:
        be_px = round(entry * (1.0 + total_buffer) if side_l == "buy" else entry * (1.0 - total_buffer), 8)
    else:
        be_px = entry

    safe_for_be = (side_l == "buy" and mark_px > be_px) or (side_l == "sell" and mark_px < be_px)

    if safe_for_be:
        return {
            "action": "AMEND_SL_TO_BREAKEVEN",
            "tool": "swap_amend_algo_order",
            "breakeven_px": be_px,
            "params": {
                "instId": sym,
                "algoId": algo_id,
                "newSlTriggerPx": str(be_px),
                "newSlOrdPx": "-1",
                "simulatedTrading": simulated_trading,
            },
        }

    close_params: dict[str, Any] = {
        "instId": sym,
        "mgnMode": mgn_mode,
        "autoCxl": True,
        "simulatedTrading": simulated_trading,
    }
    if pos_side in ("long", "short"):
        close_params["posSide"] = pos_side

    return {
        "action": "MARKET_CLOSE_REMAINING",
        "tool": "swap_close_position",
        "breakeven_px": be_px,
        "params": close_params,
        "reason": f"markPx={mark_px} already crossed breakeven={be_px} on losing side for {side_l}",
    }


def plan_trailing_stop_amend(
    inst_id: str,
    side: str,
    current_sl: float,
    candidate_sl: float,
    mark_px: float,
    algo_id: str,
    entry: float | None = None,
    initial_sl: float | None = None,
    min_step_r: float = 0.25,
    simulated_trading: bool = True,
) -> dict:
    """Plan an atomic trailing-SL tighten amend; never widens SL or places SL across markPx."""
    sym = inst_id.strip().upper()
    side_l = side.strip().lower()
    improves = (candidate_sl > current_sl) if side_l == "buy" else (candidate_sl < current_sl)
    if not improves:
        return {
            "action": "NO_TRAILING_AMEND",
            "reason": f"candidate_sl={candidate_sl} does not tighten current_sl={current_sl} for {side_l}",
        }

    if entry is not None and initial_sl is not None:
        one_r = abs(entry - initial_sl)
        if one_r > 0 and abs(candidate_sl - current_sl) < min_step_r * one_r:
            return {
                "action": "NO_TRAILING_AMEND",
                "reason": f"step {abs(candidate_sl - current_sl):.4f} < min_step_r ({min_step_r} * {one_r:.4f})",
            }

    safe_mark = (mark_px > candidate_sl) if side_l == "buy" else (mark_px < candidate_sl)
    if not safe_mark:
        return {
            "action": "NO_TRAILING_AMEND",
            "reason": f"markPx={mark_px} is already at/beyond candidate_sl={candidate_sl} for {side_l}",
        }

    return {
        "action": "AMEND_TRAILING_SL",
        "tool": "swap_amend_algo_order",
        "params": {
            "instId": sym,
            "algoId": algo_id,
            "newSlTriggerPx": str(candidate_sl),
            "newSlOrdPx": "-1",
            "simulatedTrading": simulated_trading,
        },
    }


def evaluate_realized_slippage(
    entry: float,
    sl: float,
    exit_px: float,
    side: str,
    top5_depth_usd: float | None = None,
    notional_usd: float | None = None,
) -> dict:
    """Detect >1.5R gap slippage or in-position liquidity collapse (Playbook 4)."""
    one_r = abs(entry - sl)
    if one_r <= 0:
        raise ValueError("entry and sl must differ")
    side_l = side.strip().lower()
    realized_r = ((exit_px - entry) / one_r) if side_l == "buy" else ((entry - exit_px) / one_r)
    gap_slippage_breach = realized_r < -1.5
    liquidity_collapse = (
        top5_depth_usd is not None
        and notional_usd is not None
        and float(top5_depth_usd) < float(notional_usd)
    )
    return {
        "realized_r": round(realized_r, 4),
        "gap_slippage_breach": gap_slippage_breach,
        "liquidity_collapse": liquidity_collapse,
        "halt_instrument": gap_slippage_breach or liquidity_collapse,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Live State vs Risk Ledger Reconciler")
    parser.add_argument("--input", required=True, help="JSON snapshot with positions, open_orders, algo_orders")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--live", action="store_true", help="Set simulatedTrading=false")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        snap = json.load(f)

    result = reconcile_state(
        positions=snap.get("positions", []),
        open_orders=snap.get("open_orders", []),
        algo_orders=snap.get("algo_orders", []),
        ledger_path=args.ledger,
        simulated_trading=not args.live,
        default_sl_by_inst=snap.get("default_sl_by_inst"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
