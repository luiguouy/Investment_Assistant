"""Performance attribution and journal audit report for okx_agent + risk_ledger."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from okx_agent.journal import DEFAULT_JOURNAL_PATH, read_journal
from risk_ledger import DEFAULT_LEDGER_PATH, load_ledger


def generate_performance_report(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    journal_path: str | Path = DEFAULT_JOURNAL_PATH,
) -> dict[str, Any]:
    """Compute expectancy, win rate, drawdown, per-symbol attribution, and journal audit stats."""
    ledger = load_ledger(ledger_path)
    closed = ledger.get("closed_trades", [])

    total_trades = len(closed)
    wins = [float(t["pnl_r"]) for t in closed if float(t.get("pnl_r", 0.0)) > 0]
    losses = [float(t["pnl_r"]) for t in closed if float(t.get("pnl_r", 0.0)) < 0]
    scratches = total_trades - len(wins) - len(losses)

    win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
    avg_win_r = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_r = (sum(losses) / len(losses)) if losses else 0.0
    expectancy_r = (sum(float(t.get("pnl_r", 0.0)) for t in closed) / total_trades) if total_trades > 0 else 0.0

    gross_profit_r = sum(wins)
    gross_loss_r = abs(sum(losses))
    profit_factor = (gross_profit_r / gross_loss_r) if gross_loss_r > 0 else (float("inf") if gross_profit_r > 0 else 0.0)

    cum_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0
    by_symbol: dict[str, dict[str, Any]] = {}

    for t in closed:
        r = float(t.get("pnl_r", 0.0))
        cum_r += r
        peak_r = max(peak_r, cum_r)
        max_dd_r = min(max_dd_r, cum_r - peak_r)

        sym = str(t.get("symbol", "UNKNOWN")).upper()
        bucket = by_symbol.setdefault(sym, {"trades": 0, "wins": 0, "total_r": 0.0})
        bucket["trades"] += 1
        if r > 0:
            bucket["wins"] += 1
        bucket["total_r"] = round(bucket["total_r"] + r, 4)

    for sym, stats in by_symbol.items():
        n = stats["trades"]
        stats["win_rate"] = round(stats["wins"] / n, 4) if n > 0 else 0.0
        stats["expectancy_r"] = round(stats["total_r"] / n, 4) if n > 0 else 0.0

    journal_rows = read_journal(journal_path=journal_path)
    event_counts: dict[str, int] = {}
    slippage_deltas: list[float] = []
    for row in journal_rows:
        ev = str(row.get("event_type", "UNKNOWN"))
        event_counts[ev] = event_counts.get(ev, 0) + 1
        payload = row.get("payload") or {}
        if isinstance(payload, dict) and "planned_r" in payload and "realized_r" in payload:
            slippage_deltas.append(float(payload["realized_r"]) - float(payload["planned_r"]))

    avg_r_slippage = (sum(slippage_deltas) / len(slippage_deltas)) if slippage_deltas else None

    return {
        "summary": {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "scratches": scratches,
            "win_rate": round(win_rate, 4),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "expectancy_r": round(expectancy_r, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "cumulative_r": round(cum_r, 4),
            "peak_r": round(peak_r, 4),
            "max_drawdown_r": round(max_dd_r, 4),
            "current_drawdown_r": round(cum_r - peak_r, 4),
            "open_positions_count": len(ledger.get("open_positions", [])),
        },
        "by_symbol": by_symbol,
        "journal_audit": {
            "total_events": len(journal_rows),
            "event_counts": event_counts,
            "avg_r_slippage_vs_plan": round(avg_r_slippage, 4) if avg_r_slippage is not None else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Agent Performance & Journal Attribution Report")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH))
    args = parser.parse_args()

    report = generate_performance_report(ledger_path=args.ledger, journal_path=args.journal)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
