"""AI Agent trading infrastructure for OKX perpetual swaps."""

from okx_agent.journal import append_journal_entry, read_journal
from okx_agent.market import analyze_okx_market, check_four_way_consistency, parse_okx_candles
from okx_agent.preflight import evaluate_preflight
from okx_agent.reconcile import evaluate_realized_slippage, plan_breakeven_amend, reconcile_state
from okx_agent.universe import get_instrument_policy, load_universe
from okx_agent.watcher import evaluate_price_alerts, run_watcher

__all__ = [
    "analyze_okx_market",
    "append_journal_entry",
    "check_four_way_consistency",
    "evaluate_preflight",
    "evaluate_price_alerts",
    "evaluate_realized_slippage",
    "get_instrument_policy",
    "load_universe",
    "parse_okx_candles",
    "plan_breakeven_amend",
    "read_journal",
    "reconcile_state",
    "run_watcher",
]
