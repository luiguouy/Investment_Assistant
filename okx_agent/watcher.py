"""Zero-context 15s background price watcher with hysteresis and circuit breaker."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Callable

from data_sources import http_get

OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_okx_ticker_price(inst_id: str, timeout: int = 10) -> float:
    """Fetch latest traded price (`last`) for inst_id from OKX public ticker."""
    url = f"{OKX_TICKER_URL}?instId={inst_id.strip().upper()}"
    txt, meta = http_get(url, timeout=timeout)
    if not txt:
        raise RuntimeError(f"Empty response fetching ticker for {inst_id}: {meta}")
    payload = json.loads(txt)
    if str(payload.get("code")) != "0" or not payload.get("data"):
        raise RuntimeError(f"OKX ticker API error for {inst_id}: {payload}")
    return float(payload["data"][0]["last"])


def evaluate_price_alerts(
    price: float,
    side: str,
    entry: float | None = None,
    tp1: float | None = None,
    sl: float | None = None,
    buffer_pct: float = 0.002,
    hysteresis_pct: float = 0.001,
    watch_entry: bool = False,
    armed: bool = True,
) -> dict:
    """Check whether price entered LIMIT_ENTRY, TP1_BE, or SL_NEAR alert zone with hysteresis."""
    side_l = side.strip().lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}'; expected 'buy' or 'sell'")

    zones: list[tuple[str, float, bool]] = []
    rearm_outside_all = True
    clear_pct = buffer_pct + hysteresis_pct

    if side_l == "buy":
        if watch_entry and entry is not None:
            thresh = entry * (1.0 + buffer_pct)
            zones.append(("LIMIT_ENTRY", thresh, price <= thresh))
            if price <= entry * (1.0 + clear_pct):
                rearm_outside_all = False
        if tp1 is not None:
            thresh = tp1 * (1.0 - buffer_pct)
            zones.append(("TP1_BE", thresh, price >= thresh))
            if price >= tp1 * (1.0 - clear_pct):
                rearm_outside_all = False
        if sl is not None:
            thresh = sl * (1.0 + buffer_pct)
            zones.append(("SL_NEAR", thresh, price <= thresh))
            if price <= sl * (1.0 + clear_pct):
                rearm_outside_all = False
    else:
        if watch_entry and entry is not None:
            thresh = entry * (1.0 - buffer_pct)
            zones.append(("LIMIT_ENTRY", thresh, price >= thresh))
            if price >= entry * (1.0 - clear_pct):
                rearm_outside_all = False
        if tp1 is not None:
            thresh = tp1 * (1.0 + buffer_pct)
            zones.append(("TP1_BE", thresh, price <= thresh))
            if price <= tp1 * (1.0 + clear_pct):
                rearm_outside_all = False
        if sl is not None:
            thresh = sl * (1.0 - buffer_pct)
            zones.append(("SL_NEAR", thresh, price >= thresh))
            if price >= sl * (1.0 - clear_pct):
                rearm_outside_all = False

    if not armed:
        return {
            "triggered": False,
            "event": None,
            "threshold": None,
            "next_armed": rearm_outside_all,
        }

    for event_name, thresh, hit in zones:
        if hit:
            return {
                "triggered": True,
                "event": event_name,
                "threshold": round(thresh, 6),
                "next_armed": False,
            }

    return {
        "triggered": False,
        "event": None,
        "threshold": None,
        "next_armed": True,
    }


def run_watcher(
    inst_id: str,
    side: str,
    entry: float | None = None,
    tp1: float | None = None,
    sl: float | None = None,
    buffer_pct: float = 0.002,
    hysteresis_pct: float = 0.001,
    watch_entry: bool = False,
    start_armed: bool = True,
    interval_sec: float = 15.0,
    max_consecutive_errors: int = 5,
    max_polls: int = 0,
    price_fetcher: Callable[[str], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict:
    """Poll public ticker until an alert fires, max_polls is hit, or network errors trip circuit breaker."""
    fetcher = price_fetcher or fetch_okx_ticker_price
    sleeper = sleep_fn or time.sleep
    sym = inst_id.strip().upper()
    armed = start_armed
    consecutive_errors = 0
    polls = 0

    while True:
        polls += 1
        try:
            price = float(fetcher(sym))
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                return {
                    "event": "WATCHER_ERROR_CIRCUIT_BREAKER",
                    "exit_code": 3,
                    "inst_id": sym,
                    "side": side,
                    "consecutive_errors": consecutive_errors,
                    "last_error": str(exc),
                    "polls": polls,
                    "timestamp_utc": _now_utc_iso(),
                }
            if max_polls > 0 and polls >= max_polls:
                return {
                    "event": "MAX_POLLS_REACHED",
                    "exit_code": 4,
                    "inst_id": sym,
                    "side": side,
                    "polls": polls,
                    "timestamp_utc": _now_utc_iso(),
                }
            sleeper(interval_sec)
            continue

        res = evaluate_price_alerts(
            price=price,
            side=side,
            entry=entry,
            tp1=tp1,
            sl=sl,
            buffer_pct=buffer_pct,
            hysteresis_pct=hysteresis_pct,
            watch_entry=watch_entry,
            armed=armed,
        )
        armed = res["next_armed"]

        if res["triggered"]:
            return {
                "event": res["event"],
                "exit_code": 0,
                "inst_id": sym,
                "side": side,
                "price": price,
                "threshold": res["threshold"],
                "polls": polls,
                "timestamp_utc": _now_utc_iso(),
            }

        if max_polls > 0 and polls >= max_polls:
            return {
                "event": "MAX_POLLS_REACHED",
                "exit_code": 4,
                "inst_id": sym,
                "side": side,
                "price": price,
                "polls": polls,
                "timestamp_utc": _now_utc_iso(),
            }

        sleeper(interval_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Zero-Context Background Price Watcher")
    parser.add_argument("--inst-id", required=True, help="e.g. BTC-USDT-SWAP")
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--entry", type=float, default=None)
    parser.add_argument("--tp1", type=float, default=None)
    parser.add_argument("--sl", type=float, default=None)
    parser.add_argument("--buffer-pct", type=float, default=0.002)
    parser.add_argument("--hysteresis-pct", type=float, default=0.001)
    parser.add_argument("--watch-entry", action="store_true")
    parser.add_argument("--disarmed", action="store_true", help="Start disarmed to require hysteresis exit first")
    parser.add_argument("--interval-sec", type=float, default=15.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument("--max-polls", type=int, default=0)
    args = parser.parse_args()

    result = run_watcher(
        inst_id=args.inst_id,
        side=args.side,
        entry=args.entry,
        tp1=args.tp1,
        sl=args.sl,
        buffer_pct=args.buffer_pct,
        hysteresis_pct=args.hysteresis_pct,
        watch_entry=args.watch_entry,
        start_armed=not args.disarmed,
        interval_sec=args.interval_sec,
        max_consecutive_errors=args.max_consecutive_errors,
        max_polls=args.max_polls,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
