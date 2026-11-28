"""Zero-context 15s background price watcher with hysteresis and circuit breaker."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Callable

from data_sources import http_get

OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"


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


def fetch_okx_quote_with_1m(inst_id: str, timeout: int = 10) -> dict[str, float]:
    """Fetch last price plus latest 1m candle high/low to prevent inter-poll wick misses."""
    sym = inst_id.strip().upper()
    last = fetch_okx_ticker_price(sym, timeout=timeout)
    high = last
    low = last
    try:
        c_txt, _ = http_get(f"{OKX_CANDLES_URL}?instId={sym}&bar=1m&limit=1", timeout=timeout)
        if c_txt:
            c_payload = json.loads(c_txt)
            if str(c_payload.get("code")) == "0" and c_payload.get("data"):
                row = c_payload["data"][0]
                high = max(last, float(row[2]))
                low = min(last, float(row[3]))
    except Exception:
        pass
    return {"last": last, "high": high, "low": low}


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
    high_px: float | None = None,
    low_px: float | None = None,
    trail_milestone_px: float | None = None,
) -> dict:
    """Check whether price (or 1m high/low wick) entered alert zones with hysteresis."""
    side_l = side.strip().lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}'; expected 'buy' or 'sell'")

    hi = max(price, high_px) if high_px is not None else price
    lo = min(price, low_px) if low_px is not None else price

    zones: list[tuple[str, float, bool]] = []
    rearm_outside_all = True
    clear_pct = buffer_pct + hysteresis_pct

    if side_l == "buy":
        if watch_entry and entry is not None:
            thresh = entry * (1.0 + buffer_pct)
            zones.append(("LIMIT_ENTRY", thresh, lo <= thresh))
            if lo <= entry * (1.0 + clear_pct):
                rearm_outside_all = False
        if tp1 is not None:
            thresh = tp1 * (1.0 - buffer_pct)
            zones.append(("TP1_BE", thresh, hi >= thresh))
            if hi >= tp1 * (1.0 - clear_pct):
                rearm_outside_all = False
        if trail_milestone_px is not None:
            zones.append(("TRAIL_MILESTONE", trail_milestone_px, hi >= trail_milestone_px))
            if hi >= trail_milestone_px * (1.0 - hysteresis_pct):
                rearm_outside_all = False
        if sl is not None:
            thresh = sl * (1.0 + buffer_pct)
            zones.append(("SL_NEAR", thresh, lo <= thresh))
            if lo <= sl * (1.0 + clear_pct):
                rearm_outside_all = False
    else:
        if watch_entry and entry is not None:
            thresh = entry * (1.0 - buffer_pct)
            zones.append(("LIMIT_ENTRY", thresh, hi >= thresh))
            if hi >= entry * (1.0 - clear_pct):
                rearm_outside_all = False
        if tp1 is not None:
            thresh = tp1 * (1.0 + buffer_pct)
            zones.append(("TP1_BE", thresh, lo <= thresh))
            if lo <= tp1 * (1.0 + clear_pct):
                rearm_outside_all = False
        if trail_milestone_px is not None:
            zones.append(("TRAIL_MILESTONE", trail_milestone_px, lo <= trail_milestone_px))
            if lo <= trail_milestone_px * (1.0 + hysteresis_pct):
                rearm_outside_all = False
        if sl is not None:
            thresh = sl * (1.0 - buffer_pct)
            zones.append(("SL_NEAR", thresh, hi >= thresh))
            if hi >= sl * (1.0 - clear_pct):
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
    max_hold_sec: float = 0.0,
    trail_milestone_px: float | None = None,
    price_fetcher: Callable[[str], float | dict[str, float]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    clock_fn: Callable[[], float] | None = None,
) -> dict:
    """Poll quote + 1m extremes until an alert fires, time-stop hits, max_polls is hit, or errors trip circuit breaker."""
    fetcher = price_fetcher or fetch_okx_quote_with_1m
    sleeper = sleep_fn or time.sleep
    clocker = clock_fn or time.monotonic
    sym = inst_id.strip().upper()
    armed = start_armed
    consecutive_errors = 0
    polls = 0
    started_at = clocker()

    while True:
        if max_hold_sec > 0:
            elapsed_sec = clocker() - started_at
            if elapsed_sec >= max_hold_sec:
                return {
                    "event": "TIME_STOP_REVIEW",
                    "exit_code": 0,
                    "inst_id": sym,
                    "side": side,
                    "elapsed_sec": round(elapsed_sec, 2),
                    "polls": polls,
                    "timestamp_utc": _now_utc_iso(),
                }

        polls += 1
        try:
            quote = fetcher(sym)
            if isinstance(quote, dict):
                price = float(quote["last"])
                high_px = float(quote.get("high", price))
                low_px = float(quote.get("low", price))
            else:
                price = float(quote)
                high_px = price
                low_px = price
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
            high_px=high_px,
            low_px=low_px,
            trail_milestone_px=trail_milestone_px,
        )
        armed = res["next_armed"]

        if res["triggered"]:
            return {
                "event": res["event"],
                "exit_code": 0,
                "inst_id": sym,
                "side": side,
                "price": price,
                "high_px": high_px,
                "low_px": low_px,
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
    parser.add_argument("--trail-milestone-px", type=float, default=None)
    parser.add_argument("--buffer-pct", type=float, default=0.002)
    parser.add_argument("--hysteresis-pct", type=float, default=0.001)
    parser.add_argument("--watch-entry", action="store_true")
    parser.add_argument("--disarmed", action="store_true", help="Start disarmed to require hysteresis exit first")
    parser.add_argument("--interval-sec", type=float, default=15.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument("--max-polls", type=int, default=0)
    parser.add_argument("--max-hold-hours", type=float, default=0.0, help="Trigger TIME_STOP_REVIEW after N hours")
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
        max_hold_sec=max(0.0, float(args.max_hold_hours) * 3600.0),
        trail_milestone_px=args.trail_milestone_px,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
