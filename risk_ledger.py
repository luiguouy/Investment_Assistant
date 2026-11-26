"""Persistent portfolio risk ledger, correlation buckets, and circuit breakers."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent / "data" / "risk_ledger.json"

CORRELATION_BUCKETS = {
    "us_mega_tech": {
        "AAPL", "NVDA", "MSFT", "META", "GOOGL", "GOOG", "AMZN", "TSLA",
        "QQQ", "SPY", "SMH", "AVGO", "AMD", "NFLX",
        "AAPL-USDT-SWAP", "NVDA-USDT-SWAP", "TSLA-USDT-SWAP",
    },
    "crypto_majors": {
        "BTC", "ETH", "SOL", "BTC-USD", "ETH-USD", "SOL-USD",
        "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "MSTR", "COIN",
    },
    "us_value_industrial": {
        "DIA", "CAT", "BA", "GS", "JPM", "BAC", "XOM", "CVX", "UNH", "JNJ", "WMT",
    },
    "us_small_cap": {"IWM"},
}


def get_bucket(symbol: str) -> str:
    """Return canonical correlation bucket name for a ticker or perp symbol."""
    sym = symbol.strip().upper()
    for bucket, members in CORRELATION_BUCKETS.items():
        if sym in members:
            return bucket
    base = sym.replace("-USDT-SWAP", "").replace("-USD", "")
    for bucket, members in CORRELATION_BUCKETS.items():
        if base in members:
            return bucket
    return f"other:{base}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _default_state() -> dict:
    return {
        "open_positions": [],
        "closed_trades": [],
        "daily_stats": {},
        "consecutive_losses": 0,
        "cooldown_until": None,
    }


def load_ledger(ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> dict:
    path = Path(ledger_path)
    if not path.exists():
        return _default_state()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    state = _default_state()
    state.update(data)
    return state


def save_ledger(state: dict, ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def check_pre_trade_risk(
    symbol: str,
    risk_pct: float,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    """Enforce single-trade cap, portfolio open risk, correlation bucket, daily loss, and cooldown."""
    state = load_ledger(ledger_path)
    consec = int(state.get("consecutive_losses", 0))
    max_single_risk = 0.005 if consec >= 3 else 0.01

    if risk_pct <= 0 or risk_pct > max_single_risk:
        return {
            "passed": False,
            "reason": "SINGLE_TRADE_RISK_CAP_EXCEEDED",
            "detail": f"risk_pct={risk_pct:.4f} exceeds max allowed {max_single_risk:.4f} (consecutive_losses={consec})",
        }

    cooldown_until = state.get("cooldown_until")
    if cooldown_until:
        cd_dt = datetime.fromisoformat(cooldown_until)
        if _now_utc() < cd_dt:
            return {
                "passed": False,
                "reason": "CONSECUTIVE_LOSS_COOLDOWN",
                "detail": f"Active 24h cooldown until {cooldown_until} after {consec} consecutive losses",
            }

    today = state.get("daily_stats", {}).get(_today_str(), {"pnl_pct": 0.0, "pnl_r": 0.0})
    if float(today.get("pnl_pct", 0.0)) <= -0.02 or float(today.get("pnl_r", 0.0)) <= -2.5:
        return {
            "passed": False,
            "reason": "DAILY_LOSS_CIRCUIT_BREAKER",
            "detail": f"Today pnl_pct={today.get('pnl_pct', 0.0):.4f}, pnl_r={today.get('pnl_r', 0.0):.2f} hit daily stop",
        }

    open_pos = state.get("open_positions", [])
    total_open_risk = sum(float(p.get("risk_pct", 0.0)) for p in open_pos)
    if total_open_risk + risk_pct > 0.03 + 1e-12:
        return {
            "passed": False,
            "reason": "TOTAL_OPEN_RISK_CAP_EXCEEDED",
            "detail": f"open_risk={total_open_risk:.4f} + new={risk_pct:.4f} > 0.0300",
        }

    bucket = get_bucket(symbol)
    bucket_count = sum(1 for p in open_pos if p.get("bucket") == bucket)
    if bucket_count >= 2:
        return {
            "passed": False,
            "reason": "CORRELATION_BUCKET_CAP_EXCEEDED",
            "detail": f"bucket '{bucket}' already has {bucket_count} open positions (max 2)",
        }

    return {
        "passed": True,
        "reason": "OK",
        "bucket": bucket,
        "total_open_risk_after": round(total_open_risk + risk_pct, 6),
        "bucket_count_after": bucket_count + 1,
    }


def record_open(
    symbol: str,
    risk_pct: float,
    entry: float,
    sl: float,
    venue: str = "okx_swap",
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    gate = check_pre_trade_risk(symbol, risk_pct, ledger_path=ledger_path)
    if not gate["passed"]:
        raise ValueError(f"Pre-trade risk gate blocked open: {gate['reason']} ({gate['detail']})")

    state = load_ledger(ledger_path)
    pos = {
        "symbol": symbol.strip().upper(),
        "bucket": get_bucket(symbol),
        "risk_pct": float(risk_pct),
        "entry": float(entry),
        "sl": float(sl),
        "venue": venue,
        "opened_at": _now_utc().isoformat(),
    }
    state["open_positions"].append(pos)
    save_ledger(state, ledger_path)
    return pos


def record_close(
    symbol: str,
    pnl_r: float,
    pnl_pct: float,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    state = load_ledger(ledger_path)
    sym = symbol.strip().upper()
    removed = False
    remaining = []
    for p in state["open_positions"]:
        if not removed and p.get("symbol") == sym:
            removed = True
            continue
        remaining.append(p)
    state["open_positions"] = remaining

    day = _today_str()
    daily = state.setdefault("daily_stats", {}).setdefault(day, {"pnl_r": 0.0, "pnl_pct": 0.0, "trades": 0})
    daily["pnl_r"] = round(float(daily["pnl_r"]) + float(pnl_r), 4)
    daily["pnl_pct"] = round(float(daily["pnl_pct"]) + float(pnl_pct), 6)
    daily["trades"] = int(daily["trades"]) + 1

    if pnl_r <= -0.5:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
        if state["consecutive_losses"] >= 3:
            state["cooldown_until"] = (_now_utc() + timedelta(hours=24)).isoformat()
    elif pnl_r > 0:
        state["consecutive_losses"] = 0
        state["cooldown_until"] = None

    trade = {
        "symbol": sym,
        "pnl_r": float(pnl_r),
        "pnl_pct": float(pnl_pct),
        "closed_at": _now_utc().isoformat(),
    }
    state["closed_trades"].append(trade)
    save_ledger(state, ledger_path)
    return {"trade": trade, "daily": daily, "consecutive_losses": state["consecutive_losses"], "cooldown_until": state["cooldown_until"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio Risk Ledger & Circuit Breakers")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Path to ledger JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_chk = sub.add_parser("check")
    p_chk.add_argument("--symbol", required=True)
    p_chk.add_argument("--risk-pct", type=float, required=True)

    p_open = sub.add_parser("open")
    p_open.add_argument("--symbol", required=True)
    p_open.add_argument("--risk-pct", type=float, required=True)
    p_open.add_argument("--entry", type=float, required=True)
    p_open.add_argument("--sl", type=float, required=True)
    p_open.add_argument("--venue", default="okx_swap")

    p_cls = sub.add_parser("close")
    p_cls.add_argument("--symbol", required=True)
    p_cls.add_argument("--pnl-r", type=float, required=True)
    p_cls.add_argument("--pnl-pct", type=float, required=True)

    args = parser.parse_args()
    if args.cmd == "status":
        print(json.dumps(load_ledger(args.ledger), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "check":
        res = check_pre_trade_risk(args.symbol, args.risk_pct, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["passed"] else 1
    if args.cmd == "open":
        res = record_open(args.symbol, args.risk_pct, args.entry, args.sl, venue=args.venue, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "close":
        res = record_close(args.symbol, args.pnl_r, args.pnl_pct, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
