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
        "AAPL-USDT-SWAP", "NVDA-USDT-SWAP", "TSLA-USDT-SWAP", "GOOGL-USDT-SWAP",
        "QQQ-USDT-SWAP", "SPY-USDT-SWAP", "AMD-USDT-SWAP", "MU-USDT-SWAP", "SOXX-USDT-SWAP",
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
        "rolling_loss_r": 0.0,
        "cumulative_r": 0.0,
        "peak_r": 0.0,
        "drawdown_r": 0.0,
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
    side: str | None = None,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    """Enforce single-trade cap, portfolio open/net-directional risk, correlation bucket, daily loss, drawdown, and cooldown."""
    state = load_ledger(ledger_path)
    consec = int(state.get("consecutive_losses", 0))
    rolling_loss_r = float(state.get("rolling_loss_r", 0.0))
    drawdown_r = float(state.get("drawdown_r", 0.0))
    max_single_risk = 0.005 if (consec >= 3 or rolling_loss_r <= -2.5) else 0.01

    if drawdown_r <= -5.0:
        return {
            "passed": False,
            "reason": "MAX_DRAWDOWN_CIRCUIT_BREAKER",
            "detail": f"Account drawdown_r={drawdown_r:.2f} breached -5.0R HWM limit; manual review required",
        }

    today = state.get("daily_stats", {}).get(_today_str(), {"pnl_pct": 0.0, "pnl_r": 0.0})
    if float(today.get("pnl_pct", 0.0)) <= -0.02 or float(today.get("pnl_r", 0.0)) <= -2.5:
        return {
            "passed": False,
            "reason": "DAILY_LOSS_CIRCUIT_BREAKER",
            "detail": f"Today pnl_pct={today.get('pnl_pct', 0.0):.4f}, pnl_r={today.get('pnl_r', 0.0):.2f} hit daily stop",
        }

    cooldown_until = state.get("cooldown_until")
    if cooldown_until:
        cd_dt = datetime.fromisoformat(cooldown_until)
        if _now_utc() < cd_dt:
            return {
                "passed": False,
                "reason": "CONSECUTIVE_LOSS_COOLDOWN",
                "detail": f"Active 24h cooldown until {cooldown_until} after {consec} consecutive losses (rolling_loss_r={rolling_loss_r:.2f})",
            }

    if risk_pct <= 0 or risk_pct > max_single_risk:
        return {
            "passed": False,
            "reason": "SINGLE_TRADE_RISK_CAP_EXCEEDED",
            "detail": f"risk_pct={risk_pct:.4f} exceeds max allowed {max_single_risk:.4f} (consecutive_losses={consec}, rolling_loss_r={rolling_loss_r:.2f})",
        }

    open_pos = state.get("open_positions", [])
    total_open_risk = sum(float(p.get("risk_pct", 0.0)) for p in open_pos)
    if total_open_risk + risk_pct > 0.03 + 1e-12:
        return {
            "passed": False,
            "reason": "TOTAL_OPEN_RISK_CAP_EXCEEDED",
            "detail": f"open_risk={total_open_risk:.4f} + new={risk_pct:.4f} > 0.0300",
        }

    norm_side = str(side or "buy").strip().lower()
    dir_sign = -1.0 if norm_side == "sell" else 1.0
    net_dir_risk = sum(
        float(p.get("risk_pct", 0.0)) * (-1.0 if str(p.get("side", "buy")).strip().lower() == "sell" else 1.0)
        for p in open_pos
    )
    projected_net_dir = abs(net_dir_risk + dir_sign * risk_pct)
    if projected_net_dir > 0.025 + 1e-12:
        return {
            "passed": False,
            "reason": "NET_DIRECTIONAL_RISK_CAP_EXCEEDED",
            "detail": f"|net_directional_risk|={projected_net_dir:.4f} > 0.0250",
        }

    bucket = get_bucket(symbol)
    bucket_pos = [p for p in open_pos if p.get("bucket") == bucket]
    same_dir_bucket_count = sum(
        1 for p in bucket_pos if str(p.get("side", "buy")).strip().lower() == norm_side
    )
    if same_dir_bucket_count >= 2 or len(bucket_pos) >= 3:
        return {
            "passed": False,
            "reason": "CORRELATION_BUCKET_CAP_EXCEEDED",
            "detail": f"bucket '{bucket}' has {same_dir_bucket_count} same-side ({norm_side}) / {len(bucket_pos)} total open positions",
        }

    return {
        "passed": True,
        "reason": "OK",
        "bucket": bucket,
        "total_open_risk_after": round(total_open_risk + risk_pct, 6),
        "net_directional_risk_after": round(net_dir_risk + dir_sign * risk_pct, 6),
        "bucket_count_after": len(bucket_pos) + 1,
    }


def record_open(
    symbol: str,
    risk_pct: float,
    entry: float,
    sl: float,
    venue: str = "okx_swap",
    side: str = "buy",
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    gate = check_pre_trade_risk(symbol, risk_pct, side=side, ledger_path=ledger_path)
    if not gate["passed"]:
        raise ValueError(f"Pre-trade risk gate blocked open: {gate['reason']} ({gate['detail']})")

    state = load_ledger(ledger_path)
    pos = {
        "symbol": symbol.strip().upper(),
        "bucket": get_bucket(symbol),
        "side": str(side).strip().lower(),
        "risk_pct": float(risk_pct),
        "entry": float(entry),
        "sl": float(sl),
        "venue": venue,
        "opened_at": _now_utc().isoformat(),
    }
    state["open_positions"].append(pos)
    save_ledger(state, ledger_path)
    return pos


def update_position_risk(
    symbol: str,
    new_risk_pct: float,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict:
    """Update open risk_pct for an active position (e.g., set to 0.0 after TP1 breakeven)."""
    if new_risk_pct < 0:
        raise ValueError("new_risk_pct must be >= 0")
    state = load_ledger(ledger_path)
    sym = symbol.strip().upper()
    updated = None
    for p in state.get("open_positions", []):
        if p.get("symbol") == sym:
            p["risk_pct"] = float(new_risk_pct)
            p["risk_updated_at"] = _now_utc().isoformat()
            updated = p
            break
    if updated is None:
        raise ValueError(f"No open position found in ledger for {sym}")
    save_ledger(state, ledger_path)
    return updated


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

    cum_r = round(float(state.get("cumulative_r", 0.0)) + float(pnl_r), 4)
    peak_r = max(float(state.get("peak_r", 0.0)), cum_r)
    state["cumulative_r"] = cum_r
    state["peak_r"] = round(peak_r, 4)
    state["drawdown_r"] = round(cum_r - peak_r, 4)

    if pnl_r < 0:
        state["rolling_loss_r"] = round(float(state.get("rolling_loss_r", 0.0)) + float(pnl_r), 4)
        if pnl_r <= -0.5:
            state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
        if state["consecutive_losses"] >= 3 or state["rolling_loss_r"] <= -2.5:
            state["cooldown_until"] = (_now_utc() + timedelta(hours=24)).isoformat()
    elif pnl_r > 0:
        state["consecutive_losses"] = 0
        state["rolling_loss_r"] = 0.0
        state["cooldown_until"] = None

    trade = {
        "symbol": sym,
        "pnl_r": float(pnl_r),
        "pnl_pct": float(pnl_pct),
        "closed_at": _now_utc().isoformat(),
    }
    state["closed_trades"].append(trade)
    save_ledger(state, ledger_path)
    return {
        "trade": trade,
        "daily": daily,
        "consecutive_losses": state["consecutive_losses"],
        "rolling_loss_r": state["rolling_loss_r"],
        "drawdown_r": state["drawdown_r"],
        "cooldown_until": state["cooldown_until"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio Risk Ledger & Circuit Breakers")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Path to ledger JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_chk = sub.add_parser("check")
    p_chk.add_argument("--symbol", required=True)
    p_chk.add_argument("--risk-pct", type=float, required=True)
    p_chk.add_argument("--side", default="buy", choices=["buy", "sell"])

    p_open = sub.add_parser("open")
    p_open.add_argument("--symbol", required=True)
    p_open.add_argument("--risk-pct", type=float, required=True)
    p_open.add_argument("--entry", type=float, required=True)
    p_open.add_argument("--sl", type=float, required=True)
    p_open.add_argument("--side", default="buy", choices=["buy", "sell"])
    p_open.add_argument("--venue", default="okx_swap")

    p_cls = sub.add_parser("close")
    p_cls.add_argument("--symbol", required=True)
    p_cls.add_argument("--pnl-r", type=float, required=True)
    p_cls.add_argument("--pnl-pct", type=float, required=True)

    p_upd = sub.add_parser("update-risk")
    p_upd.add_argument("--symbol", required=True)
    p_upd.add_argument("--risk-pct", type=float, required=True)

    args = parser.parse_args()
    if args.cmd == "status":
        print(json.dumps(load_ledger(args.ledger), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "check":
        res = check_pre_trade_risk(args.symbol, args.risk_pct, side=args.side, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["passed"] else 1
    if args.cmd == "open":
        res = record_open(args.symbol, args.risk_pct, args.entry, args.sl, venue=args.venue, side=args.side, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "close":
        res = record_close(args.symbol, args.pnl_r, args.pnl_pct, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "update-risk":
        res = update_position_risk(args.symbol, args.risk_pct, ledger_path=args.ledger)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
