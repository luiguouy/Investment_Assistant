"""OKX multi-timeframe candle adapter, pa_engine runner, and four-way consistency gate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import pa_engine as pe
from data_sources import http_get
from okx_agent.universe import DEFAULT_UNIVERSE_PATH, get_instrument_policy

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"


def parse_okx_candles(rows: list, drop_unconfirmed: bool = True) -> pd.DataFrame:
    """Parse OKX raw candle rows [ts, o, h, l, c, vol, ...] into an ascending OHLCV DataFrame."""
    if not rows:
        raise ValueError("Empty candle rows")

    records = []
    for row in rows:
        if len(row) < 6:
            raise ValueError(f"Malformed OKX candle row: {row}")
        if drop_unconfirmed and len(row) >= 9 and str(row[8]) == "0":
            continue
        ts_ms = int(row[0])
        o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        if h < max(o, c) or l > min(o, c):
            raise ValueError(f"Corrupted OHLC at ts={ts_ms}: o={o}, h={h}, l={l}, c={c}")
        dt = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_localize(None)
        records.append({
            "Date": dt,
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v,
        })

    if not records:
        raise ValueError("No confirmed candles remain after filtering")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_okx_candles(
    inst_id: str,
    bar: str = "1H",
    limit: int = 200,
    demo: bool = False,
    timeout: int = 15,
) -> pd.DataFrame:
    """Fetch public OKX candles with pagination up to `limit` bars."""
    collected: list = []
    after: str | None = None
    remaining = max(1, int(limit))

    while remaining > 0:
        page_limit = min(100, remaining)
        url = f"{OKX_CANDLES_URL}?instId={inst_id}&bar={bar}&limit={page_limit}"
        if after is not None:
            url += f"&after={after}"
        txt, meta = http_get(url, timeout=timeout)
        if not txt:
            raise RuntimeError(f"Failed to fetch OKX candles for {inst_id} ({bar}): {meta}")
        payload = json.loads(txt)
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX API error for {inst_id} ({bar}): {payload}")
        batch = payload.get("data") or []
        if not batch:
            break
        collected.extend(batch)
        remaining -= len(batch)
        if len(batch) < page_limit:
            break
        after = str(batch[-1][0])

    return parse_okx_candles(collected, drop_unconfirmed=True)


def _mss_direction(mss_latest: dict | None) -> int:
    if not mss_latest:
        return 0
    t = str(mss_latest.get("type", ""))
    if "Bullish MSS" in t:
        return 1
    if "Bearish MSS" in t:
        return -1
    return 0


def _channel_direction(channel: dict) -> int:
    ctype = str(channel.get("channel_type", "horizontal")).lower()
    if ctype == "ascending":
        return 1
    if ctype == "descending":
        return -1
    return 0


def _sanitize_swings(swings: list, tail: int = 6) -> list[dict]:
    out = []
    for s in swings[-tail:]:
        t = s.get("time")
        out.append({
            "type": s.get("type"),
            "label": s.get("label", s.get("type")),
            "price": round(float(s["price"]), 6),
            "time": str(t) if t is not None else None,
        })
    return out


def analyze_timeframe(
    df: pd.DataFrame,
    swing_order: int = 5,
    channel_lookback: int = 30,
) -> dict:
    """Run pa_engine structure, MSS, channel, box, and naked-K trigger analysis on one timeframe."""
    swings = pe.extract_swings(df, order=swing_order, confirmed_only=True)
    bias = int(pe.dow_bias(swings, df=df))
    dow_phase = "bullish" if bias > 0 else ("bearish" if bias < 0 else "neutral")

    mss_events = pe.detect_mss(swings, df=df)
    mss_latest = mss_events[-1] if mss_events else None
    mss_dir = _mss_direction(mss_latest)

    channel = pe.detect_price_channel(df, swings=swings, lookback_bars=min(channel_lookback, len(df)))
    channel_dir = _channel_direction(channel)

    box = pe.detect_consolidation_box(df, swings=swings, lookback_bars=min(40, len(df)))
    bar_rng = float(pe.median_bar_range(df))

    key_levels = [
        {"price": float(s["price"]), "label": s.get("label", s["type"]), "tier": 3}
        for s in swings[-6:]
    ]
    pa_signals = pe.scan_price_action(df, key_levels=key_levels, bar_rng=bar_rng)

    return {
        "bars": int(len(df)),
        "last_ts": df.index[-1].isoformat(),
        "last_close": float(df["Close"].iloc[-1]),
        "bar_range": round(bar_rng, 6),
        "dow_phase": dow_phase,
        "dow_bias": bias,
        "mss_dir": mss_dir,
        "mss_latest": mss_latest,
        "channel_dir": channel_dir,
        "channel": channel,
        "box": box,
        "recent_swings": _sanitize_swings(swings),
        "pa_signals": pa_signals,
    }


def check_four_way_consistency(
    htf: dict,
    mtf: dict,
    perp_type: str = "crypto",
    external_pa_bias: int | None = None,
) -> dict:
    """Enforce AGENTS.md Section 3 Four-Way Consistency Check across HTF/MTF/MSS/Channel."""
    htf_bias = int(htf.get("dow_bias", 0))
    mtf_bias = int(mtf.get("dow_bias", 0))
    mtf_mss_dir = int(mtf.get("mss_dir", 0))
    mtf_channel_dir = int(mtf.get("channel_dir", 0))

    signals = [
        ("htf_bias", htf_bias),
        ("mtf_bias", mtf_bias),
        ("mtf_mss", mtf_mss_dir),
        ("mtf_channel", mtf_channel_dir),
    ]
    positives = [name for name, val in signals if val > 0]
    negatives = [name for name, val in signals if val < 0]

    conflicts: list[dict[str, Any]] = []
    if positives and negatives:
        for p_name in positives:
            for n_name in negatives:
                conflicts.append({
                    "code": "FOUR_WAY_DIRECTION_CONFLICT",
                    "detail": f"{p_name}=+1 opposes {n_name}=-1",
                })

    okx_dir = 1 if positives and not negatives else (-1 if negatives and not positives else 0)
    if external_pa_bias is not None and int(external_pa_bias) != 0 and okx_dir != 0:
        if int(external_pa_bias) != okx_dir:
            conflicts.append({
                "code": "EXTERNAL_PA_ENGINE_CONFLICT",
                "detail": f"OKX live structure ({okx_dir:+d}) opposes external pa_engine bias ({int(external_pa_bias):+d}) for {perp_type}",
            })

    if conflicts:
        return {
            "verdict": "CONFLICT_HALT",
            "allowed_direction": "NONE",
            "conflicts": conflicts,
        }

    if htf_bias == 0 and mtf_bias == 0:
        return {
            "verdict": "NEUTRAL_WAIT",
            "allowed_direction": "NONE",
            "conflicts": [],
        }

    allowed = "LONG" if okx_dir > 0 else ("SHORT" if okx_dir < 0 else "NONE")
    return {
        "verdict": "ALLOW" if allowed != "NONE" else "NEUTRAL_WAIT",
        "allowed_direction": allowed,
        "conflicts": [],
    }


def analyze_okx_market(
    inst_id: str,
    htf_df: pd.DataFrame,
    mtf_df: pd.DataFrame,
    ltf_df: pd.DataFrame | None = None,
    external_pa_bias: int | None = None,
    swing_order: int = 5,
    universe_path: str | Path = DEFAULT_UNIVERSE_PATH,
) -> dict:
    """Analyze multi-timeframe OKX candles and run mandatory Four-Way Consistency Check."""
    sym = inst_id.strip().upper()
    policy = get_instrument_policy(sym, universe_path=universe_path)

    htf = analyze_timeframe(htf_df, swing_order=swing_order)
    mtf = analyze_timeframe(mtf_df, swing_order=swing_order)
    ltf = analyze_timeframe(ltf_df, swing_order=swing_order) if ltf_df is not None else None

    consistency = check_four_way_consistency(
        htf,
        mtf,
        perp_type=policy["perp_type"],
        external_pa_bias=external_pa_bias,
    )

    entry_triggers = (ltf["pa_signals"] if ltf is not None else []) + mtf["pa_signals"]

    return {
        "inst_id": sym,
        "perp_type": policy["perp_type"],
        "bucket": policy["bucket"],
        "verdict": consistency["verdict"],
        "allowed_direction": consistency["allowed_direction"],
        "conflicts": consistency["conflicts"],
        "htf": htf,
        "mtf": mtf,
        "ltf": ltf,
        "entry_triggers": entry_triggers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Multi-Timeframe Market Analyzer & Consistency Gate")
    parser.add_argument("--input", help="Offline JSON file with inst_id, htf_candles, mtf_candles, [ltf_candles]")
    parser.add_argument("--inst-id", help="OKX instrument ID, e.g. BTC-USDT-SWAP")
    parser.add_argument("--htf", default="4H", help="Higher timeframe bar (default 4H)")
    parser.add_argument("--mtf", default="1H", help="Medium/execution timeframe bar (default 1H)")
    parser.add_argument("--ltf", default=None, help="Optional lower timeframe bar, e.g. 15m")
    parser.add_argument("--limit", type=int, default=200, help="Candles per timeframe (default 200)")
    parser.add_argument("--swing-order", type=int, default=5, help="Swing order (default 5)")
    parser.add_argument("--external-pa-bias", type=int, default=None, choices=[-1, 0, 1])
    args = parser.parse_args()

    if args.input:
        with Path(args.input).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        inst_id = payload["inst_id"]
        htf_df = parse_okx_candles(payload["htf_candles"], drop_unconfirmed=True)
        mtf_df = parse_okx_candles(payload["mtf_candles"], drop_unconfirmed=True)
        ltf_df = parse_okx_candles(payload["ltf_candles"], drop_unconfirmed=True) if payload.get("ltf_candles") else None
        ext_bias = payload.get("external_pa_bias", args.external_pa_bias)
        swing_order = int(payload.get("swing_order", args.swing_order))
    elif args.inst_id:
        inst_id = args.inst_id
        htf_df = fetch_okx_candles(inst_id, bar=args.htf, limit=args.limit)
        mtf_df = fetch_okx_candles(inst_id, bar=args.mtf, limit=args.limit)
        ltf_df = fetch_okx_candles(inst_id, bar=args.ltf, limit=args.limit) if args.ltf else None
        ext_bias = args.external_pa_bias
        swing_order = args.swing_order
    else:
        parser.error("Provide either --input or --inst-id")
        return 1

    result = analyze_okx_market(
        inst_id=inst_id,
        htf_df=htf_df,
        mtf_df=mtf_df,
        ltf_df=ltf_df,
        external_pa_bias=ext_bias,
        swing_order=swing_order,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result["verdict"] == "CONFLICT_HALT" else 0


if __name__ == "__main__":
    sys.exit(main())
