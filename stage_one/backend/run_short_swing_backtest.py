#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单标的短波段回测诊断（适配每周约 1 次出手节奏）
  - 美股 (QQQ/GOOGL/MSFT): 1h 执行 + 日线 (1d) 因果顺势门禁
  - 加密 (BTC-USD):        4h 执行 + 日线 (1d) 因果顺势门禁
路径: stage_one/backend/run_short_swing_backtest.py

运行示例 (PowerShell):
  cd stage_one/backend
  python run_short_swing_backtest.py --tickers BTC-USD,QQQ,GOOGL,MSFT --days 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from strategy_core import pa_engine as pe
from strategy_core import strategy_crypto as sc
from run_diagnostic_backtest import _passes_box_channel_gate

# 短波段核心参数
SHORT_SWING_ORDER = 4
SHORT_MAX_WAIT_BARS = 12
SHORT_MAX_HOLD_BARS = 36
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 1.0


def fetch_short_swing_bars(ticker: str, days: int = 720) -> pd.DataFrame:
    """拉取 1h 复权 K 线；加密资产自动重采样为 4h。"""
    is_crypto = sc.is_crypto_asset(ticker)
    print(f"[数据] {ticker}: 拉取近 {days} 天 1h K线 ({'重采样为 4h' if is_crypto else '保持 1h'})...")
    df = yf.Ticker(ticker).history(period=f"{days}d", interval="1h", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    if is_crypto:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        df = df.resample("4h").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return df


def build_daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample("1D").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def check_daily_htf_trend_causal(df: pd.DataFrame, t: int, side: int, daily_df: pd.DataFrame) -> bool:
    """
    严格因果的日线 (1d) 高周期顺势门禁：
      - 仅使用 < 当天日期的已完整收盘日线 + 当天截至 t 时刻的局部聚合日线，零未来函数。
    """
    cutoff_ts = df.index[t]
    cutoff_day = cutoff_ts.normalize()

    completed = daily_df.loc[daily_df.index < cutoff_day]
    today_sub = df.iloc[:t + 1]
    today_bars = today_sub.loc[today_sub.index >= cutoff_day]

    if not today_bars.empty:
        today_row = pd.DataFrame([{
            "Open": float(today_bars["Open"].iloc[0]),
            "High": float(today_bars["High"].max()),
            "Low": float(today_bars["Low"].min()),
            "Close": float(today_bars["Close"].iloc[-1]),
            **({"Volume": float(today_bars["Volume"].sum())} if "Volume" in today_bars.columns else {}),
        }], index=[cutoff_day])
        causal_daily = pd.concat([completed, today_row])
    else:
        causal_daily = completed

    if len(causal_daily) < 25:
        return True

    d_swings = pe.extract_swings(causal_daily, order=3, confirmed_only=True, min_swing_k=1.5)
    bias = pe.dow_bias(d_swings, df=causal_daily, buffer_k=0.0)
    return bool(bias == side)


def run_short_swing_for_ticker(ticker: str, df: pd.DataFrame, use_htf_gate: bool = True) -> List[Dict[str, Any]]:
    is_crypto = sc.is_crypto_asset(ticker)
    profile = sc.ASSET_PROFILES["crypto" if is_crypto else "us_stock"]
    daily_df = build_daily_from_intraday(df)

    buffer_k = sc.CRYPTO_STOP_BUFFER_K if is_crypto else 1.0
    slippage_bps = 5.0 if is_crypto else 3.0
    commission_bps = 5.0 if is_crypto else 1.0
    funding_bps = 3.0 if is_crypto else 0.0

    n = len(df)
    warmup = 120
    trades: List[Dict[str, Any]] = []
    active_exit_t = -1
    last_setup_key = None

    for t in range(warmup, n - 2):
        if t <= active_exit_t:
            continue

        # 滑动窗口截断（取近 220 根保持因果与运行速度平衡）
        win_start = max(0, t - 220)
        sub = df.iloc[win_start:t + 1]
        swings = pe.extract_swings(sub, order=SHORT_SWING_ORDER, confirmed_only=True)
        if len(swings) < 3:
            continue

        side = pe.dow_bias(swings, df=sub, buffer_k=0.0)
        if side == 0:
            continue
        if profile.get("long_only", False) and side == -1:
            continue

        box = pe.detect_consolidation_box(sub, swings)
        chan = pe.detect_price_channel(sub, swings)
        if not _passes_box_channel_gate(side, box, chan):
            continue

        # 把相对索引还原为全局索引，避免重复触发同一结构
        for s in swings:
            s["index"] = int(s["index"]) + win_start
        anchor_sw = [s for s in swings if s["type"] == ("High" if side == 1 else "Low")]
        if not anchor_sw:
            continue
        setup_key = (side, anchor_sw[-1]["index"])
        if setup_key == last_setup_key:
            continue

        lvls = sc.extract_crypto_levels(df, t, side, swings, buffer_k=buffer_k)
        if not lvls:
            continue

        # 触发扫描（使用日线因果门禁替代周线门禁）
        end_k = min(t + SHORT_MAX_WAIT_BARS, n - 2)
        trig_hit = None
        for k in range(t + 1, end_k + 1):
            l_k, h_k = float(df["Low"].iloc[k]), float(df["High"].iloc[k])
            if (side == 1 and l_k <= lvls["structural_stop"]) or (side == -1 and h_k >= lvls["structural_stop"]):
                break

            trig = sc.detect_crypto_trigger_on_bar(df, k, side, lvls["swing_retest"], lvls["bar_rng"])
            if not trig.get("triggered"):
                continue
            if side == -1 and trig["pattern"] == "engulfing":
                continue
            if (not profile["allow_2b"]) and trig["pattern"] == "2B_sweep":
                continue
            if use_htf_gate and not check_daily_htf_trend_causal(df, k, side, daily_df):
                continue

            entry_t = k + 1
            entry_px = float(df["Open"].iloc[entry_t])
            tight_stop = float(trig["tight_stop"])
            if profile["stop_mode"] == "tight":
                final_stop = tight_stop
            else:
                final_stop = min(lvls["structural_stop"], tight_stop) if side == 1 else max(lvls["structural_stop"], tight_stop)

            if (side == 1 and entry_px <= final_stop) or (side == -1 and entry_px >= final_stop):
                continue
            risk = abs(entry_px - final_stop)
            if profile["max_stop_atr"] is not None and (risk / lvls["bar_rng"]) > profile["max_stop_atr"]:
                continue

            trig_hit = (entry_t, entry_px, final_stop, trig["pattern"])
            break

        if not trig_hit:
            continue

        entry_t, entry_px, stop_px, pat = trig_hit
        risk = abs(entry_px - stop_px)
        tp1_r = float(profile["tp1_r"])
        tp1_target = round(entry_px + side * tp1_r * risk, 2)

        res = sc.simulate_crypto_trade(
            df=df,
            entry_t=entry_t,
            entry_price=entry_px,
            side=side,
            stop_price=stop_px,
            tp1_price=tp1_target,
            tp2_price=lvls["tp2_target"],
            impulse_leg=lvls["impulse_leg"],
            max_hold_bars=SHORT_MAX_HOLD_BARS,
            min_rr=tp1_r,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            funding_rate_daily_bps=funding_bps,
        )
        last_setup_key = setup_key

        if res.get("status") != "completed":
            continue

        active_exit_t = int(res["exit_t"])
        stop_dist_pct = risk / entry_px
        pos_w = min(MAX_LEVERAGE, RISK_PER_TRADE / max(stop_dist_pct, 1e-6))
        hold_days = (df.index[res["exit_t"]] - df.index[res["entry_t"]]).total_seconds() / 86400.0
        # 跳空穿损检测：单笔净亏损超过 -1.15R（扣除摩擦后显著超额亏损）
        gap_slipped = bool(res["net_r"] < -1.15)

        trades.append({
            "ticker": ticker,
            "tf": "4h" if is_crypto else "1h",
            "htf_gate": use_htf_gate,
            "side": side,
            "pattern": pat,
            "entry_time": str(df.index[res["entry_t"]]),
            "exit_time": str(df.index[res["exit_t"]]),
            "stop_pct": round(stop_dist_pct * 100, 2),
            "pos_weight": round(pos_w, 2),
            "net_pnl_pct": float(res["net_pnl_pct"]),
            "sized_ret_pct": pos_w * float(res["net_pnl_pct"]),
            "net_r": float(res["net_r"]),
            "hold_days": round(hold_days, 2),
            "gap_slipped": gap_slipped,
            "exit_reason": res["exit_reason"],
            "win": bool(res["net_pnl_pct"] > 0),
        })

    return trades


def main() -> int:
    parser = argparse.ArgumentParser(description="单标的 1h/4h 短波段回测诊断")
    parser.add_argument("--tickers", type=str, default="BTC-USD,QQQ,GOOGL,MSFT")
    parser.add_argument("--days", type=int, default=720)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    all_trades: List[Dict[str, Any]] = []

    print("\n" + "=" * 108)
    print("单标的短波段诊断表（美股 1h / BTC 4h | 单笔 2% 等风险，上限 1x 现货）")
    print("=" * 108)
    print(f"{'标的':<9} {'周期':<4} {'日线门禁':<6} | {'总笔':>4} {'笔/月':>5} {'胜率':>6} {'净PF':>6} {'均R':>6} {'等险收益':>8} {'最大回撤':>8} {'均持天':>6} {'穿损%':>6}")
    print("-" * 108)

    for tk in tickers:
        df = fetch_short_swing_bars(tk, days=args.days)
        if len(df) < 200:
            print(f"[跳过] {tk}: 数据不足")
            continue
        months = max(1.0, (df.index[-1] - df.index[0]).days / 30.44)
        tf_lbl = "4h" if sc.is_crypto_asset(tk) else "1h"

        for use_gate in [True, False]:
            trs = run_short_swing_for_ticker(tk, df, use_htf_gate=use_gate)
            all_trades.extend(trs)
            if not trs:
                print(f"{tk:<9} {tf_lbl:<4} {('ON' if use_gate else 'OFF'):<6} | {'0':>4} {'0.0':>5} {'0.0%':>6} {'0.00':>6} {'+0.00':>6} {'+0.0%':>8} {'0.0%':>8} {'0.0':>6} {'0.0%':>6}")
                continue
            d = pd.DataFrame(trs)
            w = d.loc[d["sized_ret_pct"] > 0, "sized_ret_pct"].sum()
            l = abs(d.loc[d["sized_ret_pct"] <= 0, "sized_ret_pct"].sum())
            pf = float(w / l) if l > 0 else float("inf")
            eq = (1.0 + d["sized_ret_pct"]).cumprod()
            max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())
            cum_ret = float(eq.iloc[-1] - 1.0)
            pf_s = f"{pf:6.2f}" if np.isfinite(pf) else "   inf"
            print(
                f"{tk:<9} {tf_lbl:<4} {('ON' if use_gate else 'OFF'):<6} | {len(d):>4d} {len(d)/months:>5.1f} "
                f"{d['win'].mean()*100:>5.1f}% {pf_s} {d['net_r'].mean():>+6.2f} "
                f"{cum_ret*100:>+7.1f}% {max_dd*100:>7.1f}% {d['hold_days'].mean():>6.1f} {d['gap_slipped'].mean()*100:>5.1f}%"
            )
        print("-" * 108)

    if all_trades:
        out_path = BASE_DIR / "results/diagnostics/short_swing_trades.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_trades).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[导出] 短波段逐笔明细已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
