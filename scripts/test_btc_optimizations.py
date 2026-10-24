#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

import pa_engine as pe
import backtest as bt
import strategy_crypto as sc

def test_regime_and_blackswan():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    fwd = bt.Forward(df)

    records = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto", asset_type="crypto", ticker="BTC-USD")
    trades = records.drop_duplicates(subset=["t", "side", "entry_price"]).copy()
    trades["entry_date"] = pd.to_datetime(trades["date"])
    trades["exit_date"] = [df.index[min(t + hold, len(df)-1)] for t, hold in zip(trades["t"], trades["hold_bars"])]

    # Compute 200-day EMA and 50-day EMA
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # Black Swan windows:
    # 1. Covid 312: 2020-03-08 to 2020-03-25
    # 2. 519 Crash: 2021-05-15 to 2021-05-30
    # 3. Luna Collapse: 2022-05-05 to 2022-05-25
    # 4. FTX Collapse: 2022-11-05 to 2022-11-25
    black_swans = [
        ("Covid Crash", "2020-03-08", "2020-03-25"),
        ("519 Liquidation", "2021-05-15", "2021-05-30"),
        ("Luna/UST Collapse", "2022-05-05", "2022-05-25"),
        ("FTX Insolvency", "2022-11-05", "2022-11-25"),
    ]

    # Check if entry falls in black swan window
    is_bs_entry = pd.Series(False, index=trades.index)
    for name, s_d, e_d in black_swans:
        in_win = (trades["entry_date"] >= s_d) & (trades["entry_date"] <= e_d)
        is_bs_entry = is_bs_entry | in_win

    # Check if trade overlaps with black swan window
    is_bs_overlap = pd.Series(False, index=trades.index)
    for name, s_d, e_d in black_swans:
        in_win = (trades["entry_date"] <= e_d) & (trades["exit_date"] >= s_d)
        is_bs_overlap = is_bs_overlap | in_win

    # Regime check for each trade:
    # Bull regime: Close[entry_t-1] > EMA200[entry_t-1]
    trades["trend_aligned"] = False
    for idx, r in trades.iterrows():
        t = r["t"]
        side = r["side"]
        c = df["Close"].iloc[t - 1]
        ema200 = df["EMA200"].iloc[t - 1]
        if side == 1 and c > ema200:
            trades.loc[idx, "trend_aligned"] = True
        elif side == -1 and c < ema200:
            trades.loc[idx, "trend_aligned"] = True

    print("=" * 80)
    print("1. BASELINE (ALL 60 TRADES)")
    report(trades)

    print("=" * 80)
    print("2. EXCLUDE BLACK SWAN ENTRY (No new trades initiated during black swan windows)")
    report(trades[~is_bs_entry])

    print("=" * 80)
    print("3. EXCLUDE BLACK SWAN OVERLAP (Strictly remove any trades active during black swan crashes)")
    report(trades[~is_bs_overlap])

    print("=" * 80)
    print("4. TREND-ALIGNED ONLY (Dow Primary Trend: Long only > EMA200, Short only < EMA200)")
    report(trades[trades["trend_aligned"]])

    print("=" * 80)
    print("5. TREND-ALIGNED + EXCLUDE BLACK SWAN OVERLAP")
    report(trades[trades["trend_aligned"] & ~is_bs_overlap])

    print("=" * 80)
    print("6. 2B SWEEP ONLY (Highest confidence trigger)")
    report(trades[trades["pattern"] == "2B_sweep"])

    print("=" * 80)
    print("7. TREND-ALIGNED + 2B SWEEP ONLY + EXCLUDE BLACK SWAN")
    report(trades[trades["trend_aligned"] & (trades["pattern"] == "2B_sweep") & ~is_bs_overlap])

def report(t_df):
    if len(t_df) == 0:
        print("No trades")
        return
    wins = t_df[t_df["pnl_pct"] > 0]
    losses = t_df[t_df["pnl_pct"] <= 0]
    wr = len(wins) / len(t_df)
    pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
    avg_r = t_df["r_multiple"].mean()
    exp = t_df["pnl_pct"].mean()
    longs = t_df[t_df["side"] == 1]
    shorts = t_df[t_df["side"] == -1]
    w_l = (longs["pnl_pct"] > 0).sum()
    w_s = (shorts["pnl_pct"] > 0).sum()
    wr_l = w_l / len(longs) if len(longs) > 0 else 0
    wr_s = w_s / len(shorts) if len(shorts) > 0 else 0

    print(f"Total: {len(t_df):2d} | WinRate: {wr:5.1%} ({len(wins)}/{len(t_df)}) | PF: {pf:4.2f} | "
          f"Avg_R: {avg_r:+5.2f}R | Exp: {exp:+6.2%} | "
          f"Longs: {len(longs)} (WR {wr_l:4.1%}) | Shorts: {len(shorts)} (WR {wr_s:4.1%})")

if __name__ == "__main__":
    test_regime_and_blackswan()
