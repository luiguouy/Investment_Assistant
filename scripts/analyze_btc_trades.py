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

def analyze():
    df = bt.load_ticker("BTC-USD")
    print(f"Loaded BTC-USD: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    print(f"Total swing breakout events: {len(events)}")
    fwd = bt.Forward(df)

    records = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto", asset_type="crypto", ticker="BTC-USD")
    trades = records.drop_duplicates(subset=["t", "side", "entry_price"]).copy()
    print(f"Total strategy_crypto trades: {len(trades)}")

    # Add datetime
    trades["entry_date"] = pd.to_datetime(trades["date"])
    trades["exit_date"] = [df.index[min(t + hold, len(df)-1)] for t, hold in zip(trades["t"], trades["hold_bars"])]

    print("\n--- ALL TRADES ---")
    calc_stats(trades)

    print("\n--- LONGS (side=1) ---")
    longs = trades[trades["side"] == 1]
    calc_stats(longs)

    print("\n--- SHORTS (side=-1) ---")
    shorts = trades[trades["side"] == -1]
    calc_stats(shorts)

    # Let's inspect known black swans
    # 2020 Covid 312: 2020-03-08 ~ 2020-03-20
    # 2021 519 crash: 2021-05-15 ~ 2021-05-25
    # 2022 Luna collapse: 2022-05-05 ~ 2022-05-16
    # 2022 FTX collapse: 2022-11-05 ~ 2022-11-15
    black_swans = [
        ("Covid Crash (312)", "2020-03-08", "2020-03-22"),
        ("China Mining Ban / 519", "2021-05-15", "2021-05-25"),
        ("Luna / UST Implosion", "2022-05-05", "2022-05-18"),
        ("FTX Bankruptcy", "2022-11-05", "2022-11-18"),
    ]

    print("\n--- BLACK SWAN IMPACT CHECK ---")
    for name, s_d, e_d in black_swans:
        active = trades[
            (trades["entry_date"] <= e_d) & 
            (trades["exit_date"] >= s_d)
        ]
        print(f"\n{name} ({s_d} ~ {e_d}): {len(active)} active trades")
        for idx, r in active.iterrows():
            print(f"  Trade date={r['entry_date'].strftime('%Y-%m-%d')} side={'LONG' if r['side']==1 else 'SHORT'} "
                  f"entry={r['entry_price']} exit={r['exit_price']} pnl={r['pnl_pct']:+.2%} R={r['r_multiple']:+.2f}R "
                  f"exit_reason={r['exit_reason']} hold={r['hold_bars']}")

    # Let's check performance without black swan active trades
    is_bs = pd.Series(False, index=trades.index)
    for name, s_d, e_d in black_swans:
        in_bs = (trades["entry_date"] <= e_d) & (trades["exit_date"] >= s_d)
        is_bs = is_bs | in_bs
    
    clean_trades = trades[~is_bs]
    print(f"\n--- AFTER REMOVING BLACK SWAN OVERLAPPING TRADES ({len(clean_trades)} trades, removed {is_bs.sum()}) ---")
    calc_stats(clean_trades)
    print("\n--- CLEAN LONGS ---")
    calc_stats(clean_trades[clean_trades["side"] == 1])
    print("\n--- CLEAN SHORTS ---")
    calc_stats(clean_trades[clean_trades["side"] == -1])

    # Print all trades for deep inspection
    print("\n--- ALL TRADES DETAIL ---")
    for idx, r in trades.sort_values("entry_date").iterrows():
        print(f"{r['entry_date'].strftime('%Y-%m-%d')} | {'LONG ' if r['side']==1 else 'SHORT'} | "
              f"Entry: {r['entry_price']:8.1f} | Exit: {r['exit_price']:8.1f} | PnL: {r['pnl_pct']:+6.2%} | "
              f"R: {r['r_multiple']:+5.2f}R | Exit: {r['exit_reason']:<22} | Win: {r['pnl_pct']>0}")

def calc_stats(df_t):
    if len(df_t) == 0:
        print("No trades")
        return
    wins = df_t[df_t["pnl_pct"] > 0]
    losses = df_t[df_t["pnl_pct"] <= 0]
    wr = len(wins) / len(df_t)
    pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
    avg_r = df_t["r_multiple"].mean()
    exp = df_t["pnl_pct"].mean()
    mae_med = df_t["trade_mae"].median()
    mfe_med = df_t["trade_mfe"].median()
    print(f"Count: {len(df_t)} | WinRate: {wr:.1%} ({len(wins)}/{len(df_t)}) | PF: {pf:.2f} | "
          f"Avg_R: {avg_r:+.2f}R | Expectancy: {exp:+.2%} | MAE_med: {mae_med:.2%} | MFE_med: {mfe_med:+.2%}")

if __name__ == "__main__":
    analyze()
