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

def test_volume_and_exits():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    fwd = bt.Forward(df)

    # Let's test volume filter
    records_vol = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto",
                                  asset_type="crypto", ticker="BTC-USD", volume_filter=True)
    trades_vol = records_vol.drop_duplicates(subset=["t", "side", "entry_price"]).copy()

    records_no_vol = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto",
                                     asset_type="crypto", ticker="BTC-USD", volume_filter=False)
    trades_no_vol = records_no_vol.drop_duplicates(subset=["t", "side", "entry_price"]).copy()

    print(f"No volume filter: {len(trades_no_vol)} trades")
    calc(trades_no_vol, "NO VOL FILTER")

    print(f"\nWith volume filter: {len(trades_vol)} trades")
    calc(trades_vol, "WITH VOL FILTER")

def calc(t_df, label):
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
    wr_l = (longs["pnl_pct"] > 0).sum() / len(longs) if len(longs) > 0 else 0
    wr_s = (shorts["pnl_pct"] > 0).sum() / len(shorts) if len(shorts) > 0 else 0
    print(f"{label}: N={len(t_df)} | WR={wr:.1%} ({len(wins)}/{len(t_df)}) | PF={pf:.2f} | Avg_R={avg_r:+.2f}R | Exp={exp:+.2%} | Longs WR={wr_l:.1%} ({len(longs)}) | Shorts WR={wr_s:.1%} ({len(shorts)})")

if __name__ == "__main__":
    test_volume_and_exits()
