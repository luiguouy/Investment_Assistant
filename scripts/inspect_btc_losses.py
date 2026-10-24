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

def inspect_losses():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    fwd = bt.Forward(df)

    records = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto", asset_type="crypto", ticker="BTC-USD")
    trades = records.drop_duplicates(subset=["t", "side", "entry_price"]).copy()
    trades["entry_date"] = pd.to_datetime(trades["date"])

    print("=== BREAKDOWN OF EXIT REASONS ===")
    print(trades["exit_reason"].value_counts())

    print("\n=== PNL DISTRIBUTION BY EXIT REASON ===")
    for reason, grp in trades.groupby("exit_reason"):
        wins = grp[grp["pnl_pct"] > 0]
        losses = grp[grp["pnl_pct"] <= 0]
        print(f"Reason: {reason:<26} | Count: {len(grp):2d} | Wins: {len(wins):2d} | Losses: {len(losses):2d} | Mean PnL: {grp['pnl_pct'].mean():+6.2%}")

    print("\n=== TIME HORIZON TRADES (60 BARS) ===")
    th = trades[trades["exit_reason"] == "time_horizon"]
    for idx, r in th.iterrows():
        print(f"Date: {r['entry_date'].strftime('%Y-%m-%d')} | {'LONG ' if r['side']==1 else 'SHORT'} | Entry: {r['entry_price']} | Exit: {r['exit_price']} | PnL: {r['pnl_pct']:+6.2%} | MAE: {r['trade_mae']:+6.2%} | MFE: {r['trade_mfe']:+6.2%}")

    print("\n=== STOP LOSS TRADES ===")
    sl_trades = trades[trades["exit_reason"] == "stop_loss"]
    for idx, r in sl_trades.iterrows():
        print(f"Date: {r['entry_date'].strftime('%Y-%m-%d')} | {'LONG ' if r['side']==1 else 'SHORT'} | Entry: {r['entry_price']} | Exit: {r['exit_price']} | PnL: {r['pnl_pct']:+6.2%} | MAE: {r['trade_mae']:+6.2%} | MFE: {r['trade_mfe']:+6.2%}")

if __name__ == "__main__":
    inspect_losses()
