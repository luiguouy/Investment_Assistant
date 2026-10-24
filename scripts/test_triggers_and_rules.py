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

def test_triggers_and_rules():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    fwd = bt.Forward(df)

    # Let's inspect the triggers directly
    recs = []
    for e in events:
        t, side = e["t"], e["side"]
        sub_d = df.iloc[:t + 1]
        sw = pe.extract_swings(sub_d, order=bt.ORDER_DAILY)
        lvls = sc.extract_crypto_levels(sub_d, t, side, sw)
        if not lvls:
            continue
        key_level = lvls["swing_retest"]
        trig = sc.simulate_crypto_trigger_entry(df, t, side, key_level, lvls["bar_rng"], lvls["structural_stop"])
        if trig.get("triggered"):
            entry_t = trig["entry_t"]
            entry_p = trig["entry_price"]
            stop_p = trig["stop_price"]
            risk = abs(entry_p - stop_p)
            tp1_p = lvls.get("tp1_target")
            if side == 1:
                tp1_p = max(tp1_p, round(entry_p + 2.0 * risk, 2))
            else:
                tp1_p = min(tp1_p, round(entry_p - 2.0 * risk, 2))
            tp2_p = lvls.get("tp2_target")
            impulse = lvls.get("impulse_leg")

            trade = sc.simulate_crypto_trade(df, entry_t, entry_p, side, stop_p,
                                             tp1_price=tp1_p, tp2_price=tp2_p,
                                             impulse_leg=impulse, min_rr=2.0)
            if trade.get("status") == "completed":
                trade["pattern"] = trig["pattern"]
                trade["is_2b"] = trig["is_2b"]
                trade["date"] = df.index[entry_t]
                trade["arm"] = e["arm"]
                recs.append(trade)

    tdf = pd.DataFrame(recs).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()
    print(f"Total unique trades: {len(tdf)}")
    print("\n--- PERFORMANCE BY PATTERN ---")
    for pat, grp in tdf.groupby("pattern"):
        wins = grp[grp["pnl_pct"] > 0]
        wr = len(wins) / len(grp)
        pf = wins["pnl_pct"].sum() / abs(grp[grp["pnl_pct"] <= 0]["pnl_pct"].sum())
        print(f"Pattern: {pat:<12} | Count: {len(grp):2d} | WinRate: {wr:5.1%} | PF: {pf:4.2f} | Avg PnL: {grp['pnl_pct'].mean():+6.2%}")

    print("\n--- PERFORMANCE BY SIDE ---")
    for s, grp in tdf.groupby("side"):
        side_name = "LONG" if s == 1 else "SHORT"
        wins = grp[grp["pnl_pct"] > 0]
        wr = len(wins) / len(grp)
        pf = wins["pnl_pct"].sum() / abs(grp[grp["pnl_pct"] <= 0]["pnl_pct"].sum())
        print(f"Side: {side_name:<5} | Count: {len(grp):2d} | WinRate: {wr:5.1%} | PF: {pf:4.2f} | Avg PnL: {grp['pnl_pct'].mean():+6.2%}")

if __name__ == "__main__":
    test_triggers_and_rules()
