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

def test_dynamic_targets():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)

    # Let's test different TP1 multipliers and trailing breakeven triggers:
    # Option A: min_rr = 1.5 (easier to hit TP1 on volatile BTC swings)
    # Option B: min_rr = 2.0 (current)
    # Option C: Move stop to BE when price reaches +1.0R
    # Option D: ATR trailing stop after 10 bars
    
    for test_rr in [1.5, 1.8, 2.0]:
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
                    tp1_p = max(tp1_p, round(entry_p + test_rr * risk, 2))
                else:
                    tp1_p = min(tp1_p, round(entry_p - test_rr * risk, 2))
                tp2_p = lvls.get("tp2_target")
                impulse = lvls.get("impulse_leg")

                trade = sc.simulate_crypto_trade(df, entry_t, entry_p, side, stop_p,
                                                 tp1_price=tp1_p, tp2_price=tp2_p,
                                                 impulse_leg=impulse, min_rr=test_rr)
                if trade.get("status") == "completed":
                    trade["date"] = df.index[entry_t]
                    recs.append(trade)

        tdf = pd.DataFrame(recs).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()
        wins = tdf[tdf["pnl_pct"] > 0]
        losses = tdf[tdf["pnl_pct"] <= 0]
        wr = len(wins) / len(tdf) if len(tdf) > 0 else 0
        pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
        avg_r = tdf["r_multiple"].mean() if len(tdf) > 0 else 0
        exp = tdf["pnl_pct"].mean() if len(tdf) > 0 else 0

        longs = tdf[tdf["side"] == 1]
        shorts = tdf[tdf["side"] == -1]
        wr_l = (longs["pnl_pct"] > 0).sum() / len(longs) if len(longs) > 0 else 0
        wr_s = (shorts["pnl_pct"] > 0).sum() / len(shorts) if len(shorts) > 0 else 0

        print(f"Target min_rr = {test_rr:.1f} | N = {len(tdf):2d} | WinRate: {wr:5.1%} ({len(wins)}/{len(tdf)}) | PF: {pf:4.2f} | Avg_R: {avg_r:+5.2f}R | Exp: {exp:+6.2%} | Long WR: {wr_l:5.1%} | Short WR: {wr_s:5.1%}")

if __name__ == "__main__":
    test_dynamic_targets()
