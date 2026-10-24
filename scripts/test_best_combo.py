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

def test_best_combo():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)

    # Black swans
    black_swans = [
        ("Covid 312", "2020-03-08", "2020-03-22"),
        ("China 519", "2021-05-15", "2021-05-25"),
        ("Luna UST", "2022-05-05", "2022-05-18"),
        ("FTX BankRun", "2022-11-05", "2022-11-18"),
    ]

    def overlaps_bs(s_d, e_d):
        s = pd.to_datetime(s_d).tz_localize(None) if hasattr(pd.to_datetime(s_d), 'tz_localize') else pd.to_datetime(s_d)
        e = pd.to_datetime(e_d).tz_localize(None) if hasattr(pd.to_datetime(e_d), 'tz_localize') else pd.to_datetime(e_d)
        for _, b_s, b_e in black_swans:
            bs = pd.to_datetime(b_s)
            be = pd.to_datetime(b_e)
            if not (e < bs or s > be):
                return True
        return False

    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

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
            # Filter: only 2B and Pin Bar (reject engulfing)
            if trig["pattern"] not in ("2B_sweep", "pin_bar"):
                continue

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
                trade["entry_date"] = df.index[entry_t]
                trade["exit_date"] = df.index[trade["exit_t"]]
                trade["pattern"] = trig["pattern"]
                c_prev = float(df["Close"].iloc[entry_t - 1])
                ema200 = float(df["EMA200"].iloc[entry_t - 1])
                trade["trend_aligned"] = (side == 1 and c_prev > ema200) or (side == -1 and c_prev < ema200)
                trade["overlaps_bs"] = overlaps_bs(trade["entry_date"], trade["exit_date"])
                recs.append(trade)

    tdf = pd.DataFrame(recs).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()

    def pr(df_s, title):
        wins = df_s[df_s["pnl_pct"] > 0]
        losses = df_s[df_s["pnl_pct"] <= 0]
        wr = len(wins) / len(df_s) if len(df_s) > 0 else 0
        pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
        avg_r = df_s["r_multiple"].mean() if len(df_s) > 0 else 0
        exp = df_s["pnl_pct"].mean() if len(df_s) > 0 else 0
        longs = df_s[df_s["side"] == 1]
        shorts = df_s[df_s["side"] == -1]
        wr_l = (longs["pnl_pct"] > 0).sum() / len(longs) if len(longs) > 0 else 0
        wr_s = (shorts["pnl_pct"] > 0).sum() / len(shorts) if len(shorts) > 0 else 0
        print(f"{title}: N={len(df_s)} | WinRate={wr:.1%} ({len(wins)}/{len(df_s)}) | PF={pf:.2f} | Avg_R={avg_r:+.2f}R | Exp={exp:+.2%} | Longs={len(longs)} (WR {wr_l:.1%}) | Shorts={len(shorts)} (WR {wr_s:.1%})")

    print("--- 2B & PIN BAR ONLY ---")
    pr(tdf, "All 2B/Pin")
    pr(tdf[~tdf["overlaps_bs"]], "2B/Pin - Exclude BS")
    pr(tdf[tdf["trend_aligned"]], "2B/Pin + Trend Aligned")
    pr(tdf[tdf["trend_aligned"] & ~tdf["overlaps_bs"]], "2B/Pin + Trend Aligned - Exclude BS")
    pr(tdf[(tdf["side"] == 1) & ~tdf["overlaps_bs"]], "Clean 2B/Pin Longs Only")

if __name__ == "__main__":
    test_best_combo()
