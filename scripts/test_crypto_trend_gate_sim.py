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

def test_crypto_with_trend_gate():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="h2", buffer_k=0.0, volume_filter=True)
    fwd = bt.Forward(df)

    # Calculate 200 EMA
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    for arm in ["MTF", "MTF+HTF"]:
        arm_events = [e for e in events if e["arm"] == arm]
        recs = []
        for e in arm_events:
            t, side = e["t"], e["side"]
            sub_d = df.iloc[:t + 1]
            sw = pe.extract_swings(sub_d, order=bt.ORDER_DAILY)
            lvls = sc.extract_crypto_levels(sub_d, t, side, sw)
            if not lvls:
                continue
            key_level = lvls["swing_retest"]
            
            # Simulate trigger search
            end_k = min(t + sc.CRYPTO_MAX_WAIT_BARS, len(df) - 2)
            for k in range(t + 1, end_k + 1):
                l_k, h_k = float(df["Low"].iloc[k]), float(df["High"].iloc[k])
                if side == 1 and l_k <= lvls["structural_stop"]:
                    break
                if side == -1 and h_k >= lvls["structural_stop"]:
                    break

                trig = sc.detect_crypto_trigger_on_bar(df, k, side, key_level, lvls["bar_rng"])
                if trig.get("triggered"):
                    # Disallow short engulfing
                    if side == -1 and trig.get("pattern") == "engulfing":
                        continue

                    # Dow Primary Trend Gate (200 EMA)
                    c_k = float(df["Close"].iloc[k])
                    ema_k = float(df["EMA200"].iloc[k])
                    if side == 1 and c_k < ema_k:
                        continue
                    if side == -1 and c_k > ema_k:
                        continue

                    entry_t = k + 1
                    entry_p = float(df["Open"].iloc[entry_t])
                    tight_stop = float(trig["tight_stop"])
                    if side == 1:
                        final_stop = min(lvls["structural_stop"], tight_stop)
                        if entry_p <= final_stop:
                            continue
                    else:
                        final_stop = max(lvls["structural_stop"], tight_stop)
                        if entry_p >= final_stop:
                            continue

                    risk = abs(entry_p - final_stop)
                    tp1_p = lvls.get("tp1_target")
                    if side == 1:
                        tp1_p = max(tp1_p, round(entry_p + 2.0 * risk, 2))
                    else:
                        tp1_p = min(tp1_p, round(entry_p - 2.0 * risk, 2))
                    tp2_p = lvls.get("tp2_target")
                    impulse = lvls.get("impulse_leg")

                    trade = sc.simulate_crypto_trade(df, entry_t, entry_p, side, final_stop,
                                                     tp1_price=tp1_p, tp2_price=tp2_p,
                                                     impulse_leg=impulse, min_rr=2.0)
                    if trade.get("status") == "completed":
                        trade["pattern"] = trig["pattern"]
                        trade["date"] = df.index[entry_t]
                        trade["arm"] = arm
                        recs.append(trade)
                    break

        tdf = pd.DataFrame(recs).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()
        wins = tdf[tdf["pnl_pct"] > 0]
        losses = tdf[tdf["pnl_pct"] <= 0]
        wr = len(wins) / len(tdf) if len(tdf) > 0 else 0
        pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
        avg_r = tdf["r_multiple"].mean() if len(tdf) > 0 else 0
        exp = tdf["pnl_pct"].mean() if len(tdf) > 0 else 0

        longs = tdf[tdf["side"] == 1]
        shorts = tdf[tdf["side"] == -1]
        w_l = (longs["pnl_pct"] > 0).sum()
        w_s = (shorts["pnl_pct"] > 0).sum()
        wr_l = w_l / len(longs) if len(longs) > 0 else 0
        wr_s = w_s / len(shorts) if len(shorts) > 0 else 0
        pf_l = longs[longs["pnl_pct"] > 0]["pnl_pct"].sum() / abs(longs[longs["pnl_pct"] <= 0]["pnl_pct"].sum()) if len(longs[longs["pnl_pct"] <= 0]) > 0 else float("inf")
        pf_s = shorts[shorts["pnl_pct"] > 0]["pnl_pct"].sum() / abs(shorts[shorts["pnl_pct"] <= 0]["pnl_pct"].sum()) if len(shorts[shorts["pnl_pct"] <= 0]) > 0 else float("inf")

        print(f"\n[{arm} 固化道氏 200 EMA 大趋势门禁 + 剔除做空吞没]")
        print(f"  总交易: {len(tdf)} 笔 | 胜率: {wr:.1%} ({len(wins)} 胜 / {len(losses)} 负)")
        print(f"  盈亏比 (PF): {pf:.2f} | 平均 R 倍数: {avg_r:+.2f} R | 单笔期望收益: {exp:+.2%}")
        print(f"  多头(Long):  {len(longs):2d} 笔 | 胜率: {wr_l:5.1%} | PF: {pf_l:4.2f} | 期望: {longs['pnl_pct'].mean():+6.2%}")
        print(f"  空头(Short): {len(shorts):2d} 笔 | 胜率: {wr_s:5.1%} | PF: {pf_s:4.2f} | 期望: {shorts['pnl_pct'].mean():+6.2%}")

if __name__ == "__main__":
    test_crypto_with_trend_gate()
