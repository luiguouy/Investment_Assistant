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

def simulate_trade_with_be_at_1r(df, entry_t, entry_price, side, stop_price,
                                 tp1_price, tp2_price, impulse_leg, max_hold_bars=60):
    n = len(df)
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return {"status": "invalid"}

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    end_t = min(entry_t + max_hold_bars, n - 1)
    stage = 0   # 0: initial, 0.5: moved to BE (reached +1.0R), 1: TP1 hit, 2: TP2 hit
    curr_stop = stop_price
    rem_pos = 1.0
    exit_records = []
    final_exit_bar = entry_t

    # Trigger price for moving stop to breakeven: +1.0R
    be_trigger_p = entry_price + 1.0 * risk if side == 1 else entry_price - 1.0 * risk

    for bar in range(entry_t, end_t + 1):
        o, h, l, c = opens[bar], highs[bar], lows[bar], closes[bar]

        if side == 1:
            # 1. Check if we hit curr_stop
            hit_stop = (l <= curr_stop)
            hit_tp1 = (h >= tp1_price)

            if stage == 0:
                if hit_stop and hit_tp1:
                    fill_stop = min(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_stop, "weight": 1.0, "reason": "stop_loss_conservative"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_stop:
                    fill_stop = min(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_stop, "weight": 1.0, "reason": "stop_loss"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_tp1:
                    fill_tp1 = max(o, tp1_price)
                    exit_records.append({"bar": bar, "price": fill_tp1, "weight": 0.50, "reason": "tp1_half"})
                    rem_pos -= 0.50
                    stage = 1
                    curr_stop = entry_price
                elif h >= be_trigger_p:
                    # Trailing to Breakeven at +1.0R!
                    stage = 0.5
                    curr_stop = entry_price
            elif stage == 0.5:
                # In BE protection stage
                if hit_stop:
                    fill_be = min(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_be, "weight": 1.0, "reason": "be_at_1r_protected"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_tp1:
                    fill_tp1 = max(o, tp1_price)
                    exit_records.append({"bar": bar, "price": fill_tp1, "weight": 0.50, "reason": "tp1_half"})
                    rem_pos -= 0.50
                    stage = 1
                    curr_stop = entry_price
            elif stage == 1:
                if hit_stop:
                    fill_be = min(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_be, "weight": rem_pos, "reason": "tp1_then_be"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif h >= tp2_price:
                    fill_tp2 = max(o, tp2_price)
                    exit_records.append({"bar": bar, "price": fill_tp2, "weight": 0.25, "reason": "tp2_quarter"})
                    rem_pos -= 0.25
                    stage = 2
                    curr_stop = tp1_price
            elif stage == 2:
                trail_low = float(np.min(lows[max(entry_t, bar - 10):bar])) if bar > entry_t else tp1_price
                curr_stop = max(curr_stop, trail_low)
                if l <= curr_stop:
                    fill_r = min(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_r, "weight": rem_pos, "reason": "runner_stop"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
        else:
            # Short
            hit_stop = (h >= curr_stop)
            hit_tp1 = (l <= tp1_price)

            if stage == 0:
                if hit_stop and hit_tp1:
                    fill_stop = max(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_stop, "weight": 1.0, "reason": "stop_loss_conservative"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_stop:
                    fill_stop = max(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_stop, "weight": 1.0, "reason": "stop_loss"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_tp1:
                    fill_tp1 = min(o, tp1_price)
                    exit_records.append({"bar": bar, "price": fill_tp1, "weight": 0.50, "reason": "tp1_half"})
                    rem_pos -= 0.50
                    stage = 1
                    curr_stop = entry_price
                elif l <= be_trigger_p:
                    stage = 0.5
                    curr_stop = entry_price
            elif stage == 0.5:
                if hit_stop:
                    fill_be = max(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_be, "weight": 1.0, "reason": "be_at_1r_protected"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif hit_tp1:
                    fill_tp1 = min(o, tp1_price)
                    exit_records.append({"bar": bar, "price": fill_tp1, "weight": 0.50, "reason": "tp1_half"})
                    rem_pos -= 0.50
                    stage = 1
                    curr_stop = entry_price
            elif stage == 1:
                if hit_stop:
                    fill_be = max(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_be, "weight": rem_pos, "reason": "tp1_then_be"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break
                elif l <= tp2_price:
                    fill_tp2 = min(o, tp2_price)
                    exit_records.append({"bar": bar, "price": fill_tp2, "weight": 0.25, "reason": "tp2_quarter"})
                    rem_pos -= 0.25
                    stage = 2
                    curr_stop = tp1_price
            elif stage == 2:
                trail_high = float(np.max(highs[max(entry_t, bar - 10):bar])) if bar > entry_t else tp1_price
                curr_stop = min(curr_stop, trail_high)
                if h >= curr_stop:
                    fill_r = max(o, curr_stop)
                    exit_records.append({"bar": bar, "price": fill_r, "weight": rem_pos, "reason": "runner_stop"})
                    rem_pos = 0.0
                    final_exit_bar = bar
                    break

    if rem_pos > 1e-5:
        final_exit_bar = end_t
        exit_records.append({"bar": end_t, "price": closes[end_t], "weight": rem_pos, "reason": "time_horizon"})

    tot_weight = sum(r["weight"] for r in exit_records)
    avg_exit_p = sum(r["price"] * r["weight"] for r in exit_records) / tot_weight
    weighted_pnl = sum(((r["price"] / entry_price - 1.0) * side) * r["weight"] for r in exit_records) / tot_weight
    weighted_r = sum((((r["price"] - entry_price) * side) / risk) * r["weight"] for r in exit_records) / tot_weight

    return {
        "status": "completed",
        "entry_t": entry_t,
        "exit_t": final_exit_bar,
        "entry_price": entry_price,
        "exit_price": avg_exit_p,
        "pnl_pct": weighted_pnl,
        "r_multiple": weighted_r,
        "win": weighted_pnl > 0,
        "side": side,
        "exit_reason": exit_records[-1]["reason"]
    }

def test_runner():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)

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

            trade = simulate_trade_with_be_at_1r(df, entry_t, entry_p, side, stop_p, tp1_p, tp2_p, impulse)
            if trade.get("status") == "completed":
                trade["date"] = df.index[entry_t]
                recs.append(trade)

    tdf = pd.DataFrame(recs).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    wr = len(wins) / len(tdf)
    pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())
    avg_r = tdf["r_multiple"].mean()
    exp = tdf["pnl_pct"].mean()
    longs = tdf[tdf["side"] == 1]
    shorts = tdf[tdf["side"] == -1]
    wr_l = (longs["pnl_pct"] > 0).sum() / len(longs)
    wr_s = (shorts["pnl_pct"] > 0).sum() / len(shorts)

    print(f"WITH +1.0R BREAKEVEN PROTECTION:")
    print(f"Total: {len(tdf)} | WinRate: {wr:.1%} ({len(wins)}/{len(tdf)}) | PF: {pf:.2f} | Avg_R: {avg_r:+.2f}R | Exp: {exp:+.2%}")
    print(f"Longs: {len(longs)} (WR {wr_l:.1%}) | Shorts: {len(shorts)} (WR {wr_s:.1%})")
    print("\nExit reason counts:")
    print(tdf["exit_reason"].value_counts())

if __name__ == "__main__":
    test_runner()
