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

def run_experiments():
    df = bt.load_ticker("BTC-USD")
    wf = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, wf, method="swing", buffer_k=0.0)
    
    # Precompute indicators
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Black swan dates
    black_swans = [
        ("Covid 312", "2020-03-08", "2020-03-22"),
        ("China 519", "2021-05-15", "2021-05-25"),
        ("Luna UST", "2022-05-05", "2022-05-18"),
        ("FTX BankRun", "2022-11-05", "2022-11-18"),
    ]

    def in_black_swan(date_str):
        d = pd.to_datetime(date_str).tz_localize(None) if hasattr(pd.to_datetime(date_str), 'tz_localize') else pd.to_datetime(date_str)
        for _, s, e in black_swans:
            s_d = pd.to_datetime(s)
            e_d = pd.to_datetime(e)
            if s_d <= d <= e_d:
                return True
        return False

    def trade_overlaps_bs(entry_d, exit_d):
        e_in = pd.to_datetime(entry_d).tz_localize(None) if hasattr(pd.to_datetime(entry_d), 'tz_localize') else pd.to_datetime(entry_d)
        x_in = pd.to_datetime(exit_d).tz_localize(None) if hasattr(pd.to_datetime(exit_d), 'tz_localize') else pd.to_datetime(exit_d)
        for _, s, e in black_swans:
            s_d = pd.to_datetime(s)
            e_d = pd.to_datetime(e)
            if not (x_in < s_d or e_in > e_d):
                return True
        return False

    # Simulate base trades with rich metadata
    base_trades = []
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
                trade["entry_date"] = df.index[entry_t]
                trade["exit_date"] = df.index[trade["exit_t"]]
                trade["entry_t"] = entry_t
                trade["arm"] = e["arm"]
                
                # Check 200 EMA at entry
                c_prev = float(df["Close"].iloc[entry_t - 1])
                ema200 = float(df["EMA200"].iloc[entry_t - 1])
                ema50 = float(df["EMA50"].iloc[entry_t - 1])
                trade["above_ema200"] = bool(c_prev > ema200)
                trade["above_ema50"] = bool(c_prev > ema50)
                trade["is_bs_entry"] = in_black_swan(trade["entry_date"])
                trade["is_bs_overlap"] = trade_overlaps_bs(trade["entry_date"], trade["exit_date"])
                base_trades.append(trade)

    tdf = pd.DataFrame(base_trades).drop_duplicates(subset=["entry_t", "side", "entry_price"]).copy()
    print(f"Loaded {len(tdf)} unique simulated trades on BTC-USD.")

    experiments = [
        ("0. 当前基准 (Baseline - All Trades)", tdf),
        ("1. 剔除黑天鹅暴跌期入场 (No Entry in Black Swan)", tdf[~tdf["is_bs_entry"]]),
        ("2. 彻底剔除黑天鹅在途持仓 (Strictly Exclude Black Swans)", tdf[~tdf["is_bs_overlap"]]),
        ("3. 纯多头策略 (Long Only)", tdf[tdf["side"] == 1]),
        ("4. 纯多头 + 剔除黑天鹅 (Clean Long Only)", tdf[(tdf["side"] == 1) & ~tdf["is_bs_overlap"]]),
        ("5. 纯空头策略 (Short Only)", tdf[tdf["side"] == -1]),
        ("6. 纯空头 + 剔除黑天鹅 (Clean Short Only)", tdf[(tdf["side"] == -1) & ~tdf["is_bs_overlap"]]),
        ("7. 道氏大趋势顺势过滤 (Primary Trend EMA200: Long > 200EMA, Short < 200EMA)", 
         tdf[((tdf["side"] == 1) & tdf["above_ema200"]) | ((tdf["side"] == -1) & ~tdf["above_ema200"])]),
        ("8. 顺势过滤 + 剔除黑天鹅 (Trend Aligned + Exclude Black Swans)", 
         tdf[(((tdf["side"] == 1) & tdf["above_ema200"]) | ((tdf["side"] == -1) & ~tdf["above_ema200"])) & ~tdf["is_bs_overlap"]]),
        ("9. 仅保留 2B 扫损与锤子线 (2B Sweep & Pin Bar Only, No Engulfing)", 
         tdf[tdf["pattern"].isin(["2B_sweep", "pin_bar"])]),
        ("10. 仅 2B/Pin + 剔除黑天鹅 (2B/Pin + Exclude Black Swans)", 
         tdf[tdf["pattern"].isin(["2B_sweep", "pin_bar"]) & ~tdf["is_bs_overlap"]]),
    ]

    for name, subset in experiments:
        eval_subset(name, subset)

def eval_subset(name, df_s):
    if len(df_s) == 0:
        print(f"\n{name}: [无交易]")
        return
    wins = df_s[df_s["pnl_pct"] > 0]
    losses = df_s[df_s["pnl_pct"] <= 0]
    wr = len(wins) / len(df_s)
    pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float("inf")
    avg_r = df_s["r_multiple"].mean()
    exp = df_s["pnl_pct"].mean()
    longs = df_s[df_s["side"] == 1]
    shorts = df_s[df_s["side"] == -1]
    w_l = (longs["pnl_pct"] > 0).sum()
    w_s = (shorts["pnl_pct"] > 0).sum()
    wr_l = w_l / len(longs) if len(longs) > 0 else 0
    wr_s = w_s / len(shorts) if len(shorts) > 0 else 0
    mae = df_s["mae"].median()
    mfe = df_s["mfe"].median()

    print(f"\n{'='*75}")
    print(f"【{name}】")
    print(f"  总交易笔数: {len(df_s):2d} 笔 | 胜率 (Win Rate): {wr:5.1%} ({len(wins)}赢/{len(losses)}输)")
    print(f"  盈亏比 (PF): {pf:4.2f} | 平均 R 倍数: {avg_r:+5.2f} R | 单笔期望收益: {exp:+6.2%}")
    print(f"  多头表现: {len(longs):2d} 笔 (胜率 {wr_l:5.1%}) | 空头表现: {len(shorts):2d} 笔 (胜率 {wr_s:5.1%})")
    print(f"  风险收益偏度: MAE 中位 {mae:6.2%} | MFE 中位 {mfe:+6.2%}")

if __name__ == "__main__":
    run_experiments()
