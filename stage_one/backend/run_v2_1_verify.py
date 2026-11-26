#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2.1 收口前精准验证：BTC 宽止损解耦 TP1 + 美股 2B 缓冲修复"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from strategy_core import strategy_crypto as sc
from run_diagnostic_backtest import load_daily_ohlcv, build_causal_snapshots, _passes_box_channel_gate
from run_v2_experiment import summarize_sized, RISK_PER_TRADE, MAX_LEVERAGE


def run_verify(ticker: str, df: pd.DataFrame, weekly_df: pd.DataFrame,
               snapshots: dict, oos_ts: pd.Timestamp, cfg: dict) -> list:
    is_crypto = sc.is_crypto_asset(ticker)
    buffer_k = sc.CRYPTO_STOP_BUFFER_K if is_crypto else 1.0
    slippage_bps = 5.0 if is_crypto else 3.0
    commission_bps = 5.0 if is_crypto else 1.0
    funding_bps = 3.0 if is_crypto else 0.0

    trades, active_exit_t, last_setup_key = [], -1, None
    n = len(df)

    for t in sorted(snapshots.keys()):
        if t <= active_exit_t:
            continue
        snap = snapshots[t]
        side = int(snap["bias"])
        swings = snap["swings"]

        if (not is_crypto) and side == -1:
            continue
        if not _passes_box_channel_gate(side, snap["box"], snap["channel"]):
            continue

        anchor_sw = [s for s in swings if s["type"] == ("High" if side == 1 else "Low")]
        if not anchor_sw:
            continue
        setup_key = (side, anchor_sw[-1]["index"])
        if setup_key == last_setup_key:
            continue

        lvls = sc.extract_crypto_levels(df, t, side, swings, buffer_k=buffer_k)
        if not lvls:
            continue

        # 触发扫描
        end_k = min(t + sc.CRYPTO_MAX_WAIT_BARS, n - 2)
        trig_hit = None
        for k in range(t + 1, end_k + 1):
            l_k, h_k = float(df["Low"].iloc[k]), float(df["High"].iloc[k])
            if (side == 1 and l_k <= lvls["structural_stop"]) or (side == -1 and h_k >= lvls["structural_stop"]):
                break
            trig = sc.detect_crypto_trigger_on_bar(df, k, side, lvls["swing_retest"], lvls["bar_rng"])
            if not trig.get("triggered"):
                continue
            if side == -1 and trig["pattern"] == "engulfing":
                continue
            if cfg.get("drop_2b") and trig["pattern"] == "2B_sweep":
                continue
            if not sc.check_crypto_primary_trend(df, k, side, weekly_df=weekly_df).get("aligned", True):
                continue

            entry_t = k + 1
            entry_px = float(df["Open"].iloc[entry_t])

            # 止损模式
            if cfg["stop_mode"] == "wide":
                final_stop = min(lvls["structural_stop"], trig["tight_stop"]) if side == 1 else max(lvls["structural_stop"], trig["tight_stop"])
            else:
                if trig["pattern"] == "2B_sweep" and cfg.get("us_2b_buf_k"):
                    buf = cfg["us_2b_buf_k"] * lvls["bar_rng"]
                    final_stop = round(trig["trigger_bar_low"] - buf, 2) if side == 1 else round(trig["trigger_bar_high"] + buf, 2)
                else:
                    final_stop = float(trig["tight_stop"])

            if (side == 1 and entry_px <= final_stop) or (side == -1 and entry_px >= final_stop):
                continue
            risk = abs(entry_px - final_stop)
            if cfg.get("max_stop_atr") and (risk / lvls["bar_rng"]) > cfg["max_stop_atr"]:
                continue

            trig_hit = (entry_t, entry_px, final_stop, trig["pattern"])
            break

        if not trig_hit:
            continue

        entry_t, entry_px, stop_px, pat = trig_hit
        risk = abs(entry_px - stop_px)
        tp1_r = float(cfg["tp1_r"])
        tp1_target = round(entry_px + side * tp1_r * risk, 2)

        res = sc.simulate_crypto_trade(
            df=df, entry_t=entry_t, entry_price=entry_px, side=side, stop_price=stop_px,
            tp1_price=tp1_target, tp2_price=lvls["tp2_target"], impulse_leg=lvls["impulse_leg"],
            max_hold_bars=sc.CRYPTO_MAX_HOLD_BARS, min_rr=tp1_r,
            slippage_bps=slippage_bps, commission_bps=commission_bps, funding_rate_daily_bps=funding_bps,
        )
        last_setup_key = setup_key
        if res.get("status") != "completed":
            continue

        active_exit_t = int(res["exit_t"])
        entry_dt = pd.Timestamp(df.index[res["entry_t"]])
        pos_w = min(MAX_LEVERAGE, RISK_PER_TRADE / max(risk / entry_px, 1e-6))
        trades.append({
            "ticker": ticker, "config": cfg["name"], "is_oos": bool(entry_dt >= oos_ts),
            "sized_ret_pct": pos_w * float(res["net_pnl_pct"]),
            "net_r": float(res["net_r"]), "win": bool(res["net_pnl_pct"] > 0),
            "exit_reason": res["exit_reason"],
        })
    return trades


def main():
    oos_ts = pd.Timestamp("2025-01-01")
    btc_cfgs = [
        {"name": "BTC_Wide_20R", "stop_mode": "wide", "tp1_r": 2.0},
        {"name": "BTC_Wide_18R", "stop_mode": "wide", "tp1_r": 1.8},
        {"name": "BTC_Wide_15R", "stop_mode": "wide", "tp1_r": 1.5},
    ]
    us_cfgs = [
        {"name": "US_V2b_Orig",   "stop_mode": "tight", "max_stop_atr": 3.0, "tp1_r": 1.5},
        {"name": "US_2B_Buf10",   "stop_mode": "tight", "max_stop_atr": 3.0, "tp1_r": 1.5, "us_2b_buf_k": 1.0},
        {"name": "US_No2B_PinEn", "stop_mode": "tight", "max_stop_atr": 3.0, "tp1_r": 1.5, "drop_2b": True},
    ]

    print(f"\n{'标的':<9} {'实验组':<15} | {'全量笔':>5} {'净PF':>6} {'均R':>6} {'收益/DD':>7} {'僵尸%':>6} | {'OOS笔':>5} {'OOS_PF':>6} {'OOS均R':>7}")
    print("-" * 92)
    for ticker in ["BTC-USD", "QQQ", "GOOGL", "MSFT"]:
        df = load_daily_ohlcv(ticker, start="2020-01-01")
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        w_df = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
        snaps = build_causal_snapshots(df, warmup=60, swing_order=5)
        cfgs = btc_cfgs if sc.is_crypto_asset(ticker) else us_cfgs

        for cfg in cfgs:
            tr_df = pd.DataFrame(run_verify(ticker, df, w_df, snaps, oos_ts, cfg))
            m_all = summarize_sized(tr_df)
            m_oos = summarize_sized(tr_df[tr_df["is_oos"]]) if not tr_df.empty else summarize_sized(tr_df)
            pf_a = f"{m_all['net_pf']:6.2f}" if np.isfinite(m_all["net_pf"]) else "   inf"
            pf_o = f"{m_oos['net_pf']:6.2f}" if np.isfinite(m_oos["net_pf"]) else "   inf"
            rdd = f"{m_all['ret_dd']:7.2f}" if np.isfinite(m_all["ret_dd"]) else "    inf"
            print(
                f"{ticker:<9} {cfg['name']:<15} | {m_all['trades']:>5d} {pf_a} {m_all['avg_r']:>+6.2f} "
                f"{rdd} {m_all['zombie_pct']:>5.1f}% | {m_oos['trades']:>5d} {pf_o} {m_oos['avg_r']:>+7.2f}"
            )
        print("-" * 92)


if __name__ == "__main__":
    main()
