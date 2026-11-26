#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 止损/止盈机制改造 + 等风险仓位 + OOS 逐笔解剖实验脚本
路径: stage_one/backend/run_v2_experiment.py

运行示例 (PowerShell):
  cd stage_one/backend
  python run_v2_experiment.py --tickers BTC-USD,QQQ,GOOGL,MSFT --start 2020-01-01 --oos-date 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from strategy_core import strategy_crypto as sc
from run_diagnostic_backtest import (
    load_daily_ohlcv,
    build_causal_snapshots,
    _passes_box_channel_gate,
)

# 统一风控口径：单笔固定风险 2%，现货上限 1.0x（不加杠杆）
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 1.0

V2_CONFIGS = [
    {
        "name": "V1_Baseline",
        "desc": "原版宽止损 + TP1=2.0R（BTC多空 / 美股仅多）",
        "stop_mode": "wide",
        "max_stop_atr": None,
        "tp1_r": 2.0,
    },
    {
        "name": "V2a_Tight_20R",
        "desc": "形态紧凑止损(上限3.0ATR) + TP1=2.0R",
        "stop_mode": "tight",
        "max_stop_atr": 3.0,
        "tp1_r": 2.0,
    },
    {
        "name": "V2b_Tight_15R",
        "desc": "形态紧凑止损(上限3.0ATR) + TP1=1.5R减半推保本",
        "stop_mode": "tight",
        "max_stop_atr": 3.0,
        "tp1_r": 1.5,
    },
]


def simulate_trigger_v2(
    df: pd.DataFrame,
    t: int,
    side: int,
    key_level: float,
    bar_rng: float,
    structural_stop: float,
    weekly_df: pd.DataFrame,
    stop_mode: str = "wide",
    max_stop_atr: float | None = None,
    max_wait_bars: int = sc.CRYPTO_MAX_WAIT_BARS,
) -> dict:
    """
    在保持严格零未来函数（k 根确认，k+1 根开盘入场）的前提下，
    支持切换宽止损 (wide) 与形态紧凑止损 (tight + 最大止损宽度过滤)。
    """
    n = len(df)
    if t >= n - 2 or key_level <= 0 or bar_rng <= 0:
        return {"triggered": False}

    end_k = min(t + max_wait_bars, n - 2)
    for k in range(t + 1, end_k + 1):
        l_k, h_k = float(df["Low"].iloc[k]), float(df["High"].iloc[k])
        if side == 1 and l_k <= structural_stop:
            return {"triggered": False}
        if side == -1 and h_k >= structural_stop:
            return {"triggered": False}

        trig = sc.detect_crypto_trigger_on_bar(df, k, side, key_level, bar_rng)
        if not trig.get("triggered"):
            continue

        # 保留已验证有效的两项门禁：剔除空头吞没 + 周线顺势
        if side == -1 and trig.get("pattern") == "engulfing":
            continue
        if not sc.check_crypto_primary_trend(df, k, side, weekly_df=weekly_df).get("aligned", True):
            continue

        entry_t = k + 1
        entry_price = float(df["Open"].iloc[entry_t])
        tight_stop = float(trig["tight_stop"])

        if stop_mode == "tight":
            # 优先使用形态紧凑止损；若针尖极值本身就在结构位之外则自然覆盖
            final_stop = tight_stop
        else:
            final_stop = min(structural_stop, tight_stop) if side == 1 else max(structural_stop, tight_stop)

        if (side == 1 and entry_price <= final_stop) or (side == -1 and entry_price >= final_stop):
            continue

        # 畸形宽止损过滤：止损超过 N 倍中位振幅直接放弃，避免锁死盈亏比
        risk = abs(entry_price - final_stop)
        if max_stop_atr is not None and (risk / bar_rng) > max_stop_atr:
            continue

        return {
            "triggered": True,
            "entry_t": entry_t,
            "entry_price": round(entry_price, 2),
            "stop_price": round(final_stop, 2),
            "pattern": trig["pattern"],
            "is_2b": trig.get("is_2b", False),
        }

    return {"triggered": False}


def run_v2_config(
    ticker: str,
    df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snapshots: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
    oos_date: pd.Timestamp,
) -> List[Dict[str, Any]]:
    is_crypto = sc.is_crypto_asset(ticker)
    buffer_k = sc.CRYPTO_STOP_BUFFER_K if is_crypto else 1.0
    slippage_bps = 5.0 if is_crypto else 3.0
    commission_bps = 5.0 if is_crypto else 1.0
    funding_bps = 3.0 if is_crypto else 0.0

    trades: List[Dict[str, Any]] = []
    active_exit_t = -1
    last_setup_key = None

    for t in sorted(snapshots.keys()):
        if t <= active_exit_t:
            continue

        snap = snapshots[t]
        side = int(snap["bias"])
        swings = snap["swings"]

        # 锁定已验证结论：美股仅做多，BTC 多空双向
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

        trig = simulate_trigger_v2(
            df=df,
            t=t,
            side=side,
            key_level=lvls["swing_retest"],
            bar_rng=lvls["bar_rng"],
            structural_stop=lvls["structural_stop"],
            weekly_df=weekly_df,
            stop_mode=cfg["stop_mode"],
            max_stop_atr=cfg["max_stop_atr"],
        )
        if not trig.get("triggered"):
            continue

        entry_px = float(trig["entry_price"])
        stop_px = float(trig["stop_price"])
        risk = abs(entry_px - stop_px)
        tp1_r = float(cfg["tp1_r"])

        # V2 模式下直接以 tp1_r * risk 锚定 TP1（解开远端结构位锁死），TP2 仍交由扩展目标与 Runner 捕捉大趋势
        if cfg["stop_mode"] == "tight":
            tp1_target = round(entry_px + side * tp1_r * risk, 2)
        else:
            tp1_target = lvls["tp1_target"]

        res = sc.simulate_crypto_trade(
            df=df,
            entry_t=trig["entry_t"],
            entry_price=entry_px,
            side=side,
            stop_price=stop_px,
            tp1_price=tp1_target,
            tp2_price=lvls["tp2_target"],
            impulse_leg=lvls["impulse_leg"],
            max_hold_bars=sc.CRYPTO_MAX_HOLD_BARS,
            min_rr=tp1_r,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            funding_rate_daily_bps=funding_bps,
        )
        last_setup_key = setup_key

        if res.get("status") != "completed":
            continue

        active_exit_t = int(res["exit_t"])
        entry_dt = pd.Timestamp(df.index[res["entry_t"]])
        exit_dt = pd.Timestamp(df.index[res["exit_t"]])
        bar_rng = max(float(lvls["bar_rng"]), 1e-6)

        net_pnl = float(res["net_pnl_pct"])
        stop_dist_pct = risk / entry_px
        # 单笔 2% 固定风险仓位权重（现货上限 1.0x）
        pos_weight = min(MAX_LEVERAGE, RISK_PER_TRADE / max(stop_dist_pct, 1e-6))
        sized_ret = pos_weight * net_pnl

        trades.append({
            "ticker": ticker,
            "is_crypto": is_crypto,
            "config": cfg["name"],
            "side": side,
            "pattern": trig["pattern"],
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "is_oos": bool(entry_dt >= oos_date),
            "entry_price": entry_px,
            "stop_price": stop_px,
            "exit_price": float(res["exit_price"]),
            "stop_atr": round(risk / bar_rng, 2),
            "stop_pct": round(stop_dist_pct * 100, 2),
            "pos_weight": round(pos_weight, 2),
            "net_pnl_pct": net_pnl,
            "sized_ret_pct": sized_ret,
            "net_r": float(res["net_r"]),
            "hold_bars": int(res["hold_bars"]),
            "mae_pct": float(res["mae"]),
            "mfe_r": round((float(res["mfe"]) * entry_px) / max(risk, 1e-6), 2),
            "exit_reason": res["exit_reason"],
            "win": bool(net_pnl > 0),
        })

    return trades


def calc_buy_and_hold(df: pd.DataFrame, warmup: int = 60) -> Dict[str, float]:
    sub = df.iloc[warmup:]
    closes = sub["Close"].to_numpy(dtype=float)
    eq = closes / closes[0]
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - peak) / peak))
    tot_ret = float(eq[-1] - 1.0)
    ret_dd = tot_ret / abs(max_dd) if max_dd < 0 else float("inf")
    return {"cum_ret": tot_ret, "max_dd": max_dd, "ret_dd": ret_dd}


def summarize_sized(df_tr: pd.DataFrame) -> Dict[str, Any]:
    if df_tr.empty:
        return {
            "trades": 0, "win_rate": 0.0, "net_pf": 0.0, "avg_r": 0.0,
            "sized_cum": 0.0, "sized_dd": 0.0, "ret_dd": 0.0,
            "zombie_pct": 0.0, "tp_hit_pct": 0.0,
        }
    wins = df_tr.loc[df_tr["sized_ret_pct"] > 0, "sized_ret_pct"].sum()
    losses = abs(df_tr.loc[df_tr["sized_ret_pct"] <= 0, "sized_ret_pct"].sum())
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    eq = (1.0 + df_tr["sized_ret_pct"]).cumprod()
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    cum_ret = float(eq.iloc[-1] - 1.0)
    ret_dd = cum_ret / abs(max_dd) if max_dd < 0 else (float("inf") if cum_ret > 0 else 0.0)

    zombie_pct = float((df_tr["exit_reason"] == "time_horizon").mean() * 100)
    tp_hit_pct = float(df_tr["exit_reason"].str.contains("tp").mean() * 100)

    return {
        "trades": int(len(df_tr)),
        "win_rate": float(df_tr["win"].mean()),
        "net_pf": pf,
        "avg_r": float(df_tr["net_r"].mean()),
        "sized_cum": cum_ret,
        "sized_dd": max_dd,
        "ret_dd": ret_dd,
        "zombie_pct": zombie_pct,
        "tp_hit_pct": tp_hit_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 紧凑止损 + 1.5R保本 + 等风险仓位对照实验")
    parser.add_argument("--tickers", type=str, default="BTC-USD,QQQ,GOOGL,MSFT")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--oos-date", type=str, default="2025-01-01")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    oos_ts = pd.Timestamp(args.oos_date)

    all_trades: List[Dict[str, Any]] = []
    bh_map: Dict[str, Dict[str, float]] = {}

    for ticker in tickers:
        df = load_daily_ohlcv(ticker, start=args.start)
        if len(df) < 100:
            continue
        bh_map[ticker] = calc_buy_and_hold(df, warmup=60)

        agg_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg_map["Volume"] = "sum"
        weekly_df = df.resample("W-FRI").agg(agg_map).dropna(subset=["Open", "High", "Low", "Close"])
        snapshots = build_causal_snapshots(df, warmup=60, swing_order=5)

        for cfg in V2_CONFIGS:
            all_trades.extend(run_v2_config(ticker, df, weekly_df, snapshots, cfg, oos_ts))

    if not all_trades:
        print("无有效交易记录。")
        return 1

    all_df = pd.DataFrame(all_trades)

    # 表 1：V1 vs V2 等风险仓位对照 + Buy & Hold 基准
    print("\n" + "=" * 108)
    print("一、 V1 vs V2 止损/止盈改造对照表（单笔 2% 固定风险仓位，上限 1x 现货）")
    print("=" * 108)
    print(f"{'标的':<9} {'版本':<15} {'笔数':>4} {'胜率':>6} {'净PF':>6} {'均R':>6} {'等险收益':>8} {'等险回撤':>8} {'收益/回撤':>8} {'僵尸单%':>7} {'触TP%':>6}")
    print("-" * 108)
    for ticker in tickers:
        if ticker not in bh_map:
            continue
        sub_t = all_df[all_df["ticker"] == ticker]
        for cfg in V2_CONFIGS:
            m = summarize_sized(sub_t[sub_t["config"] == cfg["name"]])
            pf_s = f"{m['net_pf']:6.2f}" if np.isfinite(m["net_pf"]) else "   inf"
            rdd_s = f"{m['ret_dd']:8.2f}" if np.isfinite(m["ret_dd"]) else "     inf"
            print(
                f"{ticker:<9} {cfg['name']:<15} {m['trades']:>4d} {m['win_rate']*100:>5.1f}% "
                f"{pf_s} {m['avg_r']:>+6.2f} {m['sized_cum']*100:>+7.1f}% {m['sized_dd']*100:>7.1f}% "
                f"{rdd_s} {m['zombie_pct']:>6.1f}% {m['tp_hit_pct']:>5.1f}%"
            )
        bh = bh_map[ticker]
        print(
            f"{ticker:<9} {'[Buy&Hold基准]':<15} {'-':>4} {'-':>6} {'-':>6} {'-':>6} "
            f"{bh['cum_ret']*100:>+7.1f}% {bh['max_dd']*100:>7.1f}% {bh['ret_dd']:>8.2f} {'-':>7} {'-':>6}"
        )
        print("-" * 108)

    # 表 2：IS vs OOS 对比（重点对比 V1_Baseline vs V2b_Tight_15R）
    print(f"\n二、 样本内 (< {args.oos_date}) vs 样本外 (>= {args.oos_date}) 对比 [V1_Baseline vs V2b_Tight_15R]")
    print("-" * 108)
    for cfg_name in ["V1_Baseline", "V2b_Tight_15R"]:
        print(f"▶ 配置: {cfg_name}")
        sub_c = all_df[all_df["config"] == cfg_name]
        for ticker in tickers:
            sub_t = sub_c[sub_c["ticker"] == ticker]
            m_is = summarize_sized(sub_t[~sub_t["is_oos"]])
            m_oos = summarize_sized(sub_t[sub_t["is_oos"]])
            print(
                f"  {ticker:<9} | IS : {m_is['trades']:>2d}笔 胜率 {m_is['win_rate']*100:>5.1f}% PF {m_is['net_pf']:>5.2f} 均R {m_is['avg_r']:>+5.2f} 回撤 {m_is['sized_dd']*100:>6.1f}% "
                f"| OOS: {m_oos['trades']:>2d}笔 胜率 {m_oos['win_rate']*100:>5.1f}% PF {m_oos['net_pf']:>5.2f} 均R {m_oos['avg_r']:>+5.2f} 回撤 {m_oos['sized_dd']*100:>6.1f}%"
            )

    # 表 3：2025-2026 OOS 逐笔交易明细（解剖 OOS 真实死因）
    print(f"\n三、 2025-2026 样本外 (OOS) 逐笔交易明细 [V2b_Tight_15R]")
    print("-" * 108)
    oos_df = all_df[(all_df["config"] == "V2b_Tight_15R") & (all_df["is_oos"])]
    if oos_df.empty:
        print("  无 OOS 交易记录。")
    else:
        cols = ["ticker", "side", "pattern", "entry_date", "exit_date", "stop_pct", "pos_weight", "net_r", "mfe_r", "hold_bars", "exit_reason"]
        print(oos_df[cols].to_string(index=False))

    out_path = BASE_DIR / "results/diagnostics/trades_v2_experiment.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[导出] V2 对照明细已保存至: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
