#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地多标的消融回测与 MAE/MFE 参数标定诊断脚本
路径: stage_one/backend/run_diagnostic_backtest.py

运行示例 (PowerShell):
  cd stage_one/backend
  python run_diagnostic_backtest.py --tickers BTC-USD,QQQ,GOOGL,MSFT --start 2020-01-01 --oos-date 2025-01-01
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from strategy_core import pa_engine as pe
from strategy_core import strategy_crypto as sc


# ==============================================================================
# 1. 数据加载（优先本地 data/<SYMBOL>.csv，缺失则 yfinance 自动拉取复权数据）
# ==============================================================================

def load_daily_ohlcv(ticker: str, start: str = "2020-01-01") -> pd.DataFrame:
    data_dir = BASE_DIR / "data"
    clean_name = ticker.upper().replace("/", "-")
    local_candidates = [
        data_dir / f"{clean_name}.csv",
        data_dir / f"{clean_name.split('-')[0]}.csv",
    ]
    df = pd.DataFrame()
    for path in local_candidates:
        if path.exists():
            try:
                tmp = pd.read_csv(path, index_col=0, parse_dates=True)
                col_map = {c.lower(): c for c in tmp.columns}
                if {"open", "high", "low", "close"}.issubset(col_map):
                    cols = [col_map[c] for c in ["open", "high", "low", "close"]]
                    if "volume" in col_map:
                        cols.append(col_map["volume"])
                    df = tmp[cols].copy()
                    df.columns = ["Open", "High", "Low", "Close"] + (["Volume"] if "volume" in col_map else [])
                    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.loc[df.index >= pd.Timestamp(start)]
                    if len(df) >= 120:
                        print(f"[数据] {ticker}: 从本地加载 {path.name} ({len(df)} 根日线)")
                        return df
            except Exception as e:
                print(f"[警告] 读取本地 {path.name} 失败，改用 yfinance: {e}", file=sys.stderr)

    print(f"[数据] {ticker}: 从 yfinance 拉取日线 (start={start}, auto_adjust=True)...")
    df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ==============================================================================
# 2. 严格因果结构预计算缓存（避免 4 组消融重复跑 extract_swings，100% 零前视）
# ==============================================================================

def build_causal_snapshots(df: pd.DataFrame, warmup: int = 60, swing_order: int = 5) -> Dict[int, Dict[str, Any]]:
    """
    对每根决策 K 线 t 严格基于 df.iloc[:t+1] 提取道氏拐点、偏向、箱体与通道。
    由于各消融组共用相同的历史行情，只缓存纯因果快照可提速 4 倍且绝不引入未来信息。
    """
    n = len(df)
    snapshots: Dict[int, Dict[str, Any]] = {}
    for t in range(warmup, n - 2):
        sub = df.iloc[:t + 1]
        swings = pe.extract_swings(sub, order=swing_order, confirmed_only=True)
        if len(swings) < 3:
            continue
        bias = pe.dow_bias(swings, df=sub, buffer_k=0.0)
        if bias == 0:
            continue
        snapshots[t] = {
            "swings": swings,
            "bias": bias,
            "box": pe.detect_consolidation_box(sub, swings),
            "channel": pe.detect_price_channel(sub, swings),
        }
    return snapshots


def _passes_box_channel_gate(side: int, box: dict, chan: dict) -> bool:
    """复用 pa_engine.build_trade_plan 的箱体与通道防追高门禁逻辑。"""
    if box and box.get("is_box"):
        b_state = box.get("state", "IN_BOX")
        b_pos = float(box.get("box_pos", 0.5))
        if side == 1 and (b_state == "SWEEP_HIGH" or (b_state == "IN_BOX" and b_pos >= 0.70)):
            return False
        if side == -1 and (b_state == "SWEEP_LOW" or (b_state == "IN_BOX" and b_pos <= 0.30)):
            return False

    if chan:
        c_type = chan.get("channel_type", "horizontal")
        c_pos = float(chan.get("channel_pos", 0.5))
        if side == 1:
            if (c_type == "ascending" and c_pos > 1.25) or \
               (c_type == "descending" and 0.65 <= c_pos <= 1.00) or \
               (c_type == "horizontal" and c_pos >= 0.85):
                return False
        else:
            if (c_type == "descending" and c_pos < -0.30) or \
               (c_type == "ascending" and 0.0 <= c_pos <= 0.25) or \
               (c_type == "horizontal" and c_pos <= 0.15):
                return False
    return True


# ==============================================================================
# 3. 单组配置 Walk-Forward 回测执行器
# ==============================================================================

ABLATION_CONFIGS = [
    {
        "name": "A0_Baseline",
        "desc": "无门禁基线（纯道氏突破回踩）",
        "use_trend_gate": False,
        "disallow_short_engulfing": False,
        "use_box_channel_gate": False,
        "long_only_us": False,
    },
    {
        "name": "A1_HTF_Gate",
        "desc": "+ 周线结构顺势门禁",
        "use_trend_gate": True,
        "disallow_short_engulfing": False,
        "use_box_channel_gate": False,
        "long_only_us": False,
    },
    {
        "name": "A2_Asym_Short",
        "desc": "+ 剔除空头吞没陷阱",
        "use_trend_gate": True,
        "disallow_short_engulfing": True,
        "use_box_channel_gate": False,
        "long_only_us": False,
    },
    {
        "name": "A3_Full_Gates",
        "desc": "+ 箱体与通道防追高门禁",
        "use_trend_gate": True,
        "disallow_short_engulfing": True,
        "use_box_channel_gate": True,
        "long_only_us": False,
    },
    {
        "name": "A4_US_LongOnly",
        "desc": "完整门禁 + 美股仅做多（对照）",
        "use_trend_gate": True,
        "disallow_short_engulfing": True,
        "use_box_channel_gate": True,
        "long_only_us": True,
    },
]


def run_single_config(
    ticker: str,
    df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snapshots: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
    oos_date: pd.Timestamp,
) -> List[Dict[str, Any]]:
    is_crypto = sc.is_crypto_asset(ticker)

    # 资产差异化成本与缓冲：BTC 宽幅防插针 + 资费，美股大票低滑点 + 0 资费
    buffer_k = sc.CRYPTO_STOP_BUFFER_K if is_crypto else 1.0
    min_rr = 2.0 if is_crypto else 2.0
    slippage_bps = 5.0 if is_crypto else 3.0
    commission_bps = 5.0 if is_crypto else 1.0
    funding_bps = 3.0 if is_crypto else 0.0

    # 检查 simulate_crypto_trade 是否已应用摩擦补丁
    sim_sig = inspect.signature(sc.simulate_crypto_trade).parameters
    has_friction_patch = "slippage_bps" in sim_sig

    trades: List[Dict[str, Any]] = []
    active_exit_t = -1
    last_setup_key = None

    for t in sorted(snapshots.keys()):
        if t <= active_exit_t:
            continue

        snap = snapshots[t]
        side = int(snap["bias"])
        swings = snap["swings"]

        if cfg["long_only_us"] and (not is_crypto) and side == -1:
            continue

        if cfg["use_box_channel_gate"]:
            if not _passes_box_channel_gate(side, snap["box"], snap["channel"]):
                continue

        # 防止同一波段拐点在连续 K 线上重复发射信号
        anchor_sw = [s for s in swings if s["type"] == ("High" if side == 1 else "Low")]
        if not anchor_sw:
            continue
        setup_key = (side, anchor_sw[-1]["index"])
        if setup_key == last_setup_key:
            continue

        lvls = sc.extract_crypto_levels(df, t, side, swings, buffer_k=buffer_k)
        if not lvls:
            continue

        trig = sc.simulate_crypto_trigger_entry(
            df=df,
            t=t,
            side=side,
            key_level=lvls["swing_retest"],
            bar_rng=lvls["bar_rng"],
            structural_stop=lvls["structural_stop"],
            max_wait_bars=sc.CRYPTO_MAX_WAIT_BARS,
            use_trend_gate=cfg["use_trend_gate"],
            disallow_short_engulfing=cfg["disallow_short_engulfing"],
            weekly_df=weekly_df,
        )
        if not trig.get("triggered"):
            continue

        sim_kwargs = dict(
            df=df,
            entry_t=trig["entry_t"],
            entry_price=trig["entry_price"],
            side=side,
            stop_price=trig["stop_price"],
            tp1_price=lvls["tp1_target"],
            tp2_price=lvls["tp2_target"],
            impulse_leg=lvls["impulse_leg"],
            max_hold_bars=sc.CRYPTO_MAX_HOLD_BARS,
            min_rr=min_rr,
        )
        if has_friction_patch:
            sim_kwargs.update(
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
                funding_rate_daily_bps=funding_bps,
            )

        res = sc.simulate_crypto_trade(**sim_kwargs)
        last_setup_key = setup_key

        if res.get("status") != "completed":
            continue

        active_exit_t = int(res["exit_t"])
        entry_dt = pd.Timestamp(df.index[res["entry_t"]])
        exit_dt = pd.Timestamp(df.index[res["exit_t"]])
        bar_rng = max(float(lvls["bar_rng"]), 1e-6)
        entry_px = float(res["entry_price"])
        risk = max(float(res["risk"]), 1e-6)

        net_pnl = float(res.get("net_pnl_pct", res["pnl_pct"]))
        gross_pnl = float(res.get("gross_pnl_pct", res["pnl_pct"]))
        net_r = float(res.get("net_r", res["r_multiple"]))
        gross_r = float(res.get("gross_r", res["r_multiple"]))

        mae_pct = float(res["mae"])
        mfe_pct = float(res["mfe"])
        mae_atr = abs(mae_pct * entry_px) / bar_rng
        mfe_r = (mfe_pct * entry_px) / risk

        trades.append({
            "ticker": ticker,
            "is_crypto": is_crypto,
            "config": cfg["name"],
            "side": side,
            "pattern": trig["pattern"],
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "is_oos": bool(entry_dt >= oos_date),
            "overlaps_black_swan": bool(sc.trade_overlaps_black_swan(entry_dt, exit_dt)) if is_crypto else False,
            "entry_price": entry_px,
            "stop_price": float(trig["stop_price"]),
            "exit_price": float(res["exit_price"]),
            "bar_rng": bar_rng,
            "risk_atr": round(risk / bar_rng, 2),
            "planned_rr": round(float(res["planned_rr"]), 2),
            "gross_pnl_pct": gross_pnl,
            "net_pnl_pct": net_pnl,
            "gross_r": gross_r,
            "net_r": net_r,
            "friction_drag_pct": float(res.get("friction_drag_pct", gross_pnl - net_pnl)),
            "funding_drag_pct": float(res.get("funding_drag_pct", 0.0)),
            "hold_bars": int(res["hold_bars"]),
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "mae_atr": round(mae_atr, 2),
            "mfe_r": round(mfe_r, 2),
            "win": bool(net_pnl > 0),
            "exit_reason": res["exit_reason"],
        })

    return trades


# ==============================================================================
# 4. 指标汇总与 MAE/MFE 诊断报告渲染
# ==============================================================================

def calc_metrics(df_tr: pd.DataFrame) -> Dict[str, Any]:
    if df_tr.empty:
        return {
            "trades": 0, "win_rate": 0.0, "net_pf": 0.0, "avg_net_r": 0.0,
            "exp_net_pct": 0.0, "max_dd_pct": 0.0, "cum_ret_pct": 0.0,
        }
    wins = df_tr.loc[df_tr["net_pnl_pct"] > 0, "net_pnl_pct"].sum()
    losses = abs(df_tr.loc[df_tr["net_pnl_pct"] <= 0, "net_pnl_pct"].sum())
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    eq = (1.0 + df_tr["net_pnl_pct"]).cumprod()
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    cum_ret = float(eq.iloc[-1] - 1.0)

    return {
        "trades": int(len(df_tr)),
        "win_rate": float((df_tr["net_pnl_pct"] > 0).mean()),
        "net_pf": pf,
        "avg_net_r": float(df_tr["net_r"].mean()),
        "exp_net_pct": float(df_tr["net_pnl_pct"].mean()),
        "max_dd_pct": max_dd,
        "cum_ret_pct": cum_ret,
    }


def print_ablation_report(all_df: pd.DataFrame, oos_date_str: str) -> None:
    print("\n" + "=" * 96)
    print("一、 模块消融对比表 (Ablation Study - 净收益口径)")
    print("=" * 96)
    header = f"{'标的':<9} {'配置':<15} {'笔数':>4} {'胜率':>7} {'净PF':>6} {'均R':>6} {'单笔期望':>8} {'最大回撤':>8} {'累计净值':>8}"
    print(header)
    print("-" * 96)

    for ticker in all_df["ticker"].unique():
        sub_t = all_df[all_df["ticker"] == ticker]
        for cfg in ABLATION_CONFIGS:
            # BTC 跳过美股仅做多对照组
            if sc.is_crypto_asset(ticker) and cfg["name"] == "A4_US_LongOnly":
                continue
            m = calc_metrics(sub_t[sub_t["config"] == cfg["name"]])
            pf_s = f"{m['net_pf']:6.2f}" if np.isfinite(m["net_pf"]) else "   inf"
            print(
                f"{ticker:<9} {cfg['name']:<15} {m['trades']:>4d} "
                f"{m['win_rate']*100:>6.1f}% {pf_s} {m['avg_net_r']:>+6.2f} "
                f"{m['exp_net_pct']*100:>+7.2f}% {m['max_dd_pct']*100:>7.1f}% {m['cum_ret_pct']*100:>+7.1f}%"
            )
        print("-" * 96)

    # 样本内外对比 (以 A3_Full_Gates 为主评估对象)
    print(f"\n二、 样本内 (< {oos_date_str}) vs 样本外 (>= {oos_date_str}) 盲测对比 [A3_Full_Gates]")
    print("-" * 96)
    a3 = all_df[all_df["config"] == "A3_Full_Gates"]
    for ticker in a3["ticker"].unique():
        sub_t = a3[a3["ticker"] == ticker]
        m_is = calc_metrics(sub_t[~sub_t["is_oos"]])
        m_oos = calc_metrics(sub_t[sub_t["is_oos"]])
        print(
            f"{ticker:<9} | IS : {m_is['trades']:>2d}笔 胜率 {m_is['win_rate']*100:>5.1f}% "
            f"PF {m_is['net_pf']:>5.2f} 均R {m_is['avg_net_r']:>+5.2f} 回撤 {m_is['max_dd_pct']*100:>6.1f}% "
            f"| OOS: {m_oos['trades']:>2d}笔 胜率 {m_oos['win_rate']*100:>5.1f}% "
            f"PF {m_oos['net_pf']:>5.2f} 均R {m_oos['avg_net_r']:>+5.2f} 回撤 {m_oos['max_dd_pct']*100:>6.1f}%"
        )

    # BTC 黑天鹅双轨评估
    btc_a3 = a3[a3["is_crypto"]]
    if not btc_a3.empty:
        print("\n三、 加密资产黑天鹅双轨评估 [A3_Full_Gates]")
        print("-" * 96)
        for ticker in btc_a3["ticker"].unique():
            sub_c = btc_a3[btc_a3["ticker"] == ticker]
            m_all = calc_metrics(sub_c)
            m_ex = calc_metrics(sub_c[~sub_c["overlaps_black_swan"]])
            n_bs = int(sub_c["overlaps_black_swan"].sum())
            print(
                f"{ticker:<9} 含黑天鹅 ({m_all['trades']}笔, 命中{n_bs}笔): "
                f"胜率 {m_all['win_rate']*100:.1f}% | 净PF {m_all['net_pf']:.2f} | "
                f"均R {m_all['avg_net_r']:+.2f} | 最大回撤 {m_all['max_dd_pct']*100:.1f}%"
            )
            print(
                f"{ticker:<9} 剔黑天鹅 ({m_ex['trades']}笔)      : "
                f"胜率 {m_ex['win_rate']*100:.1f}% | 净PF {m_ex['net_pf']:.2f} | "
                f"均R {m_ex['avg_net_r']:+.2f} | 最大回撤 {m_ex['max_dd_pct']*100:.1f}%"
            )

    # MAE / MFE 止损止盈精准标定
    print("\n四、 MAE / MFE 止损缓冲与止盈效率标定 [A3_Full_Gates]")
    print("-" * 96)
    for ticker in a3["ticker"].unique():
        sub_t = a3[a3["ticker"] == ticker]
        if sub_t.empty:
            continue
        wins = sub_t[sub_t["win"]]
        losses = sub_t[~sub_t["win"]]

        w_mae_p50 = wins["mae_atr"].median() if not wins.empty else 0.0
        w_mae_p80 = wins["mae_atr"].quantile(0.80) if not wins.empty else 0.0
        w_mae_pct = wins["mae_pct"].median() * 100 if not wins.empty else 0.0

        l_mfe_ge_1r = float((losses["mfe_r"] >= 1.0).mean() * 100) if not losses.empty else 0.0
        l_mfe_ge_15r = float((losses["mfe_r"] >= 1.5).mean() * 100) if not losses.empty else 0.0
        be_ratio = float(sub_t["exit_reason"].str.contains("be_protected|break_even").mean() * 100)

        print(f"▶ {ticker} (共 {len(sub_t)} 笔 | 胜 {len(wins)} / 负 {len(losses)})")
        print(
            f"   • 盈利单最大浮亏 MAE: 中位数 {w_mae_p50:.2f}x ATR ({w_mae_pct:.2f}%), "
            f"80分位 {w_mae_p80:.2f}x ATR  → 若远低于当前止损缓冲，可尝试收紧止损拉高 R"
        )
        print(
            f"   • 亏损单曾浮盈 MFE  : {l_mfe_ge_1r:.1f}% 曾浮盈>=1.0R, "
            f"{l_mfe_ge_15r:.1f}% 曾浮盈>=1.5R | 保本离场占比 {be_ratio:.1f}%"
        )
        exits = sub_t["exit_reason"].value_counts().to_dict()
        print(f"   • 出场原因分布      : {exits}")


# ==============================================================================
# 5. CLI 主入口
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="多标的消融回测与 MAE/MFE 参数标定诊断")
    parser.add_argument("--tickers", type=str, default="BTC-USD,QQQ,GOOGL,MSFT",
                        help="逗号分隔标的列表")
    parser.add_argument("--start", type=str, default="2020-01-01", help="回测起始日期")
    parser.add_argument("--oos-date", type=str, default="2025-01-01", help="样本外切分日期")
    parser.add_argument("--out-dir", type=str, default="results/diagnostics", help="CSV 导出目录")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    oos_ts = pd.Timestamp(args.oos_date)
    all_trades: List[Dict[str, Any]] = []

    for ticker in tickers:
        df = load_daily_ohlcv(ticker, start=args.start)
        if len(df) < 100:
            print(f"[跳过] {ticker}: 有效 K 线不足 ({len(df)} 根)")
            continue

        agg_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg_map["Volume"] = "sum"
        weekly_df = df.resample("W-FRI").agg(agg_map).dropna(subset=["Open", "High", "Low", "Close"])

        print(f"[计算] {ticker}: 构建逐根因果结构快照 ({len(df)} 根)...")
        snapshots = build_causal_snapshots(df, warmup=60, swing_order=5)

        for cfg in ABLATION_CONFIGS:
            if sc.is_crypto_asset(ticker) and cfg["name"] == "A4_US_LongOnly":
                continue
            tr_list = run_single_config(ticker, df, weekly_df, snapshots, cfg, oos_ts)
            all_trades.extend(tr_list)

    if not all_trades:
        print("[结束] 未产生任何有效成交记录，请检查数据源或起始日期。")
        return 1

    all_df = pd.DataFrame(all_trades)
    print_ablation_report(all_df, args.oos_date)

    out_dir = BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trades_diagnostic.csv"
    all_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[导出] 全量逐笔诊断明细已写入: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
