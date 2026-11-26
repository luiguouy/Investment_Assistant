#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨股票池盲测 (Out-of-Universe Test) + 多标的共享资金池日频盯市组合回测
路径: stage_one/backend/run_portfolio_oou_test.py

运行示例 (PowerShell):
  cd stage_one/backend
  python run_portfolio_oou_test.py --tickers NVDA,AMD,MU,TSM,AAPL,META,AMZN --start 2020-01-01 --oos-date 2025-01-01
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

# 组合级资金与风控参数
INIT_EQUITY = 100_000.0
RISK_PER_TRADE = 0.02       # 单笔风险预算占当前总权益 2%
MAX_SINGLE_WEIGHT = 0.35    # 单只标的最大仓位 35%（防单票暴雷）
MAX_TOTAL_EXPOSURE = 1.00   # 组合总仓位上限 100%（不加融资杠杆）
MAX_CONCURRENT_POS = 5      # 最多同时持有 5 只标的


def generate_oou_trades_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    oos_ts: pd.Timestamp,
) -> List[Dict[str, Any]]:
    """直接调用生产级 strategy_crypto.ASSET_PROFILES['us_stock'] 生成单标的交易记录。"""
    profile = sc.ASSET_PROFILES["us_stock"]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    weekly_df = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    snapshots = build_causal_snapshots(df, warmup=60, swing_order=5)

    trades: List[Dict[str, Any]] = []
    active_exit_t = -1
    last_setup_key = None

    for t in sorted(snapshots.keys()):
        if t <= active_exit_t:
            continue

        snap = snapshots[t]
        side = int(snap["bias"])
        swings = snap["swings"]

        if profile.get("long_only", True) and side == -1:
            continue
        if not _passes_box_channel_gate(side, snap["box"], snap["channel"]):
            continue

        anchor_sw = [s for s in swings if s["type"] == ("High" if side == 1 else "Low")]
        if not anchor_sw:
            continue
        setup_key = (side, anchor_sw[-1]["index"])
        if setup_key == last_setup_key:
            continue

        lvls = sc.extract_crypto_levels(df, t, side, swings, buffer_k=1.0)
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
            use_trend_gate=True,
            disallow_short_engulfing=True,
            weekly_df=weekly_df,
            stop_mode=profile["stop_mode"],
            max_stop_atr=profile["max_stop_atr"],
            allow_2b=profile["allow_2b"],
        )
        if not trig.get("triggered"):
            continue

        entry_px = float(trig["entry_price"])
        stop_px = float(trig["stop_price"])
        risk = abs(entry_px - stop_px)
        tp1_r = float(profile["tp1_r"])
        tp1_target = round(entry_px + side * tp1_r * risk, 2)

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
            slippage_bps=3.0,
            commission_bps=1.0,
            funding_rate_daily_bps=0.0,
        )
        last_setup_key = setup_key

        if res.get("status") != "completed":
            continue

        active_exit_t = int(res["exit_t"])
        entry_dt = pd.Timestamp(df.index[res["entry_t"]])
        exit_dt = pd.Timestamp(df.index[res["exit_t"]])

        # 将 exit_records 中的整数索引转换为具体日期，供日频组合盯市精确分批平仓
        dated_exits = [
            {
                "date": pd.Timestamp(df.index[r["bar"]]),
                "price": float(r["price"]),
                "weight": float(r["weight"]),
                "reason": r["reason"],
            }
            for r in res["exit_records"]
        ]

        trades.append({
            "ticker": ticker,
            "side": side,
            "pattern": trig["pattern"],
            "entry_date": entry_dt,
            "exit_date": exit_dt,
            "is_oos": bool(entry_dt >= oos_ts),
            "entry_price": entry_px,
            "stop_price": stop_px,
            "risk_pct": risk / entry_px,
            "net_pnl_pct": float(res["net_pnl_pct"]),
            "net_r": float(res["net_r"]),
            "hold_bars": int(res["hold_bars"]),
            "exit_reason": res["exit_reason"],
            "win": bool(res["net_pnl_pct"] > 0),
            "dated_exits": dated_exits,
        })

    return trades


def simulate_shared_portfolio(
    price_map: Dict[str, pd.DataFrame],
    all_trades: List[Dict[str, Any]],
    slippage_bps: float = 3.0,
    commission_bps: float = 1.0,
) -> Dict[str, Any]:
    """
    共享资金池日频盯市回测：
      - 严格按真实日历推进，多票共享现金池，上限 100% 现金不加杠杆；
      - 忠实还原分批止盈（TP1 减半 -> TP2 减 25% -> Runner 追踪）与并发持仓回撤。
    """
    all_dates = sorted(set().union(*(df.index for df in price_map.values())))
    entries_by_date: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for tr in all_trades:
        entries_by_date.setdefault(tr["entry_date"], []).append(tr)

    slip = slippage_bps / 10000.0
    comm = commission_bps / 10000.0

    cash = INIT_EQUITY
    open_positions: List[Dict[str, Any]] = []
    daily_records = []
    executed_count = 0
    skipped_cash_count = 0

    for dt in all_dates:
        # 1. 开盘核算当前总权益（估算新开仓仓位用）
        def _mtm_positions(col: str = "Close") -> float:
            val = 0.0
            for pos in open_positions:
                df_t = price_map[pos["ticker"]]
                if dt in df_t.index:
                    px = float(df_t.loc[dt, col])
                    pos["last_px"] = px
                else:
                    px = pos["last_px"]
                val += pos["shares"] * px
            return val

        est_equity = cash + _mtm_positions("Open")

        # 2. 处理当日新开仓信号（按风险百分比升序优先，先挑止损更紧凑的高质量信号）
        day_entries = sorted(entries_by_date.get(dt, []), key=lambda x: x["risk_pct"])
        for tr in day_entries:
            if len(open_positions) >= MAX_CONCURRENT_POS:
                skipped_cash_count += 1
                continue
            if any(p["ticker"] == tr["ticker"] for p in open_positions):
                continue

            target_w = min(MAX_SINGLE_WEIGHT, RISK_PER_TRADE / max(tr["risk_pct"], 1e-6))
            max_investable = max(0.0, est_equity * MAX_TOTAL_EXPOSURE - _mtm_positions("Open"))
            alloc_cash = min(est_equity * target_w, cash, max_investable)

            if alloc_cash < est_equity * 0.03:  # 剩余可用资金不足 3% 则跳过
                skipped_cash_count += 1
                continue

            eff_entry = tr["entry_price"] * (1.0 + slip)
            shares = (alloc_cash * (1.0 - comm)) / eff_entry
            cash -= alloc_cash
            executed_count += 1

            open_positions.append({
                "ticker": tr["ticker"],
                "init_shares": shares,
                "shares": shares,
                "last_px": tr["entry_price"],
                "pending_exits": list(tr["dated_exits"]),
            })

        # 3. 处理当日触发的分批/全部出场
        still_open = []
        for pos in open_positions:
            rem_exits = []
            for ex in pos["pending_exits"]:
                if ex["date"] <= dt:
                    sell_shares = min(pos["shares"], pos["init_shares"] * ex["weight"])
                    eff_exit = ex["price"] * (1.0 - slip)
                    proceeds = sell_shares * eff_exit * (1.0 - comm)
                    cash += proceeds
                    pos["shares"] -= sell_shares
                else:
                    rem_exits.append(ex)
            pos["pending_exits"] = rem_exits
            if pos["shares"] > 1e-8 and rem_exits:
                still_open.append(pos)
            elif pos["shares"] > 1e-8 and not rem_exits:
                # 兜底清零残余极小浮点份额
                cash += pos["shares"] * pos["last_px"]
        open_positions = still_open

        # 4. 收盘日频盯市
        pos_val = _mtm_positions("Close")
        total_eq = cash + pos_val
        exposure = pos_val / total_eq if total_eq > 0 else 0.0
        daily_records.append({
            "date": dt,
            "equity": total_eq,
            "exposure": exposure,
            "open_count": len(open_positions),
        })

    eq_df = pd.DataFrame(daily_records).set_index("date")
    return {
        "eq_df": eq_df,
        "executed": executed_count,
        "skipped": skipped_cash_count,
    }


def calc_curve_stats(eq_series: pd.Series) -> Dict[str, float]:
    eq = eq_series.dropna()
    if len(eq) < 2:
        return {"cum_ret": 0.0, "cagr": 0.0, "max_dd": 0.0, "calmar": 0.0, "sharpe": 0.0}
    days = max(1, (eq.index[-1] - eq.index[0]).days)
    years = days / 365.25
    cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0
    return {"cum_ret": cum_ret, "cagr": cagr, "max_dd": max_dd, "calmar": calmar, "sharpe": sharpe}


def main() -> int:
    parser = argparse.ArgumentParser(description="跨股票池盲测 + 共享资金池组合净值模拟")
    parser.add_argument("--tickers", type=str, default="NVDA,AMD,MU,TSM,AAPL,META,AMZN")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--oos-date", type=str, default="2025-01-01")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    oos_ts = pd.Timestamp(args.oos_date)

    price_map: Dict[str, pd.DataFrame] = {}
    all_trades: List[Dict[str, Any]] = []

    print(f"[跨池盲测] 加载标的: {', '.join(tickers)} (规则: ASSET_PROFILES['us_stock'] 一字未改)")
    for tk in tickers:
        df = load_daily_ohlcv(tk, start=args.start)
        if len(df) < 120:
            print(f"[跳过] {tk}: 数据不足")
            continue
        price_map[tk] = df
        all_trades.extend(generate_oou_trades_for_ticker(tk, df, oos_ts))

    if not all_trades:
        print("无有效交易记录。")
        return 1

    tr_df = pd.DataFrame(all_trades)

    # ---- 表 1：单标的跨池盲测表现 ----
    print("\n" + "=" * 96)
    print("一、 跨股票池单标的盲测表（Out-of-Universe，规则零微调）")
    print("=" * 96)
    print(f"{'标的':<7} | {'全量笔':>5} {'胜率':>6} {'净PF':>6} {'均R':>6} {'僵尸%':>6} | {'OOS笔':>5} {'OOS胜率':>7} {'OOS_PF':>6} {'OOS均R':>7}")
    print("-" * 96)
    for tk in price_map.keys():
        sub = tr_df[tr_df["ticker"] == tk]
        oos = sub[sub["is_oos"]]

        def _row_stat(d: pd.DataFrame):
            if d.empty:
                return 0, 0.0, 0.0, 0.0, 0.0
            w = d.loc[d["net_pnl_pct"] > 0, "net_pnl_pct"].sum()
            l = abs(d.loc[d["net_pnl_pct"] <= 0, "net_pnl_pct"].sum())
            pf = float(w / l) if l > 0 else (float("inf") if w > 0 else 0.0)
            z = float((d["exit_reason"] == "time_horizon").mean() * 100)
            return len(d), float(d["win"].mean() * 100), pf, float(d["net_r"].mean()), z

        n_a, wr_a, pf_a, r_a, z_a = _row_stat(sub)
        n_o, wr_o, pf_o, r_o, _ = _row_stat(oos)
        pf_as = f"{pf_a:6.2f}" if np.isfinite(pf_a) else "   inf"
        pf_os = f"{pf_o:6.2f}" if np.isfinite(pf_o) else "   inf"
        print(f"{tk:<7} | {n_a:>5d} {wr_a:>5.1f}% {pf_as} {r_a:>+6.2f} {z_a:>5.1f}% | {n_o:>5d} {wr_o:>6.1f}% {pf_os} {r_o:>+7.2f}")

    # 汇总全部 7 票盲测池整体指标
    w_all = tr_df.loc[tr_df["net_pnl_pct"] > 0, "net_pnl_pct"].sum()
    l_all = abs(tr_df.loc[tr_df["net_pnl_pct"] <= 0, "net_pnl_pct"].sum())
    pf_pool = w_all / l_all if l_all > 0 else float("inf")
    print("-" * 96)
    print(
        f"7票合计 | 共 {len(tr_df)} 笔 | 胜率 {tr_df['win'].mean()*100:.1f}% | "
        f"整体净PF {pf_pool:.2f} | 均R {tr_df['net_r'].mean():+.2f} | "
        f"僵尸单 {(tr_df['exit_reason'] == 'time_horizon').mean()*100:.1f}%"
    )

    # ---- 表 2：共享资金池组合净值 vs 基准对比 ----
    sim = simulate_shared_portfolio(price_map, all_trades)
    eq_df = sim["eq_df"]
    # 剔除前 60 天 warmup 再算组合绩效与基准
    eval_start = eq_df.index[min(60, len(eq_df) - 1)]
    eq_eval = eq_df.loc[eval_start:]
    p_all = calc_curve_stats(eq_eval["equity"])
    p_oos = calc_curve_stats(eq_eval.loc[eq_eval.index >= oos_ts, "equity"])

    # 等权买入持有基准（7 票等权）
    norm_closes = pd.DataFrame({
        tk: df["Close"].reindex(eq_eval.index).ffill()
        for tk, df in price_map.items()
    }).dropna()
    ew_bh = (norm_closes / norm_closes.iloc[0]).mean(axis=1)
    bh_all = calc_curve_stats(ew_bh)
    bh_oos = calc_curve_stats(ew_bh.loc[ew_bh.index >= oos_ts])

    print("\n" + "=" * 96)
    print(f"二、 共享资金池日频盯市组合绩效 (单笔风险2% / 单票上限35% / 总仓上限100%)")
    print("=" * 96)
    print(
        f"  • 成交统计: 实际开仓 {sim['executed']} 笔 | 因资金/仓位上限跳过 {sim['skipped']} 笔 | "
        f"平均资金利用率 {eq_eval['exposure'].mean()*100:.1f}% | 平均同时持仓 {eq_eval['open_count'].mean():.2f} 只"
    )
    print("-" * 96)
    print(f"{'组合/基准':<24} | {'区间':<5} | {'累计收益':>8} {'年化CAGR':>8} {'最大回撤':>8} {'卡玛比':>7} {'夏普':>6}")
    print("-" * 96)
    for label, period_tag, st in [
        ("策略共享资金池 (7票)", "全周期", p_all),
        ("等权 Buy&Hold (7票)", "全周期", bh_all),
        ("策略共享资金池 (7票)", "OOS", p_oos),
        ("等权 Buy&Hold (7票)", "OOS", bh_oos),
    ]:
        print(
            f"{label:<24} | {period_tag:<5} | {st['cum_ret']*100:>+7.1f}% {st['cagr']*100:>+7.1f}% "
            f"{st['max_dd']*100:>7.1f}% {st['calmar']:>7.2f} {st['sharpe']:>6.2f}"
        )

    out_csv = BASE_DIR / "results/diagnostics/oou_portfolio_trades.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tr_df.drop(columns=["dated_exits"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[导出] 跨池盲测逐笔明细已保存至: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
