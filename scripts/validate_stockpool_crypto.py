#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股池泛化验证: 加密引擎 (2B + 分级止盈 + 周线结构门禁) × 30 只美股个股池

目的 (2026-10-24 会话决议 ①):
  指数 (SPY/QQQ) 验证显示多头方向 81.8% 胜率为全项目首个显著结果, 但样本仅
  11-12 笔且高相关。个股池 (config/tickers.txt, 30 只 × 约 1600-5000 根日线,
  本地缓存) 是现有约束下最大的未开发样本源。本脚本回答:
    1. 加密引擎在个股上的成交规模与分布 (多空/出场/每标的);
    2. 池化期望/胜率/PF 的 bootstrap 置信区间 (全部 + 仅多头);
    3. 相对管理机制匹配随机入场的对照 (N1 无条件 / N2 周线结构匹配, 同口径
       剔除美股黑天鹅窗口, 逐笔标的/方向/几何匹配);
    4. 周线结构门禁消融;
    5. 季度聚类稳健统计 (个股高相关, 名义样本 ≠ 有效样本) 与生存者偏差警示。

复用 validate_crypto_stats / validate_index_stats 的统计组件, 零侵入。
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pa_engine as pe
import backtest as bt
import strategy_layer as sl

from validate_crypto_stats import (lifecycle_stats, bootstrap_cis, binom_two_sided,
                                   control_verdict, fee_stress)
from validate_index_stats import random_entry_control_multi

N_TRIALS = 2_000
RNG_SEED = 20261024
JSON_OUT = os.path.join("..", "results", "stockpool_crypto_validation.json")


def load_pool(tickers_file):
    with open(tickers_file, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    data, fwd_by_tk, events_by_tk, meta = {}, {}, {}, {}
    for tk in names:
        df = bt.load_ticker(tk)
        if df.empty or len(df) < 150:
            meta[tk] = {"bars": 0 if df.empty else len(df), "skipped": True}
            continue
        data[tk] = df
        fwd_by_tk[tk] = bt.Forward(df)
        w = bt.weekly_from_daily(df)
        ev = bt.enumerate_events(df, w, method="h2", buffer_k=0.0, volume_filter=True)
        ev = [e for e in ev if not sl.is_stock_black_swan(e["date"])]
        events_by_tk[tk] = {arm: [e for e in ev if e["arm"] == arm]
                            for arm in ("MTF", "MTF+HTF")}
        meta[tk] = {"bars": len(df), "range": [str(df.index[0].date()), str(df.index[-1].date())],
                    "events_mtf": len(events_by_tk[tk]["MTF"])}
    return data, fwd_by_tk, events_by_tk, meta


def collect(data, fwd_by_tk, events_by_tk, arm, trend_gate=True):
    frames = []
    for tk, ev in events_by_tk.items():
        if not ev.get(arm):
            continue
        rec = bt.build_records(fwd_by_tk[tk], ev[arm], df=data[tk], entry_mode="strategy_crypto",
                               retest_level="swing", min_rr=2.0, max_wait_bars=15,
                               volume_filter=True, asset_type="crypto", ticker=tk,
                               exclude_black_swans=True, use_crypto_trend_gate=trend_gate)
        if rec is None or rec.empty:
            continue
        tr = rec.drop_duplicates(subset=["t", "side", "entry_price"]).copy()
        tr["ticker"] = tk
        frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cluster_quarterly(tr):
    """季度聚类稳健统计: 个股高相关, 同季入场本质上是同一个市场状态观测。"""
    if tr.empty:
        return {}
    g = tr.copy()
    g["quarter"] = pd.PeriodIndex(g["date"], freq="Q")
    per_q = g.groupby("quarter")["pnl_pct"].agg(["mean", "count"])
    n_q = len(per_q)
    if n_q < 3:
        return {"quarters": n_q, "note": "季度数不足, 无法聚类推断"}
    m = float(per_q["mean"].mean())
    s = float(per_q["mean"].std(ddof=1))
    t_stat = m / (s / np.sqrt(n_q)) if s > 0 else float("inf")
    return {
        "quarters": int(n_q),
        "unique_entry_dates": int(g["date"].nunique()),
        "cluster_mean": m,
        "cluster_sd": s,
        "cluster_t": t_stat,
        "sig_05": bool(abs(t_stat) > 2.0),
    }


def main() -> int:
    t0 = time.time()
    pe.reset_fetch_state()
    tickers_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "tickers.txt")
    data, fwd_by_tk, events_by_tk, meta = load_pool(tickers_file)
    pool = list(data.keys())
    total_bars = sum(m["bars"] for m in meta.values())
    total_events = sum(m.get("events_mtf", 0) for m in meta.values())

    print("=" * 96)
    print(f"个股池泛化验证: 加密引擎 × {len(pool)} 只美股 | 总 {total_bars:,} 根日线 | "
          f"MTF 事件 {total_events} 个 (全部剔除美股黑天鹅窗口)")
    print("⚠️  生存者偏差警示: 名单为 2026 年存续大市值股, 任何做多为主的结果都被系统性高估")
    print("=" * 96)

    report = {"meta": {"tickers": pool, "ticker_meta": meta, "total_bars": total_bars,
                       "total_events_mtf": total_events,
                       "survivorship_bias_warning": True,
                       "n_trials": N_TRIALS, "seed": RNG_SEED}}

    # ---- [1] 基线 ----
    print("\n[1] 基线 (周线结构门禁开启)")
    trades_by_arm = {}
    for arm in ("MTF", "MTF+HTF"):
        tr = collect(data, fwd_by_tk, events_by_tk, arm)
        trades_by_arm[arm] = tr
        st = lifecycle_stats(tr)
        lo = lifecycle_stats(tr[tr["side"] == 1]) if not tr.empty else {}
        sh = lifecycle_stats(tr[tr["side"] == -1]) if not tr.empty else {}
        print(f"  [{arm}] {st['trades']} 笔 | 胜率 {st['win_rate']:.1%} | PF {st['profit_factor']:.2f} | "
              f"{st['avg_r']:+.2f}R | 期望 {st['expectancy']:+.2%} | 多 {st['long_trades']} / 空 {st['short_trades']}")
        if lo:
            print(f"    多头: {lo['trades']} 笔 | 胜率 {lo['win_rate']:.1%} | PF {lo['profit_factor']:.2f} | 期望 {lo['expectancy']:+.2%}")
        if sh:
            print(f"    空头: {sh['trades']} 笔 | 胜率 {sh['win_rate']:.1%} | PF {sh['profit_factor']:.2f} | 期望 {sh['expectancy']:+.2%}")
        cq = cluster_quarterly(tr)
        print(f"    季度聚类: {cq.get('quarters', 0)} 个季度 / {cq.get('unique_entry_dates', 0)} 个独立入场日 | "
              f"簇均值 {cq.get('cluster_mean', 0):+.2%} | 簇t {cq.get('cluster_t', 0):+.2f}"
              f"{' (|t|>2 显著)' if cq.get('sig_05') else ''}")
        report[f"baseline_{arm}"] = {"all": st, "long": lo, "short": sh, "cluster": cq}

    mtf_tr = trades_by_arm["MTF"]
    if mtf_tr.empty:
        print("无成交, 终止")
        return 1

    # 每标的分布
    per = mtf_tr.groupby("ticker").agg(trades=("pnl_pct", "size"), exp=("pnl_pct", "mean"),
                                       wr=("pnl_pct", lambda x: (x > 0).mean()))
    print("\n  每标的分布 (MTF):")
    for tk, r in per.sort_values("trades", ascending=False).iterrows():
        print(f"    {tk:>6}: {int(r['trades']):>3} 笔 | 胜率 {r['wr']:.0%} | 期望 {r['exp']:+.2%}")
    report["per_ticker_mtf"] = {tk: {"trades": int(r["trades"]), "win_rate": float(r["wr"]),
                                     "expectancy": float(r["exp"])}
                                for tk, r in per.iterrows()}

    # ---- [2] Bootstrap ----
    print(f"\n[2] Bootstrap ({10000:,} 次) — MTF 池化")
    boot_out = {}
    for label, tr in (("全部", mtf_tr), ("仅多头", mtf_tr[mtf_tr["side"] == 1])):
        if tr.empty:
            continue
        bs = bootstrap_cis(tr)
        st = lifecycle_stats(tr)
        bs["binomial_vs_coinflip"] = {"wins": st["n_wins"], "n": st["trades"],
                                      "p_value": binom_two_sided(st["n_wins"], st["trades"])}
        boot_out[label] = bs
        print(f"\n  [{label}] n={st['trades']} (胜 {st['n_wins']})")
        for key in ("win_rate", "expectancy", "avg_r", "profit_factor"):
            ci = bs[key]
            if "p50" not in ci:
                continue
            flag = "✓ 排除零假设" if ci["ci_excludes_null"] else "✗ 无法排除"
            print(f"    {ci['label']:<18} 95%CI [{ci['p2_5']:+.4f}, {ci['p97_5']:+.4f}] → {flag}")
        print(f"    胜率 vs 50% 二项 p = {bs['binomial_vs_coinflip']['p_value']:.4f}")
    report["bootstrap_mtf"] = boot_out

    # ---- [3] 随机对照 ----
    print(f"\n[3] 随机入场对照 ({N_TRIALS:,} 次, 逐笔标的/方向/几何匹配, 同口径剔除黑天鹅)")
    rc_out = {}
    prep_cache = {}   # 逐标的预计算缓存, N1/N2 两次对照复用
    for tag, gate in (("N1_无条件随机", False), ("N2_结构门禁匹配", True)):
        ctrl = random_entry_control_multi(data, mtf_tr, gate_match=gate,
                                          n_trials=N_TRIALS, prep_cache=prep_cache)
        st = lifecycle_stats(mtf_tr)
        v = {"expectancy": control_verdict(st["expectancy"], ctrl["expectancy"]),
             "win_rate": control_verdict(st["win_rate"], ctrl["win_rate"]),
             "profit_factor": control_verdict(st["profit_factor"], ctrl["profit_factor"])}
        rc_out[tag] = {"n_trials": ctrl["n_trials"], "verdicts": v}
        ev, wr_v = v["expectancy"], v["win_rate"]
        print(f"\n  [{tag}] 对照 {ctrl['n_trials']:,} 次")
        print(f"    期望 {st['expectancy']:+.2%} | 对照 p5~p95: {ev['ctrl_p5']:+.2%} ~ {ev['ctrl_p95']:+.2%} "
              f"(中位 {ev['ctrl_p50']:+.2%}) | 分位 {ev['percentile']:.3f} → {ev['verdict']}")
        print(f"    胜率 {st['win_rate']:.1%} | 对照 p5~p95: {wr_v['ctrl_p5']:.1%} ~ {wr_v['ctrl_p95']:.1%} "
              f"(中位 {wr_v['ctrl_p50']:.1%}) | 分位 {wr_v['percentile']:.3f} → {wr_v['verdict']}")
    report["random_control_mtf"] = rc_out

    # ---- [4] 门禁消融 ----
    print("\n[4] 周线结构门禁消融 (MTF)")
    abl = {}
    for tag, gate in (("门禁开启(基线)", True), ("门禁关闭", False)):
        tr = collect(data, fwd_by_tk, events_by_tk, "MTF", trend_gate=gate)
        st = lifecycle_stats(tr) if not tr.empty else {"trades": 0}
        abl[tag] = st
        if st["trades"]:
            lo = lifecycle_stats(tr[tr["side"] == 1])
            print(f"  {tag}: {st['trades']} 笔 | 胜率 {st['win_rate']:.1%} | PF {st['profit_factor']:.2f} | "
                  f"期望 {st['expectancy']:+.2%} | 多 {st['long_trades']}/空 {st['short_trades']} "
                  f"| 多头期望 {lo['expectancy']:+.2%}")
        else:
            print(f"  {tag}: 无成交")
    report["gate_ablation_mtf"] = abl

    # ---- [5] 费率 ----
    fs = fee_stress(mtf_tr)
    report["fees_mtf"] = fs
    print("\n[5] 费率: ", " | ".join(f"{f['fee_one_side']:.2%}→期望{f['expectancy']:+.2%}" for f in fs))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), JSON_OUT)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入: {out_path}")
    print(f"总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
