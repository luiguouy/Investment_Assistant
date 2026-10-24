#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股池验证补跑: 多头方向随机对照 (N1/N2) + 门禁消融 + 费率。

主验证 (validate_stockpool_crypto.py) 的混合多空对照已完成:
  基线 MTF 168 笔 (多97/空71), 多头期望 +3.76% (bootstrap 排除零假设),
  混合 N1 分位 0.059。本脚本补齐决策关键的两块:
    1. 仅多头槽位的 N1/N2 对照 —— 推荐形态是 long-only, 归因问题应问多头;
    2. 周线结构门禁消融 (gate-off) 与费率。
速度优化: N2 不做拒绝采样, 直接从预计算的「周线结构偏向=+1」合格柱数组中抽样;
对照试验数 800 (分位分辨率 ~0.1%, 足够 0.95 判定); 每 200 次打印进度;
每节完成即增量写入 JSON。
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
import strategy_crypto as sc

from validate_crypto_stats import lifecycle_stats, control_verdict, fee_stress
from validate_index_stats import _prev_week_structure_bias
from validate_stockpool_crypto import load_pool, collect, cluster_quarterly

N_TRIALS = 800
RNG_SEED = 20261025
JSON_OUT = os.path.join("..", "results", "stockpool_crypto_followup.json")


def control_long_only(data, long_trades, gate_match, n_trials, seed, max_hold=60, min_rr=2.0):
    """仅多头槽位对照: N1 任意柱, N2 直接从 bias==+1 合格柱数组抽样 (零拒绝重试)。"""
    prep = {}
    for tk, df in data.items():
        n = len(df)
        bias = _prev_week_structure_bias(df)
        k_lo, k_hi = 250, n - max_hold - 2
        ks = np.arange(k_lo, k_hi + 1)
        ok = ks[ks < n - 1]
        prep[tk] = {
            "df": df, "opens": df["Open"].to_numpy(float),
            "bar_rng": pd.Series(df["High"] - df["Low"]).rolling(14).median().to_numpy(float),
            "dates": df.index,
            "k_any": ok,
            "k_gate": ok[bias[ok] == 1] if gate_match else ok,
        }
    geo = []
    for _, tr in long_trades.iterrows():
        p = prep[tr["ticker"]]
        t = int(tr["t"])
        entry, stop, tp1 = float(tr["entry_price"]), float(tr["stop_price"]), float(tr["tp_price"])
        brng = p["bar_rng"][t]
        risk = abs(entry - stop)
        if np.isfinite(brng) and brng > 0 and risk > 0:
            geo.append((risk / brng, abs(tp1 - entry) / risk))
    slots = [str(r["ticker"]) for _, r in long_trades.iterrows()]
    rng = np.random.default_rng(seed)

    wins_, pfs_, exps_, rs_ = [], [], [], []
    t_start = time.time()
    for trial in range(n_trials):
        pnls, rs = [], []
        for ticker in slots:
            p = prep[ticker]
            if len(p["k_gate"]) == 0 or not geo:
                continue
            for _attempt in range(40):
                k = int(p["k_gate"][rng.integers(0, len(p["k_gate"]))])
                entry_t = k + 1
                entry_date = p["dates"][entry_t]
                if sl.is_stock_black_swan(entry_date):
                    continue
                entry = p["opens"][entry_t]
                brng = p["bar_rng"][k]
                if not np.isfinite(brng) or brng <= 0 or entry <= 0:
                    continue
                stop_atr, rr1 = geo[int(rng.integers(0, len(geo)))]
                stop = entry - stop_atr * brng
                risk = entry - stop
                tp1 = entry + rr1 * risk
                tp2 = entry + max(rr1 + 0.5, min_rr + 1.5) * risk
                trade = sc.simulate_crypto_trade(p["df"], entry_t, entry, 1, stop,
                                                 tp1_price=tp1, tp2_price=tp2,
                                                 impulse_leg=None, max_hold_bars=max_hold,
                                                 min_rr=min_rr)
                if trade.get("status") != "completed":
                    continue
                exit_date = p["dates"][trade["exit_t"]]
                if sl.trade_overlaps_stock_black_swan(entry_date, exit_date):
                    continue
                pnls.append(trade["pnl_pct"])
                rs.append(trade["r_multiple"])
                break
        if not pnls:
            continue
        w = sum(1 for x in pnls if x > 0)
        gl = abs(sum(x for x in pnls if x <= 0))
        wins_.append(w / len(pnls))
        pfs_.append((sum(x for x in pnls if x > 0) / gl) if gl > 0 else float("inf"))
        exps_.append(float(np.mean(pnls)))
        rs_.append(float(np.mean(rs)))
        if (trial + 1) % 200 == 0:
            print(f"    ... {trial + 1}/{n_trials} 次 ({time.time() - t_start:.0f}s)", flush=True)
    return {"n_trials": len(exps_), "gate_match": gate_match,
            "win_rate": np.array(wins_), "profit_factor": np.array(pfs_),
            "expectancy": np.array(exps_), "avg_r": np.array(rs_)}


def main() -> int:
    t0 = time.time()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), JSON_OUT)
    report = {"meta": {"n_trials": N_TRIALS, "seed": RNG_SEED,
                       "note": "多头槽位对照(直接抽样) + 门禁消融 + 费率; 混合口径见主验证日志"}}

    def flush_json():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    pe.reset_fetch_state()
    tickers_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "tickers.txt")
    data, fwd_by_tk, events_by_tk, meta = load_pool(tickers_file)
    mtf_tr = collect(data, fwd_by_tk, events_by_tk, "MTF")
    longs = mtf_tr[mtf_tr["side"] == 1]
    st = lifecycle_stats(mtf_tr)
    lo = lifecycle_stats(longs)
    print(f"基线复现: 全部 {st['trades']} 笔 ({st['expectancy']:+.2%}) | 多头 {lo['trades']} 笔 "
          f"({lo['win_rate']:.1%}, {lo['expectancy']:+.2%})", flush=True)
    report["baseline_mtf"] = {"all": st, "long": lo}
    flush_json()

    for tag, gate in (("N1_long_任意随机", False), ("N2_long_结构匹配随机", True)):
        print(f"\n[{tag}] {N_TRIALS} 次 (多头槽位)", flush=True)
        ctrl = control_long_only(data, longs, gate, N_TRIALS, RNG_SEED)
        v = {"expectancy": control_verdict(lo["expectancy"], ctrl["expectancy"]),
             "win_rate": control_verdict(lo["win_rate"], ctrl["win_rate"]),
             "profit_factor": control_verdict(lo["profit_factor"], ctrl["profit_factor"])}
        report[tag] = {"n_trials": ctrl["n_trials"], "verdicts": v}
        ev, wr_v, pf_v = v["expectancy"], v["win_rate"], v["profit_factor"]
        print(f"  期望 {lo['expectancy']:+.2%} | 对照 p5~p95: {ev['ctrl_p5']:+.2%} ~ {ev['ctrl_p95']:+.2%} "
              f"(中位 {ev['ctrl_p50']:+.2%}) | 分位 {ev['percentile']:.3f} → {ev['verdict']}", flush=True)
        print(f"  胜率 {lo['win_rate']:.1%} | 对照 p5~p95: {wr_v['ctrl_p5']:.1%} ~ {wr_v['ctrl_p95']:.1%} "
              f"(中位 {wr_v['ctrl_p50']:.1%}) | 分位 {wr_v['percentile']:.3f} → {wr_v['verdict']}", flush=True)
        print(f"  PF   {lo['profit_factor']:.2f} | 对照 p5~p95: {pf_v['ctrl_p5']:.2f} ~ {pf_v['ctrl_p95']:.2f} "
              f"(中位 {pf_v['ctrl_p50']:.2f}) | 分位 {pf_v['percentile']:.3f} → {pf_v['verdict']}", flush=True)
        flush_json()

    print("\n[门禁消融 gate-off]", flush=True)
    tr_off = collect(data, fwd_by_tk, events_by_tk, "MTF", trend_gate=False)
    st_off = lifecycle_stats(tr_off)
    lo_off = lifecycle_stats(tr_off[tr_off["side"] == 1]) if not tr_off.empty else {}
    report["gate_off_mtf"] = {"all": st_off, "long": lo_off}
    print(f"  gate-off: {st_off['trades']} 笔 | 胜率 {st_off['win_rate']:.1%} | PF {st_off['profit_factor']:.2f} | "
          f"期望 {st_off['expectancy']:+.2%} | 多 {st_off['long_trades']}/空 {st_off['short_trades']}", flush=True)
    if lo_off:
        print(f"  gate-off 多头: {lo_off['trades']} 笔 | 胜率 {lo_off['win_rate']:.1%} | "
              f"期望 {lo_off['expectancy']:+.2%}", flush=True)
    report["gate_off_cluster"] = cluster_quarterly(tr_off[tr_off["side"] == 1]) if not tr_off.empty else {}
    flush_json()

    report["fees_long"] = fee_stress(longs)
    for f in report["fees_long"]:
        print(f"  费率 {f['fee_one_side']:.3%} → 多头期望 {f['expectancy']:+.2%} | PF {f['profit_factor']:.2f}", flush=True)
    flush_json()
    print(f"\n完成, 总耗时 {time.time() - t0:.0f}s → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
