#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数 ETF (SPY/QQQ) 策略统计验证 — 全程剔除美股黑天鹅极端行情窗口

背景:
  应用户要求, 测试标的从 BTC-USD 切换为 SPY 与 QQQ, 且测试全程忽略黑天鹅极端行情
  (strategy_layer.STOCK_BLACK_SWANS 定义的六大美股系统性危机窗口: 2008 GFC、
  2011 美债降级、2015-16 闪崩、2018 Q4、2020 熔断、2022 加息熊市)。
  描述性对比 (results/backtest_index_blackswan.json) 显示: 美股四层闭环与道氏波浪
  闭环在指数上均为负期望 (PF 0.85~0.91), 唯加密引擎 (2B+分级止盈+200 EMA 门禁)
  转正 (MTF 17 笔, 胜率 58.8%, PF 1.50, 期望 +1.36%; 多头 83.3% 胜率, 空头 0/5)。
  本脚本对该结果做统计检验: bootstrap 置信区间 + 管理机制匹配的随机入场对照
  (对照同样剔除黑天鹅窗口, 保证口径一致), 并做 200 EMA 门禁消融。

复用 scripts/validate_crypto_stats.py 的统计组件 (零侵入)。
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pa_engine as pe
import backtest as bt
import strategy_crypto as sc
import strategy_layer as sl

from validate_crypto_stats import (lifecycle_stats, bootstrap_cis, binom_two_sided,
                                   control_verdict, fee_stress, crypto_params)

# ==============================================================================
# 配置
# ==============================================================================

TICKERS = ["SPY", "QQQ"]
N_BOOT = 10_000
N_TRIALS = 2_000
RNG_SEED = 20261024
WARMUP = 250
JSON_OUT = os.path.join("..", "results", "index_stat_validation.json")


# ==============================================================================
# 多标的随机入场对照 (黑天鹅感知 + 逐笔标的/方向匹配 + 管理机制一致)
# ==============================================================================

def random_entry_control_multi(data: dict, real_trades: pd.DataFrame,
                               gate_match: bool, n_trials: int = N_TRIALS,
                               seed: int = RNG_SEED, max_hold: int = 60,
                               min_rr: float = 2.0, ema_span: int = 200) -> dict:
    """
    与 validate_crypto_stats.random_entry_control 同构, 三点扩展:
      - 多标的: 每个槽位在自己所属标的的序列上抽随机触发根 (保持标的构成);
      - 黑天鹅感知: 入场日落在美股危机窗口内、或持仓跨越窗口的抽样直接作废重抽
        (与被检验策略的剔除口径完全一致);
      - 几何池按方向跨标的池化 (止损距离/振幅与 RR 均为无量纲量)。
    """
    prep = {}
    for tk, df in data.items():
        n = len(df)
        prep[tk] = {
            "df": df,
            "n": n,
            "opens": df["Open"].to_numpy(dtype=float),
            "closes": df["Close"].to_numpy(dtype=float),
            "ema": df["Close"].ewm(span=ema_span, adjust=False).mean().to_numpy(dtype=float),
            "bar_rng": pd.Series(df["High"] - df["Low"]).rolling(14).median().to_numpy(dtype=float),
            "dates": df.index,
            "k_lo": WARMUP,
            "k_hi": n - max_hold - 2,
        }

    geo = {1: [], -1: []}
    for _, tr in real_trades.iterrows():
        side, t = int(tr["side"]), int(tr["t"])
        entry, stop, tp1 = float(tr["entry_price"]), float(tr["stop_price"]), float(tr["tp_price"])
        brng = prep[tr["ticker"]]["bar_rng"][t]
        risk = abs(entry - stop)
        if not np.isfinite(brng) or brng <= 0 or risk <= 0 or entry <= 0:
            continue
        geo[side].append((risk / brng, abs(tp1 - entry) / risk))

    slots = [(str(r_["ticker"]), int(r_["side"])) for _, r_ in real_trades.iterrows()]
    rng = np.random.default_rng(seed)

    wins_, pfs_, exps_, rs_ = [], [], [], []
    for _trial in range(n_trials):
        pnls, rs = [], []
        for ticker, side in slots:
            pool = geo.get(side) or []
            if not pool:
                continue
            p = prep[ticker]
            for _attempt in range(80):
                k = int(rng.integers(p["k_lo"], p["k_hi"] + 1))
                if gate_match:
                    if side == 1 and p["closes"][k] < p["ema"][k]:
                        continue
                    if side == -1 and p["closes"][k] > p["ema"][k]:
                        continue
                entry_t = k + 1
                entry_date = p["dates"][entry_t]
                if sl.is_stock_black_swan(entry_date):
                    continue
                entry = p["opens"][entry_t]
                brng = p["bar_rng"][k]
                if not np.isfinite(brng) or brng <= 0 or entry <= 0:
                    continue
                stop_atr, rr1 = pool[int(rng.integers(0, len(pool)))]
                if side == 1:
                    stop = entry - stop_atr * brng
                    risk = entry - stop
                    tp1 = entry + rr1 * risk
                    tp2 = entry + max(rr1 + 0.5, min_rr + 1.5) * risk
                else:
                    stop = entry + stop_atr * brng
                    risk = stop - entry
                    tp1 = entry - rr1 * risk
                    tp2 = entry - max(rr1 + 0.5, min_rr + 1.5) * risk
                trade = sc.simulate_crypto_trade(p["df"], entry_t, entry, side, stop,
                                                 tp1_price=tp1, tp2_price=tp2,
                                                 impulse_leg=None,
                                                 max_hold_bars=max_hold, min_rr=min_rr)
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

    return {"n_trials": len(exps_), "gate_match": gate_match,
            "win_rate": np.array(wins_), "profit_factor": np.array(pfs_),
            "expectancy": np.array(exps_), "avg_r": np.array(rs_)}


# ==============================================================================
# 主流程
# ==============================================================================

def collect_trades(entry_mode: str, events_by_tk: dict, data: dict, fwd_by_tk: dict,
                   arm: str, trend_gate: bool = True) -> pd.DataFrame:
    """跨标的池化某策略模式的成交交易 (附带 ticker 列)。"""
    frames = []
    for tk in TICKERS:
        ev = events_by_tk[tk].get(arm, [])
        if not ev:
            continue
        kw = dict(fwd=fwd_by_tk[tk], events=ev, df=data[tk], entry_mode=entry_mode,
                  retest_level="swing", min_rr=2.0, max_wait_bars=15,
                  volume_filter=True, asset_type="stock" if entry_mode == "strategy" else "crypto",
                  ticker=tk, exclude_black_swans=True)
        if entry_mode == "strategy_crypto":
            kw["use_crypto_trend_gate"] = trend_gate
        rec = bt.build_records(**kw)
        if rec is None or rec.empty:
            continue
        tr = rec.drop_duplicates(subset=["t", "side", "entry_price"])
        tr = tr.copy()
        tr["ticker"] = tk
        frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="指数 ETF 策略统计验证 (剔除美股黑天鹅)")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    t0 = time.time()
    pe.reset_fetch_state()
    data, fwd_by_tk, events_by_tk = {}, {}, {}
    for tk in TICKERS:
        df = bt.load_ticker(tk)
        if df.empty:
            print(f"[错误] 无法加载 {tk}")
            return 1
        data[tk] = df
        fwd_by_tk[tk] = bt.Forward(df)
        w = bt.weekly_from_daily(df)
        ev = bt.enumerate_events(df, w, method="h2", buffer_k=0.0, volume_filter=True)
        ev = [e for e in ev if not sl.is_stock_black_swan(e["date"])]  # 危机窗口事件剔除
        events_by_tk[tk] = {arm: [e for e in ev if e["arm"] == arm] for arm in ("MTF", "MTF+HTF")}

    print("=" * 96)
    for tk in TICKERS:
        df = data[tk]
        print(f"指数统计验证 | {tk} | {len(df)} 根日线 ({df.index[0].date()} → {df.index[-1].date()}) | "
              f"剔除黑天鹅后事件 MTF={len(events_by_tk[tk]['MTF'])} / MTF+HTF={len(events_by_tk[tk]['MTF+HTF'])}")
    print("=" * 96)

    report = {"meta": {"tickers": TICKERS, "n_boot": N_BOOT, "n_trials": args.trials,
                       "seed": RNG_SEED,
                       "black_swan_windows": [f'{b["name"]} [{b["start"]}~{b["end"]}]'
                                              for b in sl.STOCK_BLACK_SWANS]}}

    # ---- [1] 基线: 两个闭环引擎 (加密引擎 + 美股四层) ----
    print("\n[1] 基线 (全部剔除黑天鹅窗口, SPY+QQQ 池化)")
    trades_pool = {}
    for arm in ("MTF", "MTF+HTF"):
        cr = collect_trades("strategy_crypto", events_by_tk, data, fwd_by_tk, arm)
        st = collect_trades("strategy", events_by_tk, data, fwd_by_tk, arm)
        trades_pool[(arm, "crypto")] = cr
        trades_pool[(arm, "stock")] = st
        for name, tr in (("加密引擎", cr), ("美股四层", st)):
            if tr.empty:
                print(f"  [{arm} · {name}] 无成交")
                continue
            s = lifecycle_stats(tr)
            per = " / ".join(f"{tk}:{(tr['ticker'] == tk).sum()}" for tk in TICKERS)
            print(f"  [{arm} · {name}] {s['trades']} 笔 ({per}) | 胜率 {s['win_rate']:.1%} | "
                  f"PF {s['profit_factor']:.2f} | {s['avg_r']:+.2f}R | 期望 {s['expectancy']:+.2%} | "
                  f"多 {s['long_trades']} / 空 {s['short_trades']}")
    report["baseline"] = {f"{arm}_{name}": lifecycle_stats(trades_pool[(arm, name)])
                          for arm in ("MTF", "MTF+HTF") for name in ("crypto", "stock")}

    # ---- [2] Bootstrap + 二项检验 (加密引擎 MTF, 含多头子集) ----
    print(f"\n[2] Bootstrap 置信区间 ({N_BOOT:,} 次) — 加密引擎 MTF (SPY+QQQ 池化)")
    boot_out = {}
    for label, tr in (("全部", trades_pool[("MTF", "crypto")]),
                      ("仅多头", trades_pool[("MTF", "crypto")]
                       [trades_pool[("MTF", "crypto")]["side"] == 1] if not trades_pool[("MTF", "crypto")].empty else pd.DataFrame())):
        if tr.empty:
            continue
        bs = bootstrap_cis(tr)
        st = lifecycle_stats(tr)
        bs["binomial_vs_coinflip"] = {"wins": st["n_wins"], "n": st["trades"],
                                      "p_value": binom_two_sided(st["n_wins"], st["trades"])}
        boot_out[label] = bs
        print(f"\n  [{label}] n={st['trades']} (胜 {st['n_wins']} / 负 {st['n_losses']})")
        for key in ("win_rate", "expectancy", "avg_r", "profit_factor"):
            ci = bs[key]
            if "p50" not in ci:
                continue
            flag = "✓ 排除零假设" if ci["ci_excludes_null"] else "✗ 无法排除"
            print(f"    {ci['label']:<18} 95%CI [{ci['p2_5']:+.4f}, {ci['p97_5']:+.4f}] "
                  f"中位 {ci['p50']:+.4f} → {flag}")
        print(f"    胜率 vs 50% (精确二项): p = {bs['binomial_vs_coinflip']['p_value']:.3f}")
    report["bootstrap_crypto_mtf"] = boot_out

    # ---- [3] 随机入场对照 (黑天鹅感知, 2000 次) ----
    print(f"\n[3] 随机入场对照 ({args.trials:,} 次试验, 逐笔匹配标的/方向/几何/出场机制, 同口径剔除黑天鹅)")
    rc_out = {}
    for arm in ("MTF",):
        tr = trades_pool[(arm, "crypto")]
        if tr.empty:
            continue
        arm_out = {}
        for tag, gate in (("N1_无条件随机", False), ("N2_门禁匹配随机", True)):
            ctrl = random_entry_control_multi(data, tr, gate_match=gate, n_trials=args.trials)
            st = lifecycle_stats(tr)
            v = {"expectancy": control_verdict(st["expectancy"], ctrl["expectancy"]),
                 "profit_factor": control_verdict(st["profit_factor"], ctrl["profit_factor"]),
                 "win_rate": control_verdict(st["win_rate"], ctrl["win_rate"])}
            arm_out[tag] = {"n_trials": ctrl["n_trials"], "verdicts": v}
            ev, pf_v, wr_v = v["expectancy"], v["profit_factor"], v["win_rate"]
            print(f"\n  [{arm} · {tag}] 对照 {ctrl['n_trials']:,} 次")
            print(f"    期望 {st['expectancy']:+.2%} | 对照 p5~p95: {ev['ctrl_p5']:+.2%} ~ {ev['ctrl_p95']:+.2%} "
                  f"(中位 {ev['ctrl_p50']:+.2%}) | 分位 {ev['percentile']:.3f} → {ev['verdict']}")
            print(f"    PF   {st['profit_factor']:.2f} | 对照 p5~p95: {pf_v['ctrl_p5']:.2f} ~ {pf_v['ctrl_p95']:.2f} "
                  f"(中位 {pf_v['ctrl_p50']:.2f}) | 分位 {pf_v['percentile']:.3f} → {pf_v['verdict']}")
            print(f"    胜率 {st['win_rate']:.1%} | 对照 p5~p95: {wr_v['ctrl_p5']:.1%} ~ {wr_v['ctrl_p95']:.1%} "
                  f"(中位 {wr_v['ctrl_p50']:.1%}) | 分位 {wr_v['percentile']:.3f} → {wr_v['verdict']}")
        rc_out[arm] = arm_out
    report["random_control"] = rc_out

    # ---- [4] 200 EMA 门禁消融 ----
    print("\n[4] 200 EMA 门禁消融 (加密引擎, MTF)")
    abl = {}
    for tag, gate in (("门禁开启(基线)", True), ("门禁关闭", False)):
        tr = collect_trades("strategy_crypto", events_by_tk, data, fwd_by_tk, "MTF", trend_gate=gate)
        s = lifecycle_stats(tr) if not tr.empty else {"trades": 0}
        abl[tag] = s
        if s["trades"]:
            print(f"  {tag}: {s['trades']} 笔 | 胜率 {s['win_rate']:.1%} | PF {s['profit_factor']:.2f} | "
                  f"期望 {s['expectancy']:+.2%} | 多 {s['long_trades']} / 空 {s['short_trades']}")
        else:
            print(f"  {tag}: 无成交")
    report["trend_gate_ablation"] = abl

    # ---- [5] 费率压力 ----
    print("\n[5] 交易成本压力 (每笔双向 ≈ 2×单边费率, 加密引擎 MTF)")
    tr = trades_pool[("MTF", "crypto")]
    if not tr.empty:
        fs = fee_stress(tr)
        report["fees_crypto_mtf"] = fs
        for f in fs:
            print(f"    单边 {f['fee_one_side']:.3%} → 胜率 {f['win_rate']:.1%} | "
                  f"PF {f['profit_factor']:.2f} | 期望 {f['expectancy']:+.2%}")

    # ---- 落盘 ----
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), JSON_OUT)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入: {out_path}")
    print(f"总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
