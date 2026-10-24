#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密专用策略统计有效性验证 (Statistical Validation of the Crypto Native Strategy)

背景:
  HANDOFF (2026-10-24) 报告的 BTC-USD 全周期生命周期指标 (MTF 组 13 笔, 胜率 53.8%,
  PF 1.59, 期望 +4.89%) 是描述统计。样本 n=13 时二项 95% 置信区间约 [27.8%, 77.6%],
  在宣称「有效」之前必须回答: 这些数字与运气/随机入场的差距有多大, 以及对参数扰动的
  稳健性如何 (docs/strategy_theory.md 约束 2/3/6 的自我要求)。

本脚本在同一数据、同一事件集上完成四项检验, 全部零未来函数:

  [1] Bootstrap 置信区间 (10,000 次重抽样): 胜率 / 期望 / 平均R / PF 的抽样不确定性;
      附加精确二项检验 (胜率 vs 50% 抛硬币)。
  [2] 随机入场对照 (2,000 次试验, 管理机制完全匹配):
      N1 无条件随机 —— 检验「门禁+入场形态+分级止盈」整体是否优于随机进场;
      N2 门禁匹配随机 —— 随机时点仍须与 200 EMA 同侧对齐, 剥离趋势门禁的贡献,
      检验 2B/结构入场形态本身的边际价值。
      对照组逐笔复刻真实交易的几何结构 (止损距离=同方向真实分布重抽样, TP1 同步,
      TP2 沿用引擎的 3.5R 下限阶梯), 跑同一个 simulate_crypto_trade 生命周期。
  [3] 交易成本压力: 每笔交易入场+出场总名义约 2 倍仓位, 按 0 / 0.05% / 0.1% / 0.2%
      单边费率扣减后的期望与 PF。
  [4] 参数敏感性单变量邻域扫描: 六个核心参数逐个扰动, 其余保持基线。

用法:
  python scripts/validate_crypto_stats.py            # 完整验证, 结果写入 results/crypto_stat_validation.json
  python scripts/validate_crypto_stats.py --quick    # 跳过敏感性扫描 (只跑统计检验)
"""

import sys
import os
import json
import math
import time
import argparse
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

import pa_engine as pe
import backtest as bt
import strategy_crypto as sc

# ==============================================================================
# 配置
# ==============================================================================

TICKER = "BTC-USD"
N_BOOT = 10_000          # bootstrap 重抽样次数
N_TRIALS = 2_000         # 随机对照试验次数
RNG_SEED = 20261024
WARMUP = 250             # 随机对照的最小历史暖机 (>= 200 EMA + 余量)
VERDICT_HI = 0.95        # 对照分位 >= 0.95 判定显著优于随机
VERDICT_LO = 0.05        # 对照分位 <= 0.05 判定显著劣于随机
JSON_OUT = os.path.join("..", "results", "crypto_stat_validation.json")

# 单边费率 (吃单): 0=理想化基线, 0.05%≈合约吃单, 0.1%≈现货吃单, 0.2%≈保守含滑点
FEE_RATES = [0.0, 0.0005, 0.001, 0.002]

# 单变量敏感性扫描网格 (* 为基线值)
SCAN_GRID = [
    ("stop_buffer_k", "结构止损缓冲 (k×振幅)", "k",
     [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8], 1.4),
    ("buf_2b", "2B 针尖止损缓冲 (k×振幅)", "k",
     [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], 0.35),
    ("min_rr", "盈亏比硬闸门", "R",
     [1.5, 1.8, 2.0, 2.2, 2.5, 3.0], 2.0),
    ("max_wait", "回踩等待根数", "根",
     [8, 10, 12, 15, 20, 25], 15),
    ("max_hold", "最大持仓根数", "根",
     [30, 40, 50, 60, 80, 100], 60),
    ("trend_gate", "周线结构顺势门禁", "",
     [True, False], True),
]

# HANDOFF (2026-10-24) 报告的基线, 用于管线接线自检 (MTF 组)
HANDOFF_BASELINE_MTF = {"trades": 13, "win_rate": 0.538, "profit_factor": 1.59,
                        "expectancy": 0.0489}


# ==============================================================================
# 1. 生命周期指标 (与 backtest.run_comparison 同口径)
# ==============================================================================

def lifecycle_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    n_total = len(trades)
    wr = len(wins) / n_total
    gross_w = float(wins["pnl_pct"].sum())
    gross_l = float(abs(losses["pnl_pct"].sum()))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    return {
        "trades": n_total,
        "win_rate": wr,
        "profit_factor": pf,
        "avg_r": float(trades["r_multiple"].mean()),
        "expectancy": float(trades["pnl_pct"].mean()),
        "hold_bars": float(trades["hold_bars"].mean()),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "long_trades": int((trades["side"] == 1).sum()),
        "short_trades": int((trades["side"] == -1).sum()),
    }


# ==============================================================================
# 2. 参数化运行 (对 strategy_crypto 做可恢复的运行时注入)
# ==============================================================================

@contextlib.contextmanager
def crypto_params(stop_buffer_k=None, buf_2b=None, max_hold=None):
    """
    运行时注入加密策略参数。三个注入点:
      - CRYPTO_2B_STOP_BUFFER_K / CRYPTO_STOP_BUFFER_K: 模块属性, 调用时读取, 直接可补丁;
      - extract_crypto_levels(buffer_k=...): 默认参数在定义时绑定, 须包装注入
        (影响结构止损锚点, 与 CRYPTO_STOP_BUFFER_K 的针形态止损同步联动);
      - simulate_crypto_trade(max_hold_bars=...): 同理, 影响最大持仓。
    退出时全部恢复, 保证对已验证引擎零侵入。
    (原 ema_span 注入点已随 2026-10-24 引擎手术移除: 200 EMA 门禁被周线结构门禁替代。)
    """
    saved = {}
    try:
        if buf_2b is not None:
            saved["attr_2b"] = sc.CRYPTO_2B_STOP_BUFFER_K
            sc.CRYPTO_2B_STOP_BUFFER_K = buf_2b
        if stop_buffer_k is not None:
            saved["attr_struct"] = sc.CRYPTO_STOP_BUFFER_K
            sc.CRYPTO_STOP_BUFFER_K = stop_buffer_k
            saved["fn_levels"] = sc.extract_crypto_levels
            sc.extract_crypto_levels = (lambda df_, t_, side_, swings_,
                                        _bk=stop_buffer_k: saved["fn_levels"](
                                            df_, t_, side_, swings_, buffer_k=_bk))
        if max_hold is not None:
            saved["fn_sim"] = sc.simulate_crypto_trade
            def _sim_with_hold(*a, _mh=max_hold, **kw):
                kw.setdefault("max_hold_bars", _mh)
                return saved["fn_sim"](*a, **kw)
            sc.simulate_crypto_trade = _sim_with_hold
        yield
    finally:
        if "attr_2b" in saved:
            sc.CRYPTO_2B_STOP_BUFFER_K = saved["attr_2b"]
        if "attr_struct" in saved:
            sc.CRYPTO_STOP_BUFFER_K = saved["attr_struct"]
        if "fn_levels" in saved:
            sc.extract_crypto_levels = saved["fn_levels"]
        if "fn_sim" in saved:
            sc.simulate_crypto_trade = saved["fn_sim"]


def run_crypto_arm(events: list, df: pd.DataFrame, fwd: bt.Forward,
                   min_rr: float = 2.0, max_wait: int = 15,
                   trend_gate: bool = True, **patch_kw) -> pd.DataFrame:
    """在给定事件集上运行加密专用全闭环, 返回去重后的成交交易 DataFrame。"""
    with crypto_params(**patch_kw):
        rec = bt.build_records(fwd, events, df=df, entry_mode="strategy_crypto",
                               retest_level="swing", min_rr=min_rr,
                               max_wait_bars=max_wait, volume_filter=True,
                               asset_type="crypto", ticker=TICKER,
                               use_crypto_trend_gate=trend_gate)
    if rec is None or rec.empty:
        return pd.DataFrame()
    return rec.drop_duplicates(subset=["t", "side", "entry_price"])


# ==============================================================================
# 3. Bootstrap 置信区间与精确二项检验
# ==============================================================================

def _ci(arr: np.ndarray, null_value: float, label: str) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"label": label, "n_finite": 0}
    lo, mid, hi = np.percentile(arr, [2.5, 50, 97.5])
    excludes = bool(lo > null_value or hi < null_value)
    return {
        "label": label,
        "p2_5": float(lo), "p50": float(mid), "p97_5": float(hi),
        "null_value": null_value,
        "ci_excludes_null": excludes,
        "direction": "above" if mid > null_value else "below",
    }


def bootstrap_cis(trades: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    pnl = trades["pnl_pct"].to_numpy(dtype=float)
    r = trades["r_multiple"].to_numpy(dtype=float)
    n = len(pnl)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bp, br = pnl[idx], r[idx]
    is_win = bp > 0
    gross_w = np.where(is_win, bp, 0.0).sum(axis=1)
    gross_l = np.abs(np.where(~is_win, bp, 0.0).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = gross_w / gross_l
    return {
        "n_boot": n_boot,
        "win_rate": _ci(is_win.mean(axis=1), 0.5, "胜率 vs 50%"),
        "expectancy": _ci(bp.mean(axis=1), 0.0, "单笔期望 vs 0"),
        "avg_r": _ci(br.mean(axis=1), 0.0, "平均R vs 0"),
        "profit_factor": _ci(pf, 1.0, "盈亏比 PF vs 1.0"),
        "pf_inf_share": float(np.mean(~np.isfinite(pf))),
    }


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """胜率 vs 抛硬币的精确双侧二项检验 p 值。"""
    pmf = np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
    p_hi = float(pmf[k:].sum())
    p_lo = float(pmf[:k + 1].sum())
    return min(1.0, 2.0 * min(p_hi, p_lo))


# ==============================================================================
# 4. 随机入场对照 (管理机制匹配)
# ==============================================================================

def random_entry_control(df: pd.DataFrame, real_trades: pd.DataFrame,
                         gate_match: bool, n_trials: int = N_TRIALS,
                         seed: int = RNG_SEED, max_hold: int = 60,
                         min_rr: float = 2.0) -> dict:
    """
    零假设对照: 把真实交易的入场时点换成随机时点, 其余一切保持一致 ——
      - 逐笔保留真实交易的方向 (多空结构与真实样本一致);
      - 止损距离(以决策根中位振幅为单位)与 TP1 盈亏比, 从同方向真实交易的重抽样;
      - TP2 沿用引擎的阶梯下限 max(rr1+0.5, min_rr+1.5)×risk;
      - 出场交给同一个 simulate_crypto_trade (分级止盈/保本/止损优先/最大持仓)。
    gate_match=False (N1): 触发根完全随机 —— 检验系统整体;
    gate_match=True  (N2): 触发根须与周线结构偏向同侧 (与引擎现行周线结构门禁同构,
                           以「上一个已完结周」的偏向近似, 零未来函数) —— 剥离趋势
                           门禁贡献, 检验 2B/结构入场形态的边际价值。
    (2026-10-24 前此对照用 200 EMA 对齐; 引擎门禁已替换为周线结构门禁, 对照同步替换。)
    """
    n = len(df)
    opens = df["Open"].to_numpy(dtype=float)
    highs, lows = df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float)
    # 与 pe.median_bar_range(df.iloc[:t+1], n=14) 等价: 最近14根 High-Low 的中位数
    bar_rng_arr = pd.Series(highs - lows).rolling(14).median().to_numpy(dtype=float)
    # 预计算每根日线「上一个已完结周」的结构偏向 (order=3 周线摆动 + MSS), 供 N2 匹配
    weekly = bt.weekly_from_daily(df)
    week_bias = np.array([pe.dow_bias(pe.extract_swings(weekly.iloc[:i + 1], order=3),
                                      df=weekly.iloc[:i + 1], buffer_k=0.0)
                          for i in range(len(weekly))], dtype=int)
    struct_bias = np.zeros(n, dtype=int)
    week_starts = pd.DatetimeIndex([d - pd.Timedelta(days=d.weekday()) for d in df.index])
    for i, ws in enumerate(week_starts):
        pos = weekly.index.searchsorted(ws, side="left")
        if pos > 0:
            struct_bias[i] = week_bias[pos - 1]

    # 真实交易几何池 (按方向分组): (止损距离/振幅, TP1盈亏比)
    geo = {1: [], -1: []}
    for _, tr in real_trades.iterrows():
        side, t = int(tr["side"]), int(tr["t"])
        entry, stop, tp1 = float(tr["entry_price"]), float(tr["stop_price"]), float(tr["tp_price"])
        brng = bar_rng_arr[t]
        risk = abs(entry - stop)
        if not np.isfinite(brng) or brng <= 0 or risk <= 0 or entry <= 0:
            continue
        geo[side].append((risk / brng, abs(tp1 - entry) / risk))

    slots = [int(s) for s in real_trades["side"]]
    rng = np.random.default_rng(seed)
    # 触发根 k ∈ [WARMUP, n-max_hold-2]: 入场 k+1, 保证完整 max_hold 窗口 (对对照组略偏保守)
    k_lo, k_hi = WARMUP, n - max_hold - 2

    wins_, pfs_, exps_, rs_ = [], [], [], []
    for _trial in range(n_trials):
        pnls, rs = [], []
        for side in slots:
            pool = geo.get(side) or []
            if not pool:
                continue
            for _attempt in range(60):
                k = int(rng.integers(k_lo, k_hi + 1))
                if gate_match:
                    if side == 1 and struct_bias[k] != 1:
                        continue
                    if side == -1 and struct_bias[k] != -1:
                        continue
                entry_t = k + 1
                entry = opens[entry_t]
                brng = bar_rng_arr[k]
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
                trade = sc.simulate_crypto_trade(df, entry_t, entry, side, stop,
                                                 tp1_price=tp1, tp2_price=tp2,
                                                 impulse_leg=None,
                                                 max_hold_bars=max_hold, min_rr=min_rr)
                if trade.get("status") == "completed":
                    pnls.append(trade["pnl_pct"])
                    rs.append(trade["r_multiple"])
                    break
        if not pnls:
            continue
        w = sum(1 for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        wins_.append(w / len(pnls))
        pfs_.append((sum(p for p in pnls if p > 0) / gl) if gl > 0 else float("inf"))
        exps_.append(float(np.mean(pnls)))
        rs_.append(float(np.mean(rs)))

    return {
        "n_trials": len(exps_),
        "gate_match": gate_match,
        "win_rate": np.array(wins_), "profit_factor": np.array(pfs_),
        "expectancy": np.array(exps_), "avg_r": np.array(rs_),
    }


def control_verdict(observed: float, ctrl: np.ndarray, hi: float = VERDICT_HI,
                    lo: float = VERDICT_LO) -> dict:
    ctrl = ctrl[np.isfinite(ctrl)]
    if len(ctrl) == 0:
        return {"percentile": None, "verdict": "对照无效"}
    pct = float((ctrl < observed).mean())
    verdict = ("显著优于随机" if pct >= hi else
               "显著劣于随机" if pct <= lo else "无显著差异")
    q5, q50, q95 = np.percentile(ctrl, [5, 50, 95])
    return {"percentile": pct, "verdict": verdict,
            "ctrl_p5": float(q5), "ctrl_p50": float(q50), "ctrl_p95": float(q95)}


# ==============================================================================
# 5. 交易成本压力
# ==============================================================================

def fee_stress(trades: pd.DataFrame, fee_rates=FEE_RATES) -> list:
    """入场名义 1.0 + 出场名义合计 1.0 → 每笔总成本 ≈ 2×单边费率 (近似, 忽略分级价位差)。"""
    out = []
    pnl = trades["pnl_pct"].to_numpy(dtype=float)
    for fee in fee_rates:
        pnl_net = pnl - 2.0 * fee
        wins = pnl_net > 0
        gl = abs(pnl_net[~wins].sum())
        out.append({
            "fee_one_side": fee,
            "drag_per_trade": 2.0 * fee,
            "win_rate": float(wins.mean()),
            "profit_factor": float(pnl_net[wins].sum() / gl) if gl > 0 else float("inf"),
            "expectancy": float(pnl_net.mean()),
        })
    return out


# ==============================================================================
# 6. 主流程
# ==============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="加密专用策略统计有效性验证")
    ap.add_argument("--quick", action="store_true", help="跳过参数敏感性扫描")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    t0 = time.time()
    pe.reset_fetch_state()
    df = bt.load_ticker(TICKER)
    if df.empty:
        print(f"[错误] 无法加载 {TICKER} 本地缓存")
        return 1
    w = bt.weekly_from_daily(df)
    events = bt.enumerate_events(df, w, method="h2", buffer_k=0.0, volume_filter=True)
    fwd = bt.Forward(df)
    events_by_arm = {arm: [e for e in events if e["arm"] == arm]
                     for arm in ("MTF", "MTF+HTF")}

    print("=" * 96)
    print(f"加密专用策略统计有效性验证 | {TICKER} | {len(df)} 根日线 "
          f"({df.index[0].date()} → {df.index[-1].date()}) | "
          f"事件数 MTF={len(events_by_arm['MTF'])} / MTF+HTF={len(events_by_arm['MTF+HTF'])}")
    print("=" * 96)

    report = {"meta": {
        "ticker": TICKER, "bars": len(df),
        "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "events_mtf": len(events_by_arm["MTF"]),
        "events_mtf_htf": len(events_by_arm["MTF+HTF"]),
        "n_boot": N_BOOT, "n_trials": args.trials, "seed": RNG_SEED,
    }}

    # ---- [0] 基线复现 ----
    print("\n[0] 基线复现 (须与 HANDOFF 2026-10-24 对齐, 否则管线接线有误)")
    baseline = {}
    trades_by_arm = {}
    for arm in ("MTF", "MTF+HTF"):
        trades = run_crypto_arm(events_by_arm[arm], df, fwd)
        trades_by_arm[arm] = trades
        st = lifecycle_stats(trades)
        baseline[arm] = st
        print(f"  [{arm}] {st['trades']} 笔 | 胜率 {st['win_rate']:.1%} | "
              f"PF {st['profit_factor']:.2f} | {st['avg_r']:+.2f}R | "
              f"期望 {st['expectancy']:+.2%} | 多 {st['long_trades']} / 空 {st['short_trades']}")
    mtf = baseline["MTF"]
    aligned = (mtf["trades"] == HANDOFF_BASELINE_MTF["trades"]
               and abs(mtf["win_rate"] - HANDOFF_BASELINE_MTF["win_rate"]) < 0.01
               and abs(mtf["profit_factor"] - HANDOFF_BASELINE_MTF["profit_factor"]) < 0.05)
    print(f"  与 HANDOFF 基线对齐: {'✓ 通过' if aligned else '✗ 不一致, 结果仅代表本管线'}")
    report["baseline"] = baseline
    report["handoff_aligned"] = bool(aligned)

    # ---- [1] Bootstrap + 二项检验 ----
    print(f"\n[1] Bootstrap 置信区间 ({N_BOOT:,} 次重抽样) + 精确二项检验")
    report["bootstrap"] = {}
    for arm in ("MTF", "MTF+HTF"):
        trades = trades_by_arm[arm]
        if trades.empty:
            continue
        bs = bootstrap_cis(trades)
        st = baseline[arm]
        p_binom = binom_two_sided(st["n_wins"], st["trades"])
        bs["binomial_vs_coinflip"] = {
            "wins": st["n_wins"], "n": st["trades"], "p_value": p_binom}
        report["bootstrap"][arm] = bs
        print(f"\n  [{arm}] n={st['trades']} (胜 {st['n_wins']} / 负 {st['n_losses']})")
        for key, null_v in (("win_rate", 0.5), ("expectancy", 0.0),
                            ("avg_r", 0.0), ("profit_factor", 1.0)):
            ci = bs[key]
            if "p50" not in ci:
                continue
            flag = "✓ 排除零假设" if ci["ci_excludes_null"] else "✗ 无法排除"
            print(f"    {ci['label']:<18} 95%CI [{ci['p2_5']:+.4f}, {ci['p97_5']:+.4f}] "
                  f"中位 {ci['p50']:+.4f}  → {flag}")
        print(f"    胜率 vs 50% 抛硬币 (精确二项): p = {p_binom:.3f}")

    # ---- [2] 随机入场对照 ----
    print(f"\n[2] 随机入场对照 ({args.trials:,} 次试验, 逐笔匹配方向/风险几何/出场机制)")
    report["random_control"] = {}
    for arm in ("MTF", "MTF+HTF"):
        trades = trades_by_arm[arm]
        if trades.empty:
            continue
        arm_out = {}
        for tag, gate in (("N1_无条件随机", False), ("N2_门禁匹配随机", True)):
            ctrl = random_entry_control(df, trades, gate_match=gate,
                                        n_trials=args.trials)
            st = baseline[arm]
            v = {
                "win_rate": control_verdict(st["win_rate"], ctrl["win_rate"]),
                "expectancy": control_verdict(st["expectancy"], ctrl["expectancy"]),
                "profit_factor": control_verdict(st["profit_factor"], ctrl["profit_factor"]),
                "avg_r": control_verdict(st["avg_r"], ctrl["avg_r"]),
            }
            arm_out[tag] = {"n_trials": ctrl["n_trials"], "verdicts": v}
            ev = v["expectancy"]
            pf_v = v["profit_factor"]
            print(f"\n  [{arm} · {tag}] 对照 {ctrl['n_trials']:,} 次")
            print(f"    期望 {st['expectancy']:+.2%} | 对照分布 p5~p95: "
                  f"{ev['ctrl_p5']:+.2%} ~ {ev['ctrl_p95']:+.2%} (中位 {ev['ctrl_p50']:+.2%}) "
                  f"| 分位 {ev['percentile']:.3f} → {ev['verdict']}")
            print(f"    PF   {st['profit_factor']:.2f} | 对照分布 p5~p95: "
                  f"{pf_v['ctrl_p5']:.2f} ~ {pf_v['ctrl_p95']:.2f} (中位 {pf_v['ctrl_p50']:.2f}) "
                  f"| 分位 {pf_v['percentile']:.3f} → {pf_v['verdict']}")
            wr_v = v["win_rate"]
            print(f"    胜率 {st['win_rate']:.1%} | 对照分布 p5~p95: "
                  f"{wr_v['ctrl_p5']:.1%} ~ {wr_v['ctrl_p95']:.1%} (中位 {wr_v['ctrl_p50']:.1%}) "
                  f"| 分位 {wr_v['percentile']:.3f} → {wr_v['verdict']}")
        report["random_control"][arm] = arm_out

    # ---- [3] 交易成本压力 ----
    print("\n[3] 交易成本压力 (每笔双向成本 ≈ 2×单边费率)")
    report["fees"] = {}
    for arm in ("MTF", "MTF+HTF"):
        trades = trades_by_arm[arm]
        if trades.empty:
            continue
        fs = fee_stress(trades)
        report["fees"][arm] = fs
        print(f"\n  [{arm}]")
        print(f"    {'单边费率':>8} {'每笔拖累':>9} {'胜率':>7} {'PF':>7} {'期望':>9}")
        for f in fs:
            print(f"    {f['fee_one_side']:>8.3%} {f['drag_per_trade']:>9.3%} "
                  f"{f['win_rate']:>7.1%} {f['profit_factor']:>7.2f} {f['expectancy']:>+9.2%}")

    # ---- [4] 参数敏感性扫描 ----
    if not args.quick:
        print("\n[4] 参数敏感性单变量邻域扫描 (* 为基线)")
        report["sensitivity"] = {}
        for key, label, unit, values, base_v in SCAN_GRID:
            rows = []
            for v in values:
                patch_kw = {}
                pass_kw = {"min_rr": 2.0, "max_wait": 15, "trend_gate": True}
                if key == "stop_buffer_k":
                    patch_kw["stop_buffer_k"] = v
                elif key == "buf_2b":
                    patch_kw["buf_2b"] = v
                elif key == "min_rr":
                    pass_kw["min_rr"] = v
                elif key == "max_wait":
                    pass_kw["max_wait"] = v
                elif key == "max_hold":
                    patch_kw["max_hold"] = v
                elif key == "trend_gate":
                    pass_kw["trend_gate"] = v
                arm_st = {}
                for arm in ("MTF", "MTF+HTF"):
                    trades = run_crypto_arm(events_by_arm[arm], df, fwd,
                                            **pass_kw, **patch_kw)
                    arm_st[arm] = lifecycle_stats(trades)
                rows.append({"value": v, **{a: arm_st[a] for a in arm_st}})
            report["sensitivity"][key] = {"label": label, "unit": unit,
                                          "base": base_v, "rows": rows}
            print(f"\n  ▸ {label} (基线 {base_v}{unit})")
            print(f"    {'取值':>8} | {'MTF 笔数':>8} {'胜率':>7} {'PF':>6} {'平均R':>7} {'期望':>8} "
                  f"| {'HTF 笔数':>8} {'胜率':>7} {'PF':>6} {'期望':>8}")
            for row in rows:
                mark = "*" if row["value"] == base_v else " "
                m, h = row["MTF"], row["MTF+HTF"]
                if m["trades"] == 0:
                    print(f"    {str(row['value']):>7}{mark} | {'无成交':>8}")
                    continue
                print(f"    {str(row['value']):>7}{mark} | {m['trades']:>8} {m['win_rate']:>7.1%} "
                      f"{m['profit_factor']:>6.2f} {m['avg_r']:>+7.2f} {m['expectancy']:>+8.2%} "
                      f"| {h.get('trades', 0):>8} {h.get('win_rate', 0):>7.1%} "
                      f"{h.get('profit_factor', 0):>6.2f} {h.get('expectancy', 0):>+8.2%}")

    # ---- 汇总落盘 ----
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
                exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), JSON_OUT)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入: {out_path}")
    print(f"总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
