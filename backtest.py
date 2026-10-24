#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向层回测引擎：检验「道氏结构方向」是否具备预测力

理论依据见 strategy_theory.md：
  方向层（道氏）决定允不允许做 → 触发层（裸K）决定何时做。
  本脚本只测方向层。理由：若这一层没有 edge，在其上叠加裸K触发只是随机挑选入场时点，
  触发层的精细化无法产生正期望。这是「分层测试、不许打包」原则的第一层。

三条硬规则（对应 strategy_theory.md 第四节）：
  1. Walk-forward：第 t 根收盘做决策，第 t+1 根开盘入场；任何时刻只使用 <= t 的数据。
  2. 状态不是信号，状态转移才是。同一波趋势只算一个样本，避免样本重叠制造虚假显著。
  3. 必须有对照组。信号收益必须与「同期、同规模、同方向的随机入场」比较。

用法：
    python backtest.py --synthetic random      # 已知答案：随机游走，edge 应≈0
    python backtest.py --synthetic trend       # 已知答案：动量持续，edge 应显著>0
    python backtest.py --synthetic range       # 已知答案：均值回归，edge 应显著<0

    python backtest.py --tickers AAPL,MSFT,NVDA --download
    python backtest.py --tickers AAPL,MSFT,NVDA --offline
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl
import strategy_crypto as sc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HORIZONS = (5, 10, 20, 40)
ORDER_DAILY = 5        # 与 swing 模式 MTF 一致
ORDER_WEEKLY = 3       # 与 swing 模式 HTF 一致
MIN_BARS = 80
TRIALS = 400


# ==============================================================================
# 1. 数据层
# ==============================================================================

TICKER_ALIASES = {
    "GOOGLE": "GOOGL",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
}


def normalize_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    return TICKER_ALIASES.get(t, t)


def cache_path(ticker: str) -> str:
    tk = normalize_ticker(ticker)
    return os.path.join(DATA_DIR, f"{tk}.csv")


def load_ticker(ticker: str, download: bool = False) -> pd.DataFrame:
    """
    缓存优先；--download 时交由 data_sources 做多源降级（东财 → 腾讯 → yfinance → stooq）。
    取数细节与来源留痕见 data_sources.py / data/_meta.json。
    """
    tk = normalize_ticker(ticker)
    try:
        import data_sources as ds
    except ImportError:
        p = cache_path(tk)
        return pd.read_csv(p, index_col=0, parse_dates=True) if os.path.exists(p) else pd.DataFrame()

    df = ds.read_cache(tk)
    if not df.empty:
        return df
    if not download:
        return pd.DataFrame()

    df, note = ds.fetch_history(tk, bars=1600, allow_network=True)
    if df.empty:
        print(f"[取数失败] {tk}: {note}", file=sys.stderr)
    else:
        print(f"[取数] {tk}: {note}", file=sys.stderr)
    return df


def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """日线聚合为周线（W-FRI 标签）。调用方按 label <= 当日 截断，避免用未走完的这一周。"""
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Open", "High", "Low", "Close"])


# ==============================================================================
# 2. 前向收益矩阵（向量化，避免逐样本切片）
# ==============================================================================

class Forward:
    """
    以「决策根 t」为索引，预计算 t+1 开盘入场后 h 根的收益 / 最大不利偏移 / 最大有利偏移。
    只做一次滚动计算，事件统计与随机对照都从矩阵取值。
    """

    def __init__(self, df: pd.DataFrame, horizons=HORIZONS):
        self.n = len(df)
        self.entry = df["Open"].shift(-1).to_numpy(dtype=float)
        self.closes, self.minlow, self.maxhigh = {}, {}, {}
        for h in horizons:
            self.closes[h] = df["Close"].shift(-(h + 1)).to_numpy(dtype=float)
            self.minlow[h] = df["Low"].rolling(h + 1).min().shift(-(h + 1)).to_numpy(dtype=float)
            self.maxhigh[h] = df["High"].rolling(h + 1).max().shift(-(h + 1)).to_numpy(dtype=float)

    def valid_from(self, h: int) -> int:
        return int(np.argmax(np.isfinite(self.closes[h]) & np.isfinite(self.entry)))

    def ret_array(self, side: int, h: int, start: int = MIN_BARS) -> np.ndarray:
        """全部可用决策根在方向 side 下的 h 根前向收益，供随机对照抽样"""
        base = self.closes[h] / self.entry - 1.0
        ok = np.isfinite(base)
        ok[:start] = False
        return base[ok] * side

    def stats(self, t: int, side: int, h: int):
        if t < 0 or t >= self.n:
            return None
        e = self.entry[t]
        c, lo, hi = self.closes[h][t], self.minlow[h][t], self.maxhigh[h][t]
        if not (np.isfinite(e) and e > 0 and np.isfinite(c) and np.isfinite(lo) and np.isfinite(hi)):
            return None
        ret, mae, mfe = c / e - 1.0, lo / e - 1.0, hi / e - 1.0
        if side < 0:
            ret, mae, mfe = -ret, -mfe, -mae
        return {"ret": float(ret), "mae": float(min(mae, 0.0)), "mfe": float(max(mfe, 0.0))}


# ==============================================================================
# 3. 事件枚举（walk-forward，无未来函数）
# ==============================================================================

def enumerate_events(df_daily: pd.DataFrame, df_weekly: pd.DataFrame,
                     order_daily: int = ORDER_DAILY,
                     order_weekly: int = ORDER_WEEKLY,
                     method: str = "h2",
                     buffer_k: float = 0.0,
                     volume_filter: bool = False) -> list:
    """
    逐根推进，只产生「结构偏向发生转移」的事件（walk-forward，无未来函数）。
    - method="h1": 拐点标签计数法 (pe.dow_bias_count)
    - method="h2": 结构突破穿越法 (pe.dow_bias 带 df 判定)
    - buffer_k: 结构突破穿越的缓冲倍数
    - volume_filter: 道氏第6条量能验证（成交量需高于20日均量1.2倍）
    """
    dates = df_daily.index
    events, prev_d, prev_both = [], 0, 0
    w_swings, w_len = [], -1
    w_bias = 0

    for t in range(MIN_BARS, len(df_daily) - 1):
        sub_d = df_daily.iloc[:t + 1]
        d_sw = pe.extract_swings(sub_d, order=order_daily)
        if len(d_sw) < 2:
            prev_d = prev_both = 0
            continue

        if method == "h1":
            b_d = pe.dow_bias_count(d_sw)
        else:
            b_d = pe.dow_bias(d_sw, df=sub_d, buffer_k=buffer_k)

        w_prefix = df_weekly[df_weekly.index <= dates[t]]
        if len(w_prefix) != w_len:
            w_swings = pe.extract_swings(w_prefix, order=order_weekly) if len(w_prefix) else []
            if method == "h1":
                w_bias = pe.dow_bias_count(w_swings) if len(w_swings) else 0
            else:
                w_bias = pe.dow_bias(w_swings, df=w_prefix, buffer_k=buffer_k) if len(w_prefix) else 0
            w_len = len(w_prefix)
        b_w = w_bias

        both = b_d if (b_d != 0 and b_d == b_w) else 0

        # 量能确认过滤（道氏第6条）
        vol_ok = True
        if volume_filter and 'Volume' in df_daily:
            vol_t = float(df_daily['Volume'].iloc[t])
            vol_hist = df_daily['Volume'].iloc[max(0, t - 20):t]
            vol_ma = float(vol_hist.mean()) if len(vol_hist) else 0.0
            vol_ok = (vol_ma == 0 or vol_t >= vol_ma * 1.2)

        if b_d != 0 and b_d != prev_d:
            if not volume_filter or vol_ok:
                events.append({"t": t, "date": dates[t], "arm": "MTF", "side": int(b_d),
                               "bias_mtf": int(b_d), "bias_htf": int(b_w)})
        if both != 0 and both != prev_both:
            if not volume_filter or vol_ok:
                events.append({"t": t, "date": dates[t], "arm": "MTF+HTF", "side": int(both),
                               "bias_mtf": int(b_d), "bias_htf": int(b_w)})
        prev_d, prev_both = b_d, both

    return events


def build_records(fwd: Forward, events: list, horizons=HORIZONS,
                  df: pd.DataFrame = None,
                  entry_mode: str = "market",
                  retest_level: str = "swing",
                  min_rr: float = sl.DEFAULT_MIN_RR,
                  max_wait_bars: int = sl.DEFAULT_MAX_WAIT_BARS,
                  volume_filter: bool = False,
                  market_data: dict = None,
                  use_market_gate: bool = False,
                  use_wave_filter: bool = False,
                  asset_type: str = "auto",
                  ticker: str = "",
                  exclude_black_swans: bool = False,
                  use_crypto_trend_gate: bool = True) -> pd.DataFrame:
    """
    根据 entry_mode 构建收益与执行记录：
      - 'market': 原版市价追入（t+1 根开盘入场）
      - 'pullback_limit': 位置层挂单回踩关键位（Broken Swing / Fib 50% / Fib 61.8%），支持缩量洗盘过滤
      - 'trigger': 触发层裸K确认形态入场（Pin Bar / 2B / Engulfing）
      - 'strategy': 全闭环策略执行（Direction -> Location -> Trigger -> RR>=2.0 闸门与止损止盈执行），
        可联动大盘道氏印证硬闸门与艾略特波浪位置狙击过滤
      - 'strategy_crypto': 加密专用全闭环执行（2B流动性猎杀+宽幅防插针止损+分级止盈），资产彻底解耦
    """
    rows = []
    for e in events:
        if entry_mode == "market" or df is None:
            for h in horizons:
                s = fwd.stats(e["t"], e["side"], h)
                if s is None:
                    continue
                rows.append({"arm": e["arm"], "side": e["side"], "date": e["date"],
                             "horizon": h, "t": e["t"], **s})
        elif entry_mode == "pullback_limit":
            t, side = e["t"], e["side"]
            sub_d = df.iloc[:t + 1]
            sw = pe.extract_swings(sub_d, order=ORDER_DAILY)
            lvls = sl.extract_breakout_levels(sub_d, t, side, sw)
            if not lvls:
                continue
            lim = lvls["swing_retest"] if retest_level == "swing" else lvls.get(retest_level, lvls["swing_retest"])
            res = sl.simulate_limit_order(df, t, side, lim, lvls["structural_stop"],
                                          max_wait_bars=max_wait_bars, volume_filter=volume_filter)
            if res.get("filled"):
                entry_t = res["entry_t"]
                entry_p = res["entry_price"]
                for h in horizons:
                    s = sl.calc_entry_horizon_stats(df, entry_t, entry_p, side, h)
                    if s is None:
                        continue
                    rows.append({"arm": e["arm"], "side": side, "date": df.index[entry_t],
                                 "horizon": h, "t": entry_t, "entry_price": entry_p,
                                 "stop_price": res["stop_price"], "wait_bars": res["wait_bars"],
                                 "entry_mode": "pullback_limit", **s})
        elif entry_mode == "trigger":
            t, side = e["t"], e["side"]
            sub_d = df.iloc[:t + 1]
            sw = pe.extract_swings(sub_d, order=ORDER_DAILY)
            lvls = sl.extract_breakout_levels(sub_d, t, side, sw)
            if not lvls:
                continue
            key_level = lvls["swing_retest"] if retest_level == "swing" else lvls.get(retest_level, lvls["swing_retest"])
            trig = sl.simulate_trigger_entry(df, t, side, key_level, lvls["bar_rng"], lvls["structural_stop"], max_wait_bars=max_wait_bars)
            if trig.get("triggered"):
                entry_t = trig["entry_t"]
                entry_p = trig["entry_price"]
                for h in horizons:
                    s = sl.calc_entry_horizon_stats(df, entry_t, entry_p, side, h)
                    if s is None:
                        continue
                    rows.append({"arm": e["arm"], "side": side, "date": df.index[entry_t],
                                 "horizon": h, "t": entry_t, "entry_price": entry_p,
                                 "stop_price": trig["stop_price"], "pattern": trig["pattern"],
                                 "wait_bars": trig["wait_bars"], "entry_mode": "trigger", **s})
        elif entry_mode in ("strategy", "strategy_crypto"):
            use_crypto = (entry_mode == "strategy_crypto" or
                          asset_type == "crypto" or
                          (asset_type == "auto" and sc.is_crypto_asset(ticker)))
            if use_crypto:
                t, side = e["t"], e["side"]
                sub_d = df.iloc[:t + 1]
                sw = pe.extract_swings(sub_d, order=ORDER_DAILY)
                lvls = sc.extract_crypto_levels(sub_d, t, side, sw)
                if not lvls:
                    continue
                key_level = lvls["swing_retest"] if retest_level == "swing" else lvls.get(retest_level, lvls["swing_retest"])
                trig = sc.simulate_crypto_trigger_entry(df, t, side, key_level, lvls["bar_rng"],
                                                        lvls["structural_stop"], max_wait_bars=max_wait_bars,
                                                        use_trend_gate=use_crypto_trend_gate)
                if trig.get("triggered"):
                    entry_t = trig["entry_t"]
                    entry_p = trig["entry_price"]
                    entry_date = df.index[entry_t]

                    # 黑天鹅事件入场过滤（若开启）：按标的资产类型选择对应危机窗口字典
                    # （加密标的走 CRYPTO_BLACK_SWANS，美股/指数标的走 STOCK_BLACK_SWANS）
                    if exclude_black_swans:
                        _is_bs = (sc.is_crypto_black_swan if sc.is_crypto_asset(ticker)
                                  else sl.is_stock_black_swan)
                        if _is_bs(entry_date):
                            continue

                    stop_p = trig["stop_price"]
                    risk = abs(entry_p - stop_p)
                    tp1_p = lvls.get("tp1_target")
                    if side == 1:
                        tp1_p = max(tp1_p, round(entry_p + min_rr * risk, 2))
                    else:
                        tp1_p = min(tp1_p, round(entry_p - min_rr * risk, 2))
                    tp2_p = lvls.get("tp2_target")
                    impulse = lvls.get("impulse_leg")

                    # 加密资产原生架构：彻底解除美股大盘指数(SPY/QQQ)拦截
                    trade = sc.simulate_crypto_trade(df, entry_t, entry_p, side, stop_p,
                                                     tp1_price=tp1_p, tp2_price=tp2_p,
                                                     impulse_leg=impulse, min_rr=min_rr)
                    if trade.get("status") == "completed":
                        exit_date = df.index[trade["exit_t"]]
                        # 黑天鹅持仓过滤（若开启）：跳过在途持仓跨越系统性危机窗口的受灾交易
                        if exclude_black_swans:
                            _overlap_bs = (sc.trade_overlaps_black_swan if sc.is_crypto_asset(ticker)
                                           else sl.trade_overlaps_stock_black_swan)
                            if _overlap_bs(entry_date, exit_date):
                                continue

                        for h in horizons:
                            s = sl.calc_entry_horizon_stats(df, entry_t, entry_p, side, h)
                            if s is None:
                                continue
                            rows.append({"arm": e["arm"], "side": side, "date": df.index[entry_t],
                                         "horizon": h, "t": entry_t, "entry_price": entry_p,
                                         "stop_price": stop_p, "tp_price": trade.get("tp1_price", tp1_p),
                                         "exit_price": trade["exit_price"], "pnl_pct": trade["pnl_pct"],
                                         "r_multiple": trade["r_multiple"], "hold_bars": trade["hold_bars"],
                                         "exit_reason": trade["exit_reason"], "pattern": trig.get("pattern", ""),
                                         "trade_mae": trade["mae"], "trade_mfe": trade["mfe"],
                                         "entry_mode": "strategy_crypto", **s})
            else:
                t, side = e["t"], e["side"]
                sub_d = df.iloc[:t + 1]
                sw = pe.extract_swings(sub_d, order=ORDER_DAILY)
                lvls = sl.extract_breakout_levels(sub_d, t, side, sw)
                if not lvls:
                    continue
                key_level = lvls["swing_retest"] if retest_level == "swing" else lvls.get(retest_level, lvls["swing_retest"])
                trig = sl.simulate_trigger_entry(df, t, side, key_level, lvls["bar_rng"], lvls["structural_stop"], max_wait_bars=max_wait_bars)
                if trig.get("triggered"):
                    entry_t = trig["entry_t"]
                    entry_p = trig["entry_price"]
                    stop_p = trig["stop_price"]
                    risk = abs(entry_p - stop_p)
                    tp_p = round(entry_p + min_rr * risk if side == 1 else entry_p - min_rr * risk, 2)

                    # 黑天鹅事件入场过滤（若开启）：跳过在美股系统性危机窗口内开仓的信号
                    if exclude_black_swans and sl.is_stock_black_swan(df.index[entry_t]):
                        continue

                    # 构造大盘历史切片（严格 <= entry_t 所在日期，杜绝未来函数）
                    m_dow = None
                    if use_market_gate and market_data:
                        curr_date = df.index[entry_t]
                        curr_dt_naive = curr_date.tz_localize(None) if hasattr(curr_date, "tz") and curr_date.tz is not None else curr_date
                        m_dow = {}
                        for sym in ("SPY", "QQQ"):
                            m_df = market_data.get(sym)
                            if m_df is not None and not m_df.empty:
                                m_idx = m_df.index.tz_localize(None) if hasattr(m_df.index, "tz") and m_df.index.tz is not None else m_df.index
                                sub_m = m_df[m_idx <= curr_dt_naive]
                                if not sub_m.empty:
                                    m_dow[sym] = sub_m

                    # 艾略特波浪位置狙击与诱多拦截
                    w_status = None
                    if use_wave_filter:
                        sub_entry = df.iloc[:entry_t + 1]
                        sw_entry = pe.extract_swings(sub_entry, order=ORDER_DAILY)
                        w_eval = pe.evaluate_wave_position(sw_entry, df=sub_entry, current_price=entry_p)
                        w_status = w_eval.get("status")

                    trade = sl.simulate_strategy_trade(df, entry_t, entry_p, side, stop_p, tp_p,
                                                       min_rr=min_rr, market_dow=m_dow, wave_status=w_status)
                    if trade.get("status") == "completed":
                        # 黑天鹅持仓过滤（若开启）：跳过在途持仓跨越美股系统性危机窗口的受灾交易
                        if exclude_black_swans and sl.trade_overlaps_stock_black_swan(
                                df.index[entry_t], df.index[trade["exit_t"]]):
                            continue
                        for h in horizons:
                            s = sl.calc_entry_horizon_stats(df, entry_t, entry_p, side, h)
                            if s is None:
                                continue
                            rows.append({"arm": e["arm"], "side": side, "date": df.index[entry_t],
                                         "horizon": h, "t": entry_t, "entry_price": entry_p,
                                         "stop_price": stop_p, "tp_price": tp_p,
                                         "exit_price": trade["exit_price"], "pnl_pct": trade["pnl_pct"],
                                         "r_multiple": trade["r_multiple"], "hold_bars": trade["hold_bars"],
                                         "exit_reason": trade["exit_reason"],
                                         "trade_mae": trade["mae"], "trade_mfe": trade["mfe"],
                                         "entry_mode": "strategy", **s})

    return pd.DataFrame(rows)


def print_strategy_lifecycle_summary(rec: pd.DataFrame, label: str = "") -> dict:
    """输出全闭环风控执行策略的生命周期指标（胜率、PF、R倍数、出场分布等）"""
    if rec.empty or "r_multiple" not in rec.columns:
        return {}
    unique_trades = rec.drop_duplicates(subset=["arm", "t", "side", "entry_price"])
    out = {}
    for arm in ("MTF", "MTF+HTF"):
        sub = unique_trades[unique_trades["arm"] == arm]
        if sub.empty:
            continue
        wins = sub[sub["pnl_pct"] > 0]
        losses = sub[sub["pnl_pct"] <= 0]
        n_total = len(sub)
        wr = len(wins) / n_total if n_total else 0.0
        pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())) if len(losses) and losses["pnl_pct"].sum() != 0 else float("inf")
        avg_r = float(sub["r_multiple"].mean())
        exp = float(sub["pnl_pct"].mean())
        mae_col = "trade_mae" if "trade_mae" in sub.columns else "mae"
        mfe_col = "trade_mfe" if "trade_mfe" in sub.columns else "mfe"
        mae_med = float(sub[mae_col].median())
        mfe_med = float(sub[mfe_col].median())
        n_stop = int((sub["exit_reason"].str.contains("stop_loss") | (sub["exit_reason"] == "stop_loss_conservative")).sum())
        n_tp_protected = int((sub["exit_reason"] == "tp1_then_be_protected").sum())
        n_tp_runner = int((sub["exit_reason"] == "tp_runner_captured").sum())
        n_tp_partial = int((sub["exit_reason"] == "tp_partial_time_horizon").sum())
        n_tp_classic = int((sub["exit_reason"] == "take_profit").sum())
        n_tp_total = n_tp_protected + n_tp_runner + n_tp_partial + n_tp_classic
        n_time = int((sub["exit_reason"] == "time_horizon").sum())

        print(f"\n[{arm} 全闭环风控生命周期 (RR >= 2.0)]")
        print(f"  交易笔数: {n_total} 笔  |  胜率: {wr:.1%} ({len(wins)} 胜 / {len(losses)} 负)")
        print(f"  盈亏比 (PF): {pf:.2f}  |  平均 R 倍数: {avg_r:+.2f} R  |  单笔期望: {exp:+.2%}")

        # 多空双向细分统计 (Long vs Short Breakdown)
        longs = sub[sub["side"] == 1]
        shorts = sub[sub["side"] == -1]

        def dir_stats(d_df):
            if d_df.empty:
                return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "avg_r": 0.0, "exp": 0.0}
            w = d_df[d_df["pnl_pct"] > 0]
            l = d_df[d_df["pnl_pct"] <= 0]
            d_pf = (w["pnl_pct"].sum() / abs(l["pnl_pct"].sum())) if len(l) and l["pnl_pct"].sum() != 0 else float("inf")
            return {
                "trades": len(d_df),
                "win_rate": float(len(w) / len(d_df)),
                "pf": float(d_pf),
                "avg_r": float(d_df["r_multiple"].mean()),
                "exp": float(d_df["pnl_pct"].mean()),
            }

        l_st = dir_stats(longs)
        s_st = dir_stats(shorts)
        if len(longs) > 0 and len(shorts) > 0:
            print(f"  多头(Long):  {l_st['trades']:2d} 笔 | 胜率: {l_st['win_rate']:5.1%} | PF: {l_st['pf']:4.2f} | 平均 R: {l_st['avg_r']:+5.2f} R | 期望: {l_st['exp']:+6.2%}")
            print(f"  空头(Short): {s_st['trades']:2d} 笔 | 胜率: {s_st['win_rate']:5.1%} | PF: {s_st['pf']:4.2f} | 平均 R: {s_st['avg_r']:+5.2f} R | 期望: {s_st['exp']:+6.2%}")

        print(f"  平均持仓: {sub['hold_bars'].mean():.1f} 根K线  |  MAE 中位: {mae_med:.2%}  |  MFE 中位: {mfe_med:+.2%}")
        if n_tp_protected or n_tp_runner or n_tp_partial:
            print(f"  出场分布: 止损 {n_stop} 笔 ({n_stop/n_total:.1%}) | "
                  f"止盈兑现 {n_tp_total} 笔 ({n_tp_total/n_total:.1%}) [TP1保本防守: {n_tp_protected}, TP+Runner: {n_tp_runner}, 部分止盈到期: {n_tp_partial}] | "
                  f"到期平仓 {n_time} 笔 ({n_time/n_total:.1%})")
        else:
            print(f"  出场分布: 止损 {n_stop} 笔 ({n_stop/n_total:.1%}) | 止盈 {n_tp_classic} 笔 ({n_tp_classic/n_total:.1%}) | 到期平仓 {n_time} 笔 ({n_time/n_total:.1%})")

        out[f"{arm}_lifecycle"] = {
            "trades": n_total, "win_rate": wr, "profit_factor": pf,
            "avg_r": avg_r, "expectancy": exp, "mae_med": mae_med, "mfe_med": mfe_med,
            "stop_exits": n_stop, "tp_exits": n_tp_total, "time_exits": n_time,
            "tp_details": {
                "tp1_protected": n_tp_protected,
                "tp_runner": n_tp_runner,
                "tp_partial_time": n_tp_partial,
                "tp_classic": n_tp_classic
            },
            "long_stats": l_st,
            "short_stats": s_st
        }
    return out


def random_control(arrs: dict, counts: dict, trials: int = TRIALS,
                   seed: int = 0) -> np.ndarray:
    """
    按「观测到的多空混合比例」抽取同规模随机入场。
    每次抽样都保持 k_long : k_short 与原样本一致，否则对照组的方向结构不同、均值不可比。
    """
    usable = {s: arrs[s] for s, k in counts.items() if k > 0 and len(arrs.get(s, []))}
    if not usable:
        return np.array([])
    rng = np.random.default_rng(seed)
    means = np.empty(trials)
    for i in range(trials):
        parts = [rng.choice(a, size=min(counts[s], len(a)), replace=False)
                 for s, a in usable.items()]
        means[i] = np.concatenate(parts).mean()
    return means


# ==============================================================================
# 4. 汇总输出
# ==============================================================================

def arrs_from_forward(fwd: Forward, horizons=HORIZONS) -> dict:
    return {h: {s: fwd.ret_array(s, h) for s in (1, -1)} for h in horizons}


def merge_arrs(items: list, horizons=HORIZONS) -> dict:
    out = {}
    for h in horizons:
        out[h] = {}
        for s in (1, -1):
            parts = [a[h][s] for a in items if a.get(h, {}).get(s) is not None and len(a[h][s])]
            out[h][s] = np.concatenate(parts) if parts else np.array([])
    return out


def report_records(rec: pd.DataFrame, arrs: dict, label: str, verbose: bool = True) -> dict:
    """rec 可以是单标的，也可以是多样本池化后的记录；对照组从 arrs 抽取"""
    n_events = rec["date"].nunique() if len(rec) else 0
    if verbose:
        print(f"\n{'=' * 82}")
        print(f"标的: {label}  |  样本 {len(rec)} 条记录  |  事件 {n_events} 个")
        print("=" * 82)
    if rec.empty:
        if verbose:
            print("无事件")
        return {}

    out = {}
    for arm in ("MTF", "MTF+HTF"):
        sub = rec[rec["arm"] == arm]
        if sub.empty:
            continue
        if verbose:
            print(f"\n[{arm}]  事件 {sub['date'].nunique()} 个")
            print(f"{'持有期':<7}{'样本':>5}{'胜率':>8}{'均值':>9}{'中位':>9}"
                  f"{'MAE中位':>9}{'MFE中位':>9}{'随机均值':>10}{'分位':>7}  结论")
        for h in sorted(sub["horizon"].unique()):
            g = sub[sub["horizon"] == h]
            mean = float(g["ret"].mean())
            counts = {int(s): int((g["side"] == s).sum()) for s in g["side"].unique()}
            ctrl = random_control(arrs[h], counts, seed=h + 7)
            rnd = float(ctrl.mean()) if len(ctrl) else float("nan")
            pct = float((ctrl < mean).mean()) if len(ctrl) else float("nan")
            if pct >= 0.95:
                verdict = "显著优于随机"
            elif pct <= 0.05:
                verdict = "显著劣于随机"
            else:
                verdict = "无显著差异"
            if verbose:
                print(f"{h:<7}{len(g):>5}{g['ret'].gt(0).mean():>7.1%}{mean:>9.2%}"
                      f"{g['ret'].median():>9.2%}{g['mae'].median():>9.2%}"
                      f"{g['mfe'].median():>9.2%}{rnd:>10.2%}{pct:>7.2f}  {verdict}")
            out[f"{arm}_h{h}"] = {"n": int(len(g)), "win": float(g["ret"].gt(0).mean()),
                                  "mean": mean, "median": float(g["ret"].median()),
                                  "mae_med": float(g["mae"].median()),
                                  "mfe_med": float(g["mfe"].median()),
                                  "random_mean": rnd, "percentile": pct,
                                  "verdict": verdict}
    return out


def report(fwd: Forward, rec: pd.DataFrame, label: str, verbose: bool = True) -> dict:
    return report_records(rec, arrs_from_forward(fwd), label, verbose)


# ==============================================================================
# 5. 合成数据（已知答案验证）
# ==============================================================================

def synthetic_series(n: int, mode: str, seed: int) -> pd.DataFrame:
    """
    random -> 几何随机游走，无漂移（趋势跟随应无 edge）
    trend  -> 制度切换：多空趋势段交替，每段 60~150 根（道氏主张的正是这种多周趋势）
    range  -> AR(1) 负自相关，均值回归（趋势跟随应为负 edge）
    """
    rng = np.random.default_rng(seed)
    if mode == "random":
        ret = rng.normal(0, 0.015, n)
    elif mode == "trend":
        ret = np.zeros(n)
        i = 0
        while i < n:
            seg = int(rng.integers(60, 150))
            mu = float(rng.choice([-1.0, 1.0])) * 0.0015
            m = min(seg, n - i)
            ret[i:i + m] = mu + rng.normal(0, 0.011, m)
            i += seg
    elif mode == "range":
        eps = rng.normal(0, 0.015, n)
        ret = np.zeros(n)
        for i in range(1, n):
            ret[i] = -0.30 * ret[i - 1] + eps[i]
    else:
        raise ValueError(mode)

    close = 100 * np.exp(np.cumsum(ret))
    spread = close * rng.uniform(0.004, 0.012, n)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    idx = pd.bdate_range("2017-01-03", periods=n)
    return pd.DataFrame({"Open": open_,
                         "High": np.maximum(open_, close) + spread,
                         "Low": np.minimum(open_, close) - spread,
                         "Close": close,
                         "Volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def run_synthetic(mode: str, n_seeds: int = 3, n: int = 1500,
                  method: str = "h2", buffer_k: float = 0.0) -> dict:
    print(f"\n{'#' * 82}")
    print(f"# 合成数据验证  mode={mode}  method={method}  buffer={buffer_k}  序列 {n_seeds} 条 × {n} 根")
    print(f"# 预期方向: random≈0  |  trend>0  |  range<0")
    print("#" * 82)

    recs, arrs_list = [], []
    for s in range(n_seeds):
        df = synthetic_series(n, mode, seed=200 + s)
        fwd = Forward(df)
        rec = build_records(fwd, enumerate_events(df, weekly_from_daily(df),
                                                  method=method, buffer_k=buffer_k))
        if not rec.empty:
            rec["seed"] = s
            recs.append(rec)
        arrs_list.append(arrs_from_forward(fwd))
        if s == 0:
            report(fwd, rec, f"synthetic-{mode}-seed0", verbose=True)

    if not recs:
        print("无事件")
        return {}

    pooled = pd.concat(recs, ignore_index=True)
    arrs = merge_arrs(arrs_list)
    print(f"\n[池化汇总] {len(recs)} 条序列共 {pooled['date'].nunique()} 个日期上产生 "
          f"{len(pooled[pooled['horizon'] == HORIZONS[0]])} 条记录")
    return report_records(pooled, arrs, f"{mode}(pooled {n_seeds} seeds, {method})", verbose=True)


# ==============================================================================
# 6. 引擎自检（直接证明 walk-forward 性质，不依赖统计显著性）
# ==============================================================================

def verify(n: int = 700, method: str = "h2", buffer_k: float = 0.0) -> int:
    fails = []

    def chk(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    df = synthetic_series(n, "trend", seed=7)
    w = weekly_from_daily(df)
    full = enumerate_events(df, w, method=method, buffer_k=buffer_k)
    print(f"  自检序列: {n} 根, 事件 {len(full)} 个 (method={method})")

    print("\n[T1] 未来函数：用子序列重算，历史决策必须完全一致")
    for T in (300, 450, 600):
        part = enumerate_events(df.iloc[:T + 1], weekly_from_daily(df.iloc[:T + 1]),
                                method=method, buffer_k=buffer_k)
        a = [(e["t"], e["arm"], e["side"]) for e in full if e["t"] <= T - 1]
        b = [(e["t"], e["arm"], e["side"]) for e in part]
        chk(f"截至 t={T - 1} 的决策集一致", a == b, f"{len(a)} 个 vs {len(b)} 个")

    print("\n[T2] 入场时点：第 t 根决策，第 t+1 根开盘入场")
    fwd = Forward(df)
    t, h = 200, 10
    s = fwd.stats(t, 1, h)
    expect = float(df["Close"].iloc[t + 1 + h]) / float(df["Open"].iloc[t + 1]) - 1.0
    chk("多头收益 = Close[t+1+h] / Open[t+1] - 1", abs(s["ret"] - expect) < 1e-12,
        f"{s['ret']:.6f} vs {expect:.6f}")
    s2 = fwd.stats(t, -1, h)
    chk("空头收益为多头取负", abs(s2["ret"] + expect) < 1e-12)

    print("\n[T3] 周线不越界：只用标签日期 <= 当日的完整周")
    bad = 0
    for e in full[:30]:
        wp = w[w.index <= df.index[e["t"]]]
        if len(wp) and wp.index[-1] > df.index[e["t"]]:
            bad += 1
    chk("无未走完的周混入", bad == 0, f"越界 {bad} 次")

    print("\n[T4] MAE/MFE 口径")
    ok = True
    for h2 in HORIZONS:
        for tt in range(MIN_BARS, n - h2 - 2, 37):
            st = fwd.stats(tt, 1, h2)
            if st is None:
                continue
            if not (st["mae"] <= 0.0 <= st["mfe"] and st["mae"] <= st["ret"] <= st["mfe"]):
                ok = False
    chk("MAE <= 0 <= MFE 且 MAE <= 收益 <= MFE", ok)

    # 交叉验证：确保 h1 模式也能独立通过自检
    print("\n[T1-H1] 历史 H1 计数模式未来函数自检")
    full_h1 = enumerate_events(df, w, method="h1")
    part_h1 = enumerate_events(df.iloc[:451], weekly_from_daily(df.iloc[:451]), method="h1")
    a_h1 = [(e["t"], e["arm"], e["side"]) for e in full_h1 if e["t"] <= 449]
    b_h1 = [(e["t"], e["arm"], e["side"]) for e in part_h1]
    chk("H1 历史决策集一致", a_h1 == b_h1, f"{len(a_h1)} 个 vs {len(b_h1)} 个")

    print("\n[T5-Limit] 位置层挂单未来函数自检：截断重算，挂单与成交时点一致")
    full_limit = build_records(fwd, full, df=df, entry_mode="pullback_limit")
    for T in (350, 500):
        sub_df_t = df.iloc[:T + 1]
        part_events = enumerate_events(sub_df_t, weekly_from_daily(sub_df_t), method=method, buffer_k=buffer_k)
        fwd_part = Forward(sub_df_t)
        part_limit = build_records(fwd_part, part_events, df=sub_df_t, entry_mode="pullback_limit")
        a_lim = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2))
                 for _, r in full_limit[full_limit["horizon"] == 5].iterrows() if r["t"] <= T - 20]
        b_lim = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2))
                 for _, r in part_limit[part_limit["horizon"] == 5].iterrows() if r["t"] <= T - 20]
        chk(f"截至 t={T - 20} 的限价挂单成交集完全一致", a_lim == b_lim, f"{len(a_lim)} 个 vs {len(b_lim)} 个")

    print("\n[T6-Trigger] 触发层形态未来函数自检：截断重算，触发与入场时点一致")
    full_trig = build_records(fwd, full, df=df, entry_mode="trigger")
    for T in (350, 500):
        sub_df_t = df.iloc[:T + 1]
        part_events = enumerate_events(sub_df_t, weekly_from_daily(sub_df_t), method=method, buffer_k=buffer_k)
        fwd_part = Forward(sub_df_t)
        part_trig = build_records(fwd_part, part_events, df=sub_df_t, entry_mode="trigger")
        a_trig = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2))
                  for _, r in full_trig[full_trig["horizon"] == 5].iterrows() if r["t"] <= T - 20]
        b_trig = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2))
                  for _, r in part_trig[part_trig["horizon"] == 5].iterrows() if r["t"] <= T - 20]
        chk(f"截至 t={T - 20} 的触发入场集完全一致", a_trig == b_trig, f"{len(a_trig)} 个 vs {len(b_trig)} 个")

    print("\n[T7-Execution] 执行层风控与保守撮合自检：同一根K线双触严格判定止损优先")
    dummy_df = pd.DataFrame({
        "Open": [100.0, 100.0],
        "High": [100.0, 115.0],
        "Low": [100.0, 90.0],
        "Close": [100.0, 105.0],
        "Volume": [1e6, 1e6]
    }, index=pd.bdate_range("2026-01-01", periods=2))
    tie_trade = sl.simulate_strategy_trade(dummy_df, 0, 100.0, 1, 95.0, 110.0, min_rr=2.0)
    chk("同根K线双触判定为止损先出场", tie_trade.get("exit_reason") == "stop_loss_conservative" and tie_trade.get("win") is False)

    print("\n[T8-RR-Gate] 盈亏比硬闸门自检：RR < 2.0 严格拒绝入场")
    bad_rr_trade = sl.simulate_strategy_trade(dummy_df, 0, 100.0, 1, 95.0, 105.0, min_rr=2.0)
    chk("RR=1.0 < 2.0 被硬闸门驳回", bad_rr_trade.get("status") == "rejected_by_min_rr")

    print("\n[T9-Strategy] 全闭环策略未来函数自检：截断重算，触发入场与出场完全一致")
    full_strat = build_records(fwd, full, df=df, entry_mode="strategy")
    for T in (350, 500):
        sub_df_t = df.iloc[:T + 1]
        part_events = enumerate_events(sub_df_t, weekly_from_daily(sub_df_t), method=method, buffer_k=buffer_k)
        fwd_part = Forward(sub_df_t)
        part_strat = build_records(fwd_part, part_events, df=sub_df_t, entry_mode="strategy")
        a_strat = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                   for _, r in full_strat[full_strat["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        b_strat = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                   for _, r in part_strat[part_strat["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        chk(f"截至 t={T - 50} 的全闭环交易集完全一致", a_strat == b_strat, f"{len(a_strat)} 个 vs {len(b_strat)} 个")

    print("\n[T10-Dow-Wave] 道氏印证与波浪过滤闭环未来函数自检：截断重算，入场与出场完全一致")
    full_dw = build_records(fwd, full, df=df, entry_mode="strategy", use_wave_filter=True, volume_filter=True)
    for T in (350, 500):
        sub_df_t = df.iloc[:T + 1]
        part_events = enumerate_events(sub_df_t, weekly_from_daily(sub_df_t), method=method, buffer_k=buffer_k)
        fwd_part = Forward(sub_df_t)
        part_dw = build_records(fwd_part, part_events, df=sub_df_t, entry_mode="strategy", use_wave_filter=True, volume_filter=True)
        a_dw = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                for _, r in full_dw[full_dw["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        b_dw = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                for _, r in part_dw[part_dw["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        chk(f"截至 t={T - 50} 的道氏波浪闭环交易集完全一致", a_dw == b_dw, f"{len(a_dw)} 个 vs {len(b_dw)} 个")

    print("\n[T11-Crypto] 加密专用策略未来函数与分级止盈自检：截断重算，触发入场与出场完全一致")
    full_cr = build_records(fwd, full, df=df, entry_mode="strategy_crypto", asset_type="crypto")
    for T in (350, 500):
        sub_df_t = df.iloc[:T + 1]
        part_events = enumerate_events(sub_df_t, weekly_from_daily(sub_df_t), method=method, buffer_k=buffer_k)
        fwd_part = Forward(sub_df_t)
        part_cr = build_records(fwd_part, part_events, df=sub_df_t, entry_mode="strategy_crypto", asset_type="crypto")
        a_cr = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                for _, r in full_cr[full_cr["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        b_cr = [(r["t"], r["arm"], r["side"], round(r["entry_price"], 2), round(r["exit_price"], 2))
                for _, r in part_cr[part_cr["horizon"] == 5].iterrows() if r["t"] <= T - 50]
        chk(f"截至 t={T - 50} 的加密专用闭环交易集完全一致", a_cr == b_cr, f"{len(a_cr)} 个 vs {len(b_cr)} 个")

    print("\n" + ("✅ 自检通过" if not fails else f"❌ {len(fails)} 项失败"))
    return 0 if not fails else 1



# ==============================================================================
# 7. 主流程
# ==============================================================================

def cluster_robust(rec: pd.DataFrame, arm: str, h: int, freq: str = "Q") -> dict:
    """
    时间簇稳健估计：30 只美股高度同涨同跌，事件在时间上成簇。
    把每个季度当作一个独立观测，先求簇内均值再对簇均值做统计，
    避免「把 500 个高度相关的事件当成 500 个独立样本」的伪显著。
    """
    g = rec[(rec["arm"] == arm) & (rec["horizon"] == h)].copy()
    if g.empty:
        return {}
    per = g.groupby(pd.PeriodIndex(g["date"], freq=freq))["ret"]
    m = per.mean()
    if len(m) < 3:
        return {"n_clusters": int(len(m)), "mean": float(m.mean()),
                "t": float("nan"), "note": "簇数不足"}
    se = m.std(ddof=1) / np.sqrt(len(m))
    return {"n_clusters": int(len(m)), "mean": float(m.mean()),
            "std": float(m.std(ddof=1)), "t": float(m.mean() / se) if se > 0 else float("nan")}


def run_comparison(tickers: list, method: str = "h2", buffer_k: float = 0.0,
                   volume_filter: bool = True, offline: bool = True, download: bool = False,
                   min_rr: float = 2.0, json_out: str = "",
                   exclude_black_swans: bool = False,
                   use_crypto_trend_gate: bool = True) -> dict:
    """
    跨层级对比分析：在相同标的样本与相同方向事件上，同时推演并横向对比：
      1. 突破市价追入 (Baseline: Market Breakout)
      2. 结构回踩限价 (Position: Pullback Limit Swing)
      3. 斐波那契回踩 (Position: Pullback Limit Fib 50%)
      4. 关键位裸K触发 (Trigger: Pin Bar / 2B / Engulfing)
      5. 全闭环风控执行 (Execution: Direction->Location->Trigger->RR>=2.0)
    """
    pe.reset_fetch_state()
    modes = [
        ("market", "1. 突破市价追入 (Baseline)", "swing"),
        ("pullback_limit_swing", "2. 结构回踩限价 (Swing)", "swing"),
        ("pullback_limit_fib50", "3. 斐波50%回踩 (Fib50)", "fib50"),
        ("trigger", "4. 关键位裸K触发 (Trigger)", "swing"),
        ("strategy", "5. 美股原装全闭环 (RR>=2.0)", "swing"),
        ("strategy_dow_wave", "6. 道氏印证+波浪过滤闭环", "swing"),
        ("strategy_crypto", "7. 加密专用全闭环 (Crypto 2B+分级止盈)", "swing"),
    ]

    # 预加载大盘指数作为道氏印证硬闸门基准
    market_data = {
        "SPY": load_ticker("SPY", download=download and not offline),
        "QQQ": load_ticker("QQQ", download=download and not offline)
    }

    stats = {arm: {m[0]: {"events": 0, "filled": 0, "records": [], "trades": []} for m in modes}
             for arm in ("MTF", "MTF+HTF")}

    n_valid_tickers = 0
    for tk in tickers:
        df = load_ticker(tk, download=download and not offline)
        if df.empty:
            continue
        n_valid_tickers += 1
        w = weekly_from_daily(df)
        events = enumerate_events(df, w, method=method, buffer_k=buffer_k, volume_filter=volume_filter)
        if not events:
            continue

        # 黑天鹅事件过滤（美股标的）：剔除决策根落在系统性危机窗口内的事件，
        # 保证横向对比中所有模式（含基线）使用一致的危机剔除口径
        if exclude_black_swans and not sc.is_crypto_asset(tk):
            events = [e for e in events if not sl.is_stock_black_swan(e["date"])]
            if not events:
                continue

        fwd = Forward(df)
        for arm in ("MTF", "MTF+HTF"):
            arm_events = [e for e in events if e["arm"] == arm]
            if not arm_events:
                continue
            for m_key, m_name, retest in modes:
                stats[arm][m_key]["events"] += len(arm_events)
                if m_key == "market":
                    rec = build_records(fwd, arm_events, entry_mode="market")
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += len(arm_events)
                        stats[arm][m_key]["records"].append(rec)
                elif m_key == "pullback_limit_swing":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="pullback_limit", retest_level="swing",
                                        volume_filter=volume_filter)
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                elif m_key == "pullback_limit_fib50":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="pullback_limit", retest_level="fib50",
                                        volume_filter=volume_filter)
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                elif m_key == "trigger":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="trigger", retest_level="swing")
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                elif m_key == "strategy":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="strategy", retest_level="swing",
                                        min_rr=min_rr, volume_filter=volume_filter, asset_type="stock", ticker=tk,
                                        exclude_black_swans=exclude_black_swans)
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                        stats[arm][m_key]["trades"].append(rec.drop_duplicates(subset=["t", "side", "entry_price"]))
                elif m_key == "strategy_dow_wave":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="strategy", retest_level="swing",
                                        min_rr=min_rr, volume_filter=True, market_data=market_data,
                                        use_market_gate=True, use_wave_filter=True, asset_type="stock", ticker=tk,
                                        exclude_black_swans=exclude_black_swans)
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                        stats[arm][m_key]["trades"].append(rec.drop_duplicates(subset=["t", "side", "entry_price"]))
                elif m_key == "strategy_crypto":
                    rec = build_records(fwd, arm_events, df=df, entry_mode="strategy_crypto", retest_level="swing",
                                        min_rr=min_rr, volume_filter=volume_filter, asset_type="crypto", ticker=tk,
                                        exclude_black_swans=exclude_black_swans,
                                        use_crypto_trend_gate=use_crypto_trend_gate)
                    if not rec.empty:
                        stats[arm][m_key]["filled"] += rec["date"].nunique()
                        stats[arm][m_key]["records"].append(rec)
                        stats[arm][m_key]["trades"].append(rec.drop_duplicates(subset=["t", "side", "entry_price"]))


    print(f"\n{'=' * 96}")
    print(f"多层级策略全流程对比报告 (标的数: {n_valid_tickers} 只, method={method}, volume_filter={volume_filter})")
    print(f"{'=' * 96}")

    summary_out = {}
    for arm in ("MTF", "MTF+HTF"):
        print(f"\n>>> 组别: [{arm}]")
        print(f"{'策略模式':<28}{'事件数':>7}{'成交数':>7}{'成交率':>8}{'h=20胜率':>10}{'h=20均值':>10}{'h=20中位':>10}{'h=20 MAE':>10}")
        print("-" * 96)
        arm_out = {}
        for m_key, m_name, _ in modes:
            d = stats[arm][m_key]
            ev = d["events"]
            fi = d["filled"]
            rate = fi / max(1, ev)
            if not d["records"]:
                print(f"{m_name:<28}{ev:>7}{fi:>7}{rate:>7.1%}{'N/A':>10}{'N/A':>10}{'N/A':>10}{'N/A':>10}")
                continue
            cat_rec = pd.concat(d["records"], ignore_index=True)
            g20 = cat_rec[cat_rec["horizon"] == 20]
            if g20.empty:
                print(f"{m_name:<28}{ev:>7}{fi:>7}{rate:>7.1%}{'N/A':>10}{'N/A':>10}{'N/A':>10}{'N/A':>10}")
                continue
            wr = float(g20["ret"].gt(0).mean())
            ret_m = float(g20["ret"].mean())
            ret_med = float(g20["ret"].median())
            mae_med = float(g20["mae"].median())
            print(f"{m_name:<28}{ev:>7}{fi:>7}{rate:>7.1%}{wr:>9.1%}{ret_m:>9.2%}{ret_med:>9.2%}{mae_med:>9.2%}")
            arm_out[m_key] = {
                "events": ev, "filled": fi, "fill_rate": rate,
                "h20_win": wr, "h20_mean": ret_m, "h20_median": ret_med, "h20_mae_med": mae_med
            }

        # 5. 全闭环风控生命周期指标
        strat_trades = stats[arm]["strategy"]["trades"]
        if strat_trades:
            all_trades = pd.concat(strat_trades, ignore_index=True)
            wins = all_trades[all_trades["pnl_pct"] > 0]
            losses = all_trades[all_trades["pnl_pct"] <= 0]
            pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())) if len(losses) and losses["pnl_pct"].sum() != 0 else float("inf")
            avg_r = float(all_trades["r_multiple"].mean())
            exp = float(all_trades["pnl_pct"].mean())
            wr = len(wins) / len(all_trades)
            mae_col = "trade_mae" if "trade_mae" in all_trades.columns else "mae"
            mfe_col = "trade_mfe" if "trade_mfe" in all_trades.columns else "mfe"
            mae_strat = float(all_trades[mae_col].median())
            mfe_strat = float(all_trades[mfe_col].median())
            print(f"\n  [{arm} 全闭环风控 (RR >= {min_rr:.1f}) 生命周期指标]")
            print(f"  成交交易: {len(all_trades)} 笔 | 胜率: {wr:.1%} ({len(wins)} 胜 / {len(losses)} 负)")
            print(f"  盈亏比 (PF): {pf:.2f} | 平均 R 倍数: {avg_r:+.2f} R | 单笔期望收益: {exp:+.2%}")
            print(f"  平均持仓周期: {all_trades['hold_bars'].mean():.1f} 根K线 | MAE 中位: {mae_strat:.2%} | MFE 中位: {mfe_strat:+.2%}")
            arm_out["strategy_lifecycle"] = {
                "trades": len(all_trades), "win_rate": wr, "profit_factor": pf,
                "avg_r": avg_r, "expectancy": exp, "hold_bars": float(all_trades['hold_bars'].mean()),
                "mae_med": mae_strat, "mfe_med": mfe_strat
            }

        # 6. 道氏印证 + 波浪位置过滤闭环生命周期指标
        strat_dw_trades = stats[arm]["strategy_dow_wave"]["trades"]
        if strat_dw_trades:
            all_dw = pd.concat(strat_dw_trades, ignore_index=True)
            wins_dw = all_dw[all_dw["pnl_pct"] > 0]
            losses_dw = all_dw[all_dw["pnl_pct"] <= 0]
            pf_dw = (wins_dw["pnl_pct"].sum() / abs(losses_dw["pnl_pct"].sum())) if len(losses_dw) and losses_dw["pnl_pct"].sum() != 0 else float("inf")
            avg_r_dw = float(all_dw["r_multiple"].mean())
            exp_dw = float(all_dw["pnl_pct"].mean())
            wr_dw = len(wins_dw) / len(all_dw)
            mae_col = "trade_mae" if "trade_mae" in all_dw.columns else "mae"
            mfe_col = "trade_mfe" if "trade_mfe" in all_dw.columns else "mfe"
            mae_dw = float(all_dw[mae_col].median())
            mfe_dw = float(all_dw[mfe_col].median())
            print(f"\n  [{arm} 道氏印证+波浪过滤闭环 (Dow+Wave Sniper) 生命周期指标]")
            print(f"  成交交易: {len(all_dw)} 笔 | 胜率: {wr_dw:.1%} ({len(wins_dw)} 胜 / {len(losses_dw)} 负)")
            print(f"  盈亏比 (PF): {pf_dw:.2f} | 平均 R 倍数: {avg_r_dw:+.2f} R | 单笔期望收益: {exp_dw:+.2%}")
            print(f"  平均持仓周期: {all_dw['hold_bars'].mean():.1f} 根K线 | MAE 中位: {mae_dw:.2%} | MFE 中位: {mfe_dw:+.2%}")
            arm_out["strategy_dow_wave_lifecycle"] = {
                "trades": len(all_dw), "win_rate": wr_dw, "profit_factor": pf_dw,
                "avg_r": avg_r_dw, "expectancy": exp_dw, "hold_bars": float(all_dw['hold_bars'].mean()),
                "mae_med": mae_dw, "mfe_med": mfe_dw
            }

        # 7. 加密专用全闭环 (Crypto 2B+分级止盈) 生命周期指标
        strat_cr_trades = stats[arm]["strategy_crypto"]["trades"]
        if strat_cr_trades:
            all_cr = pd.concat(strat_cr_trades, ignore_index=True)
            wins_cr = all_cr[all_cr["pnl_pct"] > 0]
            losses_cr = all_cr[all_cr["pnl_pct"] <= 0]
            pf_cr = (wins_cr["pnl_pct"].sum() / abs(losses_cr["pnl_pct"].sum())) if len(losses_cr) and losses_cr["pnl_pct"].sum() != 0 else float("inf")
            avg_r_cr = float(all_cr["r_multiple"].mean())
            exp_cr = float(all_cr["pnl_pct"].mean())
            wr_cr = len(wins_cr) / len(all_cr)
            mae_col = "trade_mae" if "trade_mae" in all_cr.columns else "mae"
            mfe_col = "trade_mfe" if "trade_mfe" in all_cr.columns else "mfe"
            mae_cr = float(all_cr[mae_col].median())
            mfe_cr = float(all_cr[mfe_col].median())
            n_stop_cr = int((all_cr["exit_reason"].str.contains("stop_loss") | (all_cr["exit_reason"] == "stop_loss_conservative")).sum())
            n_tp_prot_cr = int((all_cr["exit_reason"] == "tp1_then_be_protected").sum())
            n_tp_run_cr = int((all_cr["exit_reason"] == "tp_runner_captured").sum())
            n_tp_part_cr = int((all_cr["exit_reason"] == "tp_partial_time_horizon").sum())
            n_tp_tot_cr = n_tp_prot_cr + n_tp_run_cr + n_tp_part_cr
            n_time_cr = int((all_cr["exit_reason"] == "time_horizon").sum())
            n_tot_cr = len(all_cr)

            longs_cr = all_cr[all_cr["side"] == 1]
            shorts_cr = all_cr[all_cr["side"] == -1]

            def dir_stats_cr(d_df):
                if d_df.empty:
                    return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "avg_r": 0.0, "exp": 0.0}
                w = d_df[d_df["pnl_pct"] > 0]
                l = d_df[d_df["pnl_pct"] <= 0]
                d_pf = (w["pnl_pct"].sum() / abs(l["pnl_pct"].sum())) if len(l) and l["pnl_pct"].sum() != 0 else float("inf")
                return {
                    "trades": len(d_df),
                    "win_rate": float(len(w) / len(d_df)),
                    "pf": float(d_pf),
                    "avg_r": float(d_df["r_multiple"].mean()),
                    "exp": float(d_df["pnl_pct"].mean()),
                }

            l_st_cr = dir_stats_cr(longs_cr)
            s_st_cr = dir_stats_cr(shorts_cr)

            bs_label = " [已剔除黑天鹅暴跌期]" if exclude_black_swans else ""
            print(f"\n  [{arm} 加密专用全闭环 (Crypto 2B+分级止盈) 生命周期指标{bs_label}]")
            print(f"  成交交易: {n_tot_cr} 笔 | 胜率: {wr_cr:.1%} ({len(wins_cr)} 胜 / {len(losses_cr)} 负)")
            print(f"  盈亏比 (PF): {pf_cr:.2f} | 平均 R 倍数: {avg_r_cr:+.2f} R | 单笔期望收益: {exp_cr:+.2%}")
            if len(longs_cr) > 0 and len(shorts_cr) > 0:
                print(f"  多头(Long):  {l_st_cr['trades']:2d} 笔 | 胜率: {l_st_cr['win_rate']:5.1%} | PF: {l_st_cr['pf']:4.2f} | 平均 R: {l_st_cr['avg_r']:+5.2f} R | 期望: {l_st_cr['exp']:+6.2%}")
                print(f"  空头(Short): {s_st_cr['trades']:2d} 笔 | 胜率: {s_st_cr['win_rate']:5.1%} | PF: {s_st_cr['pf']:4.2f} | 平均 R: {s_st_cr['avg_r']:+5.2f} R | 期望: {s_st_cr['exp']:+6.2%}")
            print(f"  平均持仓周期: {all_cr['hold_bars'].mean():.1f} 根K线 | MAE 中位: {mae_cr:.2%} | MFE 中位: {mfe_cr:+.2%}")
            print(f"  出场分布: 止损 {n_stop_cr} 笔 ({n_stop_cr/n_tot_cr:.1%}) | "
                  f"止盈兑现 {n_tp_tot_cr} 笔 ({n_tp_tot_cr/n_tot_cr:.1%}) [TP1保本: {n_tp_prot_cr}, Runner捕捉: {n_tp_run_cr}, 部分止盈到期: {n_tp_part_cr}] | "
                  f"到期平仓 {n_time_cr} 笔 ({n_time_cr/n_tot_cr:.1%})")
            arm_out["strategy_crypto_lifecycle"] = {
                "trades": len(all_cr), "win_rate": wr_cr, "profit_factor": pf_cr,
                "avg_r": avg_r_cr, "expectancy": exp_cr, "hold_bars": float(all_cr['hold_bars'].mean()),
                "mae_med": mae_cr, "mfe_med": mfe_cr,
                "stop_exits": n_stop_cr, "tp_exits": n_tp_tot_cr, "time_exits": n_time_cr,
                "tp_details": {
                    "tp1_protected": n_tp_prot_cr,
                    "tp_runner": n_tp_run_cr,
                    "tp_partial_time": n_tp_part_cr
                },
                "long_stats": l_st_cr,
                "short_stats": s_st_cr,
                "exclude_black_swans": bool(exclude_black_swans)
            }
        summary_out[arm] = arm_out


    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary_out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[JSON] 对比结果已写入 {json_out}", file=sys.stderr)

    return summary_out


def main() -> int:
    ap = argparse.ArgumentParser(description="分层量化回测：方向层、位置层、触发层与风控执行闭环")
    ap.add_argument("--tickers", type=str, default="", help="逗号分隔，如 AAPL,MSFT,NVDA")
    ap.add_argument("--tickers-file", type=str, default="", help="每行一个代码")
    ap.add_argument("--download", action="store_true", help="联网下载并缓存日线（5 年）")
    ap.add_argument("--offline", action="store_true", help="仅使用本地缓存")
    ap.add_argument("--synthetic", type=str, default="",
                    choices=["", "random", "trend", "range"], help="合成数据验证")
    ap.add_argument("--verify", action="store_true", help="引擎自检：证明无未来函数")
    ap.add_argument("--method", type=str, default="h2", choices=["h1", "h2"],
                    help="方向判定方式：h1 标签计数，h2 收盘突破结构位（默认 h2）")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="结构穿越缓冲倍数 buffer_k * median_bar_range（默认 0.0）")
    ap.add_argument("--volume-filter", action="store_true",
                    help="道氏第6条量能确认：放量（>1.2倍20日均量）才触发入场")
    ap.add_argument("--asset-type", type=str, default="auto", choices=["auto", "stock", "crypto"],
                    help="资产类型路由：auto 自动识别，stock 美股，crypto 加密货币（默认 auto）")
    ap.add_argument("--entry-mode", type=str, default="market",
                    choices=["market", "pullback_limit", "trigger", "strategy", "strategy_crypto"],
                    help="入场模式：market 市价追入，pullback_limit 限价回踩，trigger 裸K触发，strategy 全闭环，strategy_crypto 加密专用全闭环")
    ap.add_argument("--retest-level", type=str, default="swing",
                    choices=["swing", "fib50", "fib618"],
                    help="回踩测试水平位：swing 破位结构位，fib50 斐波50%，fib618 斐波61.8%")
    ap.add_argument("--min-rr", type=float, default=2.0,
                    help="风控硬闸门最小盈亏比要求（默认 2.0）")
    ap.add_argument("--compare", action="store_true",
                    help="执行多层级全流程对比横向审计（市价突破 vs 结构回踩 vs 斐波回踩 vs 触发形态 vs 全闭环）")
    ap.add_argument("--market-gate", action="store_true",
                    help="大盘指数相互印证硬闸门（SPY/QQQ 处于空头时拦截个股多头）")
    ap.add_argument("--wave-filter", action="store_true",
                    help="艾略特波浪位置过滤（拦截第5浪冲顶诱多、识别第3浪主升）")
    ap.add_argument("--exclude-black-swans", action="store_true",
                    help="剔除加密货币历史重大系统性黑天鹅崩盘期（312、519、Luna、FTX等全网清算挤兑）")
    ap.add_argument("--no-crypto-trend-gate", action="store_true",
                    help="禁用加密资产道氏 200 EMA 大趋势顺势门禁（默认开启：牛市只做多，熊市只做空）")
    ap.add_argument("--seeds", type=int, default=3, help="合成数据的序列条数（越多统计功效越高）")
    ap.add_argument("--bars", type=int, default=1500, help="合成数据每条序列的K线根数")
    ap.add_argument("--json", type=str, default="", dest="json_out", help="结果写入 JSON")
    args = ap.parse_args()

    if args.verify:
        return verify(method=args.method, buffer_k=args.buffer)

    if args.synthetic:
        s = run_synthetic(args.synthetic, n_seeds=args.seeds, n=args.bars,
                          method=args.method, buffer_k=args.buffer)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2, default=str)
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        tf_path = args.tickers_file
        if not os.path.exists(tf_path) and os.path.exists(os.path.join("config", tf_path)):
            tf_path = os.path.join("config", tf_path)
        with open(tf_path, encoding="utf-8") as f:
            for ln in f:
                clean = ln.split("#")[0].strip().upper()
                if clean:
                    tickers.append(clean)
    if not tickers:
        print("[错误] 请指定 --tickers / --tickers-file / --synthetic", file=sys.stderr)
        return 1

    use_trend_gate = not args.no_crypto_trend_gate

    if args.compare:
        run_comparison(tickers, method=args.method, buffer_k=args.buffer,
                       volume_filter=args.volume_filter, offline=args.offline,
                       download=args.download, min_rr=args.min_rr, json_out=args.json_out,
                       exclude_black_swans=args.exclude_black_swans,
                       use_crypto_trend_gate=use_trend_gate)
        return 0

    pe.reset_fetch_state()
    pooled, arrs_list = [], []
    market_data = None
    if args.market_gate:
        market_data = {
            "SPY": load_ticker("SPY", download=args.download and not args.offline),
            "QQQ": load_ticker("QQQ", download=args.download and not args.offline)
        }

    for tk in tickers:
        df = load_ticker(tk, download=args.download and not args.offline)
        if df.empty:
            print(f"[跳过] {tk}: 无本地缓存（Yahoo 403 拦截 / 代理不稳定，详见 backtest_protocol.md 第七节）",
                  file=sys.stderr)
            continue
        fwd = Forward(df)
        events = enumerate_events(df, weekly_from_daily(df),
                                  method=args.method,
                                  buffer_k=args.buffer,
                                  volume_filter=args.volume_filter)
        rec = build_records(fwd, events, df=df,
                            entry_mode=args.entry_mode,
                            retest_level=args.retest_level,
                            min_rr=args.min_rr,
                            volume_filter=args.volume_filter,
                            market_data=market_data,
                            use_market_gate=args.market_gate,
                            use_wave_filter=args.wave_filter,
                            asset_type=args.asset_type,
                            ticker=tk,
                            exclude_black_swans=args.exclude_black_swans,
                            use_crypto_trend_gate=use_trend_gate)
        if rec.empty:
            print(f"[跳过] {tk}: 未产生事件/成交", file=sys.stderr)
            continue
        rec["ticker"] = tk
        pooled.append(rec)
        arrs_list.append(arrs_from_forward(fwd))
        report(fwd, rec, f"{tk} ({args.entry_mode})", verbose=(len(pooled) == 1))

    if not pooled:
        print(f"\n[结果] 无可用数据。把日线 CSV 放到 {DATA_DIR}/<TICKER>.csv "
              "（列：Open,High,Low,Close,Volume；索引为日期），或用 --download 联网取数。",
              file=sys.stderr)
        return 1

    allrec = pd.concat(pooled, ignore_index=True)
    print(f"\n{'#' * 82}")
    print(f"# 全样本汇总  {len(pooled)} 只标的  mode={args.entry_mode}  "
          f"{len(allrec[allrec['horizon'] == HORIZONS[0]])} 条记录（h={HORIZONS[0]}）")
    print("#" * 82)
    summary = report_records(allrec, merge_arrs(arrs_list), f"全样本 {len(pooled)} 只 ({args.entry_mode})", verbose=True)

    if args.entry_mode in ("strategy", "strategy_crypto"):
        strat_sum = print_strategy_lifecycle_summary(allrec, label="全样本")
        summary["_strategy_lifecycle"] = strat_sum

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # ---- 时间簇稳健性（不改变 H1 定义，只作为必须一并报告的诊断）----
    print(f"\n{'#' * 82}")
    print("# 时间簇稳健性（按季度聚合，避免横截面相关性造成的伪显著）")
    print("#" * 82)
    print(f"{'组':<10}{'持有期':<7}{'簇数':>5}{'簇均值':>10}{'簇间t':>8}  解读")
    for arm in ("MTF", "MTF+HTF"):
        for h in (10, 20, 40):
            r = cluster_robust(allrec, arm, h)
            if not r:
                continue
            t = r.get("t", float("nan"))
            note = ("|t|>2 稳健" if np.isfinite(t) and abs(t) > 2 else
                    "|t|<2 不稳健" if np.isfinite(t) else "簇数不足")
            print(f"{arm:<10}{h:<7}{r['n_clusters']:>5}{r['mean']:>10.2%}{t:>8.2f}  {note}")

    if args.json_out:
        summary["_cluster_robust"] = {
            f"{a}_h{h}": cluster_robust(allrec, a, h)
            for a in ("MTF", "MTF+HTF") for h in (10, 20, 40)}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[JSON] 已写入 {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
