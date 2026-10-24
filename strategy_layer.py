#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_layer.py: 位置层（Location）、触发层（Trigger）与风控执行闭环（Execution）

理论依据（见 strategy_theory.md 与 backtest_result_v2.md）：
  1. 方向层 (Direction): 仅作为守门人（Gatekeeper），提供交易许可（side: +1 多 / -1 空）。
     严禁按突破K线市价盲目追高入场（解决 MAE -5.1% 的深幅回撤痛点）。
  2. 位置层 (Location): 突破后等待回调回踩关键结构位（Broken Swing High/Low）
     或斐波那契回撤区（Fib 50% / 61.8%），挂限价单（Limit Order）入场。
  3. 触发层 (Trigger): 在关键位置等待裸K反转确认（2B 流动性猎杀、拒绝 Pin Bar、吞没形态），
     配合基于 ATR/中位振幅 的紧密结构证伪止损（Tight Invalidation Stop）。
  4. 风险报酬与执行层 (Risk-Reward & Execution):
     - 硬闸门：RR >= min_rr (默认 2.0)，低于 1:2 一票否决；
     - 保守撮合：同一根 K 线若同时触及止损与止盈，一律判定为止损先触发；
     - 零未来函数：第 t 根决策，限价单/触发监控从 t+1 开始生效，触发后次根开盘入场。
"""

import numpy as np
import pandas as pd
import pa_engine as pe

# 默认风控参数
DEFAULT_STOP_BUFFER_K = 0.8
DEFAULT_MAX_WAIT_BARS = 15
DEFAULT_MIN_RR = 2.0
DEFAULT_MAX_HOLD_BARS = 40


# ==============================================================================
# 1. 结构位与斐波那契回撤提取 (Location Layer)
# ==============================================================================

def extract_breakout_levels(df: pd.DataFrame, t: int, side: int, swings: list,
                            buffer_k: float = 0.0) -> dict:
    """
    在第 t 根决策K线上，提取被突破的结构位、结构锚点以及斐波那契回撤位。
    严格只使用 <= t 的数据。
    """
    if len(swings) < 2 or t < 0 or t >= len(df):
        return {}

    conf = [s for s in swings if s.get("confirmed")]
    if len(conf) < 2:
        conf = swings

    highs = [s for s in conf if s.get("type") == "High"]
    lows = [s for s in conf if s.get("type") == "Low"]
    if not highs or not lows:
        return {}

    bar_rng = pe.median_bar_range(df.iloc[:t + 1], n=14)
    buf = buffer_k * bar_rng

    c_curr_high = float(df["High"].iloc[t])
    c_curr_low = float(df["Low"].iloc[t])
    c_close = float(df["Close"].iloc[t])

    if side == 1:
        # 多头突破：被突破的高点阻力转支撑
        h_break = float(highs[-1]["price"])
        l_anchor = float(lows[-1]["price"])
        l_anchor_idx = int(lows[-1].get("index", 0))
        # 提取自锚点低点以来整个冲量波段的真实最高点（严格 <= t，无未来函数）
        leg_high = float(df["High"].iloc[l_anchor_idx:t + 1].max()) if l_anchor_idx <= t else c_curr_high
        peak = max(c_curr_high, c_close, h_break, leg_high)
        impulse_leg = max(peak - l_anchor, bar_rng)

        fib50 = peak - 0.50 * impulse_leg
        fib618 = peak - 0.618 * impulse_leg
        structural_stop = round(l_anchor - DEFAULT_STOP_BUFFER_K * bar_rng, 2)

        return {
            "side": 1,
            "h_break": round(h_break, 2),
            "l_anchor": round(l_anchor, 2),
            "peak": round(peak, 2),
            "fib50": round(fib50, 2),
            "fib618": round(fib618, 2),
            "structural_stop": structural_stop,
            "bar_rng": round(bar_rng, 2),
            "swing_retest": round(h_break, 2),
        }
    elif side == -1:
        # 空头跌破：被跌破的低点支撑转阻力
        l_break = float(lows[-1]["price"])
        h_anchor = float(highs[-1]["price"])
        h_anchor_idx = int(highs[-1].get("index", 0))
        # 提取自锚点高点以来整个冲量下跌波段的真实最低点（严格 <= t，无未来函数）
        leg_low = float(df["Low"].iloc[h_anchor_idx:t + 1].min()) if h_anchor_idx <= t else c_curr_low
        trough = min(c_curr_low, c_close, l_break, leg_low)
        impulse_leg = max(h_anchor - trough, bar_rng)

        fib50 = trough + 0.50 * impulse_leg
        fib618 = trough + 0.618 * impulse_leg
        structural_stop = round(h_anchor + DEFAULT_STOP_BUFFER_K * bar_rng, 2)

        return {
            "side": -1,
            "l_break": round(l_break, 2),
            "h_anchor": round(h_anchor, 2),
            "trough": round(trough, 2),
            "fib50": round(fib50, 2),
            "fib618": round(fib618, 2),
            "structural_stop": structural_stop,
            "bar_rng": round(bar_rng, 2),
            "swing_retest": round(l_break, 2),
        }
    return {}


def verify_market_dow_gate(market_dow: dict, side: int = 1) -> dict:
    """
    大盘指数相互印证硬闸门 (Market Confirmation Gate - Dow Theory Rule 4)

    在生成/执行多头计划 (side == 1) 时，必须联动验证大盘指数（SPY 与 QQQ）：
      - 若大盘指数处于道氏空头结构（LH+LL）或当日大盘处于破位空翻状态，
        个股多头交易必须被风控硬闸门拦截驳回（rejected_by_market_dow）。
    """
    if not market_dow:
        return {"passed": True, "status": "no_market_data", "reason": "未提供大盘数据，闸门默认放行"}

    indices_checked = []
    bearish_reasons = []

    for sym in ["SPY", "QQQ"]:
        if sym not in market_dow and f"{sym}_phase" not in market_dow and f"{sym}_bias" not in market_dow:
            continue

        indices_checked.append(sym)
        is_bear = False
        reason = ""

        val = market_dow.get(sym)
        if isinstance(val, dict):
            # 格式 1: 包含 mtf / htf
            mtf_info = val.get("mtf", {})
            htf_info = val.get("htf", {})
            mtf_phase = str(mtf_info.get("phase", ""))
            htf_phase = str(htf_info.get("phase", ""))

            # 检查空头结构 (LH+LL 或 包含 '空头' / '偏空')
            is_mtf_bear = ("空头" in mtf_phase or "偏空" in mtf_phase or "LH+LL" in mtf_phase)
            is_htf_bear = ("空头" in htf_phase or "LH+LL" in htf_phase)

            # 检查是否有显式破位标志
            is_breakdown = bool(val.get("is_breakdown") or mtf_info.get("is_breakdown"))

            # 检查 swings 中是否为 LH+LL
            swings_mtf = mtf_info.get("swings", [])
            has_lh_ll = any("LH" in str(s) or "LL" in str(s) for s in swings_mtf[-2:]) if swings_mtf else False

            if is_breakdown:
                is_bear = True
                reason = f"{sym} 当日收盘处于破位空翻状态"
            elif is_mtf_bear or (has_lh_ll and is_htf_bear):
                is_bear = True
                reason = f"{sym} 处于道氏空头结构 ({mtf_phase or 'LH+LL'})"
        elif isinstance(val, pd.DataFrame):
            # 格式 2: DataFrame
            if not val.empty and len(val) >= 10:
                sw = pe.extract_swings(val, order=5)
                bias = pe.dow_bias(sw, df=val)
                if bias <= -1:
                    is_bear = True
                    reason = f"{sym} 结构偏向处于空头破位 (dow_bias={bias})"
        elif isinstance(val, (int, float)):
            if val <= -1:
                is_bear = True
                reason = f"{sym} 道氏偏向为负 ({val})"
        elif isinstance(val, str):
            if "空头" in val or "bear" in val.lower() or "breakdown" in val.lower():
                is_bear = True
                reason = f"{sym} 状态标记为空头 ({val})"

        # 检查扁平键
        if not is_bear:
            phase_str = str(market_dow.get(f"{sym}_phase", ""))
            bias_val = market_dow.get(f"{sym}_bias")
            is_bd = market_dow.get(f"{sym}_breakdown", False)
            if is_bd:
                is_bear = True
                reason = f"{sym} 处于破位空翻状态"
            elif "空头" in phase_str or "bear" in phase_str.lower():
                is_bear = True
                reason = f"{sym} 处于道氏空头结构 ({phase_str})"
            elif bias_val is not None and bias_val <= -1:
                is_bear = True
                reason = f"{sym} 偏向为负 ({bias_val})"

        if is_bear:
            bearish_reasons.append(reason)

    if not indices_checked:
        return {
            "passed": True,
            "status": "no_market_data",
            "reason": "未提供有效 SPY/QQQ 大盘数据，闸门默认放行",
            "indices_checked": []
        }

    if side == 1:
        if bearish_reasons:
            return {
                "passed": False,
                "status": "rejected_by_market_dow",
                "reason": "；".join(bearish_reasons),
                "indices_checked": indices_checked
            }
        return {
            "passed": True,
            "status": "confirmed_by_market",
            "reason": f"大盘指数 ({', '.join(indices_checked)}) 未处于空头破位结构，印证通过",
            "indices_checked": indices_checked
        }
    else:
        return {
            "passed": True,
            "status": "confirmed_by_market",
            "reason": "空头计划放行",
            "indices_checked": indices_checked
        }


# ==============================================================================
# 2. 限价回踩撮合模拟 (Pullback Limit Order Simulation)
# ==============================================================================

def simulate_limit_order(df: pd.DataFrame, t: int, side: int,
                         limit_price: float, stop_price: float,
                         max_wait_bars: int = DEFAULT_MAX_WAIT_BARS,
                         volume_filter: bool = False) -> dict:
    """
    模拟在第 t 根决策后挂限价回踩单（Limit Order）：
      - 挂单从 t+1 根生效，有效期限 max_wait_bars 根；
      - 多头限价单：当 Low[k] <= limit_price 时成交；若 Open[k] <= limit_price，按 Open[k] 更有利价成交；
      - 空头限价单：当 High[k] >= limit_price 时成交；若 Open[k] >= limit_price，按 Open[k] 更有利价成交；
      - 证伪撤单：若价格在成交前或同根K线击穿 stop_price，限价单被取消（失效撤单），避免接入单边崩盘；
      - 次级折返缩量洗盘过滤 (volume_filter=True)：
        检查突破后回踩至挂单成交期间的成交量：回踩期间必须呈现缩量态势（平均成交量小于突破日放量，
        且未出现异常巨量抛售或连续放量重挫），若回踩过程伴随异常巨量抛售，判定为派发，挂单自动撤销失效。
    """
    n = len(df)
    if t >= n - 1 or limit_price <= 0 or stop_price <= 0:
        return {"status": "invalid_input", "filled": False}
    if (side == 1 and limit_price <= stop_price) or (side == -1 and limit_price >= stop_price):
        return {"status": "invalid_levels", "filled": False}

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    end_k = min(t + max_wait_bars, n - 1)

    for k in range(t + 1, end_k + 1):
        o, h, l = opens[k], highs[k], lows[k]

        if side == 1:
            # 多头：结构止损在下方
            stop_hit = (l <= stop_price)
            limit_hit = (l <= limit_price)

            if stop_hit and limit_hit:
                # 同一根触及止损和限价：保守判定为结构破坏，放弃入场
                return {"status": "cancelled_invalidated", "filled": False, "cancel_t": k}
            elif stop_hit:
                # 击穿止损结构位，撤单
                return {"status": "cancelled_invalidated", "filled": False, "cancel_t": k}
            elif limit_hit:
                # 次级折返缩量洗盘过滤
                if volume_filter and "Volume" in df.columns and t < n:
                    vol_break = float(df["Volume"].iloc[t])
                    if vol_break > 0:
                        pb_vols = df["Volume"].iloc[t + 1 : k + 1].to_numpy(dtype=float)
                        pb_closes = closes[t + 1 : k + 1]
                        pb_opens = opens[t + 1 : k + 1]

                        # 1. 回踩期间平均成交量大于等于突破日放量（无缩量特征）
                        avg_pb_vol = float(np.mean(pb_vols))
                        is_vol_expanding = (avg_pb_vol >= vol_break)

                        # 2. 单日异常巨量抛售（大阴线且成交量 >= 1.5 倍突破日量能）
                        single_spike = False
                        for i in range(len(pb_vols)):
                            if pb_closes[i] < pb_opens[i] and pb_vols[i] >= 1.5 * vol_break:
                                single_spike = True
                                break

                        # 3. 连续放量重挫（连续两根放量阴线）
                        consec_heavy = False
                        if len(pb_vols) >= 2:
                            for i in range(len(pb_vols) - 1):
                                if (pb_closes[i] < pb_opens[i] and pb_closes[i + 1] < pb_opens[i + 1] and
                                    pb_vols[i] >= 0.9 * vol_break and pb_vols[i + 1] >= 0.9 * vol_break):
                                    consec_heavy = True
                                    break

                        if is_vol_expanding or single_spike or consec_heavy:
                            return {
                                "status": "cancelled_volume_distribution",
                                "filled": False,
                                "cancel_t": k,
                                "reason": "回踩过程伴随放量抛售/派发，缩量洗盘假设破坏，挂单自动撤销"
                            }

                # 成功回踩限价位
                fill_price = min(o, limit_price)
                return {
                    "status": "filled",
                    "filled": True,
                    "entry_t": k,
                    "entry_price": round(float(fill_price), 2),
                    "wait_bars": k - t,
                    "stop_price": round(float(stop_price), 2)
                }
        else:
            # 空头：结构止损在上方
            stop_hit = (h >= stop_price)
            limit_hit = (h >= limit_price)

            if stop_hit and limit_hit:
                return {"status": "cancelled_invalidated", "filled": False, "cancel_t": k}
            elif stop_hit:
                return {"status": "cancelled_invalidated", "filled": False, "cancel_t": k}
            elif limit_hit:
                if volume_filter and "Volume" in df.columns and t < n:
                    vol_break = float(df["Volume"].iloc[t])
                    if vol_break > 0:
                        pb_vols = df["Volume"].iloc[t + 1 : k + 1].to_numpy(dtype=float)
                        pb_closes = closes[t + 1 : k + 1]
                        pb_opens = opens[t + 1 : k + 1]

                        avg_pb_vol = float(np.mean(pb_vols))
                        is_vol_expanding = (avg_pb_vol >= vol_break)

                        single_spike = False
                        for i in range(len(pb_vols)):
                            if pb_closes[i] > pb_opens[i] and pb_vols[i] >= 1.5 * vol_break:
                                single_spike = True
                                break

                        consec_heavy = False
                        if len(pb_vols) >= 2:
                            for i in range(len(pb_vols) - 1):
                                if (pb_closes[i] > pb_opens[i] and pb_closes[i + 1] > pb_opens[i + 1] and
                                    pb_vols[i] >= 0.9 * vol_break and pb_vols[i + 1] >= 0.9 * vol_break):
                                    consec_heavy = True
                                    break

                        if is_vol_expanding or single_spike or consec_heavy:
                            return {
                                "status": "cancelled_volume_distribution",
                                "filled": False,
                                "cancel_t": k,
                                "reason": "回抽过程伴随放量反扑，挂单自动撤销"
                            }

                fill_price = max(o, limit_price)
                return {
                    "status": "filled",
                    "filled": True,
                    "entry_t": k,
                    "entry_price": round(float(fill_price), 2),
                    "wait_bars": k - t,
                    "stop_price": round(float(stop_price), 2)
                }

    return {"status": "unfilled_expired", "filled": False, "wait_bars": end_k - t}


# ==============================================================================
# 3. 触发层形态检测 (Trigger Layer: 2B / Pin Bar / Engulfing)
# ==============================================================================

def detect_trigger_on_bar(df: pd.DataFrame, k: int, side: int,
                          key_level: float, bar_rng: float,
                          tol_k: float = 0.5) -> dict:
    """
    检测单根K线（k 根）在关键位 key_level 处是否出现同向裸K确认形态：
      1. Pin Bar（长影线拒绝）
      2. 2B 假突破 / 流动性猎杀（Liquidity Sweep）
      3. 吞没形态（Engulfing）
    返回检测结果及紧密证伪止损位。
    """
    if k < 1 or k >= len(df) or bar_rng <= 0:
        return {"triggered": False}

    curr = df.iloc[k]
    prev = df.iloc[k - 1]

    c_o, c_h, c_l, c_c = float(curr["Open"]), float(curr["High"]), float(curr["Low"]), float(curr["Close"])
    p_o, p_c = float(prev["Open"]), float(prev["Close"])

    rng = c_h - c_l
    if rng <= 0:
        return {"triggered": False}

    body = abs(c_c - c_o)
    tol = tol_k * bar_rng

    if side == 1:
        # 多头形态要求：下影线探底拒绝、假跌破收回、或阳包阴
        lower_shadow = min(c_o, c_c) - c_l
        # 1. 看涨 Pin Bar (锤子线)：下影线拒绝，且K线覆盖关键位容忍区 [key_level - tol, key_level + tol]
        is_pinbar = (lower_shadow >= 2.0 * body and (lower_shadow / rng) >= 0.55
                     and c_l <= key_level + tol and c_h >= key_level - tol)

        # 2. 2B 看涨假跌破 (Liquidity Sweep Low)
        is_2b = (c_l < key_level and c_c > key_level and c_c > c_o)

        # 3. 看涨吞没 (Bullish Engulfing)：实体吞没，且在关键位附近形成
        is_engulfing = (p_c < p_o and c_c > c_o and c_c >= p_o and c_o <= p_c
                        and c_l <= key_level + tol and c_h >= key_level - tol)

        if is_pinbar or is_2b or is_engulfing:
            pattern = ("2B_sweep" if is_2b else ("pin_bar" if is_pinbar else "engulfing"))
            # 紧密证伪止损：形态最低点下移 buffer
            tight_stop = round(c_l - DEFAULT_STOP_BUFFER_K * bar_rng, 2)
            return {
                "triggered": True,
                "pattern": pattern,
                "trigger_t": k,
                "trigger_bar_low": round(c_l, 2),
                "trigger_bar_high": round(c_h, 2),
                "tight_stop": tight_stop,
                "key_level": round(key_level, 2)
            }
    else:
        # 空头形态要求：上影线摸高拒绝、假突破收回、或阴包阳
        upper_shadow = c_h - max(c_o, c_c)
        # 1. 看跌 Pin Bar (流星线)：上影线拒绝，且K线覆盖关键位容忍区 [key_level - tol, key_level + tol]
        is_pinbar = (upper_shadow >= 2.0 * body and (upper_shadow / rng) >= 0.55
                     and c_h >= key_level - tol and c_l <= key_level + tol)

        # 2. 2B 看跌假突破 (Liquidity Sweep High)
        is_2b = (c_h > key_level and c_c < key_level and c_c < c_o)

        # 3. 看跌吞没 (Bearish Engulfing)：实体吞没，且在关键位附近形成
        is_engulfing = (p_c > p_o and c_c < c_o and c_c <= p_o and c_o >= p_c
                        and c_h >= key_level - tol and c_l <= key_level + tol)

        if is_pinbar or is_2b or is_engulfing:
            pattern = ("2B_sweep" if is_2b else ("pin_bar" if is_pinbar else "engulfing"))
            # 紧密证伪止损：形态最高点上移 buffer
            tight_stop = round(c_h + DEFAULT_STOP_BUFFER_K * bar_rng, 2)
            return {
                "triggered": True,
                "pattern": pattern,
                "trigger_t": k,
                "trigger_bar_low": round(c_l, 2),
                "trigger_bar_high": round(c_h, 2),
                "tight_stop": tight_stop,
                "key_level": round(key_level, 2)
            }

    return {"triggered": False}


def simulate_trigger_entry(df: pd.DataFrame, t: int, side: int,
                           key_level: float, bar_rng: float,
                           structural_stop: float,
                           max_wait_bars: int = DEFAULT_MAX_WAIT_BARS) -> dict:
    """
    在突破确认后（t 根），在关键位附近等待裸K反转确认信号：
      - 从 t+1 根开始扫描至 t+max_wait_bars 根；
      - 若某根 k 出现触发信号，在 k+1 根开盘（Open[k+1]）正式入场（绝对无未来函数）；
      - 止损锚定为触发K线极值外的紧密止损（Tight Stop），大幅压缩单位风险 R。
    """
    n = len(df)
    if t >= n - 2 or key_level <= 0 or bar_rng <= 0:
        return {"status": "invalid_input", "triggered": False}

    end_k = min(t + max_wait_bars, n - 2)

    for k in range(t + 1, end_k + 1):
        # 结构证伪检查：如果价格跌破了整体大结构止损，停止监控
        l_k, h_k = float(df["Low"].iloc[k]), float(df["High"].iloc[k])
        if side == 1 and l_k <= structural_stop:
            return {"status": "invalidated_before_trigger", "triggered": False, "t_stop": k}
        if side == -1 and h_k >= structural_stop:
            return {"status": "invalidated_before_trigger", "triggered": False, "t_stop": k}

        trig = detect_trigger_on_bar(df, k, side, key_level, bar_rng)
        if trig.get("triggered"):
            # k 根收盘确认形态，k+1 根开盘执行入场
            entry_t = k + 1
            entry_price = float(df["Open"].iloc[entry_t])
            tight_stop = float(trig["tight_stop"])

            # 极端开盘跳空过滤：如果开盘直接跳空穿越止损，放弃入场
            if side == 1 and entry_price <= tight_stop:
                continue
            if side == -1 and entry_price >= tight_stop:
                continue

            return {
                "status": "triggered",
                "triggered": True,
                "entry_t": entry_t,
                "entry_price": round(entry_price, 2),
                "stop_price": round(tight_stop, 2),
                "pattern": trig["pattern"],
                "trigger_bar_t": k,
                "wait_bars": k - t
            }

    return {"status": "no_trigger_expired", "triggered": False, "wait_bars": end_k - t}


# ==============================================================================
# 4. 真实持仓生命周期与风控闭环 (Risk-Reward & Execution Lifecycle)
# ==============================================================================

def simulate_strategy_trade(df: pd.DataFrame, entry_t: int, entry_price: float,
                            side: int, stop_price: float, tp_price: float,
                            max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
                            min_rr: float = DEFAULT_MIN_RR,
                            market_dow: dict = None,
                            wave_status: str = None) -> dict:
    """
    全闭环策略持仓模拟：
      - 大盘硬闸门：若提供 market_dow，联动验证 SPY/QQQ，若处于空头结构则驳回（rejected_by_market_dow）；
      - 波浪冲顶拦截：若 wave_status=='rejected_by_wave5_exhaustion'，严格禁止追多；
      - 盈亏比硬闸门：实际规划盈亏比 RR = |tp - entry| / |entry - stop| 必须 >= min_rr，否则放弃入场；
      - 保守成交：当同一根 K 线的 [Low, High] 同时覆盖止损价与止盈价时，严格判定为「先击穿止损」；
      - 逐根推演：实时统计 MAE（最大不利偏移）、MFE（最大有利偏移）、持有根数、实现 R 倍数与收益率。
    """
    n = len(df)
    if entry_t >= n or entry_price <= 0 or stop_price <= 0 or tp_price <= 0:
        return {"status": "invalid_trade"}

    # 1. 大盘道氏印证硬闸门
    if market_dow is not None:
        m_gate = verify_market_dow_gate(market_dow, side=side)
        if not m_gate.get("passed", True):
            return {
                "status": "rejected_by_market_dow",
                "reason": m_gate.get("reason", "rejected_by_market_dow"),
                "entry_t": entry_t,
                "entry_price": entry_price,
                "side": side
            }

    # 2. 波浪第 5 浪冲顶诱多拦截
    if wave_status == "rejected_by_wave5_exhaustion" and side == 1:
        return {
            "status": "rejected_by_wave5_exhaustion",
            "reason": "波浪第5浪冲顶诱多/量能衰竭，风控拦截做多",
            "entry_t": entry_t,
            "entry_price": entry_price,
            "side": side
        }

    risk = abs(entry_price - stop_price)
    reward = abs(tp_price - entry_price)

    if risk <= 0:
        return {"status": "zero_risk_invalid"}

    rr = reward / risk
    # 浮点数精度容差 1e-6，防止例如 9.86 / 4.93 = 1.9999999999999998 被误判驳回
    if rr < min_rr - 1e-6:
        return {
            "status": "rejected_by_min_rr",
            "rr": round(rr, 2),
            "min_rr": min_rr,
            "risk": round(risk, 2),
            "reward": round(reward, 2)
        }

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    end_t = min(entry_t + max_hold_bars, n - 1)
    exit_t = None
    exit_price = None
    exit_reason = None

    # 从入场根 entry_t 开始推演（开盘入场后，当天盘中即可触及止损或止盈）
    for bar in range(entry_t, end_t + 1):
        o, h, l, c = opens[bar], highs[bar], lows[bar], closes[bar]

        if side == 1:
            hit_stop = (l <= stop_price)
            hit_tp = (h >= tp_price)

            if hit_stop and hit_tp:
                # 极端同根K线双触：按理论保守约束5，一律按止损出场
                exit_t = bar
                exit_price = min(o, stop_price)
                exit_reason = "stop_loss_conservative"
                break
            elif hit_stop:
                exit_t = bar
                exit_price = min(o, stop_price)
                exit_reason = "stop_loss"
                break
            elif hit_tp:
                exit_t = bar
                exit_price = max(o, tp_price)
                exit_reason = "take_profit"
                break
        else:
            hit_stop = (h >= stop_price)
            hit_tp = (l <= tp_price)

            if hit_stop and hit_tp:
                exit_t = bar
                exit_price = max(o, stop_price)
                exit_reason = "stop_loss_conservative"
                break
            elif hit_stop:
                exit_t = bar
                exit_price = max(o, stop_price)
                exit_reason = "stop_loss"
                break
            elif hit_tp:
                exit_t = bar
                exit_price = min(o, tp_price)
                exit_reason = "take_profit"
                break

    if exit_t is None:
        exit_t = end_t
        exit_price = closes[end_t]
        exit_reason = "time_horizon"

    pnl_pct = (exit_price / entry_price - 1.0) * side
    r_multiple = ((exit_price - entry_price) * side) / risk
    hold_bars = exit_t - entry_t

    # 持仓区间 MAE / MFE
    if side == 1:
        sub_low = float(np.min(lows[entry_t:exit_t + 1]))
        sub_high = float(np.max(highs[entry_t:exit_t + 1]))
        mae = min(sub_low / entry_price - 1.0, 0.0)
        mfe = max(sub_high / entry_price - 1.0, 0.0)
    else:
        sub_low = float(np.min(lows[entry_t:exit_t + 1]))
        sub_high = float(np.max(highs[entry_t:exit_t + 1]))
        mae = min(-(sub_high / entry_price - 1.0), 0.0)
        mfe = max(-(sub_low / entry_price - 1.0), 0.0)

    return {
        "status": "completed",
        "entry_t": entry_t,
        "entry_price": round(entry_price, 2),
        "exit_t": exit_t,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl_pct": float(pnl_pct),
        "r_multiple": float(r_multiple),
        "hold_bars": int(hold_bars),
        "mae": float(mae),
        "mfe": float(mfe),
        "win": bool(pnl_pct > 0),
        "risk": float(risk),
        "reward": float(reward),
        "planned_rr": float(rr)
    }


# ==============================================================================
# 5. 任意入场点的前向多周期度量 (Fixed Horizons Evaluation)
# ==============================================================================

def calc_entry_horizon_stats(df: pd.DataFrame, entry_t: int, entry_price: float,
                             side: int, h: int) -> dict:
    """
    计算从 entry_t 根以 entry_price 入场后，持有 h 根的收益 / MAE / MFE。
    """
    n = len(df)
    if entry_t < 0 or entry_t + h >= n or entry_price <= 0:
        return None

    close_h = float(df["Close"].iloc[entry_t + h])
    sub_low = float(df["Low"].iloc[entry_t:entry_t + h + 1].min())
    sub_high = float(df["High"].iloc[entry_t:entry_t + h + 1].max())

    ret = (close_h / entry_price - 1.0) * side
    if side == 1:
        mae = min(sub_low / entry_price - 1.0, 0.0)
        mfe = max(sub_high / entry_price - 1.0, 0.0)
    else:
        mae = min(-(sub_high / entry_price - 1.0), 0.0)
        mfe = max(-(sub_low / entry_price - 1.0), 0.0)

    return {
        "ret": float(ret),
        "mae": float(mae),
        "mfe": float(mfe)
    }
