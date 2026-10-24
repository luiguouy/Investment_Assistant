#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股纯技术流（道氏理论 + 艾略特波浪理论 + 裸K价格行为）核心数据与渲染引擎 v2.0
严禁使用任何滞后性技术指标，100%基于价格结构、拐点序列、斐波那契与裸K形态。

v2.0 变更摘要（10项优化）：
  P0-1: HTF 大势拐点提取与输出（三级联立完整化）
  P0-2: 成交量结构分析 + 量价背离检测
  P0-3: 多腿斐波那契比例 + 共振区 (Fibonacci Cluster) 检测
  P1-4: 日内 Opening Range (ORH/ORL, 15分钟)
  P1-5: 市场结构转换 (MSS) 精确检测
  P1-6: Swing 拐点价格纳入 2B 流动性猎杀关键位
  P2-7: 批量 Watchlist 自选池扫描模式
  P2-8: LTF 入场级拐点序列输出
  P2-9: 文件同步（由外部脚本操作）
  P2-10: 大盘指数 HTF + MTF 双级验证
"""

import sys
import argparse
from datetime import time
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import yfinance as yf


# ==============================================================================
# 1. 周期配置与数据拉取模块
# ==============================================================================

TIMEFRAME_CONFIGS = {
    "swing": {
        "HTF": {"interval": "1wk", "period": "2y",  "order": 3, "prepost": False, "desc": "大势宏观周期"},
        "MTF": {"interval": "1d",  "period": "1y",  "order": 5, "prepost": False, "desc": "波浪结构周期"},
        "LTF": {"interval": "1h",  "period": "1mo", "order": 7, "prepost": False, "desc": "裸K入场周期"},
    },
    "day": {
        "HTF": {"interval": "1h",  "period": "1mo", "order": 6, "prepost": True,  "desc": "大势宏观周期"},
        "MTF": {"interval": "15m", "period": "5d",  "order": 5, "prepost": True,  "desc": "波浪结构周期"},
        "LTF": {"interval": "5m",  "period": "2d",  "order": 4, "prepost": True,  "desc": "裸K入场周期"},
    }
}


def fetch_klines(ticker: str, interval: str, period: str, prepost: bool = False) -> pd.DataFrame:
    """拉取指定标的与周期的K线数据，并清洗规整"""
    try:
        t = yf.Ticker(ticker)
        df = t.history(interval=interval, period=period, prepost=prepost, auto_adjust=False)
        if df.empty:
            # 备选拉取，兼容不同 yfinance 版本
            df = t.history(interval=interval, period=period, prepost=prepost)
        if df.empty:
            return pd.DataFrame()

        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[cols].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[警告] 拉取 {ticker} ({interval}, {period}) 失败: {e}", file=sys.stderr)
        return pd.DataFrame()


# ==============================================================================
# 2. 客观波段拐点（Swing High/Low）与道氏标签打标
# ==============================================================================

def extract_swings(df: pd.DataFrame, order: int = 5) -> list:
    """
    使用 scipy.signal.argrelextrema 提取客观波段拐点
    并进行同向极值归并，保证拐点严格高低交替
    """
    if len(df) < order * 2 + 3:
        return []

    high_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    low_idx = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]

    raw_swings = []
    for idx in high_idx:
        raw_swings.append({
            "index": idx,
            "time": df.index[idx],
            "type": "High",
            "price": float(df['High'].iloc[idx])
        })
    for idx in low_idx:
        raw_swings.append({
            "index": idx,
            "time": df.index[idx],
            "type": "Low",
            "price": float(df['Low'].iloc[idx])
        })

    # 按时间严格升序
    raw_swings.sort(key=lambda x: x["time"])

    # 归并连续相同类型的拐点（高点取更高，低点取更低）
    clean_swings = []
    for pt in raw_swings:
        if not clean_swings:
            clean_swings.append(pt)
            continue
        last = clean_swings[-1]
        if last["type"] == pt["type"]:
            if pt["type"] == "High" and pt["price"] > last["price"]:
                clean_swings[-1] = pt
            elif pt["type"] == "Low" and pt["price"] < last["price"]:
                clean_swings[-1] = pt
        else:
            clean_swings.append(pt)

    # 打上道氏结构标签 (HH, HL, LH, LL)
    last_high = None
    last_low = None
    for pt in clean_swings:
        if pt["type"] == "High":
            if last_high is None:
                pt["dow"] = "High"
            else:
                pt["dow"] = "HH" if pt["price"] >= last_high else "LH"
            last_high = pt["price"]
        else:
            if last_low is None:
                pt["dow"] = "Low"
            else:
                pt["dow"] = "HL" if pt["price"] >= last_low else "LL"
            last_low = pt["price"]

    return clean_swings


def _determine_dow_phase(swings: list) -> str:
    """根据最近拐点标签判定道氏阶段"""
    if len(swings) < 2:
        return "数据不足"
    recent = swings[-4:] if len(swings) >= 4 else swings
    labels = [s.get("dow", "") for s in recent]
    bull = labels.count("HH") + labels.count("HL")
    bear = labels.count("LH") + labels.count("LL")

    if bull >= 3:
        return "📈 多头趋势 (HH+HL 主导)"
    elif bear >= 3:
        return "📉 空头趋势 (LH+LL 主导)"
    elif bull >= 2:
        return "📈 偏多"
    elif bear >= 2:
        return "📉 偏空"
    return "⏸️ 震荡/无方向"


# ==============================================================================
# 2b. [P1-5] 市场结构转换检测 (MSS - Market Structure Shift)
# ==============================================================================

def detect_mss(swings: list) -> list:
    """
    遍历道氏标签序列，检测趋势从多翻空（HH/HL → LH+LL）或空翻多（LH/LL → HL+HH）的精确转折点。
    """
    events = []
    if len(swings) < 4:
        return events

    for i in range(3, len(swings)):
        curr = swings[i]
        prev = swings[i - 1]

        # 多翻空：先出现 LH，再出现 LL → 多头结构被破坏
        if prev.get("dow") == "LH" and curr.get("dow") == "LL":
            prior_bullish = any(
                swings[j].get("dow") in ("HH", "HL")
                for j in range(max(0, i - 5), i - 1)
            )
            if prior_bullish:
                events.append({
                    "type": "🔻 空翻 (Bearish MSS)",
                    "trigger_time": str(curr["time"])[:16],
                    "trigger_price": curr["price"],
                    "detail": f"LH ${prev['price']:.2f} + LL ${curr['price']:.2f} → 多头结构被破坏"
                })

        # 空翻多：先出现 HL，再出现 HH → 空头结构被破坏
        elif prev.get("dow") == "HL" and curr.get("dow") == "HH":
            prior_bearish = any(
                swings[j].get("dow") in ("LH", "LL")
                for j in range(max(0, i - 5), i - 1)
            )
            if prior_bearish:
                events.append({
                    "type": "🔺 多翻 (Bullish MSS)",
                    "trigger_time": str(curr["time"])[:16],
                    "trigger_price": curr["price"],
                    "detail": f"HL ${prev['price']:.2f} + HH ${curr['price']:.2f} → 空头结构被破坏"
                })

    return events


# ==============================================================================
# 3. [P2-10] 大盘道氏理论相互验证 — HTF + MTF 双级
# ==============================================================================

def check_dow_indices(mode: str) -> dict:
    """拉取 SPY, QQQ, DIA 在 HTF 与 MTF 下的最近 4 个波段拐点"""
    indices = ["SPY", "QQQ", "DIA"]
    results = {}
    htf_cfg = TIMEFRAME_CONFIGS[mode]["HTF"]
    mtf_cfg = TIMEFRAME_CONFIGS[mode]["MTF"]

    for idx_sym in indices:
        entry = {"htf": {"latest_price": 0.0, "swings": [], "phase": ""},
                 "mtf": {"swings": [], "phase": ""}}

        # HTF 级别
        df_htf = fetch_klines(idx_sym, htf_cfg["interval"], htf_cfg["period"], htf_cfg["prepost"])
        if not df_htf.empty:
            sw_htf = extract_swings(df_htf, order=htf_cfg["order"])
            recent_htf = sw_htf[-4:] if len(sw_htf) >= 4 else sw_htf
            entry["htf"]["latest_price"] = float(df_htf['Close'].iloc[-1])
            entry["htf"]["swings"] = [f"{s['dow']} (${s['price']:.2f})" for s in recent_htf]
            entry["htf"]["phase"] = _determine_dow_phase(sw_htf)

        # MTF 级别
        df_mtf = fetch_klines(idx_sym, mtf_cfg["interval"], mtf_cfg["period"], mtf_cfg["prepost"])
        if not df_mtf.empty:
            sw_mtf = extract_swings(df_mtf, order=mtf_cfg["order"])
            recent_mtf = sw_mtf[-4:] if len(sw_mtf) >= 4 else sw_mtf
            entry["mtf"]["swings"] = [f"{s['dow']} (${s['price']:.2f})" for s in recent_mtf]
            entry["mtf"]["phase"] = _determine_dow_phase(sw_mtf)

        results[idx_sym] = entry
    return results


# ==============================================================================
# 4. 关键水平位与斐波那契计算
# ==============================================================================

def calc_fibonacci_and_levels(swings: list, df_daily: pd.DataFrame = None,
                              df_intraday: pd.DataFrame = None) -> dict:
    """
    基于最近一笔主波段计算斐波那契回撤与顺势扩展，
    以及日内关键位 (PDH, PDL, PDC, PMH, PML, ORH, ORL)
    """
    levels = {}
    if len(swings) >= 2:
        p1 = swings[-2]
        p2 = swings[-1]
        start_price = p1["price"]
        end_price = p2["price"]
        diff = abs(end_price - start_price)

        is_up = end_price > start_price
        levels["trend_direction"] = "UP" if is_up else "DOWN"
        levels["base_swing"] = {
            "from": f"{p1['type']} (${start_price:.2f})",
            "to": f"{p2['type']} (${end_price:.2f})",
            "diff": round(diff, 2)
        }

        # 斐波那契回撤与扩展
        fib_ret = {}
        fib_ext = {}
        for r in [0.236, 0.382, 0.500, 0.618, 0.786]:
            if is_up:
                fib_ret[str(r)] = round(end_price - r * diff, 2)
            else:
                fib_ret[str(r)] = round(end_price + r * diff, 2)
        for e in [1.272, 1.618, 2.618]:
            if is_up:
                fib_ext[str(e)] = round(start_price + e * diff, 2)
            else:
                fib_ext[str(e)] = round(start_price - e * diff, 2)

        levels["fib_retracements"] = fib_ret
        levels["fib_extensions"] = fib_ext

    # 日内模式额外计算昨高低收与盘前高低
    if df_daily is not None and len(df_daily) >= 2:
        prev_day = df_daily.iloc[-2]
        levels["PDH"] = round(float(prev_day['High']), 2)
        levels["PDL"] = round(float(prev_day['Low']), 2)
        levels["PDC"] = round(float(prev_day['Close']), 2)

    if df_intraday is not None and not df_intraday.empty:
        try:
            df_recent = df_intraday.tail(150).copy()
            times = df_recent.index.time

            # 盘前 (04:00 - 09:30 ET)
            pre_market_mask = [(t >= time(4, 0)) and (t < time(9, 30)) for t in times]
            df_pm = df_recent[pre_market_mask]
            if not df_pm.empty:
                levels["PMH"] = round(float(df_pm['High'].max()), 2)
                levels["PML"] = round(float(df_pm['Low'].min()), 2)

            # [P1-4] 开盘区间 Opening Range (09:30 - 09:45 ET, 15分钟)
            or_mask = [(t >= time(9, 30)) and (t < time(9, 45)) for t in times]
            df_or = df_recent[or_mask]
            if not df_or.empty:
                levels["ORH"] = round(float(df_or['High'].max()), 2)
                levels["ORL"] = round(float(df_or['Low'].min()), 2)
        except Exception:
            pass

    return levels


# ==============================================================================
# 4b. [P0-3] 多腿斐波那契比例与共振区检测
# ==============================================================================

def _match_fib_ratio(ratio: float) -> str:
    """将波段比例匹配到最近的斐波那契数"""
    targets = {
        0.382: "0.382", 0.500: "0.500", 0.618: "0.618", 0.786: "0.786",
        1.000: "1.000", 1.272: "1.272", 1.618: "1.618", 2.000: "2.000", 2.618: "2.618"
    }
    best_label, best_diff = "", float('inf')
    for val, label in targets.items():
        d = abs(ratio - val)
        if d < best_diff:
            best_diff = d
            best_label = label
    if best_diff <= 0.05:
        return f"≈ Fib {best_label} ✅"
    elif best_diff <= 0.10:
        return f"≈ Fib {best_label} (近似)"
    return "无匹配"


def calc_multi_leg_fib(swings: list, tolerance_pct: float = 0.01) -> dict:
    """
    多段波浪比例测量（艾略特验证：Wave3 vs Wave1 等）
    + 多重 Fib 共振区 (Fibonacci Cluster) 检测
    """
    result = {"leg_ratios": [], "clusters": []}
    if len(swings) < 4:
        return result

    # 构建波段腿
    legs = []
    for i in range(1, len(swings)):
        legs.append({
            "from_pt": swings[i - 1],
            "to_pt": swings[i],
            "length": abs(swings[i]["price"] - swings[i - 1]["price"]),
            "direction": "UP" if swings[i]["price"] > swings[i - 1]["price"] else "DOWN"
        })

    # 同向波段比例（隔一腿比较：推动浪之间的比例关系）
    for i in range(2, len(legs)):
        if legs[i]["direction"] == legs[i - 2]["direction"] and legs[i - 2]["length"] > 0:
            ratio = legs[i]["length"] / legs[i - 2]["length"]
            result["leg_ratios"].append({
                "current": f"${legs[i]['from_pt']['price']:.2f} → ${legs[i]['to_pt']['price']:.2f}",
                "reference": f"${legs[i-2]['from_pt']['price']:.2f} → ${legs[i-2]['to_pt']['price']:.2f}",
                "ratio": round(ratio, 3),
                "match": _match_fib_ratio(ratio)
            })

    # 共振区：对最近 3 腿分别计算全套 Fib，聚类价位
    all_fib = []
    for leg in legs[-3:]:
        s, e = leg["from_pt"]["price"], leg["to_pt"]["price"]
        d = abs(e - s)
        for f in [0.382, 0.500, 0.618, 0.786, 1.272, 1.618]:
            p = (e - f * d) if leg["direction"] == "UP" else (e + f * d)
            all_fib.append(round(p, 2))

    all_fib.sort()
    clusters = []
    if all_fib:
        group = [all_fib[0]]
        for p in all_fib[1:]:
            if group and abs(p - group[-1]) / max(abs(group[-1]), 0.01) <= tolerance_pct:
                group.append(p)
            else:
                if len(group) >= 2:
                    clusters.append({
                        "zone": f"${min(group):.2f} - ${max(group):.2f}",
                        "center": round(sum(group) / len(group), 2),
                        "count": len(group),
                        "strength": "🔥 强共振" if len(group) >= 3 else "⚡ 共振"
                    })
                group = [p]
        if len(group) >= 2:
            clusters.append({
                "zone": f"${min(group):.2f} - ${max(group):.2f}",
                "center": round(sum(group) / len(group), 2),
                "count": len(group),
                "strength": "🔥 强共振" if len(group) >= 3 else "⚡ 共振"
            })

    result["clusters"] = clusters
    return result


# ==============================================================================
# 4c. 跳空缺口探测
# ==============================================================================

def detect_gaps(df: pd.DataFrame, min_gap_pct: float = 0.008) -> list:
    """
    探测客观价格结构中的跳空缺口（相邻K线高低点不重叠）
    并自动跟踪检验该缺口在后续K线中是否未被完全回补
    """
    gaps = []
    if len(df) < 5:
        return gaps

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # 向上跳空：当前最低价 > 前一根最高价
        if curr['Low'] > prev['High'] * (1 + min_gap_pct):
            gap_low = prev['High']
            gap_high = curr['Low']
            subsequent = df.iloc[i + 1:]
            filled = not subsequent.empty and subsequent['Low'].min() <= gap_low
            if not filled:
                gaps.append({
                    "date": str(curr.name)[:10],
                    "type": "向上跳空支撑缺口 (Bullish Gap)",
                    "range": f"${gap_low:.2f} - ${gap_high:.2f}",
                    "level": round(gap_low, 2),
                    "status": "未回补 (强引力/支撑防线)"
                })

        # 向下跳空：当前最高价 < 前一根最低价
        elif curr['High'] < prev['Low'] * (1 - min_gap_pct):
            gap_high = prev['Low']
            gap_low = curr['High']
            subsequent = df.iloc[i + 1:]
            filled = not subsequent.empty and subsequent['High'].max() >= gap_high
            if not filled:
                gaps.append({
                    "date": str(curr.name)[:10],
                    "type": "向下跳空阻力缺口 (Bearish Gap)",
                    "range": f"${gap_low:.2f} - ${gap_high:.2f}",
                    "level": round(gap_high, 2),
                    "status": "未回补 (强引力/阻力防线)"
                })
    return gaps


# ==============================================================================
# 5. 裸K价格行为扫描器 (Price Action Scanner)
# ==============================================================================

def scan_price_action(df: pd.DataFrame, key_levels: list = None) -> list:
    """
    扫描最近 5 根 K 线的裸K信号：
    - Pin Bar（长影线拒绝）
    - 吞没形态（Bullish/Bearish Engulfing）
    - 孕线（Inside Bar）
    - 2B假突破 / 流动性猎杀（Liquidity Sweep）
    """
    signals = []
    if len(df) < 5:
        return signals

    recent_df = df.tail(5)
    key_levels = key_levels or []

    for i in range(1, len(recent_df)):
        curr = recent_df.iloc[i]
        prev = recent_df.iloc[i - 1]
        dt_str = str(curr.name)

        c_open, c_high, c_low, c_close = curr['Open'], curr['High'], curr['Low'], curr['Close']
        p_open, p_close = prev['Open'], prev['Close']

        body = abs(c_close - c_open)
        rng = c_high - c_low
        upper_shadow = c_high - max(c_open, c_close)
        lower_shadow = min(c_open, c_close) - c_low

        if rng == 0:
            continue

        # 1. Pin Bar 扫描 (影线 >= 实体2倍且占总振幅 >= 60%)
        if lower_shadow >= 2 * body and (lower_shadow / rng) >= 0.60:
            signals.append({
                "time": dt_str, "pattern": "看涨锤子线 Pin Bar (多头拒绝下跌)",
                "price": round(c_close, 2), "key_point": f"下影探底 ${c_low:.2f}"
            })
        elif upper_shadow >= 2 * body and (upper_shadow / rng) >= 0.60:
            signals.append({
                "time": dt_str, "pattern": "看跌流星线 Pin Bar (空头拒绝上涨)",
                "price": round(c_close, 2), "key_point": f"上影摸高 ${c_high:.2f}"
            })

        # 2. 吞没形态 (Engulfing)
        if p_close < p_open and c_close > c_open:
            if c_open <= p_close and c_close >= p_open:
                signals.append({
                    "time": dt_str, "pattern": "看涨吞没形态 (Bullish Engulfing)",
                    "price": round(c_close, 2),
                    "key_point": f"阳线实体完全覆盖前阴线 (${c_open:.2f} - ${c_close:.2f})"
                })
        elif p_close > p_open and c_close < c_open:
            if c_open >= p_close and c_close <= p_open:
                signals.append({
                    "time": dt_str, "pattern": "看跌吞没形态 (Bearish Engulfing)",
                    "price": round(c_close, 2),
                    "key_point": f"阴线实体完全覆盖前阳线 (${c_close:.2f} - ${c_open:.2f})"
                })

        # 3. 孕线形态 (Inside Bar)
        if c_high <= prev['High'] and c_low >= prev['Low']:
            signals.append({
                "time": dt_str, "pattern": "孕线收敛形态 (Inside Bar)",
                "price": round(c_close, 2), "key_point": "完全被前K包含，酝酿变盘突破"
            })

        # 4. 2B假突破 / 流动性猎杀 (Liquidity Sweep)
        for lvl in key_levels:
            # 向上假突破
            if c_high > lvl and c_close < lvl and prev['High'] <= lvl:
                signals.append({
                    "time": dt_str,
                    "pattern": f"2B高位假突破/流动性猎杀 (Sweep High ${lvl:.2f})",
                    "price": round(c_close, 2), "key_point": "刺破前高后回落，诱多扫损信号"
                })
            # 向下假突破
            elif c_low < lvl and c_close > lvl and prev['Low'] >= lvl:
                signals.append({
                    "time": dt_str,
                    "pattern": f"2B低位假跌破/流动性猎杀 (Sweep Low ${lvl:.2f})",
                    "price": round(c_close, 2), "key_point": "跌破前低后强力收回，诱空扫损信号"
                })

    return signals


# ==============================================================================
# 5b. [P0-2] 成交量结构分析
# ==============================================================================

def analyze_volume(df: pd.DataFrame, lookback: int = 20, recent: int = 5) -> dict:
    """
    分析成交量结构：计算均量基准，对最近 N 根K线标注放量/缩量，
    并检测量价背离（价格新高但量缩 / 价格新低但量缩）。
    """
    result = {"avg_volume": 0, "recent_bars": [], "divergences": []}
    if len(df) < lookback:
        return result

    vol = df['Volume'].values
    avg_vol = float(np.mean(vol[-lookback:]))
    result["avg_volume"] = int(avg_vol)

    recent_df = df.tail(recent)
    for i in range(len(recent_df)):
        row = recent_df.iloc[i]
        bar_vol = int(row['Volume'])
        ratio = bar_vol / avg_vol if avg_vol > 0 else 0

        if ratio >= 1.5:
            tag = "🔴 放量 (>1.5x)"
        elif ratio >= 1.2:
            tag = "🟠 偏强 (>1.2x)"
        elif ratio <= 0.5:
            tag = "⚪ 极缩量 (<0.5x)"
        elif ratio <= 0.7:
            tag = "🔵 缩量 (<0.7x)"
        else:
            tag = "🟢 正常"

        result["recent_bars"].append({
            "time": str(row.name)[:16],
            "volume": bar_vol,
            "ratio": round(ratio, 2),
            "tag": tag,
            "candle": "阳线" if row['Close'] > row['Open'] else "阴线"
        })

    # 量价背离检测
    if len(df) >= 10:
        last_5 = df.tail(5)
        prev_5 = df.iloc[-10:-5]

        # 顶背离：价新高 + 量萎缩
        if last_5['High'].max() > prev_5['High'].max():
            if last_5['Volume'].mean() < prev_5['Volume'].mean() * 0.8:
                result["divergences"].append(
                    "⚠️ 量价顶背离：价格创近期新高，但成交量明显萎缩 → 上涨动能衰竭信号"
                )
        # 底背离：价新低 + 量萎缩
        if last_5['Low'].min() < prev_5['Low'].min():
            if last_5['Volume'].mean() < prev_5['Volume'].mean() * 0.8:
                result["divergences"].append(
                    "⚠️ 量价底背离：价格创近期新低，但卖压成交量减弱 → 下跌动能枯竭信号"
                )

    return result


# ==============================================================================
# 6. 主调度与结构化输出
# ==============================================================================

def analyze_single_ticker(ticker: str, mode: str, market_dow: dict = None,
                          is_watchlist: bool = False):
    """对单一标的执行完整三级周期分析并输出。is_watchlist=True 时输出精简版概览。"""
    config = TIMEFRAME_CONFIGS[mode]
    htf_cfg, mtf_cfg, ltf_cfg = config["HTF"], config["MTF"], config["LTF"]

    # ---- 拉取三级 K 线 ----
    df_htf = fetch_klines(ticker, htf_cfg["interval"], htf_cfg["period"], htf_cfg["prepost"])
    df_mtf = fetch_klines(ticker, mtf_cfg["interval"], mtf_cfg["period"], mtf_cfg["prepost"])
    df_ltf = fetch_klines(ticker, ltf_cfg["interval"], ltf_cfg["period"], ltf_cfg["prepost"])

    if df_mtf.empty:
        print(f"[严重错误] 未能获取 {ticker} 的核心K线数据，分析中止。")
        return

    # ---- 拐点提取 ----
    swings_htf = extract_swings(df_htf, order=htf_cfg["order"]) if not df_htf.empty else []
    swings_mtf = extract_swings(df_mtf, order=mtf_cfg["order"])
    swings_ltf = extract_swings(df_ltf, order=ltf_cfg["order"]) if not df_ltf.empty else []

    recent_htf = swings_htf[-4:] if len(swings_htf) >= 4 else swings_htf
    recent_mtf = swings_mtf[-8:] if len(swings_mtf) >= 8 else swings_mtf
    recent_ltf = swings_ltf[-6:] if len(swings_ltf) >= 6 else swings_ltf

    htf_phase = _determine_dow_phase(swings_htf)
    mtf_phase = _determine_dow_phase(swings_mtf)
    ltf_phase = _determine_dow_phase(swings_ltf)

    # ===================== Watchlist 精简模式 =====================
    if is_watchlist:
        latest_price = float(df_mtf['Close'].iloc[-1])
        # 快速 PA 扫描
        kl = [s["price"] for s in recent_mtf[-4:]]
        pa = scan_price_action(df_ltf, key_levels=kl)
        pa_summary = pa[0]["pattern"][:20] if pa else "无信号"
        # 距最近拐点
        nearest = ""
        if recent_mtf:
            ls = recent_mtf[-1]
            d_pct = (latest_price - ls["price"]) / ls["price"] * 100
            nearest = f"距{ls['dow']} ${ls['price']:.2f} → {d_pct:+.1f}%"
        print(f"  {ticker:<6} ${latest_price:<8.2f} | HTF: {htf_phase} | MTF: {mtf_phase} | PA: {pa_summary} | {nearest}")
        return

    # ===================== 完整分析模式 =====================
    print("\n" + "=" * 70)
    print(f"🚀 【美股纯技术流分析引擎 v2.0】标的: {ticker} | 模式: {mode.upper()}")
    print("=" * 70)

    # ---- [一] 大盘道氏验证 ----
    if market_dow:
        print("\n[一、 道氏大盘指数相互验证 (SPY / QQQ / DIA)]")
        for idx_name, info in market_dow.items():
            htf_sw = " -> ".join(info["htf"].get("swings", [])) or "无显著波段"
            mtf_sw = " -> ".join(info["mtf"].get("swings", [])) or "无显著波段"
            print(f"  • {idx_name:<4} 最新价: ${info['htf']['latest_price']:<7.2f} | HTF: {info['htf'].get('phase', '')}")
            print(f"         HTF 拐点: {htf_sw}")
            print(f"         MTF 拐点: {mtf_sw}")

    # ---- [二] 三级周期拐点 ----
    print(f"\n[二、 标的 {ticker} 三级周期客观波段拐点]")

    # [P0-1] HTF
    print(f"  ▶ HTF ({htf_cfg['interval']} - {htf_cfg['desc']}) 道氏阶段: {htf_phase}")
    if recent_htf:
        for s in recent_htf:
            print(f"      - {str(s['time'])[:16]} | {s['dow']} ({s['type']}): ${s['price']:.2f}")
    else:
        print("      - 拐点数据不足")

    # MTF
    print(f"  ▶ MTF ({mtf_cfg['interval']} - {mtf_cfg['desc']}) 道氏阶段: {mtf_phase}")
    for s in recent_mtf:
        print(f"      - {str(s['time'])[:16]} | {s['dow']} ({s['type']}): ${s['price']:.2f}")

    # [P2-8] LTF
    print(f"  ▶ LTF ({ltf_cfg['interval']} - {ltf_cfg['desc']}) 道氏阶段: {ltf_phase}")
    if recent_ltf:
        for s in recent_ltf:
            print(f"      - {str(s['time'])[:16]} | {s['dow']} ({s['type']}): ${s['price']:.2f}")
    else:
        print("      - 拐点数据不足")

    # ---- [二b] MSS 结构转换 [P1-5] ----
    print(f"\n[二b、 市场结构转换检测 (MSS)]")
    mss_mtf = detect_mss(swings_mtf)
    mss_ltf = detect_mss(swings_ltf)
    has_mss = False
    for e in mss_mtf[-2:]:
        print(f"  {e['type']} @ MTF {e['trigger_time']} | {e['detail']}")
        has_mss = True
    for e in mss_ltf[-2:]:
        print(f"  {e['type']} @ LTF {e['trigger_time']} | {e['detail']}")
        has_mss = True
    if not has_mss:
        print("  ✅ 近期无结构转换事件，当前趋势延续中")

    # ---- [三] 斐波那契 ----
    df_daily = fetch_klines(ticker, "1d", "1mo", False) if mode == "day" else None
    levels = calc_fibonacci_and_levels(
        recent_mtf,
        df_daily=df_daily,
        df_intraday=df_ltf if mode == "day" else None
    )

    print("\n[三、 斐波那契坐标尺与结构关键位]")
    if "base_swing" in levels:
        bs = levels["base_swing"]
        print(f"  • 测量主基准波段: {bs['from']} -> {bs['to']} (幅度: ${bs['diff']})")
        print(f"  • 斐波那契回撤参考:")
        for fib, p in levels.get("fib_retracements", {}).items():
            print(f"      - Fib {fib}: ${p:.2f}")
        print(f"  • 顺势扩展目标位 (Extension):")
        for ext, p in levels.get("fib_extensions", {}).items():
            print(f"      - Ext {ext}: ${p:.2f}")

    # [P0-3] 多腿 Fib + 共振区
    multi_fib = calc_multi_leg_fib(recent_mtf)
    if multi_fib["leg_ratios"]:
        print(f"\n  • 多腿波段比例（艾略特验证）:")
        for lr in multi_fib["leg_ratios"]:
            print(f"      - 当前腿: {lr['current']}")
            print(f"        参照腿: {lr['reference']}")
            print(f"        比例: {lr['ratio']}  {lr['match']}")
    if multi_fib["clusters"]:
        print(f"\n  • 🎯 斐波那契共振区 (Fibonacci Cluster):")
        for cl in multi_fib["clusters"]:
            print(f"      - {cl['strength']} 区间: {cl['zone']} (中枢 ${cl['center']:.2f}, {cl['count']}个Fib汇聚)")

    # ---- [四] 跳空缺口 ----
    print("\n[四、 未回补跳空缺口 (Gap 磁吸与防线)]")
    detected_gaps = detect_gaps(df_mtf)
    if detected_gaps:
        for g in detected_gaps[-4:]:
            print(f"  • [{g['date']}] {g['type']}: {g['range']} -> {g['status']}")
    else:
        print("  • MTF 周期近期无显著未回补缺口")

    # ---- [五] 日内关键位 ----
    if mode == "day":
        print("\n[五、 日内关键结构价位 (PDH/PDL/PMH/PML/ORH/ORL)]")
        for key, label in [("PDH", "昨日最高价"), ("PDL", "昨日最低价"), ("PDC", "昨日收盘价"),
                           ("PMH", "当日盘前最高"), ("PML", "当日盘前最低"),
                           ("ORH", "开盘15分钟高"), ("ORL", "开盘15分钟低")]:
            if key in levels:
                print(f"      - {label} ({key}): ${levels[key]:.2f}")

    # ---- [六] 成交量分析 [P0-2] ----
    print("\n[六、 成交量结构分析 (Volume Analysis)]")
    vol_mtf = analyze_volume(df_mtf)
    vol_ltf = analyze_volume(df_ltf)

    print(f"  ▶ MTF 近20根均量: {vol_mtf['avg_volume']:,}")
    for bar in vol_mtf.get("recent_bars", []):
        print(f"      - [{bar['time']}] {bar['candle']} | 量: {bar['volume']:,} ({bar['ratio']}x) {bar['tag']}")
    for div in vol_mtf.get("divergences", []):
        print(f"  {div}")

    print(f"  ▶ LTF 近20根均量: {vol_ltf['avg_volume']:,}")
    for div in vol_ltf.get("divergences", []):
        print(f"  {div}")

    # ---- [七] 裸K PA 扫描 [P1-6 增强 key_levels] ----
    print("\n[七、 LTF 周期裸K价格行为异动扫描]")

    # 构建关键水平位列表
    key_hlines = []
    key_hlines.extend(list(levels.get("fib_retracements", {}).values()))
    if mode == "day":
        for k in ("PDH", "PDL", "ORH", "ORL"):
            if k in levels:
                key_hlines.append(levels[k])
    for g in detected_gaps[-3:]:
        key_hlines.append(g["level"])
    # [P1-6] 纳入 Swing 拐点价格
    for s in recent_mtf[-4:]:
        key_hlines.append(s["price"])
    for s in recent_ltf[-4:]:
        key_hlines.append(s["price"])
    # 共振区中枢
    for cl in multi_fib.get("clusters", []):
        key_hlines.append(cl["center"])

    pa_signals = scan_price_action(df_ltf, key_levels=key_hlines)
    if pa_signals:
        # 构建 LTF 成交量查找表（用于附带量能标注）
        vol_lookup = {bar["time"]: bar["tag"] for bar in vol_ltf.get("recent_bars", [])}
        for sig in pa_signals:
            vol_note = ""
            sig_time_short = sig["time"][:16]
            for vt, vtag in vol_lookup.items():
                if vt in sig_time_short or sig_time_short in vt:
                    vol_note = f" | 量能: {vtag}"
                    break
            print(f"  ⚡ [{sig_time_short}] {sig['pattern']} | 现价: ${sig['price']:.2f} | {sig['key_point']}{vol_note}")
    else:
        print("  • 最近5根K线暂无标准极限K线形态，价格处于正常结构推进中。")

    print("\n" + "=" * 70)
    print("🎯 数据处理完成，请结合以上客观数据执行【道氏+波浪+裸K】报告生成。")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="美股纯技术流（道氏+波浪+裸K）分析引擎 v2.0")
    parser.add_argument("--ticker", type=str, default="",
                        help="美股股票代码 (如 NVDA, AAPL, TSM)")
    parser.add_argument("--mode", type=str, choices=["swing", "day"], default="swing",
                        help="交易模式: swing (波段) 或 day (日内)")
    parser.add_argument("--watchlist", type=str, default="",
                        help="批量扫描自选池，逗号分隔 (如 AAPL,NVDA,TSLA,META)")
    args = parser.parse_args()

    mode = args.mode.lower()

    # [P2-7] 批量 Watchlist 模式
    if args.watchlist:
        tickers = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]
        print("=" * 70)
        print(f"📋 【Watchlist 快速扫描】模式: {mode.upper()} | 标的数: {len(tickers)}")
        print("=" * 70)
        market_dow = check_dow_indices(mode)
        for idx_name, info in market_dow.items():
            phase = info["htf"].get("phase", "")
            print(f"  大盘 {idx_name:<4} ${info['htf']['latest_price']:<7.2f} | {phase}")
        print("-" * 70)
        for t in tickers:
            analyze_single_ticker(t, mode, market_dow=market_dow, is_watchlist=True)
        print("=" * 70)
        return

    # 单一标的模式
    if not args.ticker:
        print("[错误] 请指定 --ticker 或 --watchlist 参数")
        sys.exit(1)

    ticker = args.ticker.upper()
    market_dow = check_dow_indices(mode)
    analyze_single_ticker(ticker, mode, market_dow=market_dow, is_watchlist=False)


if __name__ == "__main__":
    main()
