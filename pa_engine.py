#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股纯技术流（道氏理论 + 艾略特波浪理论 + 裸K价格行为）核心数据与渲染引擎 v2.0
严禁使用任何滞后性技术指标，100%基于价格结构、拐点序列、斐波那契与裸K形态。

v2.2 变更摘要（策略层：把风控从文档搬进代码）：
  P1-1: 新增 build_trade_plan() —— 结构证伪止损、TP1/TP2、盈亏比硬闸门（RR<1:2 直接判不成立）、按风险预算计算股数
  P1-2: 关键位分级 (swing/day_extreme/gap/fib/cluster) + 按中位振幅归并去重，2B 仅接受 tier>=3
  P1-3: 形态位置过滤 —— Pin Bar / 吞没 / 孕线 必须落在关键位 ±0.25×中位振幅 内
  P1-4: 剔除仍在运行的末根日内 K 线，避免形态重绘
  P3-2: 新增 --json 结构化输出、--equity / --risk-pct 仓位入参

v2.1 修复摘要（2026-10-22 审计修复 4 项 P0 缺陷）：
  FIX-1 [P0-1] 拐点增加 confirmed 标记，结构/斐波那契只用已确认拐点，未确认的单独以「待确认」呈现
  FIX-2 [P0-2] 盘前/开盘区间改为「先锁定最后交易日」，并按 bar 周期校验窗口覆盖率；数据不足输出 N/A
  FIX-3 [P0-3] 斐波那契共振区改为「与组首比较 + 簇宽上限」，消除单链漂移导致的假强共振
  FIX-4 [P0-4] 取数失败不再静默：限流指数退避重试 + 运行内缓存 + 失败汇总 + 非零退出码

v2.0 变更摘要（10项优化）：
  P0-1: HTF 大势拐点提取与输出（三级联立完整化）
  P0-2: 成交量结构分析（量价背离检测已按裸K流派边界移除, 2026-10-24）
  P0-3: 多腿斐波那契比例 + 共振区 (Fibonacci Cluster) 检测
  P1-4: 日内 Opening Range (ORH/ORL, 15分钟)
  P1-5: 市场结构转换 (MSS) 精确检测
  P1-6: Swing 拐点价格纳入 2B 流动性猎杀关键位
  P2-7: 批量 Watchlist 自选池扫描模式
  P2-8: LTF 入场级拐点序列输出
  P2-9: 文件同步（由外部脚本操作）
  P2-10: 大盘指数 HTF + MTF 双级验证
"""

import os
import sys
import time
import json
import argparse
from datetime import time as dtime
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import yfinance as yf


# ==============================================================================
# 0. 集中参数（便于后续标定，避免散落魔法数）
# ==============================================================================

FETCH_RETRIES = 2            # 取数重试次数（yfinance 自身也在重试，外层不宜过多）
FETCH_BACKOFF_BASE = 2.0     # 指数退避基数（秒）：2s / 4s
RATE_LIMIT_BREAKER = 2       # 连续限流失败达到该次数后熔断，跳过剩余请求（避免长时间阻塞）
INTRADAY_WINDOW_COVERAGE = 0.6   # 盘前/开盘窗口最少覆盖率，低于此值不输出该关键位
EXCHANGE_TZ = "America/New_York"  # 分时窗口基准时区
CLUSTER_TOLERANCE_PCT = 0.01     # 斐波那契共振区聚类容差

# --- 风控层参数 ---
LEVEL_TIERS = {"swing": 4, "day_extreme": 3, "gap": 3, "fib": 2, "cluster": 1}
TIER_MIN_FOR_SWEEP = 3        # 2B 只接受该等级以上的水平位（Swing / 日内极值 / 缺口）
MERGE_LEVEL_K = 0.15         # 归并阈值 = K × 中位振幅
LOCATION_GATE_K = 0.25       # 形态必须落在关键位 ± K × 中位振幅 内
STOP_BUFFER_K = 0.8          # 结构止损外移 = K × 中位振幅；同时充当「最小止损距离」，
                             # 否则紧贴结构的噪声级止损会把盈亏比虚增到失真
MIN_RR = 2.0                 # 盈亏比硬闸门（SKILL.md: 低于 1:2 必须放弃）
DEFAULT_RISK_PCT = 0.01      # 单笔风险占账户比例


class DataUnavailable(RuntimeError):
    """取数失败（重试耗尽，或非重试型错误）"""


_FETCH_CACHE: dict = {}
_FETCH_FAILURES: list = []
_rate_limit_streak = 0


def reset_fetch_state() -> None:
    """重置取数缓存/失败记录/熔断计数。CLI 每次运行开始时调用，保证可重复执行。"""
    global _rate_limit_streak
    _FETCH_CACHE.clear()
    _FETCH_FAILURES.clear()
    _rate_limit_streak = 0


def median_bar_range(df: pd.DataFrame, n: int = 14) -> float:
    """
    中位振幅：最近 n 根 (High-Low) 的中位数。
    只用作尺度归一化（阈值、缓冲、归并宽度），不产生方向信号，
    因此与「禁用滞后指标」的约定不冲突。比 ATR 更抗极端值。
    """
    if df is None or len(df) < 3:
        return 0.0
    rng = (df['High'] - df['Low']).tail(n)
    return float(rng.median())


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


def fetch_klines(ticker: str, interval: str, period: str, prepost: bool = False,
                 retries: int = FETCH_RETRIES) -> pd.DataFrame:
    """
    拉取指定标的与周期的K线数据，并清洗规整。
    [FIX-4] 失败不再静默降级：限流类错误指数退避重试；成功结果在本次运行内缓存；
            最终失败记录到 _FETCH_FAILURES 供 main() 汇总并以非零退出码上报。
    复权口径统一为 auto_adjust=True，避免拆股/除权被误判为跳空缺口。
    """
    global _rate_limit_streak

    key = (ticker.upper(), interval, period, bool(prepost))
    if key in _FETCH_CACHE:
        return _FETCH_CACHE[key].copy()

    # 熔断：连续限流说明当前配额已耗尽，继续请求只会长时间阻塞
    if _rate_limit_streak >= RATE_LIMIT_BREAKER:
        _FETCH_FAILURES.append(f"{ticker} {interval}/{period}: 已熔断（连续限流），跳过请求")
        return pd.DataFrame()

    last_err = "未知错误"
    for attempt in range(max(1, retries)):
        try:
            df = yf.Ticker(ticker).history(
                interval=interval, period=period, prepost=prepost, auto_adjust=True
            )
            if df.empty:
                last_err = "返回空数据集（代码无效 / 该周期无数据 / 被限流）"
            else:
                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                # 只按价格列去空，避免因 Volume 缺失而误删有效K线
                df = df[cols].dropna(subset=['Open', 'High', 'Low', 'Close'])
                if df.empty:
                    last_err = "清洗后无有效K线"
                else:
                    _FETCH_CACHE[key] = df.copy()
                    _rate_limit_streak = 0
                    return df
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        rate_limited = ("Too Many Requests" in last_err) or ("RateLimit" in last_err)
        if rate_limited:
            _rate_limit_streak += 1
            if _rate_limit_streak >= RATE_LIMIT_BREAKER:
                last_err = f"{last_err}（连续限流 {_rate_limit_streak} 次，已熔断）"
                break
        if attempt < retries - 1:
            wait = FETCH_BACKOFF_BASE ** (attempt + 1)
            print(f"[重试] {ticker} ({interval}/{period}) 第 {attempt + 1} 次失败：{last_err}"
                  f"；{wait:.0f}s 后重试", file=sys.stderr)
            time.sleep(wait)
        elif not rate_limited:
            break

    _FETCH_FAILURES.append(f"{ticker} {interval}/{period}: {last_err}")
    print(f"[警告] 拉取 {ticker} ({interval}, {period}) 失败: {last_err}", file=sys.stderr)
    return pd.DataFrame()


# ==============================================================================
# 2. 客观波段拐点（Swing High/Low）与道氏标签打标
# ==============================================================================

def extract_swings(df: pd.DataFrame, order: int = 5, confirmed_only: bool = True) -> list:
    """
    使用 scipy.signal.argrelextrema 提取客观波段拐点
    并进行同向极值归并，保证拐点严格高低交替

    [FIX-1] argrelextrema 的 order 语义要求右侧还需 order 根K线，该点才能成立。
    因此 index > len(df)-1-order 的拐点是「未确认」的：它可能被后续K线推翻（重绘），
    且回测中引用它等于使用未来信息。默认只返回已确认拐点。
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

    # [FIX-1] 标记确认状态：右侧需满 order 根才能确认
    cutoff = len(df) - 1 - order
    for pt in clean_swings:
        pt["confirmed"] = pt["index"] <= cutoff
        pt["bars_to_confirm"] = max(0, pt["index"] + order - (len(df) - 1))

    if confirmed_only:
        # 只裁掉尾部的未确认点，历史点不受影响（标签依赖更早的点，无需重算）
        while clean_swings and not clean_swings[-1]["confirmed"]:
            clean_swings.pop()

    return clean_swings


def _split_swings(df: pd.DataFrame, order: int):
    """返回 (已确认拐点, 未确认拐点)，供报告分层呈现"""
    if df is None or len(df) == 0:
        return [], []
    all_sw = extract_swings(df, order=order, confirmed_only=False)
    return ([s for s in all_sw if s["confirmed"]],
            [s for s in all_sw if not s["confirmed"]])


def _fmt_pending(pend: list) -> str:
    """未确认拐点的提示行：明确它未计入结构判定，避免读者误当作既定结构"""
    if not pend:
        return ""
    p = pend[-1]
    return (f"      ⏳ 待确认: {str(p['time'])[:16]} | {p['type']} ${p['price']:.2f}"
            f"（还需 {max(p.get('bars_to_confirm', 1), 1)} 根K线确认，未计入道氏/斐波那契判定）")


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

def detect_mss(swings: list, df: pd.DataFrame = None, buffer_k: float = 0.0) -> list:
    """
    [P1-5 / P1-7 优化] 市场结构转换检测 (MSS - Market Structure Shift)。
    
    当提供 df 时：
      要求收盘价对前一个结构位形成真实穿越（displacement），
      不仅看相邻标签组合，且同向事件去重，消除重复触发与伪信号。
    当 df 为 None 时：
      遍历道氏标签序列，检测趋势从多翻空（HH/HL → LH+LL）或空翻多（LH/LL → HL+HH）的精确转折点。
    """
    events = []
    if len(swings) < 4:
        return events

    if df is not None and len(df) > 0:
        bar_rng = median_bar_range(df, n=14) if buffer_k > 0 else 0.0
        buf = buffer_k * bar_rng

        confirmed = [s for s in swings if s.get("confirmed", True)]
        if len(confirmed) >= 3:
            highs = [s for s in confirmed if s.get("type") == "High"]
            lows = [s for s in confirmed if s.get("type") == "Low"]

            # 多翻空条件：此前有多头特征，收盘击穿最近的波段低点
            if lows and any(s.get("dow") in ("HH", "HL") for s in confirmed[-5:]):
                last_l = lows[-1]
                l_idx = max(0, min(last_l.get("index", 0), len(df) - 1))
                sub_closes = df['Close'].iloc[l_idx:]
                hits = sub_closes[sub_closes < last_l["price"] - buf]
                if not hits.empty:
                    hit_time = str(hits.index[0])[:16]
                    hit_price = float(hits.iloc[0])
                    hit_loc = df.index.get_loc(hits.index[0])
                    events.append({
                        "type": "🔻 空翻 (Bearish MSS)",
                        "trigger_time": hit_time,
                        "trigger_price": hit_price,
                        "bars_ago": len(df) - 1 - hit_loc,
                        "detail": f"收盘跌破前结构低点 ${last_l['price']:.2f}（外溢 ${buf:.2f}）→ 多头结构被破坏"
                    })

            # 空翻多条件：此前有空头特征，收盘突破最近的波段高点
            if highs and any(s.get("dow") in ("LH", "LL") for s in confirmed[-5:]):
                last_h = highs[-1]
                h_idx = max(0, min(last_h.get("index", 0), len(df) - 1))
                sub_closes = df['Close'].iloc[h_idx:]
                hits = sub_closes[sub_closes > last_h["price"] + buf]
                if not hits.empty:
                    hit_time = str(hits.index[0])[:16]
                    hit_price = float(hits.iloc[0])
                    hit_loc = df.index.get_loc(hits.index[0])
                    events.append({
                        "type": "🔺 多翻 (Bullish MSS)",
                        "trigger_time": hit_time,
                        "trigger_price": hit_price,
                        "bars_ago": len(df) - 1 - hit_loc,
                        "detail": f"收盘突破前结构高点 ${last_h['price']:.2f}（外溢 ${buf:.2f}）→ 空头结构被破坏"
                    })
        # 若 df 模式下识别到了 MSS 事件，直接流向去重处理
    # 未识别到位移穿越时，回退到纯拐点序列判定
    if not events:
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

    # [P1-7] 同向连续重复触发去重
    deduped = []
    for ev in events:
        if not deduped or deduped[-1]["type"] != ev["type"]:
            deduped.append(ev)
        else:
            deduped[-1] = ev
    return deduped


# ==============================================================================
# 3. [P2-10] 大盘道氏理论相互验证 — HTF + MTF 双级
# ==============================================================================

def _fmt_price(value) -> str:
    """价格格式化：取数失败时显示 N/A，而不是误导性的 $0.00"""
    return f"${value:<7.2f}" if isinstance(value, (int, float)) else "N/A      "


def check_dow_indices(mode: str) -> dict:
    """拉取 SPY, QQQ, DIA 在 HTF 与 MTF 下的最近 4 个波段拐点"""
    indices = ["SPY", "QQQ", "DIA"]
    results = {}
    htf_cfg = TIMEFRAME_CONFIGS[mode]["HTF"]
    mtf_cfg = TIMEFRAME_CONFIGS[mode]["MTF"]

    for idx_sym in indices:
        entry = {"htf": {"latest_price": None, "swings": [], "phase": ""},
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

def _to_exchange_tz(df: pd.DataFrame):
    """[FIX-2] 统一到交易所时区，避免分时掩码按错误的时区切片"""
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            print("[警告] 日内K线索引非时间类型，跳过分时窗口计算", file=sys.stderr)
            return None
        if df.index.tz is None:
            print(f"[警告] 日内K线无时区信息，按 {EXCHANGE_TZ} 本地时间处理", file=sys.stderr)
            return df
        return df.tz_convert(EXCHANGE_TZ)
    except Exception as e:
        print(f"[警告] 时区转换失败：{type(e).__name__}: {e}", file=sys.stderr)
        return None


def _infer_bar_minutes(df: pd.DataFrame) -> int:
    """推断K线周期（分钟），用于校验盘前/开盘窗口是否被截断"""
    if len(df) < 3:
        return 5
    diffs = pd.Series(df.index).diff().dropna()
    diffs = diffs[diffs <= pd.Timedelta(hours=2)]   # 剔除跨日/跨周末间隔
    if diffs.empty:
        return 5
    return max(1, int(diffs.median().total_seconds() // 60))


def _prev_session_bar(df_daily: pd.DataFrame, df_intraday: pd.DataFrame):
    """确定「上一交易日」的日线：若日线末根已是当前会话，则取倒数第二根"""
    if df_daily is None or len(df_daily) < 2:
        return None
    try:
        last_date = df_daily.index[-1].date()
        if df_intraday is not None and not df_intraday.empty:
            session = df_intraday.index[-1].date()
            return df_daily.iloc[-2] if last_date == session else df_daily.iloc[-1]
        return df_daily.iloc[-2]
    except Exception:
        return df_daily.iloc[-2]


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
    levels["intraday_missing"] = []
    if df_daily is not None and len(df_daily) >= 2:
        # [FIX-2b] 不再固定取 iloc[-2]，改为与当日会话比对后确定「昨日」
        prev_day = _prev_session_bar(df_daily, df_intraday)
        if prev_day is not None:
            levels["PDH"] = round(float(prev_day['High']), 2)
            levels["PDL"] = round(float(prev_day['Low']), 2)
            levels["PDC"] = round(float(prev_day['Close']), 2)

    if df_intraday is not None and not df_intraday.empty:
        # [FIX-2] 原实现按「根数」取 tail(150) 且只按时刻过滤，会同时造成两种错误：
        #   (a) 窗口不足 1 个交易日 -> 盘前 04:00-07:30 被截断；
        #   (b) 窗口跨日 -> 前一交易日的盘前/开盘区间混入当日。
        # 改为：先锁定最后一个交易日，再按时刻取窗口，并按K线周期校验覆盖率。
        dfi = _to_exchange_tz(df_intraday)
        if dfi is None or len(dfi) < 5:
            levels["intraday_missing"].append("日内K线不可用（时区或长度异常）")
        else:
            session = dfi.index[-1].date()
            dfi = dfi[dfi.index.date == session]
            levels["session"] = str(session)
            bar_min = _infer_bar_minutes(dfi)
            t = dfi.index.time

            # 盘前 (04:00 - 09:30 ET)
            df_pm = dfi[[(x >= dtime(4, 0)) and (x < dtime(9, 30)) for x in t]]
            need_pm = int((9 * 60 + 30 - 4 * 60) // bar_min)
            if need_pm and len(df_pm) >= need_pm * INTRADAY_WINDOW_COVERAGE:
                levels["PMH"] = round(float(df_pm['High'].max()), 2)
                levels["PML"] = round(float(df_pm['Low'].min()), 2)
            else:
                levels["intraday_missing"].append(
                    f"PMH/PML（盘前仅 {len(df_pm)}/{need_pm} 根，覆盖率不足）")

            # [P1-4] 开盘区间 Opening Range (09:30 - 09:45 ET, 15分钟)
            df_or = dfi[[(x >= dtime(9, 30)) and (x < dtime(9, 45)) for x in t]]
            need_or = max(1, int(15 // bar_min))
            if len(df_or) >= max(1, need_or * INTRADAY_WINDOW_COVERAGE):
                levels["ORH"] = round(float(df_or['High'].max()), 2)
                levels["ORL"] = round(float(df_or['Low'].min()), 2)
            else:
                levels["intraday_missing"].append(
                    f"ORH/ORL（开盘区间仅 {len(df_or)}/{need_or} 根）")

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
    # [FIX-3] 原实现用 abs(p - group[-1]) 只与组内上一个元素比较，构成单链（single-linkage），
    #         相邻两点都在容差内即可无限漂移，导致一个跨度远超容差的区间被标为「强共振」。
    #         改为：与组首比较 + 簇总宽上限 + 记录来源腿（同一腿的多个 Fib 不算共振）。
    all_fib = []
    for li, leg in enumerate(legs[-3:]):
        s, e = leg["from_pt"]["price"], leg["to_pt"]["price"]
        d = abs(e - s)
        if d <= 0:
            continue
        for f in [0.382, 0.500, 0.618, 0.786, 1.272, 1.618]:
            p = (e - f * d) if leg["direction"] == "UP" else (e + f * d)
            all_fib.append({"price": round(p, 2), "leg": li, "fib": f})

    all_fib.sort(key=lambda x: x["price"])
    clusters = []

    def _flush(group):
        if len(group) < 2:
            return
        lo = min(x["price"] for x in group)
        hi = max(x["price"] for x in group)
        if (hi - lo) / max(abs(lo), 0.01) > tolerance_pct:
            return                       # 双保险：超宽容差不输出
        legs_in = {x["leg"] for x in group}
        clusters.append({
            "zone": f"${lo:.2f} - ${hi:.2f}",
            "center": round(float(np.mean([x["price"] for x in group])), 2),
            "count": len(group),
            "legs": len(legs_in),
            "strength": "🔥 强共振" if (len(group) >= 3 and len(legs_in) >= 2) else "⚡ 共振"
        })

    if all_fib:
        base = all_fib[0]["price"]        # 升序 -> base 恒为组内最低价
        group = [all_fib[0]]
        for item in all_fib[1:]:
            cand_lo = min(base, item["price"])
            cand_hi = max(base, item["price"])
            if (cand_hi - cand_lo) / max(abs(cand_lo), 0.01) <= tolerance_pct:
                group.append(item)
            else:
                _flush(group)
                group = [item]
                base = item["price"]
        _flush(group)

    # 按汇聚强度排序，便于报告只取前若干个
    clusters.sort(key=lambda c: (-c["count"], -c["legs"]))
    result["clusters"] = clusters
    return result


# ==============================================================================
# 4b-2. 艾略特波浪理论客观状态机与浪型识别 (Algorithmic Wave Engine)
# ==============================================================================

def classify_wave_pattern(swings: list, df: pd.DataFrame = None) -> dict:
    """
    基于客观确认的波段拐点自动进行浪型识别与三大铁律校验：
      * 铁律1: 浪2 回撤绝不超过 浪1 起点 (Retracement <= 100%)；
      * 铁律2: 浪3 永远不是 浪1、浪3、浪5 中最短的一浪 (L3 not shortest among 1, 3, 5)；
      * 铁律3: 浪4 终点绝不重叠进入 浪1 价格领地 (Wave 4 low > Wave 1 high for bull impulse)；
      * 交替原则 (Alternation): 浪2 与 浪4 形态/幅度差异化参考 (e.g. 浪2深度回撤 vs 浪4浅度回撤)；
      * 调整浪识别：A-B-C 浪型，计算 C/A 比例 (0.618 / 1.0 / 1.618)。
    """
    conf = [s for s in swings if s.get("confirmed", True)]
    if len(conf) < 3:
        return {
            "pattern": "unclassified",
            "direction": 0,
            "current_wave": "none",
            "is_valid_impulse": False,
            "rules": {},
            "summary": "已确认拐点不足 3 个，无法识别浪型"
        }

    # 内部辅助：校验 6 拐点是否为标准 5 浪冲量驱动
    def _check_impulse_5(cand):
        if len(cand) < 6:
            return None
        # 多头推动浪候选：Low, High, Low, High, Low, High
        if (cand[0]["type"] == "Low" and cand[1]["type"] == "High" and
            cand[2]["type"] == "Low" and cand[3]["type"] == "High" and
            cand[4]["type"] == "Low" and cand[5]["type"] == "High"):

            p0, p1, p2, p3, p4, p5 = [float(s["price"]) for s in cand]
            l1 = p1 - p0
            l3 = p3 - p2
            l5 = p5 - p4

            # 铁律 1: 浪2 不跌破 浪1 起点 (p2 >= p0)
            rule1_passed = bool(p2 >= p0 and l1 > 0)
            w2_retrace = (p1 - p2) / l1 if l1 > 0 else 0.0

            # 铁律 2: 浪3 永远不是 1, 3, 5 中最短的
            is_w3_shortest = bool(l3 < l1 and l3 < l5)
            rule2_passed = bool(not is_w3_shortest and l3 > 0 and p3 > p1)

            # 铁律 3: 浪4 终点不重叠进入 浪1 价格领地 (p4 > p1)
            rule3_passed = bool(p4 > p1)
            w4_retrace = (p3 - p4) / l3 if l3 > 0 else 0.0

            # 交替原则: 浪2 与 浪4 回撤幅度差异化
            alternation_passed = bool(abs(w2_retrace - w4_retrace) >= 0.15)
            alt_desc = (f"浪2回撤 {w2_retrace:.1%} 与 浪4回撤 {w4_retrace:.1%} 呈现明显交替差异"
                        if alternation_passed else
                        f"浪2回撤 {w2_retrace:.1%} 与 浪4回撤 {w4_retrace:.1%} 幅度接近")

            is_valid = rule1_passed and rule2_passed and rule3_passed

            return {
                "pattern": "impulse_5_wave",
                "direction": 1,
                "current_wave": "wave_5_completed" if is_valid else "impulse_failed",
                "is_valid_impulse": is_valid,
                "wave_points": {0: cand[0], 1: cand[1], 2: cand[2], 3: cand[3], 4: cand[4], 5: cand[5]},
                "rules": {
                    "rule1": {
                        "passed": rule1_passed,
                        "name": "铁律1: 浪2不破浪1起点",
                        "desc": f"浪2低点 ${p2:.2f} >= 浪1起点 ${p0:.2f} (回撤率 {w2_retrace:.1%})"
                                if rule1_passed else f"浪2低点 ${p2:.2f} 跌破浪1起点 ${p0:.2f}"
                    },
                    "rule2": {
                        "passed": rule2_passed,
                        "name": "铁律2: 浪3非最短浪",
                        "desc": f"L1={l1:.2f}, L3={l3:.2f}, L5={l5:.2f} (浪3未成为最短浪)"
                                if rule2_passed else f"浪3涨幅 {l3:.2f} 劣于浪1({l1:.2f})和浪5({l5:.2f})，为最短浪"
                    },
                    "rule3": {
                        "passed": rule3_passed,
                        "name": "铁律3: 浪4不重叠浪1领地",
                        "desc": f"浪4低点 ${p4:.2f} > 浪1高点 ${p1:.2f}"
                                if rule3_passed else f"浪4低点 ${p4:.2f} 击穿侵入浪1高点 ${p1:.2f}"
                    },
                    "alternation": {
                        "passed": alternation_passed,
                        "name": "交替原则 (Alternation)",
                        "desc": alt_desc
                    }
                },
                "ratios": {
                    "w2_retrace": round(w2_retrace, 3),
                    "w3_to_w1": round(l3 / l1, 3) if l1 > 0 else None,
                    "w4_retrace": round(w4_retrace, 3),
                    "w5_to_w1": round(l5 / l1, 3) if l1 > 0 else None
                },
                "summary": "完整多头5浪冲量驱动结构（三大铁律全部通过）" if is_valid else "多头5浪形态破坏（未通过铁律校验）"
            }

        # 空头推动浪候选：High, Low, High, Low, High, Low
        elif (cand[0]["type"] == "High" and cand[1]["type"] == "Low" and
              cand[2]["type"] == "High" and cand[3]["type"] == "Low" and
              cand[4]["type"] == "High" and cand[5]["type"] == "Low"):

            p0, p1, p2, p3, p4, p5 = [float(s["price"]) for s in cand]
            l1 = p0 - p1
            l3 = p2 - p3
            l5 = p4 - p5

            rule1_passed = bool(p2 <= p0 and l1 > 0)
            w2_retrace = (p2 - p1) / l1 if l1 > 0 else 0.0

            is_w3_shortest = bool(l3 < l1 and l3 < l5)
            rule2_passed = bool(not is_w3_shortest and l3 > 0 and p3 < p1)

            rule3_passed = bool(p4 < p1)
            w4_retrace = (p4 - p3) / l3 if l3 > 0 else 0.0

            alternation_passed = bool(abs(w2_retrace - w4_retrace) >= 0.15)
            is_valid = rule1_passed and rule2_passed and rule3_passed

            return {
                "pattern": "impulse_5_wave",
                "direction": -1,
                "current_wave": "wave_5_completed" if is_valid else "impulse_failed",
                "is_valid_impulse": is_valid,
                "wave_points": {0: cand[0], 1: cand[1], 2: cand[2], 3: cand[3], 4: cand[4], 5: cand[5]},
                "rules": {
                    "rule1": {"passed": rule1_passed, "name": "铁律1: 浪2不破浪1起点", "desc": f"浪2高点 ${p2:.2f} <= 浪1起点 ${p0:.2f}"},
                    "rule2": {"passed": rule2_passed, "name": "铁律2: 浪3非最短浪", "desc": f"L1={l1:.2f}, L3={l3:.2f}, L5={l5:.2f}"},
                    "rule3": {"passed": rule3_passed, "name": "铁律3: 浪4不重叠浪1领地", "desc": f"浪4反弹 ${p4:.2f} < 浪1低点 ${p1:.2f}"},
                    "alternation": {"passed": alternation_passed, "name": "交替原则", "desc": f"浪2回撤 {w2_retrace:.1%} vs 浪4回撤 {w4_retrace:.1%}"}
                },
                "ratios": {
                    "w2_retrace": round(w2_retrace, 3),
                    "w3_to_w1": round(l3 / l1, 3) if l1 > 0 else None,
                    "w4_retrace": round(w4_retrace, 3),
                    "w5_to_w1": round(l5 / l1, 3) if l1 > 0 else None
                },
                "summary": "完整空头5浪冲量驱动结构（三大铁律全部通过）" if is_valid else "空头5浪形态破坏"
            }
        return None

    # 若入参为专门测试的 6 个拐点，直接返回 5 浪冲量检验结果
    if len(conf) == 6:
        cand6_exact = _check_impulse_5(conf[-6:])
        if cand6_exact is not None:
            return cand6_exact

    # 若长序列下最近 6 个拐点正好构成合规的 5 浪推动完成状态
    if len(conf) >= 6:
        cand6 = _check_impulse_5(conf[-6:])
        if cand6 is not None and cand6.get("is_valid_impulse"):
            return cand6

    last_type = conf[-1].get("type")

    # 当最新已确认拐点为 Low：寻找 (1) 多头第5浪进行中 (2) 多头 A-B-C 调整结束 (3) 多头第3浪展开中
    if last_type == "Low":
        # 1. 检查进行中的第 5 浪结构 (5 个拐点: Low, High, Low, High, Low)
        if len(conf) >= 5:
            cand5 = conf[-5:]
            if (cand5[0]["type"] == "Low" and cand5[1]["type"] == "High" and
                cand5[2]["type"] == "Low" and cand5[3]["type"] == "High" and
                cand5[4]["type"] == "Low"):
                p0, p1, p2, p3, p4 = [float(s["price"]) for s in cand5]
                l1 = p1 - p0
                l3 = p3 - p2
                rule1_ok = (p2 >= p0 and l1 > 0)
                rule3_ok = (p4 > p1)
                if rule1_ok and rule3_ok and p3 > p1 and l3 > 0:
                    return {
                        "pattern": "developing_wave_5",
                        "direction": 1,
                        "current_wave": "wave_5_developing",
                        "is_valid_impulse": True,
                        "wave_points": {0: cand5[0], 1: cand5[1], 2: cand5[2], 3: cand5[3], 4: cand5[4]},
                        "rules": {
                            "rule1": {"passed": True, "name": "铁律1: 浪2不破浪1起点", "desc": f"浪2低点 ${p2:.2f} >= ${p0:.2f}"},
                            "rule2": {"passed": True, "name": "铁律2: 浪3进行中", "desc": f"L1={l1:.2f}, L3={l3:.2f}"},
                            "rule3": {"passed": True, "name": "铁律3: 浪4不重叠浪1领地", "desc": f"浪4低点 ${p4:.2f} > 浪1高点 ${p1:.2f}"}
                        },
                        "summary": f"第5浪正在推进中（浪4回踩在 ${p4:.2f} 完成，守住浪1顶 ${p1:.2f}）"
                    }

        # 2. 检查 A-B-C 调整浪结构 (4 个拐点: High, Low, High, Low)
        if len(conf) >= 4:
            cand4 = conf[-4:]
            if (cand4[0]["type"] == "High" and cand4[1]["type"] == "Low" and
                cand4[2]["type"] == "High" and cand4[3]["type"] == "Low"):
                p0, pa, pb, pc = [float(s["price"]) for s in cand4]
                la = p0 - pa
                lc = pb - pc
                if la > 0 and pb < p0 and lc > 0:
                    ca_ratio = lc / la
                    return {
                        "pattern": "corrective_abc",
                        "direction": 1,
                        "current_wave": "wave_c_completed",
                        "is_valid_impulse": False,
                        "wave_points": {"0": cand4[0], "A": cand4[1], "B": cand4[2], "C": cand4[3]},
                        "ratios": {"c_to_a": round(ca_ratio, 3)},
                        "rules": {
                            "c_projection": {
                                "passed": True,
                                "desc": f"C浪幅度 {lc:.2f} 为 A浪幅度 {la:.2f} 的 {ca_ratio:.3f} 倍"
                            }
                        },
                        "summary": f"A-B-C 调整浪形态（C浪终端 ${pc:.2f}，C/A比率 {ca_ratio:.2f}）"
                    }

        # 3. 检查进行中的第 3 浪结构 (3 个拐点: Low, High, Low)
        if len(conf) >= 3:
            cand3 = conf[-3:]
            if (cand3[0]["type"] == "Low" and cand3[1]["type"] == "High" and cand3[2]["type"] == "Low"):
                p0, p1, p2 = [float(s["price"]) for s in cand3]
                l1 = p1 - p0
                rule1_ok = (p2 >= p0 and l1 > 0)
                w2_retrace = (p1 - p2) / l1 if l1 > 0 else 0.0
                return {
                    "pattern": "developing_wave_3",
                    "direction": 1,
                    "current_wave": "wave_3_developing",
                    "is_valid_impulse": rule1_ok,
                    "wave_points": {0: cand3[0], 1: cand3[1], 2: cand3[2]},
                    "rules": {
                        "rule1": {
                            "passed": rule1_ok,
                            "name": "铁律1: 浪2不破浪1起点",
                            "desc": f"浪2低点 ${p2:.2f} >= ${p0:.2f} (回撤率 {w2_retrace:.1%})"
                                    if rule1_ok else f"浪2跌破浪1起点 ${p0:.2f}，浪型无效"
                        }
                    },
                    "ratios": {"w2_retrace": round(w2_retrace, 3)},
                    "summary": "第3浪主升展开中（浪2回踩确认完成）" if rule1_ok else "浪2跌破起点，非有效浪型"
                }

    # 当最新已确认拐点为 High：寻找空头结构
    elif last_type == "High":
        # 空头进行中第5浪 (High, Low, High, Low, High)
        if len(conf) >= 5:
            cand5 = conf[-5:]
            if (cand5[0]["type"] == "High" and cand5[1]["type"] == "Low" and
                cand5[2]["type"] == "High" and cand5[3]["type"] == "Low" and
                cand5[4]["type"] == "High"):
                p0, p1, p2, p3, p4 = [float(s["price"]) for s in cand5]
                l1 = p0 - p1
                l3 = p2 - p3
                rule1_ok = (p2 <= p0 and l1 > 0)
                rule3_ok = (p4 < p1)
                if rule1_ok and rule3_ok and p3 < p1 and l3 > 0:
                    return {
                        "pattern": "developing_wave_5",
                        "direction": -1,
                        "current_wave": "wave_5_developing",
                        "is_valid_impulse": True,
                        "wave_points": {0: cand5[0], 1: cand5[1], 2: cand5[2], 3: cand5[3], 4: cand5[4]},
                        "rules": {
                            "rule1": {"passed": True, "name": "铁律1: 浪2不破浪1起点", "desc": f"浪2高点 ${p2:.2f} <= ${p0:.2f}"},
                            "rule2": {"passed": True, "name": "铁律2: 浪3进行中", "desc": f"L1={l1:.2f}, L3={l3:.2f}"},
                            "rule3": {"passed": True, "name": "铁律3: 浪4不重叠浪1领地", "desc": f"浪4反弹 ${p4:.2f} < 浪1低点 ${p1:.2f}"}
                        },
                        "summary": f"空头第5浪正在推进中（反弹在 ${p4:.2f} 受阻，未触及浪1低点 ${p1:.2f}）"
                    }

        # 空头 A-B-C 反弹 (Low, High, Low, High)
        if len(conf) >= 4:
            cand4 = conf[-4:]
            if (cand4[0]["type"] == "Low" and cand4[1]["type"] == "High" and
                cand4[2]["type"] == "Low" and cand4[3]["type"] == "High"):
                p0, pa, pb, pc = [float(s["price"]) for s in cand4]
                la = pa - p0
                lc = pc - pb
                if la > 0 and pb > p0 and lc > 0:
                    ca_ratio = lc / la
                    return {
                        "pattern": "corrective_abc",
                        "direction": -1,
                        "current_wave": "wave_c_completed",
                        "is_valid_impulse": False,
                        "wave_points": {"0": cand4[0], "A": cand4[1], "B": cand4[2], "C": cand4[3]},
                        "ratios": {"c_to_a": round(ca_ratio, 3)},
                        "rules": {
                            "c_projection": {
                                "passed": True,
                                "desc": f"C浪反弹幅度 {lc:.2f} 为 A浪的 {ca_ratio:.3f} 倍"
                            }
                        },
                        "summary": f"空头 A-B-C 反弹浪形态（C浪终端 ${pc:.2f}，C/A比率 {ca_ratio:.2f}）"
                    }

        # 空头进行中第3浪 (High, Low, High)
        if len(conf) >= 3:
            cand3 = conf[-3:]
            if (cand3[0]["type"] == "High" and cand3[1]["type"] == "Low" and cand3[2]["type"] == "High"):
                p0, p1, p2 = [float(s["price"]) for s in cand3]
                l1 = p0 - p1
                rule1_ok = (p2 <= p0 and l1 > 0)
                w2_retrace = (p2 - p1) / l1 if l1 > 0 else 0.0
                return {
                    "pattern": "developing_wave_3",
                    "direction": -1,
                    "current_wave": "wave_3_developing",
                    "is_valid_impulse": rule1_ok,
                    "wave_points": {0: cand3[0], 1: cand3[1], 2: cand3[2]},
                    "rules": {
                        "rule1": {
                            "passed": rule1_ok,
                            "name": "铁律1: 浪2不破浪1起点",
                            "desc": f"浪2高点 ${p2:.2f} <= ${p0:.2f}" if rule1_ok else f"浪2突破起点 ${p0:.2f}"
                        }
                    },
                    "ratios": {"w2_retrace": round(w2_retrace, 3)},
                    "summary": "空头第3浪下杀展开中" if rule1_ok else "浪2突破起点，非有效浪型"
                }

    # 兜底：若前置未识别成功，检查 6 拐点
    cand6_fallback = _check_impulse_5(conf[-6:]) if len(conf) >= 6 else None
    if cand6_fallback is not None:
        return cand6_fallback

    return {
        "pattern": "unclassified",
        "direction": 0,
        "current_wave": "none",
        "is_valid_impulse": False,
        "rules": {},
        "summary": "当前波段未匹配到典型艾略特浪型"
    }


def evaluate_wave_position(swings: list, df: pd.DataFrame = None, current_price: float = None) -> dict:
    """
    波浪位置狙击决策 (Wave-Position Sniper)：
      1. 强化第 3 浪主升突破：浪2回踩完成、向上突破浪1高点时，判定为优质主升买点；
      2. 拦截第 5 浪冲顶诱多：当前处于浪5晚期且伴随量价/动能衰竭时，严格禁止追多；
      3. 捕捉 C 浪终结反转：大级别 A-B-C 调整浪末端（达到 C ≈ 0.618A 或 1.0A 且伴随 2B 假突破扫损收回）时，发出高赔率左侧波浪反转信号。
    """
    wp = classify_wave_pattern(swings, df)
    if current_price is None and df is not None and len(df) > 0:
        current_price = float(df["Close"].iloc[-1])

    # 1. 拦截第 5 浪冲顶诱多与动能衰竭 (优先级最高：保命防高位接盘)
    # 严格要求：必须是有效且同向的多头第5浪 (direction == 1, is_valid_impulse == True)
    if wp.get("direction") == 1 and wp.get("is_valid_impulse", False):
        if wp.get("pattern") in ("developing_wave_5", "impulse_5_wave"):
            pts = wp.get("wave_points", {})
            p3 = float(pts.get(3, {}).get("price", 0)) if 3 in pts else 0
            p5 = float(pts.get(5, {}).get("price", 0)) if 5 in pts else 0
            ref_high = max(p3, p5)
            if current_price is not None and ref_high > 0 and current_price >= ref_high * 0.98:
                volume_exhausted = False
                if df is not None and "Volume" in df.columns and len(df) >= 20:
                    recent_vol = float(df["Volume"].tail(5).mean())
                    base_vol = float(df["Volume"].iloc[-25:-5].mean()) if len(df) >= 25 else recent_vol
                    if base_vol > 0 and recent_vol < 0.85 * base_vol:
                        volume_exhausted = True
                elif df is None or "Volume" not in df.columns:
                    volume_exhausted = True

                if volume_exhausted:
                    return {
                        "signal": "wave5_exhaustion",
                        "action": "block_chasing",
                        "status": "rejected_by_wave5_exhaustion",
                        "reason": f"识别处于第5浪冲顶晚期（现价 ${current_price:.2f} 逼近或超越高点 ${ref_high:.2f}），且量能动能呈现衰竭，风控禁止追多",
                        "wave_pattern": wp
                    }

    # 2. 强化第 3 浪主升突破 (必须是进行中的第3浪 developing_wave_3，严禁在已完成的5浪顶部误报第3浪)
    if wp.get("direction") == 1 and wp.get("pattern") == "developing_wave_3" and wp.get("is_valid_impulse", False):
        pts = wp.get("wave_points", {})
        p0 = float(pts.get(0, {}).get("price", 0))
        p1 = float(pts.get(1, {}).get("price", 0))
        p2 = float(pts.get(2, {}).get("price", 0))

        rule1_ok = wp.get("rules", {}).get("rule1", {}).get("passed", False)
        if rule1_ok and p1 > p0 and current_price is not None:
            if current_price >= p1 * 0.99:
                return {
                    "signal": "wave3_breakout",
                    "action": "allow",
                    "status": "wave3_impulse_breakout",
                    "reason": f"浪2在 ${p2:.2f} 企稳回踩完成（守住起点 ${p0:.2f}），现价突破浪1高点 ${p1:.2f}，确认主升第3浪起爆",
                    "wave_pattern": wp
                }

    # 3. 捕捉 C 浪终结与 2B 假突破筑底反转
    if wp.get("direction") == 1 and wp.get("pattern") == "corrective_abc":
        pts = wp.get("wave_points", {})
        pa = float(pts.get("A", {}).get("price", 0))
        pc = float(pts.get("C", {}).get("price", 0))
        ca_ratio = wp.get("ratios", {}).get("c_to_a", 0.0)

        # 满足 C ≈ 0.618A / 1.0A / 1.618A 经典斐波比例
        is_ratio_matched = (0.50 <= ca_ratio <= 1.70)
        if is_ratio_matched and pa > 0 and current_price is not None:
            # 2B 假跌破扫损收回：C 浪触穿 A 浪低点 (pc <= pa * 1.01)，且现价收复至 A 浪低点附近或上方 (>= pa * 0.99)
            has_2b_sweep = (pc <= pa * 1.01 and current_price >= pa * 0.99)
            if has_2b_sweep:
                return {
                    "signal": "wave_c_reversal",
                    "action": "bottom_reversal",
                    "status": "wave_c_bottom_reversal",
                    "reason": f"大级别 A-B-C 调整浪满足 C≈{ca_ratio:.2f}A 且在 A 浪低点 ${pa:.2f} 处出现假跌破扫损收回，构成波浪筑底反转",
                    "wave_pattern": wp
                }

    return {
        "signal": "neutral",
        "action": "neutral",
        "status": "neutral",
        "reason": "未触发特殊波浪狙击条件，按常规结构规则处理",
        "wave_pattern": wp
    }


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

def scan_price_action(df: pd.DataFrame, key_levels: list = None,
                      bar_rng: float = None, require_location: bool = True,
                      sweep_min_tier: int = TIER_MIN_FOR_SWEEP) -> list:
    """
    扫描最近 5 根 K 线的裸K信号：
    - Pin Bar（长影线拒绝）
    - 吞没形态（Bullish/Bearish Engulfing）
    - 孕线（Inside Bar）
    - 2B假突破 / 流动性猎杀（Liquidity Sweep）

    [P1-2] key_levels 支持两种入参：
           - float 列表（视为 tier=3，兼容旧调用）
           - 带 kind 的 dict 列表：只有 tier >= sweep_min_tier 的水平位才允许触发 2B
    [P1-3] require_location=True 时，Pin Bar / 吞没必须发生在关键位 ±LOCATION_GATE_K×振幅 内，
           否则视为「脱离位置的形态」，不输出。
    """
    signals = []
    if len(df) < 5:
        return signals

    if bar_rng is None:
        bar_rng = median_bar_range(df)

    # 归一化水平位
    norm = []
    for lv in (key_levels or []):
        if isinstance(lv, dict):
            price = float(lv.get("price", 0) or 0)
            tier = LEVEL_TIERS.get(lv.get("kind", "swing"), 3)
            src = lv.get("src", "")
        else:
            price, tier, src = float(lv), 3, ""
        if price > 0:
            norm.append({"price": price, "tier": tier, "src": src})

    sweep_levels = []
    for lv in norm:
        if lv["tier"] >= sweep_min_tier:
            if not any(abs(lv["price"] - x) < 1e-6 for x in sweep_levels):
                sweep_levels.append(lv["price"])
    loc_prices = [lv["price"] for lv in norm]

    tol = LOCATION_GATE_K * bar_rng if bar_rng > 0 else 0.0
    recent_df = df.tail(5)
    emitted_2b = set()

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

        # 位置判定：形态极值距最近关键位的距离
        nearest = min((abs(c_close - l) for l in loc_prices), default=None) \
            if loc_prices else None
        at_level = nearest is not None and tol > 0 and nearest <= tol
        loc_note = (f" | 距关键位 ${nearest:.2f}" if nearest is not None else "")

        def gate_ok(price_probe):
            if not require_location:
                return True
            if not loc_prices:
                return False
            d = min(abs(price_probe - l) for l in loc_prices)
            return tol > 0 and d <= tol

        # 1. Pin Bar 扫描 (影线 >= 实体2倍且占总振幅 >= 60%)
        if lower_shadow >= 2 * body and (lower_shadow / rng) >= 0.60:
            if gate_ok(c_low):
                signals.append({
                    "time": dt_str, "pattern": "看涨锤子线 Pin Bar (多头拒绝下跌)",
                    "price": round(c_close, 2),
                    "key_point": f"下影探底 ${c_low:.2f}{loc_note}",
                    "at_key_level": at_level,
                })
        elif upper_shadow >= 2 * body and (upper_shadow / rng) >= 0.60:
            if gate_ok(c_high):
                signals.append({
                    "time": dt_str, "pattern": "看跌流星线 Pin Bar (空头拒绝上涨)",
                    "price": round(c_close, 2),
                    "key_point": f"上影摸高 ${c_high:.2f}{loc_note}",
                    "at_key_level": at_level,
                })

        # 2. 吞没形态 (Engulfing)
        if p_close < p_open and c_close > c_open:
            if c_open <= p_close and c_close >= p_open and gate_ok(c_low):
                signals.append({
                    "time": dt_str, "pattern": "看涨吞没形态 (Bullish Engulfing)",
                    "price": round(c_close, 2),
                    "key_point": f"阳线实体完全覆盖前阴线 (${c_open:.2f} - ${c_close:.2f}){loc_note}",
                    "at_key_level": at_level,
                })
        elif p_close > p_open and c_close < c_open:
            if c_open >= p_close and c_close <= p_open and gate_ok(c_high):
                signals.append({
                    "time": dt_str, "pattern": "看跌吞没形态 (Bearish Engulfing)",
                    "price": round(c_close, 2),
                    "key_point": f"阴线实体完全覆盖前阳线 (${c_close:.2f} - ${c_open:.2f}){loc_note}",
                    "at_key_level": at_level,
                })

        # 3. 孕线形态 (Inside Bar) —— 孕线本身只是收敛，不构成方向，仍需位置上下文
        if c_high <= prev['High'] and c_low >= prev['Low'] and gate_ok(c_close):
            signals.append({
                "time": dt_str, "pattern": "孕线收敛形态 (Inside Bar)",
                "price": round(c_close, 2),
                "key_point": f"完全被前K包含，酝酿变盘突破{loc_note}",
                "at_key_level": at_level,
            })

        # 4. 2B假突破 / 流动性猎杀 (Liquidity Sweep)
        for lvl in sweep_levels:
            key = (dt_str, round(lvl, 2))
            if key in emitted_2b:
                continue
            if c_high > lvl and c_close < lvl and prev['High'] <= lvl:
                emitted_2b.add(key)
                signals.append({
                    "time": dt_str,
                    "pattern": f"2B高位假突破/流动性猎杀 (Sweep High ${lvl:.2f})",
                    "price": round(c_close, 2), "key_point": "刺破前高后回落，诱多扫损信号",
                    "at_key_level": True,
                })
            elif c_low < lvl and c_close > lvl and prev['Low'] >= lvl:
                emitted_2b.add(key)
                signals.append({
                    "time": dt_str,
                    "pattern": f"2B低位假跌破/流动性猎杀 (Sweep Low ${lvl:.2f})",
                    "price": round(c_close, 2), "key_point": "跌破前低后强力收回，诱空扫损信号",
                    "at_key_level": True,
                })

    return signals


# ==============================================================================
# 5b. [P0-2] 成交量结构分析
# ==============================================================================

def analyze_volume(df: pd.DataFrame, lookback: int = 20, recent: int = 5) -> dict:
    """
    分析成交量结构：计算均量基准，对最近 N 根K线标注放量/缩量。
    量价背离检测已按裸K流派边界移除 (2026-10-24)：
    形态/结构/位置全部由价格决定，量能仅作为方向层突破真伪过滤器 (道氏第6条)。
    """
    result = {"avg_volume": 0, "recent_bars": [], "available": True}
    if len(df) < lookback:
        return result
    if "Volume" not in df.columns:
        # [FIX-4 附带] 部分标的/数据源不返回 Volume，原实现会直接 KeyError 崩掉整个扫描
        result["available"] = False
        result["reason"] = "数据源未提供 Volume 列"
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

    # 量价背离检测已移除 (裸K流派边界, 2026-10-24)

    return result


# ==============================================================================
# 5c. 风控层：结构偏向 / 关键位分级 / 交易计划
# ==============================================================================

def dow_bias_count(swings: list) -> int:
    """已确认拐点的标签计数偏向（H1 基准实现，供对照使用）"""
    if len(swings) < 2:
        return 0
    recent = swings[-4:] if len(swings) >= 4 else swings
    labels = [s.get("dow", "") for s in recent]
    bull = labels.count("HH") + labels.count("HL")
    bear = labels.count("LH") + labels.count("LL")
    if bull > bear and bull >= 2:
        return 1
    if bear > bull and bear >= 2:
        return -1
    return 0


def dow_bias(swings: list, df: pd.DataFrame = None, buffer_k: float = 0.0) -> int:
    """
    结构偏向判定：+1 多 / -1 空 / 0 中性。

    当提供 df 时，采用 H2 结构突破原则（收盘价穿越最新已确认波段高低点）：
      1. 收盘突破最新确认的高点（+缓冲）-> +1 (多头结构确立)
      2. 收盘跌破最新确认的低点（-缓冲）-> -1 (空头结构确立)
      3. 处于高低点之间时，遵循道氏第5条「趋势持续直到反转」，沿用最近一次突破建立的偏向

    当 df 为 None 或数据不足时，优雅回退到 dow_bias_count 标签计数，保持 100% 向后兼容。
    """
    if df is None or len(df) == 0:
        return dow_bias_count(swings)
    if len(swings) < 2:
        return 0

    highs = [s for s in swings if s.get("type") == "High"]
    lows = [s for s in swings if s.get("type") == "Low"]
    if not highs or not lows:
        return dow_bias_count(swings)

    p_h = highs[-1]["price"]
    p_l = lows[-1]["price"]
    bar_rng = median_bar_range(df, n=14) if buffer_k > 0 else 0.0
    buf = buffer_k * bar_rng
    c_close = float(df['Close'].iloc[-1])

    # 1. 当前K线直接穿越最新已确认高低点
    if c_close > p_h + buf:
        return 1
    if c_close < p_l - buf:
        return -1

    # 2. 回溯寻找自当前参考区间以来最近一次有效穿越
    # min_idx 取 min(highs[-1]['index'], lows[-1]['index'])，覆盖到前一个极值点，
    # 确保价格在突破后回调（但未破反向支撑）期间保持趋势方向（道氏第5条）
    closes = df['Close'].to_numpy()
    min_idx = min(highs[-1].get("index", 0), lows[-1].get("index", 0))
    for idx in range(len(closes) - 2, max(0, min_idx - 1), -1):
        c = closes[idx]
        if c > p_h + buf:
            return 1
        if c < p_l - buf:
            return -1

    # 3. 结构特征推导：最新拐点是 High 还是 Low
    last_sw = swings[-1]
    if last_sw.get("type") == "High":
        # 最新拐点是高点，价格在回调中
        if len(highs) >= 2 and highs[-1]["price"] > highs[-2]["price"]:
            # 形成了 Higher High，且未跌破当前低点支撑 -> 多头结构回调中
            return 1
        if len(highs) >= 2 and highs[-1]["price"] < highs[-2]["price"] and len(lows) >= 2 and lows[-1]["price"] <= lows[-2]["price"]:
            # Lower High + Lower Low -> 空头结构
            return -1
    else:
        # 最新拐点是低点，价格在反弹中
        if len(lows) >= 2 and lows[-1]["price"] < lows[-2]["price"]:
            # 形成了 Lower Low，且未突破当前高点阻力 -> 空头结构反弹中
            return -1
        if len(lows) >= 2 and lows[-1]["price"] > lows[-2]["price"] and len(highs) >= 2 and highs[-1]["price"] >= highs[-2]["price"]:
            # Higher Low + Higher High -> 多头结构
            return 1

    # 4. 前两组结构对比
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1]["price"] >= highs[-2]["price"] and lows[-1]["price"] >= lows[-2]["price"]:
            return 1
        if highs[-1]["price"] <= highs[-2]["price"] and lows[-1]["price"] <= lows[-2]["price"]:
            return -1

    # 5. 默认回退
    return dow_bias_count(swings)



def build_key_levels(levels: dict, swings_mtf: list, swings_ltf: list,
                     gaps: list, clusters: list, bar_rng: float) -> list:
    """
    [P1-2] 关键位分级 + 归并去重。
    原实现把五类异质水平平铺成一个 list（约 20 个、含 8 个重复），
    每个位都允许触发 2B，导致「流动性猎杀」几乎每次扫描都出现。
    现在：带 tier、按 tier 优先归并、记录同源汇聚。
    """
    k = max(bar_rng * MERGE_LEVEL_K, 0.01)
    raw = []

    def add(price, kind, src):
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        if p > 0:
            raw.append({"price": p, "kind": kind, "src": src})

    for s in (swings_mtf or [])[-4:]:
        add(s.get("price"), "swing", f"MTF-{s.get('dow', '')}")
    for s in (swings_ltf or [])[-4:]:
        add(s.get("price"), "swing", f"LTF-{s.get('dow', '')}")
    for key in ("PDH", "PDL", "ORH", "ORL"):
        if key in levels:
            add(levels[key], "day_extreme", key)
    for g in (gaps or [])[-3:]:
        add(g.get("level"), "gap", "Gap")
    for f, p in levels.get("fib_retracements", {}).items():
        add(p, "fib", f"Fib{f}")
    for cl in (clusters or []):
        add(cl.get("center"), "cluster", "Cluster")

    raw.sort(key=lambda x: (-LEVEL_TIERS[x["kind"]], x["price"]))
    merged = []
    for lv in raw:
        hit = next((m for m in merged if abs(lv["price"] - m["price"]) <= k), None)
        if hit:
            hit.setdefault("confluence", []).append(lv["src"])
        else:
            merged.append(dict(lv))

    merged.sort(key=lambda x: x["price"])
    return merged


def _is_last_bar_open(df: pd.DataFrame, now=None) -> bool:
    """
    [P1-4] 判断末根 K 线是否仍在运行。
    仅对日内周期判断：日线及以上无法从 OHLC 推断收盘状态，返回 False 避免误判。
    """
    if df is None or len(df) < 2:
        return False
    try:
        bar_min = _infer_bar_minutes(df)
        if bar_min >= 1440:
            return False
        now = now or pd.Timestamp.now(tz=df.index.tz)
        return bool(now < df.index[-1] + pd.Timedelta(minutes=bar_min))
    except Exception:
        return False


def build_trade_plan(mode: str, price: float, swings_mtf: list, swings_ltf: list,
                     key_levels: list, levels: dict, bar_rng: float,
                     bias_htf: int, bias_mtf: int, equity: float = None,
                     risk_pct: float = DEFAULT_RISK_PCT,
                     min_rr: float = MIN_RR,
                     market_dow: dict = None,
                     df_mtf: pd.DataFrame = None) -> dict:
    """
    [P1-1] 把「止损 / 目标位 / 盈亏比 / 仓位」从 SKILL.md 的文档承诺变成代码闸门。
    新增大盘道氏相互印证硬闸门与艾略特波浪位置狙击过滤。
    RR 低于 min_rr 时计划直接标记为不成立 —— 不再依赖 LLM 心算。
    """
    plan = {
        "bias_htf": bias_htf, "bias_mtf": bias_mtf, "side": None, "status": "",
        "entry": None, "entry_type": "", "entry_trigger": None,
        "stop": None, "stop_reason": "", "tp1": None, "tp2": None,
        "rr1": None, "rr2": None, "risk_per_share": None,
        "shares": None, "risk_budget": None, "risk_pct": risk_pct, "min_rr": min_rr,
    }

    if bias_htf * bias_mtf < 0:
        plan["status"] = "⏸️ HTF 与 MTF 结构相反（趋势冲突），不出计划"
        return plan
    bias = bias_htf if bias_htf != 0 else bias_mtf
    if bias == 0:
        plan["status"] = "⏸️ 结构无方向（震荡），不出计划"
        return plan

    side = "long" if bias > 0 else "short"
    plan["side"] = side

    # 1. 大盘道氏指数相互印证硬闸门 (Market Confirmation Gate)
    if market_dow is not None and side == "long":
        try:
            import strategy_layer as sl
            m_gate = sl.verify_market_dow_gate(market_dow, side=1)
            if not m_gate.get("passed", True):
                plan["status"] = f"❌ 大盘指数处于空头结构/破位，多头交易被风控硬闸门驳回（rejected_by_market_dow: {m_gate.get('reason')}）"
                plan["rejected_by"] = "rejected_by_market_dow"
                plan["market_dow_gate"] = m_gate
                return plan
        except Exception:
            pass

    # 2. 艾略特波浪位置狙击与冲顶诱多拦截 (Wave-Position Sniper)
    wave_eval = evaluate_wave_position(swings_mtf, df=df_mtf, current_price=price)
    plan["wave_eval"] = wave_eval
    if side == "long":
        if wave_eval.get("signal") == "wave5_exhaustion":
            plan["status"] = f"❌ 识别为第5浪冲顶诱多/量能衰竭晚期，严格禁止追多（rejected_by_wave5_exhaustion: {wave_eval.get('reason')}）"
            plan["rejected_by"] = "rejected_by_wave5_exhaustion"
            return plan
        elif wave_eval.get("signal") == "wave3_breakout":
            plan["wave_note"] = f"⭐ 第3浪主升突破: {wave_eval.get('reason')}"
        elif wave_eval.get("signal") == "wave_c_reversal":
            plan["wave_note"] = f"🎯 C浪终结反转: {wave_eval.get('reason')}"

    conf = [s for s in (swings_ltf or []) if s.get("confirmed")]
    if len(conf) < 2:
        conf = [s for s in (swings_mtf or []) if s.get("confirmed")]
    if not conf:
        plan["status"] = "❌ 无已确认拐点，止损无结构锚点"
        return plan

    buf = STOP_BUFFER_K * bar_rng

    # ---------- 止损：结构证伪位 ----------
    if side == "long":
        anchors = [s["price"] for s in conf if s["type"] == "Low"]
        if not anchors:
            plan["status"] = "❌ 近期无已确认波段低点，止损无锚点"
            return plan
        anchor = anchors[-1]          # 最近一个已确认波段低点（趋势中的最后一个 HL）
        if anchor >= price:
            plan["status"] = f"❌ 现价 ${price:.2f} 已跌破最近波段低点 ${anchor:.2f}，多头结构失效"
            return plan
        stop = round(anchor - buf, 2)
        plan["stop_reason"] = (f"最近已确认波段低点 ${anchor:.2f} 外侧 {buf:.2f}"
                               f"（{STOP_BUFFER_K}×中位振幅 ${bar_rng:.2f}）")
    else:
        anchors = [s["price"] for s in conf if s["type"] == "High"]
        if not anchors:
            plan["status"] = "❌ 近期无已确认波段高点，止损无锚点"
            return plan
        anchor = anchors[-1]          # 最近一个已确认波段高点
        if anchor <= price:
            plan["status"] = f"❌ 现价 ${price:.2f} 已突破最近波段高点 ${anchor:.2f}，空头结构失效"
            return plan
        stop = round(anchor + buf, 2)
        plan["stop_reason"] = (f"最近已确认波段高点 ${anchor:.2f} 外侧 {buf:.2f}"
                               f"（{STOP_BUFFER_K}×中位振幅 ${bar_rng:.2f}）")

    # ---------- 入场 ----------
    if side == "long":
        supports = [lv for lv in key_levels
                    if stop < lv["price"] < price
                    and LEVEL_TIERS[lv["kind"]] >= 2
                    and (price - lv["price"]) / price <= 0.05]
        if supports:
            best = max(supports, key=lambda x: x["price"])
            entry = round(best["price"], 2)
            entry_type = f"限价回踩 {best['kind']}({best['src']})"
        else:
            entry, entry_type = round(price, 2), "现价（无临近支撑可用）"

        trig = [lv["price"] for lv in key_levels
                if lv["price"] > price and LEVEL_TIERS[lv["kind"]] >= 3]
        plan["entry_trigger"] = round(min(trig), 2) if trig else None

        cands = [lv["price"] for lv in key_levels
                 if lv["price"] >= entry + 0.5 * bar_rng and LEVEL_TIERS[lv["kind"]] >= 2]
        tp1 = round(min(cands), 2) if cands else None

        ext = levels.get("fib_extensions", {}).get("1.618")
        if ext and tp1 and float(ext) > tp1:
            tp2 = round(float(ext), 2)
        elif tp1:
            far = [lv["price"] for lv in key_levels
                   if lv["price"] > tp1 and LEVEL_TIERS[lv["kind"]] >= 3]
            tp2 = round(min(far), 2) if far else None
        else:
            tp2 = None
    else:
        resists = [lv for lv in key_levels
                   if price < lv["price"] < stop
                   and LEVEL_TIERS[lv["kind"]] >= 2
                   and (lv["price"] - price) / price <= 0.05]
        if resists:
            best = min(resists, key=lambda x: x["price"])
            entry = round(best["price"], 2)
            entry_type = f"限价反抽 {best['kind']}({best['src']})"
        else:
            entry, entry_type = round(price, 2), "现价（无临近阻力可用）"

        trig = [lv["price"] for lv in key_levels
                if lv["price"] < price and LEVEL_TIERS[lv["kind"]] >= 3]
        plan["entry_trigger"] = round(max(trig), 2) if trig else None

        cands = [lv["price"] for lv in key_levels
                 if lv["price"] <= entry - 0.5 * bar_rng and LEVEL_TIERS[lv["kind"]] >= 2]
        tp1 = round(max(cands), 2) if cands else None

        ext = levels.get("fib_extensions", {}).get("1.618")
        if ext and tp1 and float(ext) < tp1:
            tp2 = round(float(ext), 2)
        elif tp1:
            far = [lv["price"] for lv in key_levels
                   if lv["price"] < tp1 and LEVEL_TIERS[lv["kind"]] >= 3]
            tp2 = round(max(far), 2) if far else None
        else:
            tp2 = None

    # ---------- 盈亏比与闸门 ----------
    risk = abs(entry - stop)
    plan.update({"entry": entry, "entry_type": entry_type, "stop": stop,
                 "tp1": tp1, "tp2": tp2, "risk_per_share": round(risk, 2),
                 "stop_distance_pct": round(risk / entry * 100, 2) if entry else None})

    if risk <= 0:
        plan["status"] = "❌ 入场与止损重合，计划无效"
        return plan

    if tp1 is not None:
        plan["rr1"] = round(abs(tp1 - entry) / risk, 2)
    if tp2 is not None:
        plan["rr2"] = round(abs(tp2 - entry) / risk, 2)

    if equity:
        budget = equity * risk_pct
        plan["risk_budget"] = round(budget, 2)
        plan["shares"] = int(budget // risk)

    if plan["rr1"] is None:
        plan["status"] = "❌ 前方无有效结构目标位，RR 无法成立"
    elif plan["rr1"] < min_rr:
        plan["status"] = f"❌ RR 1:{plan['rr1']:.2f} < 1:{min_rr:.0f}，按规则放弃"
    else:
        plan["status"] = f"✅ 可执行（RR 1:{plan['rr1']:.2f}）"

    return plan


def _print_trade_plan(plan: dict) -> None:
    """按 SKILL.md 的「狙击手交易计划」表结构输出，数值全部来自代码计算"""
    def side_cn(s):
        return "做多 (Long)" if s == "long" else ("做空 (Short)" if s == "short" else "观望 (Wait)")

    print("\n[八、 狙击手交易计划（风控闸门，RR 下限 1:2 由代码强制）]")
    print(f"  • 结构偏向: HTF {plan['bias_htf']:+d} / MTF {plan['bias_mtf']:+d}"
          f"（+1 多 / -1 空 / 0 中性）")
    print(f"  • 交易方向: {side_cn(plan['side'])}")

    if plan["entry"] is not None:
        mark = "  ← 当前计划入场" if plan["entry_type"].startswith("现价") else ""
        print(f"  • 左侧埋伏入场区: ${plan['entry']:.2f}（{plan['entry_type']}）{mark}")
    if plan["entry_trigger"] is not None:
        verb = "收盘突破" if plan["side"] == "long" else "收盘跌破"
        print(f"  • 右侧确认入场位: {verb} ${plan['entry_trigger']:.2f}")
    if plan["stop"] is not None:
        print(f"  • 硬性证伪止损位: ${plan['stop']:.2f}")
        if plan["stop_reason"]:
            print(f"        依据: {plan['stop_reason']}")
    if plan["tp1"] is not None:
        print(f"  • 第一止盈位 TP1: ${plan['tp1']:.2f}（结构位，建议减仓 50% 并推保本损）")
    else:
        print("  • 第一止盈位 TP1: N/A（前方无有效结构位）")
    if plan["tp2"] is not None:
        print(f"  • 第二止盈位 TP2: ${plan['tp2']:.2f}（斐波那契 1.618 扩展 / 远端结构位）")
    else:
        print("  • 第二止盈位 TP2: N/A")

    if plan["rr1"] is not None:
        rr2 = f" / TP2 1:{plan['rr2']:.2f}" if plan["rr2"] is not None else ""
        pct = f"，占入场价 {plan['stop_distance_pct']:.2f}%" if plan.get("stop_distance_pct") else ""
        print(f"  • 预估盈亏比: TP1 1:{plan['rr1']:.2f}{rr2}"
              f"（下限 1:{plan['min_rr']:.0f}）"
              f"  每股风险 ${plan['risk_per_share']:.2f}{pct}")
    if plan["shares"] is not None:
        print(f"  • 建议仓位: 风险预算 ${plan['risk_budget']:,.2f}"
              f"（{plan['risk_pct']:.1%}）÷ 每股风险 ${plan['risk_per_share']:.2f}"
              f" → {plan['shares']} 股")
    elif plan["risk_per_share"] is not None:
        print("  • 建议仓位: 未提供 --equity，跳过股数计算"
              "（可用 --equity <账户规模> --risk-pct 0.01 启用）")

    if plan.get("wave_note"):
        print(f"  • 波浪狙击特征: {plan['wave_note']}")
    if plan.get("wave_eval"):
        wp = plan["wave_eval"].get("wave_pattern", {})
        if wp.get("pattern") and wp.get("pattern") != "unclassified":
            print(f"  • 艾略特浪型: {wp.get('summary', '')}")

    print(f"  • 结论: {plan['status']}")


def _write_json(path: str, payload: dict) -> bool:
    """[P3-2] 结构化输出，供回测/流水线消费"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[JSON] 结构化结果已写入 {path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[错误] 写入 JSON 失败：{type(e).__name__}: {e}", file=sys.stderr)
        return False


# ==============================================================================
# 6. 主调度与结构化输出
# ==============================================================================

def analyze_single_ticker(ticker: str, mode: str, market_dow: dict = None,
                          is_watchlist: bool = False, json_out: str = None,
                          equity: float = None, risk_pct: float = DEFAULT_RISK_PCT):
    """对单一标的执行完整三级周期分析并输出。is_watchlist=True 时输出精简版概览。"""
    config = TIMEFRAME_CONFIGS[mode]
    htf_cfg, mtf_cfg, ltf_cfg = config["HTF"], config["MTF"], config["LTF"]

    # ---- 拉取三级 K 线 ----
    df_htf = fetch_klines(ticker, htf_cfg["interval"], htf_cfg["period"], htf_cfg["prepost"])
    df_mtf = fetch_klines(ticker, mtf_cfg["interval"], mtf_cfg["period"], mtf_cfg["prepost"])
    df_ltf = fetch_klines(ticker, ltf_cfg["interval"], ltf_cfg["period"], ltf_cfg["prepost"])

    if df_mtf.empty:
        print(f"[严重错误] 未能获取 {ticker} 的核心K线数据，分析中止。", file=sys.stderr)
        return "failed"

    # ---- 拐点提取（[FIX-1] 区分已确认 / 未确认）----
    swings_htf, pend_htf = _split_swings(df_htf, htf_cfg["order"])
    swings_mtf, pend_mtf = _split_swings(df_mtf, mtf_cfg["order"])
    swings_ltf, pend_ltf = _split_swings(df_ltf, ltf_cfg["order"])

    status = "ok" if (not df_htf.empty and not df_ltf.empty) else "partial"

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
        # 取最新一根K线的信号（原实现取 pa[0]，是最早的一条）
        pa_summary = pa[-1]["pattern"][:20] if pa else "无信号"
        # 距最近拐点
        nearest = ""
        if recent_mtf:
            ls = recent_mtf[-1]
            d_pct = (latest_price - ls["price"]) / ls["price"] * 100
            nearest = f"距{ls['dow']} ${ls['price']:.2f} → {d_pct:+.1f}%"
        pend_note = f" | ⏳待确认 {pend_mtf[-1]['type']} ${pend_mtf[-1]['price']:.2f}" if pend_mtf else ""
        print(f"  {ticker:<6} ${latest_price:<8.2f} | HTF: {htf_phase} | MTF: {mtf_phase} | "
              f"PA: {pa_summary} | {nearest}{pend_note}")
        return status

    # ===================== 完整分析模式 =====================
    print("\n" + "=" * 70)
    print(f"🚀 【美股纯技术流分析引擎 v2.2】标的: {ticker} | 模式: {mode.upper()}")
    print("=" * 70)

    # ---- [一] 大盘道氏验证 ----
    if market_dow:
        print("\n[一、 道氏大盘指数相互验证 (SPY / QQQ / DIA)]")
        for idx_name, info in market_dow.items():
            htf_sw = " -> ".join(info["htf"].get("swings", [])) or "无显著波段"
            mtf_sw = " -> ".join(info["mtf"].get("swings", [])) or "无显著波段"
            print(f"  • {idx_name:<4} 最新价: {_fmt_price(info['htf']['latest_price'])} | "
                  f"HTF: {info['htf'].get('phase', '') or '数据不可用'}")
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
    if pend_htf:
        print(_fmt_pending(pend_htf))

    # MTF
    print(f"  ▶ MTF ({mtf_cfg['interval']} - {mtf_cfg['desc']}) 道氏阶段: {mtf_phase}")
    for s in recent_mtf:
        print(f"      - {str(s['time'])[:16]} | {s['dow']} ({s['type']}): ${s['price']:.2f}")
    if pend_mtf:
        print(_fmt_pending(pend_mtf))

    # [P2-8] LTF
    print(f"  ▶ LTF ({ltf_cfg['interval']} - {ltf_cfg['desc']}) 道氏阶段: {ltf_phase}")
    if recent_ltf:
        for s in recent_ltf:
            print(f"      - {str(s['time'])[:16]} | {s['dow']} ({s['type']}): ${s['price']:.2f}")
    else:
        print("      - 拐点数据不足")
    if pend_ltf:
        print(_fmt_pending(pend_ltf))

    # ---- [二b] MSS 结构转换 [P1-5] ----
    print(f"\n[二b、 市场结构转换检测 (MSS)]")
    mss_mtf = detect_mss(swings_mtf, df=df_mtf)
    mss_ltf = detect_mss(swings_ltf, df=df_ltf)
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
    print("  · 基准腿取自【已确认】拐点；未确认拐点不参与结构测量")
    if "base_swing" in levels:
        bs = levels["base_swing"]
        print(f"  • 测量主基准波段: {bs['from']} -> {bs['to']} (幅度: ${bs['diff']})")
        print(f"  • 斐波那契回撤参考:")
        for fib, p in levels.get("fib_retracements", {}).items():
            print(f"      - Fib {fib}: ${p:.2f}")
        print(f"  • 顺势扩展目标位 (Extension):")
        for ext, p in levels.get("fib_extensions", {}).items():
            print(f"      - Ext {ext}: ${p:.2f}")
    else:
        print("  • 已确认拐点不足 2 个，无法确定基准波段（斐波那契坐标尺跳过）")

    # [P0-3] 多腿 Fib + 共振区
    multi_fib = calc_multi_leg_fib(recent_mtf, tolerance_pct=CLUSTER_TOLERANCE_PCT)
    if multi_fib["leg_ratios"]:
        print(f"\n  • 多腿波段比例（艾略特验证）:")
        for lr in multi_fib["leg_ratios"]:
            print(f"      - 当前腿: {lr['current']}")
            print(f"        参照腿: {lr['reference']}")
            print(f"        比例: {lr['ratio']}  {lr['match']}")
    if multi_fib["clusters"]:
        print(f"\n  • 🎯 斐波那契共振区 (Fibonacci Cluster):")
        for cl in multi_fib["clusters"]:
            print(f"      - {cl['strength']} 区间: {cl['zone']} (中枢 ${cl['center']:.2f}, "
                  f"{cl['count']}个Fib汇聚 / 来自{cl.get('legs', 1)}条腿)")
    else:
        print(f"\n  • 无有效共振区（容差 {CLUSTER_TOLERANCE_PCT:.0%} 内无多重 Fib 汇聚）")

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
        if levels.get("session"):
            print(f"      · 会话日期: {levels['session']}")
        for key, label in [("PDH", "昨日最高价"), ("PDL", "昨日最低价"), ("PDC", "昨日收盘价"),
                           ("PMH", "当日盘前最高"), ("PML", "当日盘前最低"),
                           ("ORH", "开盘15分钟高"), ("ORL", "开盘15分钟低")]:
            if key in levels:
                print(f"      - {label} ({key}): ${levels[key]:.2f}")
            else:
                # [FIX-2] 数据不足时明确输出 N/A，不给近似值
                print(f"      - {label} ({key}): N/A（数据不足）")
        for reason in levels.get("intraday_missing", []):
            print(f"      ⚠️ 未计算: {reason}")

    # ---- [六] 成交量分析 [P0-2] ----
    print("\n[六、 成交量结构分析 (Volume Analysis)]")
    vol_mtf = analyze_volume(df_mtf)
    vol_ltf = analyze_volume(df_ltf)

    if vol_mtf.get("available", True):
        print(f"  ▶ MTF 近20根均量: {vol_mtf['avg_volume']:,}")
        for bar in vol_mtf.get("recent_bars", []):
            print(f"      - [{bar['time']}] {bar['candle']} | 量: {bar['volume']:,} ({bar['ratio']}x) {bar['tag']}")
    else:
        print(f"  ▶ MTF 成交量分析跳过：{vol_mtf.get('reason', '不可用')}")

    if vol_ltf.get("available", True):
        print(f"  ▶ LTF 近20根均量: {vol_ltf['avg_volume']:,}")
    else:
        print(f"  ▶ LTF 成交量分析跳过：{vol_ltf.get('reason', '不可用')}")

    # ---- [七] 裸K PA 扫描 [P1-2 分级 / P1-3 位置过滤 / P1-4 剔除未收盘K线] ----
    print("\n[七、 LTF 周期裸K价格行为异动扫描]")

    bar_rng_ltf = median_bar_range(df_ltf)
    bar_rng_mtf = median_bar_range(df_mtf)

    # 关键位：带 tier + 归并去重
    key_levels = build_key_levels(levels, recent_mtf, recent_ltf,
                                  detected_gaps, multi_fib.get("clusters", []),
                                  bar_rng_ltf)

    pa_df = df_ltf
    if _is_last_bar_open(df_ltf):
        pa_df = df_ltf.iloc[:-1]
        print(f"  ℹ️ 末根K线（{str(df_ltf.index[-1])[:16]}）仍在运行，已从形态扫描中剔除，避免形态重绘")

    print(f"  ℹ️ 关键位 {len(key_levels)} 个（已按 tier 归并去重）；"
          f"形态须落在关键位 ±{LOCATION_GATE_K}×中位振幅（${bar_rng_ltf:.2f}）内")

    pa_signals = scan_price_action(pa_df, key_levels=key_levels, bar_rng=bar_rng_ltf)
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
        print("  • 最近5根K线在关键位上无标准极限K线形态（脱离位置的形态已过滤）。")

    # ---- [八] 交易计划与风控闸门 [P1-1] ----
    price_now = float(df_mtf['Close'].iloc[-1])
    plan = build_trade_plan(
        mode=mode, price=price_now,
        swings_mtf=swings_mtf, swings_ltf=swings_ltf,
        key_levels=key_levels, levels=levels, bar_rng=bar_rng_ltf,
        bias_htf=dow_bias(swings_htf, df=df_htf) if df_htf is not None and len(df_htf) else dow_bias(swings_htf),
        bias_mtf=dow_bias(swings_mtf, df=df_mtf) if df_mtf is not None and len(df_mtf) else dow_bias(swings_mtf),
        equity=equity, risk_pct=risk_pct,
        market_dow=market_dow,
        df_mtf=df_mtf,
    )
    _print_trade_plan(plan)

    if status == "partial":
        print()
        if df_htf.empty:
            print("  ⚠️ HTF 数据缺失：大势验证不可用，本报告结论不完整", file=sys.stderr)
        if df_ltf.empty:
            print("  ⚠️ LTF 数据缺失：裸K信号不可用，本报告结论不完整", file=sys.stderr)

    print("\n" + "=" * 70)
    print("🎯 数据处理完成，请结合以上客观数据执行【道氏+波浪+裸K】报告生成。")
    print("=" * 70)

    # ---- [P3-2] 结构化输出 ----
    if json_out:
        payload = {
            "ticker": ticker,
            "mode": mode,
            "status": status,
            "generated_at": str(pd.Timestamp.now(tz=EXCHANGE_TZ)),
            "price": price_now,
            "bar_range": {"ltf": round(bar_rng_ltf, 4), "mtf": round(bar_rng_mtf, 4)},
            "phase": {"htf": htf_phase, "mtf": mtf_phase, "ltf": ltf_phase},
            "bias": {"htf": plan["bias_htf"], "mtf": plan["bias_mtf"]},
            "swings": {
                "htf": recent_htf, "mtf": recent_mtf, "ltf": recent_ltf,
                "pending": {"htf": pend_htf, "mtf": pend_mtf, "ltf": pend_ltf},
            },
            "levels": {k: v for k, v in levels.items() if v is not None},
            "key_levels": key_levels,
            "gaps": detected_gaps,
            "volume": {"mtf": vol_mtf, "ltf": vol_ltf},
            "pa_signals": pa_signals,
            "plan": plan,
            "wave_analysis": plan.get("wave_eval"),
        }
        _write_json(json_out, payload)

    return status


def _print_run_summary(results: dict) -> int:
    """
    [FIX-4] 统一收尾：汇报取数失败与降级标的，并返回进程退出码。
    0 = 全部正常；1 = 存在失败标的（含单一标的失败）；降级仅告警不改退出码。
    """
    failed = [k for k, v in results.items() if v == "failed"]
    partial = [k for k, v in results.items() if v == "partial"]

    if _FETCH_FAILURES:
        skips = [x for x in _FETCH_FAILURES if "已熔断（连续限流），跳过请求" in x]
        nets = [x for x in _FETCH_FAILURES if x not in skips]
        extra = f"，熔断跳过 {len(skips)} 次" if skips else ""
        print(f"\n[取数失败汇总] 网络失败 {len(nets)} 次{extra}", file=sys.stderr)
        for item in nets[:12]:
            print(f"  - {item}", file=sys.stderr)
        if len(nets) > 12:
            print(f"  ... 其余 {len(nets) - 12} 条略", file=sys.stderr)

    if partial:
        print(f"[运行结果] ⚠️ 降级标的（部分周期数据缺失，结论不完整）: {', '.join(partial)}",
              file=sys.stderr)
    if failed:
        print(f"[运行结果] ❌ 失败标的: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="美股纯技术流（道氏+波浪+裸K）分析引擎 v2.2")
    parser.add_argument("--ticker", type=str, default="",
                        help="美股股票代码 (如 NVDA, AAPL, TSM)")
    parser.add_argument("--mode", type=str, choices=["swing", "day"], default="swing",
                        help="交易模式: swing (波段) 或 day (日内)")
    parser.add_argument("--watchlist", type=str, default="",
                        help="批量扫描自选池，逗号分隔 (如 AAPL,NVDA,TSLA,META)")
    parser.add_argument("--watchlist-file", type=str, default="",
                        help="从文件读取自选池标的 (如 config/watchlist.txt)")
    parser.add_argument("--json", type=str, default="", dest="json_out",
                        help="将结构化结果写入指定 JSON 文件（单标的模式）")
    parser.add_argument("--equity", type=float, default=None,
                        help="账户规模（美元），提供后按 --risk-pct 计算建议股数")
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT,
                        help=f"单笔风险占账户比例，默认 {DEFAULT_RISK_PCT}")
    args = parser.parse_args()

    reset_fetch_state()
    mode = args.mode.lower()

    if args.risk_pct <= 0 or args.risk_pct > 0.1:
        print("[错误] --risk-pct 应在 (0, 0.1] 区间内，例如 0.01 表示 1%", file=sys.stderr)
        return 1
    if args.equity is not None and args.equity <= 0:
        print("[错误] --equity 必须大于 0", file=sys.stderr)
        return 1

    # [P2-7] 批量 Watchlist 模式
    tickers = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]
    if args.watchlist_file:
        wf_path = args.watchlist_file
        if not os.path.exists(wf_path) and os.path.exists(os.path.join("config", wf_path)):
            wf_path = os.path.join("config", wf_path)
        if os.path.exists(wf_path):
            with open(wf_path, encoding="utf-8") as f:
                for ln in f:
                    clean = ln.split("#")[0].strip().upper()
                    if clean:
                        tickers.append(clean)
        else:
            print(f"[错误] 未找到自选池文件: {args.watchlist_file}", file=sys.stderr)
            return 1

    if tickers:
        if args.json_out:
            print("[提示] --json 仅支持单标的模式，本次批量扫描不输出 JSON", file=sys.stderr)
        print("=" * 70)
        print(f"📋 【Watchlist 快速扫描】模式: {mode.upper()} | 标的数: {len(tickers)}")
        print("=" * 70)
        market_dow = check_dow_indices(mode)
        for idx_name, info in market_dow.items():
            phase = info["htf"].get("phase", "") or "数据不可用"
            print(f"  大盘 {idx_name:<4} {_fmt_price(info['htf']['latest_price'])} | {phase}")
        print("-" * 70)

        results = {}
        for t in tickers:
            try:
                results[t] = analyze_single_ticker(t, mode, market_dow=market_dow,
                                                   is_watchlist=True) or "partial"
            except Exception as e:
                # 单个标的异常不再中断整批扫描
                results[t] = "failed"
                print(f"[错误] {t} 分析异常: {type(e).__name__}: {e}", file=sys.stderr)
        print("=" * 70)
        return _print_run_summary(results)

    # 单一标的模式
    if not args.ticker:
        print("[错误] 请指定 --ticker 或 --watchlist 参数", file=sys.stderr)
        return 1

    ticker = args.ticker.upper()
    market_dow = check_dow_indices(mode)
    status = analyze_single_ticker(ticker, mode, market_dow=market_dow, is_watchlist=False,
                                   json_out=args.json_out or None,
                                   equity=args.equity, risk_pct=args.risk_pct)
    return _print_run_summary({ticker: status or "failed"})


if __name__ == "__main__":
    sys.exit(main())
