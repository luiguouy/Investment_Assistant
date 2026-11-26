# -*- coding: utf-8 -*-
"""
日线波段回测与策略验证引擎 (Zero-Lookahead Daily Swing Backtester)
路径: stage_one/backend/backtest_swing.py

功能:
  1. 可选一键更新回测标的 (BTC-USD, NQmain, ESmain 及大盘参考 SPY/QQQ/IWM) 的最新日线数据
     至 SQLite 缓存 (stock_data.db) 与本地 CSV (stage_one/backend/data/)。
  2. 严格零未来函数 (t 根收盘决策 -> t+1 根开盘入场或因果限价/裸K触发撮合, 同根双触止损优先,
     扣减往返 8 bps 交易摩擦) 在最新日线数据上评测:
     - Track 1 (adapter_direct): strategy_adapter.run_strategy_analysis 直接通过门禁的顺势信号
     - Track 2 (adapter_playbook): 直接信号 + condition_playbook 严苛回踩共振挂单 (15根GTC & 1根日刷)
     - Track 3 (adapter_counter_trend): 新增 counter_trend_playbook 超跌反弹 / 2B 破底翻剧本
     - Track 4 (strategy_core_mss): strategy_core 原生 MSS + 2B/PinBar/吞没 + 分级止盈因果基准
  3. 同步执行全历史逐日不变量审计 (方向一致性、非负价格、盈亏比门禁) 与黑天鹅窗口剔除对比。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from strategy_adapter import _normalize_df, run_strategy_analysis
from strategy_core import pa_engine as pe
from strategy_core import strategy_crypto as sc
from strategy_core import strategy_layer as sl

ROUND_TRIP_FRICTION = 0.0008  # 往返 8 bps (单边 3bps 滑点 + 1bps 规费)

DEFAULT_TARGETS = [
    {"symbol": "BTC-USD", "csv": "BTC-USD.csv", "market": "crypto"},
    {"symbol": "NQmain", "csv": "NQMAIN.csv", "market": "futures"},
    {"symbol": "ESmain", "csv": "ESMAIN.csv", "market": "futures"},
]


# ==============================================================================
# 数据更新与加载
# ==============================================================================

def _merge_and_save_csv(
    path: Path, df_new: pd.DataFrame, date_col_name: str = "Date", replace: bool = False
) -> pd.DataFrame:
    df_n = df_new.rename(columns={
        "date": date_col_name,
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })[[date_col_name, "Open", "High", "Low", "Close", "Volume"]].copy()
    df_n[date_col_name] = df_n[date_col_name].astype(str).str[:10]

    if not replace and path.exists():
        df_old = pd.read_csv(path)
        old_dcol = df_old.columns[0]
        if old_dcol != date_col_name:
            df_old = df_old.rename(columns={old_dcol: date_col_name})
        df_old[date_col_name] = df_old[date_col_name].astype(str).str[:10]
        merged = pd.concat([df_old, df_n], ignore_index=True)
    else:
        merged = df_n

    merged = (
        merged.drop_duplicates(subset=[date_col_name], keep="last")
        .sort_values(date_col_name)
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


async def update_target_datasets() -> Dict[str, Dict[str, Any]]:
    """拉取并更新 BTC-USD, NQmain, ESmain 及 SPY/QQQ/IWM 最新日线数据至 DB 缓存和本地 CSV。"""
    from main import get_klines_df

    summary: Dict[str, Dict[str, Any]] = {}

    # 1. BTC-USD (全量 10y UTC 日线替换，消除历史 US_EAST 时区偏移与跨日重复棒)
    btc_df, src = await get_klines_df("BTC-USD", "1d", full=True, bypass_cache=True)
    m_btc = _merge_and_save_csv(DATA_DIR / "BTC-USD.csv", btc_df, "Datetime", replace=True)
    summary["BTC-USD"] = {
        "source": src,
        "bars": len(m_btc),
        "start": str(m_btc.iloc[0]["Datetime"]),
        "end": str(m_btc.iloc[-1]["Datetime"]),
        "last_close": round(float(m_btc.iloc[-1]["Close"]), 2),
    }

    # 2. NQmain / ESmain (由 DataProviderManager 优先 Yahoo 10y 真实成交量 + 实时行情校准并同步 DB 与 CSV)
    for sym, fname in [("NQmain", "NQMAIN.csv"), ("ESmain", "ESMAIN.csv")]:
        fut_df, src_fut = await get_klines_df(sym, "1d", full=True, bypass_cache=True)
        m_fut = _merge_and_save_csv(DATA_DIR / fname, fut_df, "Date", replace=True)
        summary[sym] = {
            "source": src_fut,
            "bars": len(m_fut),
            "start": str(m_fut.iloc[0]["Date"]),
            "end": str(m_fut.iloc[-1]["Date"]),
            "last_close": round(float(m_fut.iloc[-1]["Close"]), 2),
        }

    # 3. SPY / QQQ / IWM 指数基准同步更新 (保留 2001/2005 起完整历史)
    for idx_sym in ["SPY", "QQQ", "IWM"]:
        df_idx, src_idx = await get_klines_df("us" + idx_sym, "1d", full=True, bypass_cache=True)
        m_idx = _merge_and_save_csv(DATA_DIR / f"{idx_sym}.csv", df_idx, "Date", replace=False)
        summary[idx_sym] = {
            "source": src_idx,
            "bars": len(m_idx),
            "start": str(m_idx.iloc[0]["Date"]),
            "end": str(m_idx.iloc[-1]["Date"]),
            "last_close": round(float(m_idx.iloc[-1]["Close"]), 2),
        }

    return summary


def load_symbol_df(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"找不到历史数据文件: {path}")
    raw = pd.read_csv(path)
    df = _normalize_df(raw)
    if df is None or df.empty:
        raise ValueError(f"数据文件为空或格式异常: {path}")
    return df


# ==============================================================================
# 统计指标与黑天鹅过滤
# ==============================================================================

def _overlaps_black_swan(entry_date: Any, exit_date: Any, is_crypto: bool) -> bool:
    if is_crypto:
        try:
            e_d = pd.to_datetime(entry_date).tz_localize(None) if pd.to_datetime(entry_date).tz is not None else pd.to_datetime(entry_date)
            x_d = pd.to_datetime(exit_date).tz_localize(None) if pd.to_datetime(exit_date).tz is not None else pd.to_datetime(exit_date)
        except Exception:
            return False
        for bs in sc.CRYPTO_BLACK_SWANS:
            s = pd.to_datetime(bs["start"])
            e = pd.to_datetime(bs["end"])
            if not (x_d < s or e_d > e):
                return True
        return False
    return sl.trade_overlaps_stock_black_swan(entry_date, exit_date)


def summarize_trades(
    trades: List[Dict[str, Any]],
    df: pd.DataFrame,
    start_idx: int,
    friction: float = ROUND_TRIP_FRICTION,
    is_crypto: bool = False,
) -> Dict[str, Any]:
    start_close = float(df["Close"].iloc[start_idx])
    end_close = float(df["Close"].iloc[-1])
    bh_return_pct = (end_close / start_close - 1.0) * 100.0 if start_close > 0 else 0.0

    def _calc_subset(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub:
            return {
                "trades": 0,
                "long_trades": 0,
                "short_trades": 0,
                "win_rate_pct": 0.0,
                "long_win_rate_pct": 0.0,
                "short_win_rate_pct": 0.0,
                "gross_pf": 0.0,
                "net_pf": 0.0,
                "total_net_return_pct": 0.0,
                "total_gross_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_r": 0.0,
                "median_r": 0.0,
                "expectancy_net_pct": 0.0,
                "median_mae_pct": 0.0,
                "median_mfe_pct": 0.0,
                "avg_hold_bars": 0.0,
                "exit_reasons": {},
            }

        gross_pnls = np.array([float(t["pnl_pct"]) for t in sub], dtype=float)
        net_pnls = gross_pnls - friction
        r_mults = np.array([float(t["r_multiple"]) for t in sub], dtype=float)
        maes = np.array([float(t["mae"]) * 100.0 for t in sub], dtype=float)
        mfes = np.array([float(t["mfe"]) * 100.0 for t in sub], dtype=float)
        holds = np.array([int(t["hold_bars"]) for t in sub], dtype=float)

        wins = net_pnls > 0
        longs = [t for t in sub if t["side"] == 1]
        shorts = [t for t in sub if t["side"] == -1]

        long_wr = (sum(1 for t in longs if (t["pnl_pct"] - friction) > 0) / len(longs) * 100.0) if longs else 0.0
        short_wr = (sum(1 for t in shorts if (t["pnl_pct"] - friction) > 0) / len(shorts) * 100.0) if shorts else 0.0

        g_pos = float(gross_pnls[gross_pnls > 0].sum())
        g_neg = float(-gross_pnls[gross_pnls < 0].sum())
        n_pos = float(net_pnls[net_pnls > 0].sum())
        n_neg = float(-net_pnls[net_pnls < 0].sum())

        gross_pf = (g_pos / g_neg) if g_neg > 1e-12 else (99.0 if g_pos > 0 else 0.0)
        net_pf = (n_pos / n_neg) if n_neg > 1e-12 else (99.0 if n_pos > 0 else 0.0)

        # 复利权益曲线与最大回撤 (按单笔 100% 名义资金串行复利，非杠杆)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        gross_eq = 1.0
        for gp, npnl in zip(gross_pnls, net_pnls):
            gross_eq *= max(0.01, 1.0 + gp)
            equity *= max(0.01, 1.0 + npnl)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        exit_reasons: Dict[str, int] = {}
        for t in sub:
            r = str(t.get("exit_reason", "unknown"))
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        return {
            "trades": len(sub),
            "long_trades": len(longs),
            "short_trades": len(shorts),
            "win_rate_pct": round(float(wins.mean() * 100.0), 2),
            "long_win_rate_pct": round(long_wr, 2),
            "short_win_rate_pct": round(short_wr, 2),
            "gross_pf": round(gross_pf, 2),
            "net_pf": round(net_pf, 2),
            "total_net_return_pct": round((equity - 1.0) * 100.0, 2),
            "total_gross_return_pct": round((gross_eq - 1.0) * 100.0, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "avg_r": round(float(r_mults.mean()), 3),
            "median_r": round(float(np.median(r_mults)), 3),
            "expectancy_net_pct": round(float(net_pnls.mean() * 100.0), 3),
            "median_mae_pct": round(float(np.median(maes)), 2),
            "median_mfe_pct": round(float(np.median(mfes)), 2),
            "avg_hold_bars": round(float(holds.mean()), 1),
            "exit_reasons": exit_reasons,
        }

    ex_bs = [t for t in trades if not _overlaps_black_swan(t["entry_date"], t["exit_date"], is_crypto)]
    recent = [
        {
            "entry_date": str(t["entry_date"])[:10],
            "exit_date": str(t["exit_date"])[:10],
            "side": "LONG" if t["side"] == 1 else "SHORT",
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "net_pnl_pct": round((t["pnl_pct"] - friction) * 100.0, 2),
            "r_multiple": round(t["r_multiple"], 2),
            "exit_reason": t["exit_reason"],
            "source": t.get("signal_source", ""),
        }
        for t in trades[-5:]
    ]

    return {
        "eval_start": str(df.index[start_idx])[:10],
        "eval_end": str(df.index[-1])[:10],
        "eval_bars": len(df) - start_idx,
        "buy_and_hold_pct": round(bh_return_pct, 2),
        "all": _calc_subset(trades),
        "ex_black_swan": _calc_subset(ex_bs),
        "recent_trades": recent,
    }


# ==============================================================================
# 因果扫描与交易模拟
# ==============================================================================

def _execute_tiered_trade(
    df: pd.DataFrame,
    entry_t: int,
    entry_price: float,
    side: int,
    stop_price: float,
    tp1_price: float,
    tp2_price: Optional[float],
    is_crypto: bool,
    min_rr_guard: float = 0.5,
) -> Dict[str, Any]:
    """调用 strategy_core 的保守分级止盈生命周期模拟。"""
    if entry_price <= 0 or stop_price <= 0 or tp1_price <= 0:
        return {"status": "invalid_levels"}
    if side == 1 and (entry_price <= stop_price or tp1_price <= entry_price):
        return {"status": "gap_invalidated"}
    if side == -1 and (entry_price >= stop_price or tp1_price >= entry_price):
        return {"status": "gap_invalidated"}

    impulse_leg = abs(tp1_price - entry_price)
    if is_crypto:
        return sc.simulate_crypto_trade(
            df=df,
            entry_t=entry_t,
            entry_price=entry_price,
            side=side,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            impulse_leg=impulse_leg,
            max_hold_bars=60,
            min_rr=min_rr_guard,
        )
    return sl.simulate_strategy_trade(
        df=df,
        entry_t=entry_t,
        entry_price=entry_price,
        side=side,
        stop_price=stop_price,
        tp_price=tp1_price,
        max_hold_bars=40,
        min_rr=min_rr_guard,
        use_tiered_tp=True,
        tp2_price=tp2_price,
        impulse_leg=impulse_leg,
    )


def _check_invariants(tp: Dict[str, Any], cur_p: float, bar_idx: int) -> Optional[str]:
    dm = tp.get("direction_mode")
    if tp.get("passed_gates"):
        if dm == "LONG" and not (tp["stop_loss"] < tp["entry"] < tp["tp1"]):
            return f"bar={bar_idx}: LONG passed invariant failed sl={tp['stop_loss']}, entry={tp['entry']}, tp1={tp['tp1']}"
        if dm == "SHORT" and not (tp["tp1"] < tp["entry"] < tp["stop_loss"]):
            return f"bar={bar_idx}: SHORT passed invariant failed tp1={tp['tp1']}, entry={tp['entry']}, sl={tp['stop_loss']}"
    pb = tp.get("condition_playbook")
    if isinstance(pb, dict):
        if dm in ("LONG", "WAIT") and not (pb["stop_loss"] < pb["ideal_entry"] < pb["tp1"]):
            return f"bar={bar_idx}: COND_LONG invariant failed {pb}"
        if dm == "SHORT" and not (pb["tp1"] < pb["ideal_entry"] < pb["stop_loss"]):
            return f"bar={bar_idx}: COND_SHORT invariant failed {pb}"
    ct = tp.get("counter_trend_playbook")
    if isinstance(ct, dict):
        ez = ct["entry_zone"]
        if not (ct["stop_loss"] < ez[0] <= ez[1] < ct["tp1"] < ct["tp2"]):
            return f"bar={bar_idx}: CT invariant failed {ct}"
    return None


def scan_adapter_signals(
    symbol: str, df: pd.DataFrame, market: str, start_idx: int = 250
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """逐根严格因果扫描 strategy_adapter.run_strategy_analysis，缓存决策输出并审计不变量。"""
    plans: Dict[int, Dict[str, Any]] = {}
    invariant_errors: List[str] = []
    n = len(df)
    for t in range(start_idx, n - 1):
        res = run_strategy_analysis(symbol, period="1d", mode="swing", df=df.iloc[:t + 1], market=market)
        if "error" in res:
            invariant_errors.append(f"bar={t}: {res['error']}")
            continue
        tp = res["summary"]["trade_plan"]
        err = _check_invariants(tp, float(res["current_price"]), t)
        if err:
            invariant_errors.append(err)
        plans[t] = {
            "direction_mode": tp.get("direction_mode"),
            "passed_gates": bool(tp.get("passed_gates")),
            "entry": tp.get("entry"),
            "stop_loss": tp.get("stop_loss"),
            "tp1": tp.get("tp1"),
            "tp2": tp.get("tp2"),
            "rr_ratio": tp.get("rr_ratio"),
            "condition_playbook": tp.get("condition_playbook"),
            "counter_trend_playbook": tp.get("counter_trend_playbook"),
        }
    return plans, invariant_errors


def _is_open_end_bar_trade(res: Dict[str, Any], n: int) -> bool:
    """识别在最后一根 K 线 (n-1) 当日入场且尚未经受次日验证的未平仓头寸，避免记为 0-bar time_horizon 伪亏损。"""
    return (
        int(res.get("entry_t", -1)) >= n - 1
        and int(res.get("hold_bars", 0)) == 0
        and str(res.get("exit_reason", "")) in ("time_horizon", "tp_partial_time_horizon")
    )


def sim_adapter_direct(
    df: pd.DataFrame, plans: Dict[int, Dict[str, Any]], start_idx: int, is_crypto: bool
) -> Dict[str, Any]:
    """Track 1: 仅交易 passed_gates == True 的直接现价/突破信号 (t+1 开盘入场)。"""
    trades: List[Dict[str, Any]] = []
    open_at_end: List[Dict[str, Any]] = []
    skipped_gap = 0
    next_free_t = start_idx
    n = len(df)

    for t in range(start_idx, n - 1):
        if t < next_free_t:
            continue
        plan = plans.get(t)
        if not plan or not plan["passed_gates"]:
            continue
        dm = plan["direction_mode"]
        if dm not in ("LONG", "SHORT"):
            continue
        side = 1 if dm == "LONG" else -1
        entry_t = t + 1
        entry_price = float(df["Open"].iloc[entry_t])

        res = _execute_tiered_trade(
            df, entry_t, entry_price, side,
            float(plan["stop_loss"]), float(plan["tp1"]),
            float(plan["tp2"]) if plan.get("tp2") else None,
            is_crypto=is_crypto, min_rr_guard=0.5,
        )
        if res.get("status") != "completed":
            skipped_gap += 1
            continue
        if _is_open_end_bar_trade(res, n):
            open_at_end.append({
                "entry_date": str(df.index[res["entry_t"]])[:10],
                "side": "LONG" if side == 1 else "SHORT",
                "entry_price": res["entry_price"],
                "source": "direct",
            })
            next_free_t = res["exit_t"] + 1
            continue
        res["entry_date"] = df.index[res["entry_t"]]
        res["exit_date"] = df.index[res["exit_t"]]
        res["side"] = side
        res["signal_source"] = "direct"
        trades.append(res)
        next_free_t = res["exit_t"] + 1

    out = summarize_trades(trades, df, start_idx, is_crypto=is_crypto)
    out["skipped_gap"] = skipped_gap
    out["open_at_end"] = open_at_end
    return out


def sim_adapter_playbook(
    df: pd.DataFrame,
    plans: Dict[int, Dict[str, Any]],
    start_idx: int,
    is_crypto: bool,
    max_wait_bars: int = 15,
) -> Dict[str, Any]:
    """Track 2: 直接信号 + condition_playbook 回踩限价挂单 (默认 15 根 GTC 有效期)。"""
    trades: List[Dict[str, Any]] = []
    open_at_end: List[Dict[str, Any]] = []
    skipped_gap = 0
    orders_placed = 0
    orders_filled = 0
    next_free_t = start_idx
    n = len(df)

    for t in range(start_idx, n - 1):
        if t < next_free_t:
            continue
        plan = plans.get(t)
        if not plan:
            continue

        dm = plan["direction_mode"]
        if plan["passed_gates"] and dm in ("LONG", "SHORT"):
            side = 1 if dm == "LONG" else -1
            entry_t = t + 1
            entry_price = float(df["Open"].iloc[entry_t])
            res = _execute_tiered_trade(
                df, entry_t, entry_price, side,
                float(plan["stop_loss"]), float(plan["tp1"]),
                float(plan["tp2"]) if plan.get("tp2") else None,
                is_crypto=is_crypto, min_rr_guard=0.5,
            )
            if res.get("status") != "completed":
                skipped_gap += 1
                continue
            if _is_open_end_bar_trade(res, n):
                open_at_end.append({
                    "entry_date": str(df.index[res["entry_t"]])[:10],
                    "side": "LONG" if side == 1 else "SHORT",
                    "entry_price": res["entry_price"],
                    "source": "direct",
                })
                next_free_t = res["exit_t"] + 1
                continue
            res["entry_date"] = df.index[res["entry_t"]]
            res["exit_date"] = df.index[res["exit_t"]]
            res["side"] = side
            res["signal_source"] = "direct"
            trades.append(res)
            next_free_t = res["exit_t"] + 1
            continue

        pb = plan.get("condition_playbook")
        if isinstance(pb, dict) and pb.get("passed_gates"):
            side = 1 if float(pb["tp1"]) > float(pb["ideal_entry"]) else -1
            orders_placed += 1
            fill = sl.simulate_limit_order(
                df, t, side,
                limit_price=float(pb["ideal_entry"]),
                stop_price=float(pb["stop_loss"]),
                max_wait_bars=max_wait_bars,
                volume_filter=False,
            )
            if not fill.get("filled"):
                # 若在 max_wait_bars=1 模式下未成交，次日即可重新评估新剧本；GTC 模式严格推进至撤单/过期之后防重叠偷看
                if max_wait_bars > 1 and fill.get("cancel_t"):
                    next_free_t = int(fill["cancel_t"]) + 1
                elif max_wait_bars > 1 and fill.get("status") == "unfilled_expired":
                    next_free_t = t + max_wait_bars + 1
                continue

            orders_filled += 1
            res = _execute_tiered_trade(
                df, int(fill["entry_t"]), float(fill["entry_price"]), side,
                float(pb["stop_loss"]), float(pb["tp1"]),
                float(pb["tp2"]) if pb.get("tp2") else None,
                is_crypto=is_crypto, min_rr_guard=1.0,
            )
            if res.get("status") != "completed":
                skipped_gap += 1
                next_free_t = int(fill["entry_t"]) + 1
                continue
            if _is_open_end_bar_trade(res, n):
                open_at_end.append({
                    "entry_date": str(df.index[res["entry_t"]])[:10],
                    "side": "LONG" if side == 1 else "SHORT",
                    "entry_price": res["entry_price"],
                    "source": "condition_playbook",
                })
                next_free_t = res["exit_t"] + 1
                continue
            res["entry_date"] = df.index[res["entry_t"]]
            res["exit_date"] = df.index[res["exit_t"]]
            res["side"] = side
            res["signal_source"] = "condition_playbook"
            trades.append(res)
            next_free_t = res["exit_t"] + 1

    out = summarize_trades(trades, df, start_idx, is_crypto=is_crypto)
    out["skipped_gap"] = skipped_gap
    out["orders_placed"] = orders_placed
    out["orders_filled"] = orders_filled
    out["open_at_end"] = open_at_end
    return out


def sim_adapter_counter_trend(
    df: pd.DataFrame,
    plans: Dict[int, Dict[str, Any]],
    start_idx: int,
    is_crypto: bool,
    include_observe_only: bool = False,
    max_wait_bars: int = 5,
) -> Dict[str, Any]:
    """Track 3: counter_trend_playbook 超跌反弹 / 2B 破底翻剧本测试 (严格因果无重叠限价挂单)。"""
    trades: List[Dict[str, Any]] = []
    open_at_end: List[Dict[str, Any]] = []
    orders_placed = 0
    orders_filled = 0
    next_free_t = start_idx
    n = len(df)

    for t in range(start_idx, n - 1):
        if t < next_free_t:
            continue
        plan = plans.get(t)
        if not plan:
            continue
        ct = plan.get("counter_trend_playbook")
        if not isinstance(ct, dict):
            continue
        if ct.get("mode") != "ACTIVE" and not include_observe_only:
            continue

        orders_placed += 1
        limit_p = float(ct["entry_zone"][1])
        stop_p = float(ct["stop_loss"])
        fill = sl.simulate_limit_order(
            df, t, side=1, limit_price=limit_p, stop_price=stop_p,
            max_wait_bars=max_wait_bars, volume_filter=False,
        )
        if not fill.get("filled"):
            if max_wait_bars > 1 and fill.get("cancel_t"):
                next_free_t = int(fill["cancel_t"]) + 1
            elif max_wait_bars > 1 and fill.get("status") == "unfilled_expired":
                next_free_t = t + max_wait_bars + 1
            continue

        orders_filled += 1
        res = _execute_tiered_trade(
            df, int(fill["entry_t"]), float(fill["entry_price"]), side=1,
            stop_price=stop_p, tp1_price=float(ct["tp1"]), tp2_price=float(ct["tp2"]),
            is_crypto=is_crypto, min_rr_guard=0.5,
        )
        if res.get("status") != "completed":
            next_free_t = int(fill["entry_t"]) + 1
            continue
        if _is_open_end_bar_trade(res, n):
            open_at_end.append({
                "entry_date": str(df.index[res["entry_t"]])[:10],
                "side": "LONG",
                "entry_price": res["entry_price"],
                "source": f"counter_trend({ct.get('mode')})",
            })
            next_free_t = res["exit_t"] + 1
            continue
        res["entry_date"] = df.index[res["entry_t"]]
        res["exit_date"] = df.index[res["exit_t"]]
        res["side"] = 1
        res["signal_source"] = f"counter_trend({ct.get('mode')})"
        trades.append(res)
        next_free_t = res["exit_t"] + 1

    out = summarize_trades(trades, df, start_idx, is_crypto=is_crypto)
    out["orders_placed"] = orders_placed
    out["orders_filled"] = orders_filled
    out["open_at_end"] = open_at_end
    return out


def sim_strategy_core_mss(
    df: pd.DataFrame, start_idx: int, is_crypto: bool
) -> Dict[str, Any]:
    """Track 4: strategy_core 原生 MSS + 2B/PinBar/吞没触发 + 多重门禁 + 分级止盈因果基准。"""
    trades: List[Dict[str, Any]] = []
    open_at_end: List[Dict[str, Any]] = []
    n = len(df)
    next_free_t = start_idx
    ema200 = df["Close"].astype(float).ewm(span=200, adjust=False).mean()
    min_rr = sc.CRYPTO_MIN_RR if is_crypto else 2.5

    # 预聚合全量周线只用于校验日期对齐；实际按 <= k 切片严格重算以保持零未来函数
    signals_detected = 0
    triggered_count = 0

    for t in range(start_idx, n - 2):
        if t < next_free_t:
            continue

        sub = df.iloc[:t + 1]
        swings = pe.extract_swings(sub, order=5, confirmed_only=False)
        conf = [s for s in swings if s.get("confirmed", True)]
        if len(conf) < 4:
            continue

        mss_list = pe.detect_mss(swings, sub)
        side = 0
        # 1. 当日刚好发生位移破位 MSS (bars_ago == 0)
        for ev in reversed(mss_list):
            if ev.get("bars_ago") == 0:
                if "Bullish" in ev.get("type", "") or "多翻" in ev.get("type", ""):
                    side = 1
                elif "Bearish" in ev.get("type", "") or "空翻" in ev.get("type", ""):
                    side = -1
                break

        # 2. 或当日刚好确认了符合当前道氏主偏向的新拐点 (t - 5 == last confirmed swing index)
        if side == 0:
            last_conf_idx = int(conf[-1].get("index", -1))
            if last_conf_idx == t - 5:
                bias = pe.dow_bias(conf, sub)
                if bias in (1, -1):
                    side = bias

        if side == 0:
            continue

        signals_detected += 1

        # 位置门禁: 箱体 & 通道
        box_g = sl.verify_box_gate(df, t, side, swings=conf)
        if not box_g.get("passed", True):
            continue
        chan_g = sl.verify_channel_gate(df, t, side, swings=conf)
        if not chan_g.get("passed", True):
            continue

        if is_crypto:
            lvl = sc.extract_crypto_levels(df, t, side, conf)
            if not lvl:
                continue
            trig = sc.simulate_crypto_trigger_entry(
                df, t, side,
                key_level=lvl["swing_retest"],
                bar_rng=lvl["bar_rng"],
                structural_stop=lvl["structural_stop"],
                max_wait_bars=15,
                use_trend_gate=True,
                disallow_short_engulfing=True,
            )
            if not trig.get("triggered"):
                continue
            triggered_count += 1
            res = sc.simulate_crypto_trade(
                df,
                entry_t=int(trig["entry_t"]),
                entry_price=float(trig["entry_price"]),
                side=side,
                stop_price=float(trig["stop_price"]),
                tp1_price=float(lvl["tp1_target"]),
                tp2_price=float(lvl["tp2_target"]),
                impulse_leg=float(lvl["impulse_leg"]),
                max_hold_bars=60,
                min_rr=min_rr,
            )
        else:
            trend_g = sl.verify_primary_trend_gate(df, side=side, t=t, ema_series=ema200)
            if not trend_g.get("passed", True):
                continue
            lvl = sl.extract_breakout_levels(df, t, side, conf)
            if not lvl:
                continue
            trig = sl.simulate_trigger_entry(
                df, t, side,
                key_level=lvl["swing_retest"],
                bar_rng=lvl["bar_rng"],
                structural_stop=lvl["structural_stop"],
                max_wait_bars=15,
            )
            if not trig.get("triggered"):
                continue
            triggered_count += 1
            entry_p = float(trig["entry_price"])
            tp1 = float(lvl["peak"] if side == 1 else lvl["trough"])
            if (side == 1 and tp1 <= entry_p) or (side == -1 and tp1 >= entry_p):
                tp1 = round(entry_p + side * 1.618 * lvl["impulse_leg"], 2)
            tp2 = round(max(0.01, entry_p + side * 2.618 * lvl["impulse_leg"]), 2)
            res = sl.simulate_strategy_trade(
                df,
                entry_t=int(trig["entry_t"]),
                entry_price=entry_p,
                side=side,
                stop_price=float(trig["stop_price"]),
                tp_price=tp1,
                max_hold_bars=40,
                min_rr=min_rr,
                use_tiered_tp=True,
                tp2_price=tp2,
                impulse_leg=float(lvl["impulse_leg"]),
            )

        if res.get("status") != "completed":
            next_free_t = int(trig["entry_t"]) + 1
            continue
        if _is_open_end_bar_trade(res, n):
            open_at_end.append({
                "entry_date": str(df.index[res["entry_t"]])[:10],
                "side": "LONG" if side == 1 else "SHORT",
                "entry_price": res["entry_price"],
                "source": f"core_mss({trig.get('pattern')})",
            })
            next_free_t = res["exit_t"] + 1
            continue
        res["entry_date"] = df.index[res["entry_t"]]
        res["exit_date"] = df.index[res["exit_t"]]
        res["side"] = side
        res["signal_source"] = f"core_mss({trig.get('pattern')})"
        trades.append(res)
        next_free_t = res["exit_t"] + 1

    out = summarize_trades(trades, df, start_idx, is_crypto=is_crypto)
    out["signals_detected"] = signals_detected
    out["triggered_count"] = triggered_count
    out["open_at_end"] = open_at_end
    return out


# ==============================================================================
# 单标的完整回测调度
# ==============================================================================

def run_symbol_backtest(cfg: Dict[str, str]) -> Dict[str, Any]:
    sym = cfg["symbol"]
    mkt = cfg["market"]
    is_crypto = (mkt == "crypto")
    df = load_symbol_df(cfg["csv"])
    start_idx = min(250, max(60, len(df) // 4))

    t0 = time.time()
    plans, inv_errors = scan_adapter_signals(sym, df, mkt, start_idx=start_idx)
    t_scan = time.time() - t0

    track_direct = sim_adapter_direct(df, plans, start_idx, is_crypto)
    track_pb_gtc = sim_adapter_playbook(df, plans, start_idx, is_crypto, max_wait_bars=15)
    track_pb_1d = sim_adapter_playbook(df, plans, start_idx, is_crypto, max_wait_bars=1)
    track_ct = sim_adapter_counter_trend(
        df, plans, start_idx, is_crypto, include_observe_only=(not is_crypto)
    )
    track_core = sim_strategy_core_mss(df, start_idx, is_crypto)

    # 最新一期实盘状态 (包含最后一根 2026-11-20)
    latest_res = run_strategy_analysis(sym, period="1d", mode="swing", df=df, market=mkt)
    latest_summary = latest_res["summary"]
    latest_tp = latest_summary["trade_plan"]

    return {
        "symbol": sym,
        "market": mkt,
        "data_range": f"{str(df.index[0])[:10]} -> {str(df.index[-1])[:10]}",
        "total_bars": len(df),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "scan_seconds": round(t_scan, 2),
        "invariant_errors_count": len(inv_errors),
        "invariant_errors_sample": inv_errors[:5],
        "signal_counts": {
            "bars_scanned": len(plans),
            "passed_gates": sum(1 for p in plans.values() if p["passed_gates"]),
            "passed_long": sum(1 for p in plans.values() if p["passed_gates"] and p["direction_mode"] == "LONG"),
            "passed_short": sum(1 for p in plans.values() if p["passed_gates"] and p["direction_mode"] == "SHORT"),
            "condition_playbook": sum(1 for p in plans.values() if p.get("condition_playbook") is not None),
            "counter_trend_playbook": sum(1 for p in plans.values() if p.get("counter_trend_playbook") is not None),
        },
        "tracks": {
            "1_adapter_direct": track_direct,
            "2_adapter_playbook_15bar_gtc": track_pb_gtc,
            "3_adapter_playbook_1bar_refresh": track_pb_1d,
            "4_adapter_counter_trend": track_ct,
            "5_strategy_core_mss": track_core,
        },
        "latest_signal": {
            "date": str(df.index[-1])[:10],
            "price": latest_res["current_price"],
            "dow_phase": latest_summary["dow_phase"],
            "dow_bias": latest_summary["dow_bias"],
            "ema_200_gate": latest_summary["ema_200_gate"],
            "rsi_14": latest_summary["rsi_14"],
            "wave": latest_summary["wave"]["pattern"],
            "direction": latest_tp["direction"],
            "direction_mode": latest_tp["direction_mode"],
            "passed_gates": latest_tp["passed_gates"],
            "stop_loss": latest_tp["stop_loss"],
            "tp1": latest_tp["tp1"],
            "tp2": latest_tp["tp2"],
            "rr_ratio": latest_tp["rr_ratio"],
            "condition_playbook": latest_tp.get("condition_playbook"),
            "counter_trend_playbook": latest_tp.get("counter_trend_playbook"),
        },
    }


def _print_report(results: List[Dict[str, Any]]) -> None:
    print("=" * 92)
    print("StockVision 最新日线波段策略全量回测报告 (Zero-Lookahead Walk-Forward, 8 bps 摩擦)")
    print("=" * 92)
    for r in results:
        print(f"\n>>> [{r['symbol']}] ({r['market']}) | 数据: {r['data_range']} ({r['total_bars']} bars) | 最新收盘: ${r['last_close']} | 不变量异常: {r['invariant_errors_count']}")
        sc_cnt = r["signal_counts"]
        print(
            f"    扫描K线: {sc_cnt['bars_scanned']} | 直接信号: {sc_cnt['passed_gates']} "
            f"(L={sc_cnt['passed_long']}, S={sc_cnt['passed_short']}) | "
            f"回踩剧本: {sc_cnt['condition_playbook']} | 超跌剧本: {sc_cnt['counter_trend_playbook']}"
        )
        print(f"    {'策略轨道':<32} {'交易数':>5} {'胜率':>7} {'净PF':>6} {'净收益%':>9} {'BH%':>8} {'最大回撤%':>9} {'均R':>7} {'中位MAE%':>9}")
        for tname, tdata in r["tracks"].items():
            a = tdata["all"]
            print(
                f"    {tname:<32} {a['trades']:>5} {a['win_rate_pct']:>6.1f}% {a['net_pf']:>6.2f} "
                f"{a['total_net_return_pct']:>8.1f}% {tdata['buy_and_hold_pct']:>7.1f}% "
                f"{a['max_drawdown_pct']:>8.1f}% {a['avg_r']:>7.2f} {a['median_mae_pct']:>8.2f}%"
            )
            ex = tdata["ex_black_swan"]
            if ex["trades"] != a["trades"]:
                print(
                    f"      └─ 剔除黑天鹅:                 {ex['trades']:>5} {ex['win_rate_pct']:>6.1f}% {ex['net_pf']:>6.2f} "
                    f"{ex['total_net_return_pct']:>8.1f}%          {ex['max_drawdown_pct']:>8.1f}% {ex['avg_r']:>7.2f} {ex['median_mae_pct']:>8.2f}%"
                )
        ls = r["latest_signal"]
        print(
            f"    最新信号 ({ls['date']} @ ${ls['price']}): {ls['direction']} | {ls['dow_phase']} | "
            f"RSI(14)={ls['rsi_14']} | 200EMA={ls['ema_200_gate']['trend']}(${ls['ema_200_gate']['value']}) | "
            f"RR={ls['rr_ratio']}R"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-lookahead daily swing backtest.")
    parser.add_argument("--update-data", action="store_true", help="Refresh target data before running backtest.")
    parser.add_argument("--json-out", type=str, default=str(BACKEND_DIR / "results" / "swing_backtest_latest.json"))
    args = parser.parse_args()

    data_updates = None
    if args.update_data:
        print("[1/2] Updating target datasets (BTC-USD, NQmain, ESmain, SPY, QQQ, IWM)...")
        data_updates = asyncio.run(update_target_datasets())
        print(json.dumps(data_updates, ensure_ascii=False, indent=2))

    print("[2/2] Running parallel walk-forward backtests...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_symbol_backtest, DEFAULT_TARGETS))
    elapsed = round(time.time() - t0, 2)

    _print_report(results)
    print(f"\nTotal backtest elapsed: {elapsed:.2f}s")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"data_updates": data_updates, "elapsed_sec": elapsed, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
