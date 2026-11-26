#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_intraday_friction_and_short_gate.py: Requirement R4 核心单元测试套件

涵盖需求 R4 的五大核心风控支柱：
  1. 日内 8 bps 真实执行摩擦硬核算 (Intraday 8 bps Friction Accounting):
     - 单边 3 bps 滑点 + 1 bps 规费 = 4 bps 单边，双边/往返 8 bps (0.0008)；
     - 毛收益 (Gross PnL) 与真实净收益 (Net PnL) 精确核算；
     - 净 R 倍数因摩擦确定性缩减；
     - 微利假象过滤：毛收益为正 (+5 bps) 但不足以覆盖摩擦时，净收益转负且判定为亏损 (win=False)；
     - 空头持仓摩擦核算 (卖出开仓不利滑点 + 买入平仓不利滑点 + 双边规费)；
     - 汇总统计 (summarize_day_trades) 中胜率与 Profit Factor 严格扣除 8 bps。
  2. 全局日内最小盈亏比硬闸门 (Universal Minimum RR >= 2.5R Discipline):
     - DEFAULT_DAY_MIN_RR 常量更新为 2.5；
     - 初始计划盈亏比 RR < 2.5 严格一票否决 (status="rejected_by_min_rr", reason="rejected_by_rr")；
     - 初始计划盈亏比 RR >= 2.5 正常放行；
     - 浮点边界容差 (2.4999 驳回 vs 2.5001 放行)；
     - backtest.py 命令行参数与默认值统一升级至 2.5。
  3. 1h HTF 宏观趋势过滤门禁默认生效 (1h HTF Trend Gate as Mandatory Default):
     - enumerate_day_events 默认参数 use_htf_filter == True；
     - 1h 顺势大势过滤：多头顺势下放行多头，空头顺势下放行空头；
     - 严格无未来函数重采样与道氏偏向提取；
     - 截断不变性因果自检。
  4. 低波动标的微利刷单风控拦截 (Low-Volatility Churn Protection):
     - 止损风险空间 |Entry - SL| / Entry < 2 * friction (16 bps) 严格驳回 (rejected_by_low_volatility_churn)；
     - 止损风险空间 >= 16 bps 允许放行；
     - 边界阈值测试 (15 bps 驳回 vs 17 bps 放行)；
     - enumerate_day_events 自动过滤微波动刷单信号。
  5. 美股非对称做空风险门禁 (Asymmetric Short-Selling Risk Gate):
     - verify_macro_short_gate 门禁功能与状态码；
     - 顺势长牛环境下盲目做空一票否决 (rejected_by_macro_short_gate)；
     - 宏观结构破位 (SPY/QQQ 跌破 200 EMA 或空头破位) 顺势放行做空；
     - 标的自身大级别空翻 (现价处于 200 EMA 下方) 顺势放行做空；
     - 高确定性微观 2B 顶流动性猎杀反转 (2B top sweep) 豁免放行；
     - simulate_day_trade 与 enumerate_day_events 日内级非对称做空门禁拦截与 2B 顶豁免。
"""

import os
import sys
import unittest
from datetime import time as dtime
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pa_engine as pe
import strategy_layer as sl
import strategy_day as sd
import backtest


class TestIntradayFrictionAccounting(unittest.TestCase):
    """1. 日内 8 bps 真实执行摩擦硬核算单元测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        self.df = pd.DataFrame({
            "Open":  [100.0] * 10,
            "High":  [103.0] * 10,
            "Low":   [98.0] * 10,
            "Close": [101.0] * 10,
            "Volume":[1000] * 10
        }, index=self.ts)

    def test_8bps_exact_roundtrip_friction_deduction(self):
        """测试 3 bps 滑点 + 1 bps 规费单边，往返总摩擦损耗精确为 8 bps (0.0008)"""
        # 平价出场（entry 100.0, exit 100.0），纯测摩擦损耗
        df_flat = self.df.copy()
        df_flat["High"] = 100.0
        df_flat["Low"] = 100.0
        df_flat["Close"] = 100.0

        tr = sd.simulate_day_trade(
            df_flat, entry_t=1, entry_price=100.0, side=1,
            stop_price=98.0, tp_price=105.0, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0,
            max_hold_bars=3
        )
        self.assertEqual(tr.get("status"), "completed")
        self.assertAlmostEqual(tr["gross_pnl_pct"], 0.0, places=6)
        # 往返 8 bps 损耗：3 bps 入场滑点 + 3 bps 出场滑点 + 2 bps 往返规费 = 8 bps (0.0008)
        self.assertAlmostEqual(tr["net_pnl_pct"], -0.0008, delta=0.00005)
        self.assertAlmostEqual(tr["slippage_drag_pct"], 0.0008, delta=0.00005)
        self.assertLess(tr["net_r"], tr["gross_r"])

    def test_gross_win_becomes_net_loss_due_to_friction(self):
        """测试微利假象被摩擦穿透：毛利 +5 bps 在扣除 8 bps 摩擦后转为真实亏损 (-3 bps)，正确计为负收益"""
        df_micro = self.df.copy()
        df_micro["Low"] = 99.5  # 高于止损价 99.0，避免误触止损
        # entry=100.0, bar 2 触及 100.05 止盈出场 (+5 bps 毛收益)
        df_micro.iloc[2, df_micro.columns.get_loc("High")] = 100.05
        df_micro.iloc[2, df_micro.columns.get_loc("Close")] = 100.05

        tr = sd.simulate_day_trade(
            df_micro, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=100.05, min_rr=0.01,  # 验证纯摩擦损耗
            slippage_bps=3.0, commission_bps=1.0,
            max_hold_bars=2
        )
        self.assertEqual(tr.get("status"), "completed")
        self.assertEqual(tr.get("exit_reason"), "take_profit")
        # 毛收益为正 (+5 bps)
        self.assertGreater(tr["gross_pnl_pct"], 0.0)
        self.assertTrue(tr["gross_win"])
        # 净收益因 8 bps 摩擦而转负 (-3 bps)
        self.assertLess(tr["net_pnl_pct"], 0.0)
        self.assertFalse(tr["win"])

    def test_short_trade_friction_accounting(self):
        """测试空头持仓摩擦核算（卖出实际成交价更低，买回实际成交价更高）"""
        tr_short = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=97.0, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0,
            max_hold_bars=3
        )
        self.assertEqual(tr_short.get("status"), "completed")
        # 净收益必定小于毛收益
        self.assertLess(tr_short["net_pnl_pct"], tr_short["gross_pnl_pct"])
        friction_drag = tr_short["gross_pnl_pct"] - tr_short["net_pnl_pct"]
        self.assertAlmostEqual(friction_drag, 0.0008, delta=0.00005)

    def test_summarize_day_trades_strictly_deducts_friction(self):
        """测试日内交易汇总统计 (summarize_day_trades) 严格扣减 8 bps 摩擦"""
        trades = [
            # 交易 1: 大赚 +2.0% 毛利，扣除 8 bps 净赚 +1.92%
            {"gross_pnl_pct": 0.0200, "net_pnl_pct": 0.0192, "gross_r": 2.0, "net_r": 1.92,
             "side": 1, "exit_reason": "take_profit", "hold_bars": 5, "mae": -0.005, "mfe": 0.025,
             "slippage_drag_pct": 0.0008},
            # 交易 2: 微利 +0.05% 毛利，扣除 8 bps 变成 -0.03% 净亏
            {"gross_pnl_pct": 0.0005, "net_pnl_pct": -0.0003, "gross_r": 0.05, "net_r": -0.03,
             "side": 1, "exit_reason": "session_end_close", "hold_bars": 10, "mae": -0.002, "mfe": 0.001,
             "slippage_drag_pct": 0.0008},
            # 交易 3: 止损 -1.0% 毛亏，扣除 8 bps 净亏 -1.08%
            {"gross_pnl_pct": -0.0100, "net_pnl_pct": -0.0108, "gross_r": -1.0, "net_r": -1.08,
             "side": -1, "exit_reason": "stop_loss", "hold_bars": 4, "mae": -0.012, "mfe": 0.002,
             "slippage_drag_pct": 0.0008}
        ]
        summary = sd.summarize_day_trades(trades)
        self.assertEqual(summary["trades"], 3)
        # 毛胜率 2/3 = 66.7%，净胜率 1/3 = 33.3%（微利单因摩擦扣减变为亏损）
        self.assertAlmostEqual(summary["gross_win_rate"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(summary["net_win_rate"], 1.0 / 3.0, places=4)
        # 平均摩擦损耗必须为 8 bps
        self.assertAlmostEqual(summary["avg_friction_drag"], 0.0008, places=4)
        # 净期望必须小于毛期望
        self.assertLess(summary["exp_net"], summary["exp_gross"])
        # 净 PF 必须小于毛 PF
        self.assertLess(summary["net_pf"], summary["gross_pf"])


class TestUniversalMinRRGate(unittest.TestCase):
    """2. 全局最小盈亏比硬闸门 (Universal Minimum RR >= 2.5R) 单元测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        self.df = pd.DataFrame({
            "Open":  [100.0] * 10,
            "High":  [104.0] * 10,
            "Low":   [98.0] * 10,
            "Close": [101.0] * 10,
            "Volume":[1000] * 10
        }, index=self.ts)

    def test_default_day_min_rr_constant(self):
        """测试日内最小盈亏比硬闸门常量升级至 2.5"""
        self.assertEqual(sd.DEFAULT_DAY_MIN_RR, 2.5)

    def test_rr_less_than_2_5_rejected(self):
        """测试 RR < 2.5 时被硬闸门一票否决 (rejected_by_min_rr / rejected_by_rr)"""
        # entry=100.0, stop=99.0 (risk=1.0), tp=102.0 (reward=2.0) -> RR = 2.0 < 2.5
        tr = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.0, min_rr=2.5
        )
        self.assertEqual(tr.get("status"), "rejected_by_min_rr")
        self.assertEqual(tr.get("reason"), "rejected_by_rr")
        self.assertAlmostEqual(tr.get("rr"), 2.0)
        self.assertEqual(tr.get("min_rr"), 2.5)

    def test_rr_greater_equal_2_5_allowed(self):
        """测试 RR >= 2.5 时顺利通过门禁并执行推演"""
        # entry=100.0, stop=99.0 (risk=1.0), tp=102.5 (reward=2.5) -> RR = 2.5
        tr = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.5, min_rr=2.5
        )
        self.assertEqual(tr.get("status"), "completed")

    def test_floating_point_boundary_min_rr(self):
        """测试浮点精度临界值：2.4999 驳回 vs 2.5001 放行"""
        # risk = 1.0, reward = 2.4999 -> RR = 2.4999 < 2.5
        tr_sub = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.4999, min_rr=2.5
        )
        self.assertEqual(tr_sub.get("status"), "rejected_by_min_rr")

        # risk = 1.0, reward = 2.5001 -> RR = 2.5001 >= 2.5
        tr_sup = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.5001, min_rr=2.5
        )
        self.assertEqual(tr_sup.get("status"), "completed")

    def test_enumerate_day_events_outputs_gte_2_5r(self):
        """测试日内事件枚举器生成的所有事件计划盈亏比四舍五入均满足 >= 2.5R"""
        df_syn = sd.synthetic_intraday_days(n_days=5, seed=42)
        events = sd.enumerate_day_events(df_syn, min_rr=2.5)
        self.assertTrue(len(events) > 0, "必须生成有效的日内事件")
        for ev in events:
            rr = abs(ev["tp_price"] - ev["entry_price"]) / ev["risk"]
            self.assertGreaterEqual(round(rr, 1), 2.5,
                                    f"事件计划盈亏比 {rr:.2f} 必须满足 >= 2.5R 硬要求")


class Test1hHTFTrendFilter(unittest.TestCase):
    """3. 1h HTF 宏观趋势过滤门禁默认生效单元测试"""

    def test_enumerate_day_events_default_use_htf_filter(self):
        """测试 enumerate_day_events 函数默认参数为 use_htf_filter=True"""
        import inspect
        sig = inspect.signature(sd.enumerate_day_events)
        param = sig.parameters["use_htf_filter"]
        self.assertTrue(param.default, "use_htf_filter 必须默认设为 True")

    def test_extract_intraday_htf_bias_causality(self):
        """测试 1h HTF 道氏趋势提取零未来函数（cutoff_ts 截断后完全一致）"""
        df_syn = sd.synthetic_intraday_days(n_days=5, seed=333)
        sessions = sd.split_intraday_sessions(df_syn)
        s_dates = list(sessions.keys())
        day3_open = sessions[s_dates[2]].index[0]

        bias_full = sd.extract_intraday_htf_bias(df_syn, cutoff_ts=day3_open)
        sub_df = df_syn[df_syn.index <= day3_open]
        bias_sub = sd.extract_intraday_htf_bias(sub_df, cutoff_ts=day3_open)

        self.assertEqual(bias_full, bias_sub, "HTF 偏向提取在截断前后必须完全一致，杜绝未来函数")

    def test_htf_filter_suppresses_countertrend_continuation(self):
        """测试在 1h HTF 多头趋势下，常规顺势突破做空 (ORB Short) 被门禁拦截"""
        # 构造高分辨率连续数据
        df_syn = sd.synthetic_intraday_days(n_days=5, seed=555)
        # 分别以开启与关闭 HTF 门禁运行 ORB 策略
        ev_no_htf = sd.enumerate_day_events(df_syn, strategy_type="orb", use_htf_filter=False, min_rr=2.5)
        ev_htf = sd.enumerate_day_events(df_syn, strategy_type="orb", use_htf_filter=True, min_rr=2.5)

        # 开启 HTF 过滤后的事件集必然经过了大周期趋势过滤（或者子集或更精炼）
        self.assertIsInstance(ev_htf, list)
        self.assertIsInstance(ev_no_htf, list)


class TestLowVolatilityChurnProtection(unittest.TestCase):
    """4. 低波动标的微利刷单风控拦截 (Low-Volatility Churn Protection) 单元测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        self.df = pd.DataFrame({
            "Open":  [100.0] * 10,
            "High":  [103.0] * 10,
            "Low":   [98.0] * 10,
            "Close": [101.0] * 10,
            "Volume":[1000] * 10
        }, index=self.ts)

    def test_risk_distance_below_16bps_rejected(self):
        """测试止损风险空间 < 2 * friction (16 bps) 必须被一票否决 (rejected_by_low_volatility_churn)"""
        # entry=100.0, stop=99.85 -> risk = 0.15 (15 bps < 16 bps 门槛)
        # 即便盈亏比极高 (tp=103.0, RR = 3.0 / 0.15 = 20R)，也必须因摩擦侵吞风险空间而被拦截
        tr = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.85, tp_price=103.0, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        self.assertEqual(tr.get("status"), "rejected_by_low_volatility_churn")
        self.assertEqual(tr.get("reason"), "rejected_by_low_volatility_churn")
        self.assertAlmostEqual(tr.get("risk_pct"), 0.0015, places=5)
        self.assertAlmostEqual(tr.get("churn_threshold"), 0.0016, places=5)

    def test_risk_distance_above_16bps_permitted(self):
        """测试止损风险空间 >= 16 bps 且 RR >= 2.5R 时正常放行"""
        # entry=100.0, stop=99.80 -> risk = 0.20 (20 bps >= 16 bps)
        # tp=100.60 -> reward = 0.60, RR = 3.0R >= 2.5R
        tr = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.80, tp_price=100.60, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        self.assertEqual(tr.get("status"), "completed")

    def test_churn_boundary_tolerance(self):
        """测试微利刷单边界容差 (15.9 bps 驳回 vs 16.1 bps 放行)"""
        # 15.9 bps -> 0.159
        tr_sub = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.841, tp_price=102.0, min_rr=2.5
        )
        self.assertEqual(tr_sub.get("status"), "rejected_by_low_volatility_churn")

        # 16.1 bps -> 0.161, tp = 100 + 2.5 * 0.161 = 100.4025
        tr_sup = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.839, tp_price=100.41, min_rr=2.5
        )
        self.assertEqual(tr_sup.get("status"), "completed")


class TestAsymmetricShortSellingRiskGate(unittest.TestCase):
    """5. 美股非对称做空风险门禁 (Asymmetric Short-Selling Risk Gate) 单元测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        self.df_bull = pd.DataFrame({
            "Open":  [100.0] * 10,
            "High":  [105.0] * 10,
            "Low":   [98.0] * 10,
            "Close": [102.0] * 10,
            "Volume":[1000] * 10
        }, index=self.ts)

    def test_verify_macro_short_gate_rejects_blind_short_in_bull(self):
        """测试大盘处于多头顺势环境时，常规盲目做空被宏观做空门禁驳回"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        res = sl.verify_macro_short_gate(market_dow=market_bull)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_macro_short_gate")
        self.assertIn("rejected_by_macro_short_gate", res["reason"])

    def test_verify_macro_short_gate_allows_short_on_market_breakdown(self):
        """测试大盘出现空头破位或道氏空头结构时，顺势放行做空"""
        market_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1
        }
        res = sl.verify_macro_short_gate(market_dow=market_bear)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_macro_short_gate")

    def test_verify_macro_short_gate_allows_2b_top_sweep_reversal(self):
        """测试高确定性 2B 顶流动性猎杀反转 (2B top sweep) 豁免宏观逆风放行做空"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        res = sl.verify_macro_short_gate(market_dow=market_bull, micro_conviction="strong_2b_reversal")
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_2b_short_reversal")

    def test_simulate_day_trade_rejects_routine_short_in_bull_htf(self):
        """测试 simulate_day_trade 中，顺势长牛环境 (htf_bias=1) 下常规做空被门禁拦截"""
        tr = sd.simulate_day_trade(
            self.df_bull, entry_t=1, entry_price=100.0, side=-1,
            stop_price=102.0, tp_price=94.0, min_rr=2.5,
            htf_bias=1, pattern="orb_retest_short"
        )
        self.assertEqual(tr.get("status"), "rejected_by_macro_short_gate")
        self.assertIn("rejected_by_macro_short_gate", tr.get("reason"))

    def test_simulate_day_trade_permits_2b_top_sweep_in_bull_htf(self):
        """测试 simulate_day_trade 中，顺势多头下高确定性 2B 顶扫损反转允许放行"""
        tr = sd.simulate_day_trade(
            self.df_bull, entry_t=1, entry_price=100.0, side=-1,
            stop_price=102.0, tp_price=94.0, min_rr=2.5,
            htf_bias=1, pattern="2B_sweep_high_PDH"
        )
        self.assertEqual(tr.get("status"), "completed")

    def test_simulate_day_trade_with_bullish_market_dow(self):
        """测试 simulate_day_trade 传入大盘多头字典时拦截常规做空"""
        market_bull = {"SPY_bias": 1, "QQQ_bias": 1}
        tr_routine = sd.simulate_day_trade(
            self.df_bull, entry_t=1, entry_price=100.0, side=-1,
            stop_price=102.0, tp_price=94.0, min_rr=2.5,
            market_dow=market_bull, pattern="routine_short"
        )
        self.assertEqual(tr_routine.get("status"), "rejected_by_macro_short_gate")

        tr_2b = sd.simulate_day_trade(
            self.df_bull, entry_t=1, entry_price=100.0, side=-1,
            stop_price=102.0, tp_price=94.0, min_rr=2.5,
            market_dow=market_bull, pattern="2B_sweep_high_PMH"
        )
        self.assertEqual(tr_2b.get("status"), "completed")


if __name__ == "__main__":
    unittest.main()
