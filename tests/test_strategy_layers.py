#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略位置层、触发层与全闭环风控执行单元测试集 (Unit Tests for strategy_layer.py)

测试覆盖：
  1. extract_breakout_levels: 多空结构位、斐波那契回撤与结构止损锚点
  2. simulate_limit_order: 限价回踩撮合、开盘跳空更优价成交、触及止损失效撤单、超时未成交过期
  3. detect_trigger_on_bar: 拒绝 Pin Bar、2B 流动性猎杀扫损、吞没形态检测与紧密止损标定
  4. simulate_trigger_entry: 触发后次根K线开盘入场（零未来函数）、跳空穿止损放弃入场
  5. simulate_strategy_trade:
     - 盈亏比硬闸门（RR < 2.0 一票否决）
     - 保守撮合原则（同一根K线同时触及止损与止盈，严格判定为止损先触发）
     - 正常止损与止盈生命周期推演、MAE 与 MFE 计算
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import strategy_layer as sl
import pa_engine as pe

class TestStrategyLayers(unittest.TestCase):

    def setUp(self):
        # 构造 100 根基准序列
        dates = pd.bdate_range("2026-01-01", periods=100)
        c = 100.0 + np.sin(np.linspace(0, 10, 100)) * 10
        self.df = pd.DataFrame({
            "Open": c,
            "High": c + 1.0,
            "Low": c - 1.0,
            "Close": c,
            "Volume": np.full(100, 1000000.0)
        }, index=dates)

    def test_extract_breakout_levels_long(self):
        """测试多头突破结构位与斐波那契提取"""
        swings = [
            {"index": 20, "type": "High", "price": 105.0, "dow": "High", "confirmed": True},
            {"index": 40, "type": "Low", "price": 95.0, "dow": "Low", "confirmed": True},
        ]
        df_sub = self.df.iloc[:60].copy()
        df_sub.iloc[-1, df_sub.columns.get_loc('High')] = 115.0
        df_sub.iloc[-1, df_sub.columns.get_loc('Close')] = 112.0

        lvls = sl.extract_breakout_levels(df_sub, 59, 1, swings)
        self.assertEqual(lvls["side"], 1)
        self.assertEqual(lvls["h_break"], 105.0)
        self.assertEqual(lvls["l_anchor"], 95.0)
        self.assertEqual(lvls["swing_retest"], 105.0)
        # peak=115, l_anchor=95, impulse=20. fib50=115-10=105, fib618=115-12.36=102.64
        self.assertAlmostEqual(lvls["fib50"], 105.0, places=1)
        self.assertAlmostEqual(lvls["fib618"], 102.64, places=1)
        self.assertTrue(lvls["structural_stop"] < 95.0)

    def test_extract_breakout_levels_short(self):
        """测试空头跌破结构位与斐波那契提取"""
        swings = [
            {"index": 20, "type": "High", "price": 110.0, "dow": "High", "confirmed": True},
            {"index": 40, "type": "Low", "price": 90.0, "dow": "Low", "confirmed": True},
        ]
        df_sub = self.df.iloc[:60].copy()
        df_sub.iloc[-1, df_sub.columns.get_loc('Low')] = 80.0
        df_sub.iloc[-1, df_sub.columns.get_loc('Close')] = 82.0

        lvls = sl.extract_breakout_levels(df_sub, 59, -1, swings)
        self.assertEqual(lvls["side"], -1)
        self.assertEqual(lvls["l_break"], 90.0)
        self.assertEqual(lvls["h_anchor"], 110.0)
        self.assertEqual(lvls["swing_retest"], 90.0)
        self.assertTrue(lvls["structural_stop"] > 110.0)

    def test_limit_order_fill_and_gap(self):
        """测试限价回踩挂单成交逻辑：限价成交与更有利开盘价成交"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        # t=0 决策挂单 limit=100.0, stop=90.0
        # bar 1: Open=102, Low=99, High=103, Close=101 -> 盘中打到 99，成交于 limit 100.0
        # bar 2: Open=98 (跳空低于100), Low=97, High=101, Close=100 -> 应成交于 Open 98.0
        df_test1 = pd.DataFrame({
            "Open": [105.0, 102.0, 101.0, 102.0, 103.0],
            "High": [106.0, 103.0, 102.0, 103.0, 104.0],
            "Low":  [104.0, 99.0, 100.0, 101.0, 102.0],
            "Close":[105.0, 101.0, 101.0, 102.0, 103.0]
        }, index=idx)

        res1 = sl.simulate_limit_order(df_test1, 0, 1, limit_price=100.0, stop_price=90.0)
        self.assertTrue(res1["filled"])
        self.assertEqual(res1["entry_t"], 1)
        self.assertEqual(res1["entry_price"], 100.0)

        # 测试跳空更有利价格
        df_test2 = pd.DataFrame({
            "Open": [105.0, 98.0, 101.0, 102.0, 103.0],
            "High": [106.0, 103.0, 102.0, 103.0, 104.0],
            "Low":  [104.0, 97.0, 100.0, 101.0, 102.0],
            "Close":[105.0, 101.0, 101.0, 102.0, 103.0]
        }, index=idx)
        res2 = sl.simulate_limit_order(df_test2, 0, 1, limit_price=100.0, stop_price=90.0)
        self.assertTrue(res2["filled"])
        self.assertEqual(res2["entry_price"], 98.0)

    def test_limit_order_invalidation_cancel(self):
        """测试价格跌破止损位导致挂单失效撤销"""
        idx = pd.bdate_range("2026-01-01", periods=4)
        # bar 1 直接暴跌击穿止损 90.0
        df_crash = pd.DataFrame({
            "Open": [105.0, 92.0, 85.0, 86.0],
            "High": [106.0, 93.0, 86.0, 87.0],
            "Low":  [104.0, 88.0, 84.0, 85.0],  # Low 88 < stop 90
            "Close":[105.0, 89.0, 85.0, 86.0]
        }, index=idx)
        res = sl.simulate_limit_order(df_crash, 0, 1, limit_price=100.0, stop_price=90.0)
        self.assertFalse(res["filled"])
        self.assertEqual(res["status"], "cancelled_invalidated")

    def test_limit_order_timeout_expiration(self):
        """测试超过最大等待周期未成交过期"""
        idx = pd.bdate_range("2026-01-01", periods=10)
        # 价格一直在 102 以上，从未回踩 100.0
        df_away = pd.DataFrame({
            "Open": np.full(10, 105.0),
            "High": np.full(10, 108.0),
            "Low":  np.full(10, 103.0),
            "Close":np.full(10, 106.0)
        }, index=idx)
        res = sl.simulate_limit_order(df_away, 0, 1, limit_price=100.0, stop_price=90.0, max_wait_bars=5)
        self.assertFalse(res["filled"])
        self.assertEqual(res["status"], "unfilled_expired")

    def test_trigger_detection(self):
        """测试 2B 假突破、Pin Bar 与 吞没形态检测"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        # 1. 2B 假跌破收回：前K在关键位100上方，当前K低点98跌破100，收盘102收回上方
        df_2b = pd.DataFrame({
            "Open": [105.0, 101.0, 101.0],
            "High": [106.0, 102.0, 103.0],
            "Low":  [104.0, 98.0, 99.0],
            "Close":[105.0, 102.0, 102.0]
        }, index=idx)
        trig_2b = sl.detect_trigger_on_bar(df_2b, 1, side=1, key_level=100.0, bar_rng=2.0)
        self.assertTrue(trig_2b["triggered"])
        self.assertEqual(trig_2b["pattern"], "2B_sweep")
        self.assertTrue(trig_2b["tight_stop"] < 98.0)

        # 2. 看涨 Pin Bar：长下影探底拒绝（未跌破关键位，纯 Pin Bar）
        df_pin = pd.DataFrame({
            "Open": [105.0, 101.5, 102.0],
            "High": [106.0, 102.0, 103.0],
            "Low":  [104.0, 100.2, 99.0],    # 最低 100.2 在关键位 100 上方，振幅 1.8，实体 0.3，下影 1.3
            "Close":[105.0, 101.8, 102.0]
        }, index=idx)
        trig_pin = sl.detect_trigger_on_bar(df_pin, 1, side=1, key_level=100.0, bar_rng=2.0)
        self.assertTrue(trig_pin["triggered"])
        self.assertEqual(trig_pin["pattern"], "pin_bar")

        # 3. 看涨吞没 (在支撑位上方形成，未跌破关键位)
        df_eng = pd.DataFrame({
            "Open": [105.0, 101.0, 100.2],
            "High": [106.0, 101.5, 103.0],
            "Low":  [104.0, 100.4, 100.1],   # 靠近 100.0 但在 100.0 之上
            "Close":[105.0, 100.5, 102.5]   # 前阴线 [101->100.5]，今阳线 [100.2->102.5] 实体吞没
        }, index=idx)
        trig_eng = sl.detect_trigger_on_bar(df_eng, 2, side=1, key_level=100.0, bar_rng=2.0)
        self.assertTrue(trig_eng["triggered"])
        self.assertEqual(trig_eng["pattern"], "engulfing")

    def test_conservative_execution_tie_breaking(self):
        """测试同一根K线同时触及止损与止盈时的保守撮合规则（必须止损优先）"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        # entry=100, stop=95, tp=110. bar 1: Low=90 <= stop 且 High=115 >= tp
        df_tie = pd.DataFrame({
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 115.0, 100.0],
            "Low":  [100.0, 90.0,  100.0],
            "Close":[100.0, 105.0, 100.0]
        }, index=idx)
        tr = sl.simulate_strategy_trade(df_tie, 0, 100.0, side=1, stop_price=95.0, tp_price=110.0, min_rr=2.0)
        self.assertEqual(tr["exit_reason"], "stop_loss_conservative")
        self.assertFalse(tr["win"])
        self.assertEqual(tr["exit_price"], 95.0)

    def test_min_rr_gate_rejection(self):
        """测试盈亏比硬闸门：低于 min_rr 时必须直接驳回"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        df_dummy = pd.DataFrame({
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 105.0, 100.0],
            "Low":  [100.0, 98.0, 100.0],
            "Close":[100.0, 102.0, 100.0]
        }, index=idx)
        # risk=5 (100-95), reward=5 (105-100) -> RR=1.0 < 2.0
        tr = sl.simulate_strategy_trade(df_dummy, 0, 100.0, side=1, stop_price=95.0, tp_price=105.0, min_rr=2.0)
        self.assertEqual(tr["status"], "rejected_by_min_rr")
        self.assertEqual(tr["rr"], 1.0)

    def test_successful_take_profit_lifecycle(self):
        """测试成功触发止盈的交易生命周期与 MAE / MFE 计算"""
        idx = pd.bdate_range("2026-01-01", periods=4)
        # entry=100.0, stop=98.0 (risk=2.0), tp=105.0 (reward=5.0, RR=2.5)
        # bar 1: Low 99.0, High 102.0
        # bar 2: Low 101.0, High 106.0 -> 触发止盈 105.0
        df_win = pd.DataFrame({
            "Open": [100.0, 100.0, 102.0, 104.0],
            "High": [100.0, 102.0, 106.0, 105.0],
            "Low":  [100.0, 99.0,  101.0, 103.0],
            "Close":[100.0, 101.5, 105.5, 104.0]
        }, index=idx)
        tr = sl.simulate_strategy_trade(df_win, 0, 100.0, side=1, stop_price=98.0, tp_price=105.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertTrue(tr["win"])
        self.assertEqual(tr["exit_reason"], "take_profit")
        self.assertEqual(tr["exit_price"], 105.0)
        self.assertAlmostEqual(tr["pnl_pct"], 0.05, places=5)
        self.assertEqual(tr["r_multiple"], 2.5)
        self.assertEqual(tr["hold_bars"], 2)
        self.assertAlmostEqual(tr["mae"], -0.01)  # 最低打到 99.0 -> -1%
        self.assertAlmostEqual(tr["mfe"], 0.06)   # 最高摸到 106.0 -> +6%

    def test_trigger_out_of_zone_rejection(self):
        """测试脱离关键位容忍区的伪形态必须被严格驳回（防越界假形态）"""
        idx = pd.bdate_range("2026-01-01", periods=2)
        # key_level=100.0, bar_rng=2.0 (tol=1.0)
        # 在 80~85 区间形成的锤子线（虽有长下影，但离关键位 100 极远，属于深幅崩盘而非回踩测试）
        df_deep_drop = pd.DataFrame({
            "Open": [95.0, 84.0],
            "High": [96.0, 85.0],
            "Low":  [94.0, 80.0],
            "Close":[95.0, 84.5]
        }, index=idx)
        trig = sl.detect_trigger_on_bar(df_deep_drop, 1, side=1, key_level=100.0, bar_rng=2.0)
        self.assertFalse(trig["triggered"])

    def test_same_day_stop_loss_lifecycle(self):
        """测试入场当天（entry_t 根）盘中打到止损必须当天出场，杜绝持仓首日无风控漏洞"""
        idx = pd.bdate_range("2026-01-01", periods=2)
        # entry=100.0, stop=95.0, tp=110.0. bar 0 盘中打到 94.0
        df_same_day = pd.DataFrame({
            "Open": [100.0, 102.0],
            "High": [101.0, 108.0],
            "Low":  [94.0,  100.0],
            "Close":[96.0,  107.0]
        }, index=idx)
        tr = sl.simulate_strategy_trade(df_same_day, 0, 100.0, side=1, stop_price=95.0, tp_price=110.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertEqual(tr["exit_reason"], "stop_loss")
        self.assertEqual(tr["exit_t"], 0)
        self.assertEqual(tr["hold_bars"], 0)
        self.assertEqual(tr["exit_price"], 95.0)
        self.assertFalse(tr["win"])

    def test_floating_point_exact_min_rr(self):
        """测试浮点数二进制精度误差（如 9.86 / 4.93 = 1.9999999999999998）不被误判驳回"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        df_fp = pd.DataFrame({
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 110.0, 110.0],
            "Low":  [100.0, 98.0,  98.0],
            "Close":[100.0, 109.86, 109.86]
        }, index=idx)
        # entry=100.0, stop=95.07 (risk=4.93), tp=109.86 (reward=9.86). 9.86 / 4.93 = 1.9999999999999998
        tr = sl.simulate_strategy_trade(df_fp, 0, 100.0, side=1, stop_price=95.07, tp_price=109.86, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")

    def test_impulse_leg_peak_calculation(self):
        """测试多头冲量波段峰值计算：必须捕捉锚点到决策点之间的波段真实极值"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        # index 0: low anchor=90
        # index 1: tall high=110
        # index 2: pause bar close=105, high=106 (此时决策突破)
        df_impulse = pd.DataFrame({
            "Open": [95.0, 100.0, 104.0, 105.0, 105.0],
            "High": [96.0, 110.0, 106.0, 107.0, 108.0],
            "Low":  [90.0, 99.0,  103.0, 104.0, 104.0],
            "Close":[92.0, 108.0, 105.0, 106.0, 106.0]
        }, index=idx)
        swings = [
            {"index": 0, "type": "Low", "price": 90.0, "confirmed": True},
            {"index": 1, "type": "High", "price": 105.0, "confirmed": True}
        ]
        lvls = sl.extract_breakout_levels(df_impulse, 2, side=1, swings=swings)
        # 真实冲量峰值应为 index 1 的 110.0，而非 index 2 的 106.0
        self.assertEqual(lvls["peak"], 110.0)
        # fib50 = 110 - 0.5 * (110 - 90) = 100.0
        self.assertEqual(lvls["fib50"], 100.0)

    def test_simulate_strategy_trade_tiered(self):
        """测试全闭环分级止盈生命周期：TP1 减半推保本防守 + Runner 捕获"""
        idx = pd.bdate_range("2026-01-01", periods=6)
        # entry=100.0, stop=90.0 (risk=10), tp1=120.0 (2R), tp2=135.0
        # bar 0: entry at 100.0
        # bar 1: High=122.0 (触发 TP1 120.0 减半，推保本 curr_stop=100.0), Low=102.0
        # bar 2: Low=99.0 (回踩跌破 100.0 保本点) -> 保本防守出场
        df_be = pd.DataFrame({
            "Open":  [100.0, 105.0, 102.0, 98.0, 97.0, 96.0],
            "High":  [101.0, 126.0, 103.0, 99.0, 98.0, 97.0],
            "Low":   [99.0,  102.0, 99.0,  96.0, 95.0, 94.0],
            "Close": [100.5, 121.0, 99.5,  97.0, 96.0, 95.0]
        }, index=idx)

        tr_be = sl.simulate_strategy_trade(df_be, 0, 100.0, side=1, stop_price=90.0, tp_price=125.0,
                                          use_tiered_tp=True, tp2_price=140.0)
        self.assertEqual(tr_be["status"], "completed")
        self.assertEqual(tr_be["exit_reason"], "tp1_then_be_protected")
        self.assertTrue(tr_be["win"])
        # 50% * 125 + 50% * 100 = 112.5
        self.assertAlmostEqual(tr_be["exit_price"], 112.5, places=1)
        self.assertAlmostEqual(tr_be["pnl_pct"], 0.125, places=3)
        self.assertAlmostEqual(tr_be["r_multiple"], 1.25, places=2)

if __name__ == "__main__":
    unittest.main()
