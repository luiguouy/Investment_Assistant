#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_wave_dow_engine.py: 道氏理论深化与艾略特波浪算法引擎单元测试集

测试覆盖：
  1. 艾略特波浪三大铁律客观校验 (Three Iron Rules of Elliott Wave):
     - 铁律1: 浪2 回撤绝不超过 浪1 起点 (Retracement <= 100%)
     - 铁律2: 浪3 永远不是 浪1、浪3、浪5 中最短的一浪 (Wave 3 not shortest)
     - 铁律3: 浪4 终点绝不重叠进入 浪1 价格领地 (Wave 4 low > Wave 1 high for bull)
     - 交替原则 (Alternation): 浪2 与 浪4 形态/幅度差异化特征
  2. 波浪位置狙击决策 (Wave-Position Sniper):
     - 第 3 浪主升突破放行 (wave3_impulse_breakout)
     - 第 5 浪冲顶诱多与动能衰竭拦截 (rejected_by_wave5_exhaustion)
     - C 浪终结 2B 假突破扫损筑底反转 (wave_c_bottom_reversal)
  3. 大盘指数相互印证硬闸门 (Market Confirmation Gate - Dow Theory Rule 4):
     - 大盘处于 LH+LL 空头趋势或破位空翻时，个股多头计划必须被硬闸门一票否决 (rejected_by_market_dow)
     - 大盘健康/非空头破位时，印证通过放行
  4. 次级折返缩量洗盘过滤 (Pullback Volume Contraction Filter - Dow Theory Rule 6):
     - 回踩过程呈现缩量态势时，限价单正常成交
     - 回踩过程伴随异常巨量抛售或连续放量重挫时，判定为派发，挂单自动撤销 (cancelled_volume_distribution)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pa_engine as pe
import strategy_layer as sl


class TestWaveDowEngine(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2026-01-01", periods=100)

    # ==========================================================================
    # 1. 艾略特波浪三大铁律与交替原则校验
    # ==========================================================================

    def test_wave_iron_rule1_violation(self):
        """测试铁律1违背：浪2回撤跌破浪1起点 (Retracement > 100%)"""
        # P0=100, P1=120, P2=95 (< 100), P3=140, P4=125, P5=155
        swings = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 95.0, "confirmed": True},   # 违背铁律1: 95 < 100
            {"index": 35, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 125.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 155.0, "confirmed": True},
        ]
        res = pe.classify_wave_pattern(swings)
        self.assertFalse(res["is_valid_impulse"])
        self.assertFalse(res["rules"]["rule1"]["passed"])
        self.assertIn("跌破浪1起点", res["rules"]["rule1"]["desc"])

    def test_wave_iron_rule2_violation(self):
        """测试铁律2违背：浪3成为 1, 3, 5 中最短的一浪"""
        # P0=100, P1=140 (L1=40), P2=130, P3=150 (L3=20), P4=145, P5=180 (L5=35)
        # L3=20 < L1=40 且 L3=20 < L5=35 -> 浪3最短，违背铁律2
        swings = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 130.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 150.0, "confirmed": True},  # 浪3涨幅 20
            {"index": 45, "type": "Low", "price": 142.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 180.0, "confirmed": True},  # 浪5涨幅 38
        ]
        res = pe.classify_wave_pattern(swings)
        self.assertFalse(res["is_valid_impulse"])
        self.assertFalse(res["rules"]["rule2"]["passed"])
        self.assertIn("最短浪", res["rules"]["rule2"]["desc"])

    def test_wave_iron_rule3_violation(self):
        """测试铁律3违背：浪4终点进入浪1价格领地 (Wave 4 low <= Wave 1 high)"""
        # P0=100, P1=120, P2=110, P3=160, P4=118 (<= 120 侵入领地), P5=170
        swings = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},  # 浪1顶 120
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 160.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 118.0, "confirmed": True},  # 违背铁律3: 118 <= 120
            {"index": 55, "type": "High", "price": 170.0, "confirmed": True},
        ]
        res = pe.classify_wave_pattern(swings)
        self.assertFalse(res["is_valid_impulse"])
        self.assertFalse(res["rules"]["rule3"]["passed"])
        self.assertIn("侵入浪1高点", res["rules"]["rule3"]["desc"])

    def test_wave_valid_impulse_and_alternation(self):
        """测试标准合规 5 浪推动浪与交替原则检测"""
        # P0=100, P1=125 (L1=25), P2=110 (浪2回撤 15/25 = 60% 深回撤)
        # P3=160 (L3=50 显著延伸), P4=148 (浪4回撤 12/50 = 24% 浅回撤)
        # P5=175 (L5=27)
        # 铁律1: 110 >= 100 PASS
        # 铁律2: L1=25, L3=50, L5=27 (L3最长，非最短) PASS
        # 铁律3: P4=148 > P1=125 PASS
        # 交替原则: 60% vs 24% (差异 36% >= 15%) PASS
        swings = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 125.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 160.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 148.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 175.0, "confirmed": True},
        ]
        res = pe.classify_wave_pattern(swings)
        self.assertTrue(res["is_valid_impulse"])
        self.assertTrue(res["rules"]["rule1"]["passed"])
        self.assertTrue(res["rules"]["rule2"]["passed"])
        self.assertTrue(res["rules"]["rule3"]["passed"])
        self.assertTrue(res["rules"]["alternation"]["passed"])
        self.assertAlmostEqual(res["ratios"]["w2_retrace"], 0.60, places=2)
        self.assertAlmostEqual(res["ratios"]["w4_retrace"], 0.24, places=2)

    # ==========================================================================
    # 2. 波浪位置客观证伪与斐波坐标尺 (Wave Invalidation & Fibonacci Ruler)
    # ==========================================================================

    def test_wave3_breakout_sniper(self):
        """测试第 3 浪进行中客观证伪基准与斐波坐标尺（已剔除主观预测入场信号）"""
        # P0=100, P1=120, P2=108 (守住100), 现价=122 向上突破 P1
        swings = [
            {"index": 10, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 20, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 30, "type": "Low", "price": 108.0, "confirmed": True},
        ]
        eval_res = pe.evaluate_wave_position(swings, current_price=122.0)
        self.assertEqual(eval_res["status"], "valid")
        self.assertEqual(eval_res["wave_pattern"], "developing_wave_3")
        self.assertEqual(eval_res["wave4_invalidation_level"], 120.0)
        self.assertEqual(eval_res["rule1_invalidation_level"], 100.0)
        self.assertIn("1.618", eval_res["fib_projections"])
        self.assertAlmostEqual(eval_res["fib_projections"]["1.618"], 140.36, places=1)
        self.assertNotEqual(eval_res.get("signal"), "wave3_breakout")

    def test_wave5_exhaustion_blocking(self):
        """测试第 5 浪客观证伪止损与非一票否决特性（彻底废除 rejected_by_wave5_exhaustion）"""
        # P0=100, P1=120, P2=110, P3=160, P4=145 (进行中的第5浪)
        # 现价 165 (创新高摸顶)，但成交量大幅萎缩
        swings = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 160.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 145.0, "confirmed": True},
        ]
        # 构造顶背离成交量：前段 20 根均量 300万，末端 5 根仅 100万
        df_exhaust = pd.DataFrame({
            "Open": np.full(30, 160.0),
            "High": np.full(30, 166.0),
            "Low":  np.full(30, 159.0),
            "Close":np.full(30, 165.0),
            "Volume": np.concatenate([np.full(25, 3000000.0), np.full(5, 1000000.0)])
        }, index=pd.bdate_range("2026-01-01", periods=30))

        eval_res = pe.evaluate_wave_position(swings, df=df_exhaust, current_price=165.0)
        self.assertEqual(eval_res["status"], "valid")
        self.assertEqual(eval_res["wave_pattern"], "developing_wave_5")
        self.assertEqual(eval_res["wave4_invalidation_level"], 120.0)
        self.assertNotEqual(eval_res.get("signal"), "wave5_exhaustion")
        self.assertNotEqual(eval_res.get("status"), "rejected_by_wave5_exhaustion")

        # 验证在 simulate_strategy_trade 中不再被 wave5 一票否决拦截
        tr = sl.simulate_strategy_trade(df_exhaust, 0, 165.0, side=1, stop_price=145.0, tp_price=190.0,
                                        wave_status="rejected_by_wave5_exhaustion")
        self.assertNotEqual(tr.get("status"), "rejected_by_wave5_exhaustion")

    def test_wave_c_reversal_detection(self):
        """测试 C 浪客观结构与斐波那契坐标比例尺（已剔除主观预测入场信号）"""
        # A 浪: 150 -> 130 (LA=20)
        # B 浪: 130 -> 142
        # C 浪: 142 -> 128 (LC=14, LC/LA = 0.70 ≈ 0.618 斐波黄金比例)
        # 跌破 A 浪低点 130 后收盘回到 131 (2B 假跌破收回)
        swings = [
            {"index": 5, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 15, "type": "Low",  "price": 130.0, "confirmed": True}, # A 浪低点
            {"index": 25, "type": "High", "price": 142.0, "confirmed": True}, # B 浪反弹
            {"index": 35, "type": "Low",  "price": 128.0, "confirmed": True}, # C 浪终点 2B 探底
        ]
        eval_res = pe.evaluate_wave_position(swings, current_price=131.0)
        self.assertEqual(eval_res["status"], "valid")
        self.assertEqual(eval_res["wave_pattern"], "corrective_abc")
        self.assertIn("1.000", eval_res["fib_projections"])
        self.assertNotEqual(eval_res.get("signal"), "wave_c_reversal")

    # ==========================================================================
    # 3. 大盘指数相互印证硬闸门 (Market Confirmation Gate)
    # ==========================================================================

    def test_market_dow_gate_bearish_rejection(self):
        """测试大盘处于 LH+LL 空头结构或破位时，个股多头计划被风控硬闸门一票否决"""
        # 构造大盘数据：SPY 处于空头趋势 (LH+LL)
        market_bear = {
            "SPY": {
                "mtf": {"phase": "📉 空头趋势 (LH+LL 主导)", "swings": ["LH ($550.00)", "LL ($530.00)"]},
                "htf": {"phase": "📉 偏空", "swings": []}
            },
            "QQQ": {
                "mtf": {"phase": "📉 空头趋势 (LH+LL 主导)", "swings": ["LH ($470.00)", "LL ($450.00)"]},
                "htf": {"phase": "📉 偏空", "swings": []}
            }
        }
        gate_res = sl.verify_market_dow_gate(market_bear, side=1)
        self.assertFalse(gate_res["passed"])
        self.assertEqual(gate_res["status"], "rejected_by_market_dow")

        # 测试在 simulate_strategy_trade 中被硬闸门拦截驳回
        df_dummy = pd.DataFrame({
            "Open": [100.0, 100.0],
            "High": [100.0, 115.0],
            "Low": [100.0, 95.0],
            "Close": [100.0, 105.0]
        }, index=pd.bdate_range("2026-01-01", periods=2))
        tr = sl.simulate_strategy_trade(df_dummy, 0, 100.0, side=1, stop_price=90.0, tp_price=125.0,
                                        market_dow=market_bear)
        self.assertEqual(tr["status"], "rejected_by_market_dow")

    def test_market_dow_gate_confirmed_pass(self):
        """测试大盘健康且未破位时，相互印证通过放行"""
        market_bull = {
            "SPY": {
                "mtf": {"phase": "📈 多头趋势 (HH+HL 主导)", "swings": ["HH ($570.00)", "HL ($555.00)"]},
                "htf": {"phase": "📈 多头趋势", "swings": []}
            },
            "QQQ": {
                "mtf": {"phase": "📈 偏多", "swings": ["HH ($490.00)", "HL ($475.00)"]},
                "htf": {"phase": "📈 多头趋势", "swings": []}
            }
        }
        gate_res = sl.verify_market_dow_gate(market_bull, side=1)
        self.assertTrue(gate_res["passed"])
        self.assertEqual(gate_res["status"], "confirmed_by_market")

    # ==========================================================================
    # 4. 次级折返缩量洗盘过滤 (Pullback Volume Contraction Filter)
    # ==========================================================================

    def test_pullback_volume_contraction_healthy_fill(self):
        """测试正常缩量回踩：回踩期间量能萎缩，挂单正常成交"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        # t=0 突破日放量 3,000,000, limit=100.0, stop=90.0
        # t=1~2 回踩，成交量仅 800,000 (远小于 3,000,000)
        df_healthy = pd.DataFrame({
            "Open": [105.0, 102.0, 100.5, 102.0, 103.0],
            "High": [106.0, 103.0, 101.5, 103.0, 104.0],
            "Low":  [104.0, 100.8, 99.5, 101.0, 102.0],  # bar 2 触及 99.5 <= 100.0 成交
            "Close":[105.0, 101.5, 100.2, 102.0, 103.0],
            "Volume":[3000000.0, 800000.0, 900000.0, 1000000.0, 1200000.0]
        }, index=idx)

        res = sl.simulate_limit_order(df_healthy, 0, 1, limit_price=100.0, stop_price=90.0, volume_filter=True)
        self.assertTrue(res["filled"])
        self.assertEqual(res["status"], "filled")
        self.assertEqual(res["entry_t"], 2)

    def test_pullback_volume_distribution_panic_dump_cancel(self):
        """测试异常巨量抛售：回踩过程伴随单日暴跌巨量抛售，判定为派发，挂单自动撤销"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        # t=0 突破日成交量 2,000,000
        # t=1 出现大阴线且成交量达 4,500,000 (超过 1.5 倍突破量)
        df_dump = pd.DataFrame({
            "Open": [105.0, 104.0, 99.5, 100.0, 101.0],
            "High": [106.0, 104.5, 100.0, 101.0, 102.0],
            "Low":  [104.0, 99.0, 98.0, 99.0, 100.0],   # bar 1 虽跌至 99.0 触及 100
            "Close":[105.0, 99.2, 98.5, 100.0, 101.0],  # 实体大阴线
            "Volume":[2000000.0, 4500000.0, 2500000.0, 1000000.0, 1000000.0]
        }, index=idx)

        res = sl.simulate_limit_order(df_dump, 0, 1, limit_price=100.0, stop_price=90.0, volume_filter=True)
        self.assertFalse(res["filled"])
        self.assertEqual(res["status"], "cancelled_volume_distribution")
        self.assertIn("放量抛售/派发", res["reason"])

    def test_pullback_volume_distribution_consecutive_heavy_cancel(self):
        """测试连续放量重挫：回踩过程连续两根高量阴线，挂单撤销"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        df_consec = pd.DataFrame({
            "Open": [105.0, 104.0, 101.0, 100.0, 101.0],
            "High": [106.0, 104.0, 101.0, 101.0, 102.0],
            "Low":  [104.0, 101.5, 99.0, 99.0, 100.0],
            "Close":[105.0, 101.8, 99.2, 100.0, 101.0],
            "Volume":[2000000.0, 2100000.0, 2200000.0, 800000.0, 800000.0]
        }, index=idx)

        res = sl.simulate_limit_order(df_consec, 0, 1, limit_price=100.0, stop_price=90.0, volume_filter=True)
        self.assertFalse(res["filled"])
        self.assertEqual(res["status"], "cancelled_volume_distribution")

    # ==========================================================================
    # 5. 边界与严苛审查测试 (Edge Cases & Boundary Tests)
    # ==========================================================================

    def test_wave_classification_in_long_history(self):
        """测试在长历史序列(>6个拐点)中，准确识别正在进行中的第5浪，而不是被死锁截断为 impulse_failed"""
        swings_long = [
            {"index": 2, "type": "High", "price": 105.0, "confirmed": True},
            {"index": 6, "type": "Low", "price": 95.0, "confirmed": True},
            {"index": 12, "type": "High", "price": 115.0, "confirmed": True},
            # 以下 5 根为标准正在推进的第 5 浪 (P0=100, P1=120, P2=110, P3=160, P4=145)
            {"index": 20, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 30, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 40, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 50, "type": "High", "price": 160.0, "confirmed": True},
            {"index": 60, "type": "Low", "price": 145.0, "confirmed": True},
        ]
        res = pe.classify_wave_pattern(swings_long)
        self.assertEqual(res["pattern"], "developing_wave_5")
        self.assertTrue(res["is_valid_impulse"])
        self.assertEqual(res["direction"], 1)

    def test_wave3_not_misidentified_at_wave5_peak(self):
        """测试在已完成的第5浪顶部(P5=175)，严禁误报为第3浪突破买点"""
        swings_completed = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 125.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 160.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 148.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 175.0, "confirmed": True},
        ]
        eval_res = pe.evaluate_wave_position(swings_completed, current_price=175.0)
        # 绝不能是 wave3_breakout
        self.assertNotEqual(eval_res["signal"], "wave3_breakout")

    def test_invalid_impulse_does_not_trigger_wave5_exhaustion(self):
        """测试违背铁律的无效形态绝不错误触发第5浪衰竭诱多拦截"""
        # P0=100, P1=120, P2=95 (跌破起点，铁律1违背), P3=140, P4=125, P5=155
        swings_invalid = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 95.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 125.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 155.0, "confirmed": True},
        ]
        eval_res = pe.evaluate_wave_position(swings_invalid, current_price=155.0)
        self.assertEqual(eval_res["signal"], "neutral")

    def test_corrective_abc_in_long_history(self):
        """测试长序列下 ABC 调整浪正常识别与客观斐波坐标尺（已剔除主观预测入场信号）"""
        swings_abc = [
            {"index": 2, "type": "Low", "price": 90.0, "confirmed": True},
            {"index": 10, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 18, "type": "Low", "price": 120.0, "confirmed": True},
            # 最近 4 拐点为 ABC: 0=150, A=130, B=142, C=128
            {"index": 25, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 35, "type": "Low",  "price": 130.0, "confirmed": True},
            {"index": 45, "type": "High", "price": 142.0, "confirmed": True},
            {"index": 55, "type": "Low",  "price": 128.0, "confirmed": True},
        ]
        eval_res = pe.evaluate_wave_position(swings_abc, current_price=131.0)
        self.assertEqual(eval_res["status"], "valid")
        self.assertEqual(eval_res["wave_pattern"], "corrective_abc")
        self.assertIn("1.000", eval_res["fib_projections"])
        self.assertNotEqual(eval_res.get("signal"), "wave_c_reversal")


if __name__ == "__main__":
    unittest.main()
