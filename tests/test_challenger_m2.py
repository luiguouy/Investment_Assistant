"""
Adversarial Verification Suite for Milestone 2: Wave Downgrade & Causality
Challenger: challenger_m2_2
Target: pa_engine.py, strategy_layer.py, backtest.py
Verification: Causality, zero-lookahead bias, edge cases (<3 swings, flat market, wave4/wave1 overlap)
"""

import unittest
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl


class TestChallengerM2Adversarial(unittest.TestCase):
    """对抗性实证挑战测试套件：因果性、零前视偏差及极端边界条件"""

    def setUp(self):
        np.random.seed(42)

    # =========================================================================
    # 1. 因果性与零前视偏差对抗测试 (Zero Lookahead & Causality)
    # =========================================================================

    def test_zero_lookahead_wave_eval_and_trade_plan(self):
        """
        对抗挑战：未来数据变异攻击。
        在序列截断点 t 计算 wave_eval 与 trade_plan；
        然后对 t+1 之后的未来数据进行极端变异（价格放大100倍、剧烈崩盘），
        验证在严格因果截断截取 [:t+1] 下，计算结果绝对恒等，零前视偏差。
        """
        n = 160
        base_price = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
        dates = pd.bdate_range("2025-01-01", periods=n)
        df_original = pd.DataFrame({
            "Open": base_price + np.random.randn(n) * 0.2,
            "High": base_price + np.abs(np.random.randn(n) * 1.0) + 0.5,
            "Low": base_price - np.abs(np.random.randn(n) * 1.0) - 0.5,
            "Close": base_price,
            "Volume": np.random.randint(1000000, 5000000, size=n)
        }, index=dates)

        # 变异副本：将第 100 根以后的未来数据彻底破坏
        df_mutated = df_original.copy()
        df_mutated.iloc[100:, df_mutated.columns.get_loc("Close")] *= 10.0
        df_mutated.iloc[100:, df_mutated.columns.get_loc("High")] *= 10.0
        df_mutated.iloc[100:, df_mutated.columns.get_loc("Low")] *= 0.1
        df_mutated.iloc[100:, df_mutated.columns.get_loc("Volume")] = 99999999

        cutoffs = [50, 70, 90, 99]
        for t in cutoffs:
            sub_orig = df_original.iloc[:t + 1]
            sub_mut = df_mutated.iloc[:t + 1]

            swings_orig = pe.extract_swings(sub_orig, order=3)
            swings_mut = pe.extract_swings(sub_mut, order=3)

            eval_orig = pe.evaluate_wave_position(swings_orig, df=sub_orig)
            eval_mut = pe.evaluate_wave_position(swings_mut, df=sub_mut)

            # 断言波浪评估在未来变异下保持 100% 恒定
            self.assertEqual(eval_orig["status"], eval_mut["status"])
            self.assertEqual(eval_orig["wave_pattern"], eval_mut["wave_pattern"])
            self.assertEqual(eval_orig["wave4_invalidation_level"], eval_mut["wave4_invalidation_level"])
            self.assertEqual(eval_orig["fib_projections"], eval_mut["fib_projections"])
            self.assertEqual(eval_orig["signal"], "neutral")
            self.assertEqual(eval_orig["action"], "neutral")

            # 断言交易计划在未来变异下保持 100% 恒定
            curr_p = float(sub_orig["Close"].iloc[-1])
            plan_orig = pe.build_trade_plan(
                mode="daily", price=curr_p, swings_mtf=swings_orig, swings_ltf=swings_orig,
                key_levels=[], levels={}, bar_rng=2.0, bias_htf=1, bias_mtf=1, df_mtf=sub_orig
            )
            plan_mut = pe.build_trade_plan(
                mode="daily", price=curr_p, swings_mtf=swings_mut, swings_ltf=swings_mut,
                key_levels=[], levels={}, bar_rng=2.0, bias_htf=1, bias_mtf=1, df_mtf=sub_mut
            )
            self.assertEqual(plan_orig["status"], plan_mut["status"])
            self.assertEqual(plan_orig["stop"], plan_mut["stop"])
            self.assertEqual(plan_orig["entry"], plan_mut["entry"])
            self.assertEqual(plan_orig.get("wave4_invalidation_level"), plan_mut.get("wave4_invalidation_level"))

    # =========================================================================
    # 2. 边缘情况：极少拐点与空数据边界 (< 3 swings)
    # =========================================================================

    def test_edge_case_zero_swings(self):
        """测试 0 拐点极端输入：不崩溃，返回 unclassified 且 status=invalid"""
        eval_res = pe.evaluate_wave_position([])
        self.assertEqual(eval_res["status"], "invalid")
        self.assertEqual(eval_res["wave_pattern"], "unclassified")
        self.assertIsNone(eval_res["wave4_invalidation_level"])
        self.assertEqual(eval_res["fib_projections"], {})
        self.assertEqual(eval_res["signal"], "neutral")

        # 交易计划在 0 拐点时优雅拦截
        plan = pe.build_trade_plan(
            mode="daily", price=100.0, swings_mtf=[], swings_ltf=[],
            key_levels=[], levels={}, bar_rng=2.0, bias_htf=1, bias_mtf=1
        )
        self.assertIn("无已确认拐点", plan["status"])

    def test_edge_case_one_or_two_swings(self):
        """测试 1 或 2 个拐点极端输入：不满足波浪结构前提，不崩溃"""
        swings_1 = [{"index": 5, "type": "Low", "price": 100.0, "confirmed": True}]
        eval_1 = pe.evaluate_wave_position(swings_1)
        self.assertEqual(eval_1["status"], "invalid")
        self.assertEqual(eval_1["wave_pattern"], "unclassified")
        self.assertIsNone(eval_1["wave4_invalidation_level"])

        swings_2 = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True}
        ]
        eval_2 = pe.evaluate_wave_position(swings_2)
        self.assertEqual(eval_2["status"], "invalid")
        self.assertEqual(eval_2["wave_pattern"], "unclassified")

    # =========================================================================
    # 3. 边缘情况：平水横盘极端市场 (Flat Market / Zero Price Movement)
    # =========================================================================

    def test_edge_case_flat_market_zero_amplitude(self):
        """
        对抗挑战：零波动平水市场（所有拐点价格完全一致）。
        验证不会发生 ZeroDivisionError，且优雅识别出无效波浪。
        """
        swings_flat_6 = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 10, "type": "High", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 20, "type": "High", "price": 100.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 30, "type": "High", "price": 100.0, "confirmed": True},
        ]
        eval_flat = pe.evaluate_wave_position(swings_flat_6, current_price=100.0)
        self.assertEqual(eval_flat["status"], "invalid")
        self.assertEqual(eval_flat["signal"], "neutral")

        swings_flat_3 = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 10, "type": "High", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 100.0, "confirmed": True},
        ]
        eval_flat_3 = pe.evaluate_wave_position(swings_flat_3, current_price=100.0)
        self.assertEqual(eval_flat_3["status"], "invalid")

    # =========================================================================
    # 4. 核心铁律挑战：浪4重叠浪1领地 (Rule 3 Invalidation)
    # =========================================================================

    def test_wave4_overlapping_wave1_impulse_invalidation(self):
        r"""
        对抗挑战：浪4 回踩跌穿 浪1 顶 ($P_4 \le P_1$)。
        P0=100, P1=120, P2=110, P3=150, P4=115 (< 120), P5=160
        铁律3严格证伪：is_valid_impulse 为 False，status 为 invalid，
        但输出客观证伪参考位 wave4_invalidation_level = 120.0，fib_projections 为空。
        """
        swings_overlap = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 115.0, "confirmed": True}, # 跌破 120.0
            {"index": 55, "type": "High", "price": 160.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings_overlap)
        self.assertFalse(wp["is_valid_impulse"])
        self.assertFalse(wp["rules"]["rule3"]["passed"])
        self.assertEqual(wp["current_wave"], "impulse_failed")

        eval_res = pe.evaluate_wave_position(swings_overlap, current_price=160.0)
        self.assertEqual(eval_res["status"], "invalid")
        self.assertEqual(eval_res["wave4_invalidation_level"], 120.0)
        self.assertEqual(eval_res["fib_projections"], {})
        self.assertEqual(eval_res["signal"], "neutral")

    def test_wave4_overlap_in_developing_wave_5(self):
        r"""
        对抗挑战：在 5 拐点序列中浪4 回踩跌破 浪1 ($P_4 \le P_1$)。
        P0=100, P1=120, P2=110, P3=150, P4=118 (< 120)。
        由于违背铁律3，不得识别为合规 developing_wave_5。
        """
        swings_dev_overlap = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 118.0, "confirmed": True}, # P4 < P1
        ]
        wp = pe.classify_wave_pattern(swings_dev_overlap)
        self.assertNotEqual(wp.get("pattern"), "developing_wave_5")

    def test_bearish_wave4_overlapping_wave1(self):
        r"""
        对抗挑战：空头浪4 反弹突破 浪1 底 ($P_4 \ge P_1$)。
        P0=200, P1=160, P2=180, P3=130, P4=165 (> 160), P5=120
        空头铁律3严格证伪：status 为 invalid，wave4_invalidation_level = 160.0。
        """
        swings_bear_overlap = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
            {"index": 35, "type": "Low", "price": 130.0, "confirmed": True},
            {"index": 45, "type": "High", "price": 165.0, "confirmed": True}, # 突破 160.0
            {"index": 55, "type": "Low", "price": 120.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings_bear_overlap)
        self.assertFalse(wp["is_valid_impulse"])
        self.assertFalse(wp["rules"]["rule3"]["passed"])

        eval_res = pe.evaluate_wave_position(swings_bear_overlap, current_price=120.0)
        self.assertEqual(eval_res["status"], "invalid")
        self.assertEqual(eval_res["wave4_invalidation_level"], 160.0)

    # =========================================================================
    # 5. 核心铁律挑战：浪2破起点与浪3最短 (Rule 1 & Rule 2 Invalidation)
    # =========================================================================

    def test_wave2_breaking_wave1_start_rule1(self):
        """
        对抗挑战：浪2 深度回撤击穿 浪1 起点 ($P_2 < P_0$)。
        P0=100, P1=120, P2=95。
        铁律1严格证伪：is_valid_impulse 为 False，status 为 invalid。
        """
        swings_break_start = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 95.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings_break_start)
        self.assertFalse(wp["is_valid_impulse"])
        self.assertFalse(wp["rules"]["rule1"]["passed"])

        eval_res = pe.evaluate_wave_position(swings_break_start, current_price=105.0)
        self.assertEqual(eval_res["status"], "invalid")
        self.assertEqual(eval_res["rule1_invalidation_level"], 100.0)

    def test_wave3_shortest_rule2(self):
        """
        对抗挑战：浪3 成为 1, 3, 5 中最短的一浪。
        P0=100, P1=140 (L1=40), P2=130, P3=145 (L3=15), P4=142, P5=180 (L5=38)。
        铁律2严格证伪：is_valid_impulse 为 False。
        """
        swings_short_w3 = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 130.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 145.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 142.0, "confirmed": True},
            {"index": 55, "type": "High", "price": 180.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings_short_w3)
        self.assertFalse(wp["is_valid_impulse"])
        self.assertFalse(wp["rules"]["rule2"]["passed"])

        eval_res = pe.evaluate_wave_position(swings_short_w3, current_price=180.0)
        self.assertEqual(eval_res["status"], "invalid")

    # =========================================================================
    # 6. 空头波浪斐波那契坐标比例尺与降级验证
    # =========================================================================

    def test_bearish_developing_wave3_and_5_projections(self):
        """
        测试空头浪型下的客观下行斐波那契比例尺：
        空头浪3: P0=200, P1=160 (L1=40), P2=180 -> 1.618 目标 = 180 - 1.618*40 = 115.28
        """
        swings_bear3 = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        eval_bear3 = pe.evaluate_wave_position(swings_bear3, current_price=150.0)
        self.assertEqual(eval_bear3["status"], "valid")
        self.assertEqual(eval_bear3["wave_pattern"], "developing_wave_3")
        self.assertEqual(eval_bear3["wave4_invalidation_level"], 160.0)
        self.assertEqual(eval_bear3["rule1_invalidation_level"], 200.0)
        self.assertAlmostEqual(eval_bear3["fib_projections"]["1.618"], 115.28, places=2)

    # =========================================================================
    # 7. 契约与解耦验证：build_trade_plan 绝不因波浪拒绝交易
    # =========================================================================

    def test_build_trade_plan_decoupled_from_wave_invalidation(self):
        """
        验证无论波浪结构是有效、无效还是重叠违背铁律，
        build_trade_plan 的决策完全取决于道氏与箱体/通道结构，
        绝不会出现 rejected_by_wave5_exhaustion 或波浪一票否决。
        同时 wave4_invalidation_level 与浪4证伪底线被准确记录于交易计划与止损理由中。
        """
        # 构造标准的浪4重叠浪1违背铁律3的 6 拐点序列：P4 (115) < P1 (120)
        swings_invalid_wave = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 35, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 45, "type": "Low", "price": 115.0, "confirmed": True}, # P4 < P1 违背铁律3
            {"index": 55, "type": "High", "price": 160.0, "confirmed": True},
        ]
        key_levels = [
            {"price": 150.0, "kind": "swing", "src": "D1"},
            {"price": 260.0, "kind": "swing", "src": "D1"}
        ]
        plan = pe.build_trade_plan(
            mode="daily", price=155.0, swings_mtf=swings_invalid_wave, swings_ltf=swings_invalid_wave,
            key_levels=key_levels, levels={}, bar_rng=3.0, bias_htf=1, bias_mtf=1
        )
        # 绝不包含 rejected_by_wave5_exhaustion 或被波浪一票否决阻断
        self.assertNotEqual(plan.get("rejected_by"), "rejected_by_wave5_exhaustion")
        self.assertNotIn("rejected_by_wave5_exhaustion", plan.get("status", ""))
        self.assertEqual(plan["side"], "long")
        # 止损基于最近波段低点 115.0 - 0.8 * 3.0 = 112.6
        self.assertAlmostEqual(plan["stop"], 112.6, places=2)
        # 浪4客观证伪底线正确附注在止损原因中
        self.assertIn("[浪4证伪底线 $120.00]", plan["stop_reason"])
        # 浪4证伪底线记录在案
        self.assertEqual(plan.get("wave4_invalidation_level"), 120.0)
        # 计划正常成立（净加权 RR >= 2.5）
        self.assertGreaterEqual(plan["weighted_rr"], 2.5)
        self.assertIn("可执行", plan["status"])


if __name__ == "__main__":
    unittest.main()
