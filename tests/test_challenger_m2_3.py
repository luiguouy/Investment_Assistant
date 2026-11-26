"""
Adversarial Verification Suite for Milestone 2 Remediation
Challenger: challenger_m2_3
Target: pa_engine.py (_print_trade_plan robustness, short TP2 projections, wave_pattern types)
Verification: Empirical stress test and adversarial inputs
"""

import io
import contextlib
import unittest
import numpy as np
import pandas as pd

import pa_engine as pe


class TestChallengerM23Adversarial(unittest.TestCase):
    """对抗性实证挑战：专门针对 worker_m2_2 修复的 _print_trade_plan 与做空 TP2 斐波投影"""

    def setUp(self):
        np.random.seed(42)
        # 基础骨架交易计划
        self.base_plan = {
            "mode": "daily",
            "side": "short",
            "entry": 100.0,
            "entry_type": "现价（无临近阻力可用）",
            "entry_trigger": 98.0,
            "stop": 105.0,
            "stop_reason": "最近已确认波段高点 $104.00 外侧 1.00",
            "tp1": 90.0,
            "tp2": 80.0,
            "risk_per_share": 5.0,
            "stop_distance_pct": 5.0,
            "rr1": 2.0,
            "rr2": 4.0,
            "min_rr": 2.0,
            "equity": 50000.0,
            "risk_pct": 0.01,
            "risk_budget": 500.0,
            "shares": 100,
            "bias_htf": -1,
            "bias_mtf": -1,
            "status": "✅ 可执行（RR 1:2.00）"
        }

    # =========================================================================
    # 1. _print_trade_plan 极端与畸形 wave_pattern 输入鲁棒性挑战
    # =========================================================================

    def _call_print_plan(self, plan):
        """辅助函数：捕获 stdout 输出并确保无崩溃"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pe._print_trade_plan(plan)
        return buf.getvalue()

    def test_print_trade_plan_wave_eval_none(self):
        """挑战：plan 无 wave_eval 或 wave_eval 为 None"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = None
        output = self._call_print_plan(plan)
        self.assertIn("【交易计划】" if "【交易计划】" in output else "狙击手交易计划", output)
        self.assertIn("做空 (Short)", output)
        self.assertNotIn("艾略特浪型", output)

    def test_print_trade_plan_wave_pattern_none(self):
        """挑战：wave_pattern 为 None"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": None,
            "reason": "测试无波浪形态"
        }
        output = self._call_print_plan(plan)
        self.assertNotIn("艾略特浪型", output)

    def test_print_trade_plan_wave_pattern_empty_string(self):
        """挑战：wave_pattern 为空字符串 "" """
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "",
            "reason": "空形态字符串"
        }
        output = self._call_print_plan(plan)
        self.assertNotIn("艾略特浪型", output)

    def test_print_trade_plan_wave_pattern_unclassified(self):
        """挑战：wave_pattern 为 'unclassified' 时不应展示形态"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "invalid",
            "wave_pattern": "unclassified",
            "reason": "未识别出合规波浪结构"
        }
        output = self._call_print_plan(plan)
        self.assertNotIn("艾略特浪型", output)

    def test_print_trade_plan_wave_pattern_str_with_detail(self):
        """挑战：M2 规范 string wave_pattern 且带有 wave_pattern_detail"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "impulse_5_wave",
            "wave_pattern_detail": {
                "summary": "完整多头5浪冲量驱动结构（三大铁律全部通过）"
            }
        }
        output = self._call_print_plan(plan)
        self.assertIn("艾略特浪型: 完整多头5浪冲量驱动结构（三大铁律全部通过）", output)

    def test_print_trade_plan_wave_pattern_str_fallback_to_reason(self):
        """挑战：M2 规范 string wave_pattern，无 detail，fallback 到 reason"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "developing_wave_3",
            "reason": "处于进行中第3浪推进，斐波那契坐标投影: 1.618x=$120.00"
        }
        output = self._call_print_plan(plan)
        self.assertIn("艾略特浪型: 处于进行中第3浪推进", output)

    def test_print_trade_plan_wave_pattern_str_fallback_to_raw_str(self):
        """挑战：M2 规范 string wave_pattern，无 detail 且无 reason，回退到 p_str 自身"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "custom_exotic_wave"
        }
        output = self._call_print_plan(plan)
        self.assertIn("艾略特浪型: custom_exotic_wave", output)

    def test_print_trade_plan_wave_pattern_dict_compatibility(self):
        """挑战：向下兼容旧版 dict wave_pattern"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": {
                "pattern": "impulse_5_wave",
                "summary": "旧版字典格式波浪摘要"
            }
        }
        output = self._call_print_plan(plan)
        self.assertIn("艾略特浪型: 旧版字典格式波浪摘要", output)

    def test_print_trade_plan_wave_pattern_dict_unclassified(self):
        """挑战：旧版 dict wave_pattern 但 pattern='unclassified'"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "invalid",
            "wave_pattern": {
                "pattern": "unclassified",
                "summary": "未识别出合规波浪结构"
            }
        }
        output = self._call_print_plan(plan)
        self.assertNotIn("艾略特浪型", output)

    def test_print_trade_plan_wave_pattern_arbitrary_types(self):
        """对抗挑战：wave_pattern 为极端非法类型（int, float, list, bool）不崩溃"""
        adversarial_patterns = [123, 45.67, True, False, ["wave_list"], (1, 2)]
        for bad_pattern in adversarial_patterns:
            plan = dict(self.base_plan)
            plan["wave_eval"] = {
                "status": "valid",
                "wave_pattern": bad_pattern,
                "reason": "非法类型输入测试"
            }
            output = self._call_print_plan(plan)
            # 必须平稳打印，绝不引发 TypeError 或 AttributeError
            self.assertIn("狙击手交易计划", output)

    def test_print_trade_plan_wave_pattern_detail_non_dict(self):
        """对抗挑战：wave_pattern_detail 为非 dict 异常对象（如字符串或列表）不崩溃"""
        plan = dict(self.base_plan)
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "impulse_5_wave",
            "wave_pattern_detail": "invalid_detail_string",
            "reason": "降级原因文本"
        }
        output = self._call_print_plan(plan)
        self.assertIn("艾略特浪型: 降级原因文本", output)

    def test_print_trade_plan_wave_note_and_eval_coexistence(self):
        """挑战：wave_note 与 wave_eval 同时存在时格式正确"""
        plan = dict(self.base_plan)
        plan["wave_note"] = "🌊 客观波浪铁律证伪位: 浪4不破浪1极值 $320.28"
        plan["wave_eval"] = {
            "status": "valid",
            "wave_pattern": "impulse_5_wave",
            "wave_pattern_detail": {"summary": "多头5浪"}
        }
        output = self._call_print_plan(plan)
        self.assertIn("波浪结构与证伪位: 🌊 客观波浪铁律证伪位: 浪4不破浪1极值 $320.28", output)
        self.assertIn("艾略特浪型: 多头5浪", output)

    # =========================================================================
    # 2. 做空 TP2 斐波那契投影与多空对称性对抗挑战
    # =========================================================================

    def test_short_tp2_fallback_to_wave_fib_projections(self):
        """
        对抗挑战：空头交易且 levels 中无 fib_extensions 时，
        必须成功回退挂接 wave_eval['fib_projections']['1.618']。
        验证方向性：ext 必须严格小于 tp1，且 tp2 准确赋值为 1.618 扩展位。
        """
        swings_bear = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        key_levels = [
            {"price": 185.0, "kind": "swing", "src": "D1"},  # 阻力位（stop 依据）
            {"price": 140.0, "kind": "swing", "src": "D1"},  # TP1 支撑位
        ]
        # levels 不提供 fib_extensions
        levels = {}

        plan = pe.build_trade_plan(
            mode="daily", price=165.0, swings_mtf=swings_bear, swings_ltf=swings_bear,
            key_levels=key_levels, levels=levels, bar_rng=3.0, bias_htf=-1, bias_mtf=-1,
            market_dow={"SPY_bias": -1}
        )
        self.assertEqual(plan["side"], "short")
        self.assertEqual(plan["tp1"], 140.0)
        # 空头 developing_wave_3: P0=200, P1=160 (L1=40), P2=180
        # fib 1.618 投影 = 180 - 1.618 * 40 = 115.28 < tp1(140.0)
        self.assertIsNotNone(plan["tp2"])
        self.assertAlmostEqual(plan["tp2"], 115.28, places=2)
        self.assertLess(plan["tp2"], plan["tp1"])
        self.assertGreater(plan["rr2"], plan["rr1"])

    def test_short_tp2_fib_extension_precedence_over_wave_projections(self):
        """
        对抗挑战：当 levels['fib_extensions']['1.618'] 存在时，
        优先使用 levels 中的扩展位，wave_eval 仅作 fallback。
        """
        swings_bear = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        key_levels = [
            {"price": 185.0, "kind": "swing", "src": "D1"},
            {"price": 140.0, "kind": "swing", "src": "D1"},
        ]
        # levels 提供显式的 fib_extensions
        levels = {
            "fib_extensions": {"1.618": 125.0}
        }

        plan = pe.build_trade_plan(
            mode="daily", price=165.0, swings_mtf=swings_bear, swings_ltf=swings_bear,
            key_levels=key_levels, levels=levels, bar_rng=3.0, bias_htf=-1, bias_mtf=-1,
            market_dow={"SPY_bias": -1}
        )
        self.assertEqual(plan["tp2"], 125.0)

    def test_short_tp2_rejects_projection_greater_than_tp1(self):
        """
        对抗挑战：做空时如果波浪投影或扩展位异常高于 TP1（如反向多头投影 165.0 > TP1 140.0），
        必须被 float(ext) < tp1 拒绝，不得出现 TP2 > TP1 的倒错。
        """
        swings_bear = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        # key_levels 中提供更远的第3级支撑位 120.0
        key_levels = [
            {"price": 185.0, "kind": "swing", "src": "D1"},
            {"price": 140.0, "kind": "swing", "src": "D1"}, # TP1
            {"price": 120.0, "kind": "swing", "src": "D1"}, # Tier 3 far level
        ]
        # 异常的倒错扩展位（大于 TP1）
        levels = {
            "fib_extensions": {"1.618": 150.0}  # 150 > 140!
        }

        plan = pe.build_trade_plan(
            mode="daily", price=165.0, swings_mtf=swings_bear, swings_ltf=swings_bear,
            key_levels=key_levels, levels=levels, bar_rng=3.0, bias_htf=-1, bias_mtf=-1,
            market_dow={"SPY_bias": -1}
        )
        # 150.0 必须被驳回，回退到 far key level 120.0
        self.assertEqual(plan["tp2"], 120.0)

    def test_short_tp2_when_tp1_is_none(self):
        """
        对抗挑战：当无前方支撑导致 TP1 为 None 时，
        即使存在波浪 1.618 投影，TP2 也必须为 None，不可凭空跨级成立。
        """
        swings_bear = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        # 无任何下方支撑
        key_levels = [
            {"price": 185.0, "kind": "swing", "src": "D1"},
        ]
        plan = pe.build_trade_plan(
            mode="daily", price=165.0, swings_mtf=swings_bear, swings_ltf=swings_bear,
            key_levels=key_levels, levels={}, bar_rng=3.0, bias_htf=-1, bias_mtf=-1,
            market_dow={"SPY_bias": -1}
        )
        self.assertIsNone(plan["tp1"])
        self.assertIsNone(plan["tp2"])
        self.assertIn("前方无有效结构目标位", plan["status"])

    def test_long_short_tp2_symmetry(self):
        """
        对抗挑战：多空双方在相同斐波那契投影幅度下的几何对称性测试。
        多头：P0=100, P1=140, P2=120 -> L1=40, ext=120+1.618*40=184.72
        空头：P0=200, P1=160, P2=180 -> L1=40, ext=180-1.618*40=115.28
        验证多空两侧均能在 levels 缺省时准确触发波浪投影回退，且公式方向完全镜像对称。
        """
        swings_bull = [
            {"index": 5, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 140.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 120.0, "confirmed": True},
        ]
        key_levels_bull = [
            {"price": 115.0, "kind": "swing", "src": "D1"},  # 止损依据
            {"price": 150.0, "kind": "swing", "src": "D1"},  # TP1
        ]
        plan_bull = pe.build_trade_plan(
            mode="daily", price=135.0, swings_mtf=swings_bull, swings_ltf=swings_bull,
            key_levels=key_levels_bull, levels={}, bar_rng=3.0, bias_htf=1, bias_mtf=1
        )
        self.assertEqual(plan_bull["side"], "long")
        self.assertAlmostEqual(plan_bull["tp2"], 184.72, places=2)

        swings_bear = [
            {"index": 5, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 160.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 180.0, "confirmed": True},
        ]
        key_levels_bear = [
            {"price": 185.0, "kind": "swing", "src": "D1"},
            {"price": 140.0, "kind": "swing", "src": "D1"},
        ]
        plan_bear = pe.build_trade_plan(
            mode="daily", price=165.0, swings_mtf=swings_bear, swings_ltf=swings_bear,
            key_levels=key_levels_bear, levels={}, bar_rng=3.0, bias_htf=-1, bias_mtf=-1,
            market_dow={"SPY_bias": -1}
        )
        self.assertEqual(plan_bear["side"], "short")
        self.assertAlmostEqual(plan_bear["tp2"], 115.28, places=2)

        # 验证距离镜像绝对相等: |184.72 - 120| == 64.72, |115.28 - 180| == 64.72
        dist_bull = abs(plan_bull["tp2"] - 120.0)
        dist_bear = abs(plan_bear["tp2"] - 180.0)
        self.assertAlmostEqual(dist_bull, dist_bear, places=2)


if __name__ == "__main__":
    unittest.main()
