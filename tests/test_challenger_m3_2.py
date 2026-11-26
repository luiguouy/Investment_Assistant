#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_challenger_m3_2.py:
Adversarial Verification Suite for Milestone 3 (200 EMA Primary Trend Gate)
Challenger: challenger_m3_2 (Roles: critic, specialist)
Targets:
  - strategy_layer.py: verify_primary_trend_gate
  - pa_engine.py: build_trade_plan (trend gate integration)
  - strategy_layer.py: simulate_strategy_trade (trend gate integration)

Verification Dimensions:
  1. Strict zero lookahead test under dynamic truncation & future poisoning
  2. Extreme gaps (100x spike, 99.999% crash, flash crashes, 1e14 scale, 1e-8 scale)
  3. Zero volatility (pure flatline, sub-epsilon jitter)
  4. Boundary lengths (<10 bars, 0 bars, 1 bar, 2 bars, 199/200/201 transitions)
  5. Exact touch on 200 EMA & epsilon boundary sensitivity
  6. Negative prices, case insensitivity, malformed inputs & defensive robustness
  7. End-to-end integration and causal decision consistency
"""

import math
import unittest
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl


def _generate_synthetic_market_data(n_bars: int = 500, base_price: float = 100.0,
                                    seed: int = 42) -> pd.DataFrame:
    """生成具备代表性的价格时序，包含趋势、波动与均值回归"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)
    prices = [base_price]
    for _ in range(1, n_bars):
        ret = rng.normal(0.0005, 0.015)
        new_p = max(prices[-1] * (1.0 + ret), 0.01)
        prices.append(new_p)
    prices = np.array(prices)
    highs = prices * (1.0 + rng.uniform(0.002, 0.01, size=n_bars))
    lows = prices * (1.0 - rng.uniform(0.002, 0.01, size=n_bars))
    opens = (highs + lows) / 2.0
    vols = rng.integers(100000, 5000000, size=n_bars)

    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(prices, 2),
        "Volume": vols
    }, index=dates)
    return df


class TestCausalityZeroLookaheadAdversarial(unittest.TestCase):
    """
    1. 严格因果性与零未来函数对抗性挑战 (Strict Zero Lookahead Test)
    验证切片重算、点位传参与未来数据投毒下决策与数值的 100% 不变性。
    """

    def setUp(self):
        self.df_market = _generate_synthetic_market_data(n_bars=500, base_price=150.0, seed=123)

    def test_dense_truncation_consistency_all_regimes(self):
        """
        挑战：全区段密集截断一致性。
        在不同样本长度（极短、短、临界200、长序列）下，
        verify_primary_trend_gate(df, t=t) 与 verify_primary_trend_gate(df.iloc[:t+1])
        的 EMA 数值、现价、判定结果必须 100% 相同。
        """
        eval_indices = [
            0, 1, 2, 3, 5, 8, 9, 10, 15, 30, 50, 99, 100,
            198, 199, 200, 201, 202, 250, 300, 399, 450, 499
        ]
        for t in eval_indices:
            df_sub = self.df_market.iloc[:t + 1].copy()

            for side in [1, -1]:
                res_with_t = sl.verify_primary_trend_gate(self.df_market, side=side, ema_period=200, t=t)
                res_sub_slice = sl.verify_primary_trend_gate(df_sub, side=side, ema_period=200)

                self.assertEqual(res_with_t["passed"], res_sub_slice["passed"],
                                 f"t={t}, side={side} 截断判定不一致！")
                self.assertEqual(res_with_t["status"], res_sub_slice["status"],
                                 f"t={t}, side={side} 状态不一致！")
                self.assertAlmostEqual(res_with_t["current_price"], res_sub_slice["current_price"], places=4,
                                       msg=f"t={t} 现价不一致！")
                self.assertAlmostEqual(res_with_t["ema_val"], res_sub_slice["ema_val"], places=4,
                                       msg=f"t={t} EMA 不一致！")

    def test_future_poisoning_adversarial(self):
        """
        挑战：未来数据恶毒投毒对抗。
        截断时点 t=220，将 t+1 后的所有数据分别替换为：
          A. 超级暴涨 (+10^9)
          B. 超级暴跌 (-10^9)
          C. NaN 坏值
          D. Inf 极大值
          E. 零值
        门禁在 t 时点的输出必须完全免疫，毫无未来数据泄露！
        """
        t_cutoff = 220
        res_baseline_long = sl.verify_primary_trend_gate(self.df_market, side=1, ema_period=200, t=t_cutoff)
        res_baseline_short = sl.verify_primary_trend_gate(self.df_market, side=-1, ema_period=200, t=t_cutoff)

        poison_scenarios = [
            ("super_spike", 1e9),
            ("super_crash", -1e9),
            ("nan_values", np.nan),
            ("inf_values", np.inf),
            ("zero_values", 0.0),
        ]

        for name, poison_val in poison_scenarios:
            df_poisoned = self.df_market.copy()
            df_poisoned.iloc[t_cutoff + 1:, df_poisoned.columns.get_loc("Close")] = poison_val

            res_p_long = sl.verify_primary_trend_gate(df_poisoned, side=1, ema_period=200, t=t_cutoff)
            res_p_short = sl.verify_primary_trend_gate(df_poisoned, side=-1, ema_period=200, t=t_cutoff)

            self.assertEqual(res_p_long["passed"], res_baseline_long["passed"],
                             f"投毒 {name} 破坏了多头判定！")
            self.assertEqual(res_p_short["passed"], res_baseline_short["passed"],
                             f"投毒 {name} 破坏了空头判定！")
            self.assertEqual(res_p_long["ema_val"], res_baseline_long["ema_val"],
                             f"投毒 {name} 污染了 EMA 数值！")
            self.assertEqual(res_p_long["current_price"], res_baseline_long["current_price"],
                             f"投毒 {name} 污染了现价！")

    def test_pure_mathematical_oracle_step_by_step(self):
        """
        挑战：标准递推数学预言机（Oracle）精确比对。
        使用纯 Python 手写递推公式:
          EMA_0 = P_0
          EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}, 其中 alpha = 2 / (span + 1)
        检验 strategy_layer 中的实现与数学理论值在每一根 K 线上误差均小于 1e-8。
        """
        closes = self.df_market["Close"].values[:250]
        # 针对 span=200 验证完整 250 根序列
        span = 200
        alpha = 2.0 / (span + 1.0)
        oracle_ema = [closes[0]]
        for p in closes[1:]:
            oracle_ema.append(alpha * p + (1.0 - alpha) * oracle_ema[-1])

        df_full = pd.DataFrame({"Close": closes})
        # 验证 200 根之后（处于标准 200 周期状态）
        for t in range(199, 250):
            res = sl.verify_primary_trend_gate(df_full, side=1, ema_period=200, t=t)
            expected_ema = oracle_ema[t]
            self.assertAlmostEqual(res["ema_val"], round(expected_ema, 4), places=4,
                                   msg=f"第 {t} 根与纯数学预言机不符！")


class TestNumericalStabilityExtremeGaps(unittest.TestCase):
    """
    2. 极端价格跳空与数值稳定性挑战 (Extreme Gaps & Numerical Stability)
    测试包括 100x 跳空暴涨、99.999% 闪崩、V型剧烈反转、天文数字与微小数字。
    """

    def test_extreme_gap_up_100x(self):
        """挑战：稳定价格基线后出现瞬间 100x 巨幅跳空暴涨"""
        prices = [100.0] * 200 + [10000.0]
        df = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertFalse(res_short["passed"])
        self.assertFalse(math.isnan(res_long["ema_val"]))
        self.assertFalse(math.isinf(res_long["ema_val"]))
        # alpha = 2/201 ≈ 0.00995. EMA 应从 100 增至约 100 + 0.00995 * 9900 ≈ 198.5
        expected_approx = (1.0 - 2.0 / 201.0) * 100.0 + (2.0 / 201.0) * 10000.0
        self.assertAlmostEqual(res_long["ema_val"], round(expected_approx, 4), places=2)

    def test_extreme_crash_99_999_pct(self):
        """挑战：稳定价格基线后出现 99.999% 毁灭性闪崩至 0.001"""
        prices = [100.0] * 200 + [0.001]
        df = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df, side=-1, ema_period=200)

        self.assertFalse(res_long["passed"])
        self.assertTrue(res_short["passed"])
        self.assertGreater(res_long["ema_val"], res_long["current_price"])
        self.assertFalse(math.isnan(res_long["ema_val"]))
        self.assertFalse(math.isinf(res_long["ema_val"]))

    def test_flash_crash_and_v_reversal(self):
        """挑战：闪崩后下一根 K 线瞬间 V 型收回 (100 -> 0.01 -> 100)"""
        prices = [100.0] * 200 + [0.01, 100.0]
        df = pd.DataFrame({"Close": prices})

        # 在闪崩 bar (t = -2)
        res_crash = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=200)
        self.assertFalse(res_crash["passed"])

        # 在恢复 bar (t = -1)
        res_recovered = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=201)
        self.assertTrue(res_recovered["passed"])
        self.assertGreaterEqual(res_recovered["current_price"], res_recovered["ema_val"])

    def test_astronomical_floating_scale_1e14(self):
        """挑战：万亿级超大数字（模拟超高通胀/大市值序列），不得溢出为 Inf"""
        base_val = 1e12
        prices = [base_val] * 200 + [base_val * 1.5]
        df = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        self.assertTrue(res_long["passed"])
        self.assertFalse(math.isinf(res_long["ema_val"]))
        self.assertFalse(math.isnan(res_long["ema_val"]))
        self.assertGreater(res_long["current_price"], res_long["ema_val"])

    def test_microscopic_floating_scale_1e8(self):
        """挑战：微小浮点数（模拟加密货币聪级定价 1e-8）"""
        base_val = 1e-7
        prices = [base_val] * 200 + [base_val * 1.5]
        df = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        self.assertTrue(res_long["passed"])
        self.assertFalse(math.isnan(res_long["ema_val"]))


class TestZeroVolatilityAndFlatline(unittest.TestCase):
    """
    3. 零波动率与恒定价格区制挑战 (Zero Volatility & Flatline Regimes)
    测试方差为零、纯水平线序列与微弱抖动环境下的数学收敛与容差鲁棒性。
    """

    def test_pure_flatline_constant_price(self):
        """
        挑战：完全无波动的绝对平坦序列（250 根价格恒为 100.0）。
        在此极值下，EMA 严格收敛至 100.0，Close == EMA。
        多头门禁 (>= EMA - 1e-8) 与空头门禁 (<= EMA + 1e-8) 均应平稳判定，无除零或数值溢出。
        """
        df_flat = pd.DataFrame({"Close": [100.0] * 250})

        res_long = sl.verify_primary_trend_gate(df_flat, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_flat, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertTrue(res_short["passed"])
        self.assertEqual(res_long["current_price"], 100.0)
        self.assertEqual(res_long["ema_val"], 100.0)

    def test_sub_epsilon_jitter(self):
        """挑战：低于浮点容差的微纳级别抖动 (1e-11) 不会导致门禁发生震荡翻转"""
        jitter = 1e-11
        prices = [100.0 + (jitter if i % 2 == 0 else -jitter) for i in range(250)]
        df = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertTrue(res_short["passed"])


class TestSubTenBarsAndBoundaryLengths(unittest.TestCase):
    """
    4. 极小样本与边界长度挑战 (<10 Bars & Boundary Conditions)
    测试 0 根、1 根、2 根、<10 根及 199/200/201 状态切换点的平滑降级。
    """

    def test_zero_bars_and_none_data(self):
        """挑战：数据为 None、空 DataFrame、缺失 Close 列"""
        for empty_input in [None, pd.DataFrame(), pd.DataFrame({"Open": [100.0]})]:
            res = sl.verify_primary_trend_gate(empty_input, side=1, ema_period=200)
            self.assertFalse(res["passed"])
            self.assertEqual(res["status"], "rejected_by_missing_trend_data")
            self.assertEqual(res["current_price"], 0.0)
            self.assertEqual(res["ema_val"], 0.0)

    def test_single_bar_history(self):
        """
        挑战：仅有 1 根 K 线历史 (N=1)。
        span_val = min(1, 200) = 1，EMA 必须精确等于该唯一收盘价，触发自适应提示。
        """
        df_1 = pd.DataFrame({"Close": [150.0]})
        res_long = sl.verify_primary_trend_gate(df_1, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_1, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertTrue(res_short["passed"])
        self.assertEqual(res_long["current_price"], 150.0)
        self.assertEqual(res_long["ema_val"], 150.0)
        self.assertIn("自适应平滑代理", res_long["reason"])

    def test_two_bars_history_exact_math(self):
        """
        挑战：历史仅 2 根 K 线 (N=2)。
        Close = [100.0, 110.0]，span_val = 2，alpha = 2 / 3。
        EMA_0 = 100.0
        EMA_1 = (1/3)*100.0 + (2/3)*110.0 = 106.6667
        现价 110.0 > 106.6667，多头通过，空头驳回。
        """
        df_2 = pd.DataFrame({"Close": [100.0, 110.0]})
        res_long = sl.verify_primary_trend_gate(df_2, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_2, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertFalse(res_short["passed"])
        self.assertAlmostEqual(res_long["ema_val"], 106.6667, places=4)
        self.assertEqual(res_long["current_price"], 110.0)

    def test_sub_ten_bars_history(self):
        """挑战：历史处于 < 10 根（如 3, 5, 8, 9 根）时的数值递推严谨性"""
        for n in [3, 5, 8, 9]:
            prices = [100.0 + i * 2.0 for i in range(n)]
            df_n = pd.DataFrame({"Close": prices})

            res_long = sl.verify_primary_trend_gate(df_n, side=1, ema_period=200)
            res_short = sl.verify_primary_trend_gate(df_n, side=-1, ema_period=200)

            # 递增序列中，最新现价必然大于历史 EMA
            self.assertTrue(res_long["passed"])
            self.assertFalse(res_short["passed"])
            self.assertIn("自适应平滑代理", res_long["reason"])

    def test_transition_at_200_bars_boundary(self):
        """
        挑战：考察跨越 200 根的临界区制切换 (199 -> 200 -> 201)。
        在 199 根时应触发 is_fallback 自适应降级提示；
        在 200 根及以上时应完全消除 fallback 提示，进入原生 200 周期 EMA。
        """
        prices = [100.0 + i * 0.1 for i in range(210)]
        df_all = pd.DataFrame({"Close": prices})

        # N = 199
        res_199 = sl.verify_primary_trend_gate(df_all.iloc[:199], side=1, ema_period=200)
        self.assertIn("自适应平滑代理", res_199["reason"])

        # N = 200
        res_200 = sl.verify_primary_trend_gate(df_all.iloc[:200], side=1, ema_period=200)
        self.assertNotIn("自适应平滑代理", res_200["reason"])

        # N = 201
        res_201 = sl.verify_primary_trend_gate(df_all.iloc[:201], side=1, ema_period=200)
        self.assertNotIn("自适应平滑代理", res_201["reason"])


class TestExactTouchAndEpsilonBand(unittest.TestCase):
    """
    5. 精确触碰 200 EMA 与微小容差带挑战 (Exact Touch & Epsilon Band)
    测试价格恰好触及 EMA 以及位于 1e-8 容差带内外的决策表现。
    """

    def setUp(self):
        # 构造一条基线数据并在尾部注入精确价格
        self.df_base = _generate_synthetic_market_data(n_bars=220, base_price=100.0, seed=99)
        self.ema_prev = float(self.df_base["Close"].ewm(span=200, adjust=False).mean().iloc[-2])

    def test_exact_mathematical_touch(self):
        """
        挑战：令最后一根收盘价精确等于上一根 EMA 值。
        根据递推公式：EMA_t = alpha * EMA_{t-1} + (1-alpha) * EMA_{t-1} = EMA_{t-1}。
        此时现价与 EMA_t 绝对精确相等。
        由于门禁设置了 1e-8 容差：
          多头：Close >= EMA - 1e-8 为 True
          空头：Close <= EMA + 1e-8 为 True
        两端均应放行。
        """
        df_touch = self.df_base.copy()
        df_touch.iloc[-1, df_touch.columns.get_loc("Close")] = self.ema_prev

        res_long = sl.verify_primary_trend_gate(df_touch, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_touch, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertTrue(res_short["passed"])
        self.assertAlmostEqual(res_long["current_price"], res_long["ema_val"], places=6)

    def test_inside_epsilon_band_tolerance(self):
        """挑战：位于容差带内 (±1e-9) 时，双向平滑放行，避免微观浮点舍入误差导致的死锁"""
        df_above = self.df_base.copy()
        df_above.iloc[-1, df_above.columns.get_loc("Close")] = self.ema_prev + 1e-9

        res_long = sl.verify_primary_trend_gate(df_above, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_above, side=-1, ema_period=200)

        self.assertTrue(res_long["passed"])
        self.assertTrue(res_short["passed"])

    def test_outside_epsilon_band_distinct_rejection(self):
        """
        挑战：超出容差带 (±1e-6) 时，方向判决必须清晰互斥。
        - 略高于 EMA (EMA + 1e-6)：多头通过，空头驳回
        - 略低于 EMA (EMA - 1e-6)：多头驳回，空头通过
        """
        # 略高于 EMA
        df_high = self.df_base.copy()
        df_high.iloc[-1, df_high.columns.get_loc("Close")] = self.ema_prev + 1e-5
        res_high_long = sl.verify_primary_trend_gate(df_high, side=1, ema_period=200)
        res_high_short = sl.verify_primary_trend_gate(df_high, side=-1, ema_period=200)
        self.assertTrue(res_high_long["passed"])
        self.assertFalse(res_high_short["passed"])

        # 略低于 EMA
        df_low = self.df_base.copy()
        df_low.iloc[-1, df_low.columns.get_loc("Close")] = self.ema_prev - 1e-5
        res_low_long = sl.verify_primary_trend_gate(df_low, side=1, ema_period=200)
        res_low_short = sl.verify_primary_trend_gate(df_low, side=-1, ema_period=200)
        self.assertFalse(res_low_long["passed"])
        self.assertTrue(res_low_short["passed"])


class TestMalformedInputsAndDefensiveRobustness(unittest.TestCase):
    """
    6. 畸形输入与防御性鲁棒性挑战 (Malformed Inputs & Robustness)
    测试负价格（如原油穿底）、字段大小写兼容、非标准 side、越界 t 参数。
    """

    def test_negative_prices_wti_scenario(self):
        """挑战：负价格序列（模拟 2020 年 5 月 WTI 原油穿底至 -$37.63 极端黑天鹅）"""
        prices = [-50.0 + i * 0.1 for i in range(250)]
        df_neg = pd.DataFrame({"Close": prices})

        res_long = sl.verify_primary_trend_gate(df_neg, side=1, ema_period=200)
        res_short = sl.verify_primary_trend_gate(df_neg, side=-1, ema_period=200)

        # 价格从 -50 升至 -25.1，现价大于历史 EMA
        self.assertTrue(res_long["passed"])
        self.assertFalse(res_short["passed"])
        self.assertLess(res_long["current_price"], 0.0)
        self.assertIn("顺应道氏主要多头趋势", res_long["reason"])

    def test_column_name_case_insensitivity(self):
        """挑战：列名小写 'close' 兼容性"""
        df_lower = pd.DataFrame({"close": [100.0 + i for i in range(50)]})
        res = sl.verify_primary_trend_gate(df_lower, side=1, ema_period=200)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_primary_trend")

    def test_neutral_and_irregular_side_values(self):
        """挑战：非正规 side 参数 (side=0, side=99, side=None) 返回中性且不崩溃"""
        df = pd.DataFrame({"Close": [100.0] * 50})
        for irregular_side in [0, 99, -99, None]:
            res = sl.verify_primary_trend_gate(df, side=irregular_side, ema_period=200)
            self.assertTrue(res["passed"])
            self.assertEqual(res["status"], "neutral")

    def test_out_of_bounds_t_parameter(self):
        """挑战：传入越界 t 参数（如 t < 0 或 t >= len(df)）安全降级为全量序列"""
        df = pd.DataFrame({"Close": [100.0 + i for i in range(50)]})
        res_neg_t = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=-1)
        res_overflow_t = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=9999)
        res_full = sl.verify_primary_trend_gate(df, side=1, ema_period=200)

        self.assertEqual(res_neg_t["ema_val"], res_full["ema_val"])
        self.assertEqual(res_overflow_t["ema_val"], res_full["ema_val"])


class TestEngineIntegrationAdversarial(unittest.TestCase):
    """
    7. 顶层量化引擎与回测集成挑战 (Engine Integration)
    测试 build_trade_plan 与 simulate_strategy_trade 对 200 EMA 顺势门禁的真实联动。
    """

    def test_build_trade_plan_trend_gate_rejection(self):
        """
        挑战：在 pa_engine:build_trade_plan 中，当标的在 200 EMA 下方逆势做多时，
        交易计划必须被明确驳回为 rejected_by_primary_trend。
        """
        # 构造空头大势行情 (现价位于 200 EMA 远下方)
        df_bear = _generate_synthetic_market_data(n_bars=250, base_price=200.0, seed=42)
        swings = pe.extract_swings(df_bear)
        last_close = float(df_bear["Close"].iloc[-1])

        # 构建逆势做多计划
        plan = pe.build_trade_plan(
            mode="swing",
            price=last_close,
            swings_mtf=swings,
            swings_ltf=swings,
            key_levels=[],
            levels={},
            bar_rng=1.0,
            bias_htf=1,
            bias_mtf=1,
            df_mtf=df_bear,
            risk_pct=0.01,
            equity=50000.0,
            use_trend_gate=True
        )

        self.assertIn("rejected_by_primary_trend", plan.get("rejected_by", ""))
        self.assertIn("rejected_by_primary_trend", plan.get("status", ""))
        self.assertFalse(plan.get("trend_gate", {}).get("passed", True))

    def test_simulate_strategy_trade_trend_gate_integration(self):
        """
        挑战：在 strategy_layer:simulate_strategy_trade 中，
        开启 use_trend_gate=True 时，逆势交易被直接拦截。
        """
        df_bear = _generate_synthetic_market_data(n_bars=250, base_price=200.0, seed=42)
        entry_t = 220
        df_bear.iloc[210:220, df_bear.columns.get_loc("Close")] *= 0.6
        entry_p = float(df_bear["Close"].iloc[entry_t - 1])

        trade_res = sl.simulate_strategy_trade(
            df=df_bear,
            entry_t=entry_t,
            entry_price=entry_p,
            side=1,  # 逆势做多
            stop_price=entry_p * 0.95,
            tp_price=entry_p * 1.15,
            use_trend_gate=True
        )

        self.assertEqual(trade_res.get("status"), "rejected_by_primary_trend")


if __name__ == "__main__":
    unittest.main()
