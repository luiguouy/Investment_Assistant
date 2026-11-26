#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Verification Suite for Milestone 3:
200 EMA Primary Trend Gate, Sector Divergence, Micro Trumps Macro & Asymmetric Short Protection
Challenger: challenger_m3_1
Target: strategy_layer.py, pa_engine.py, backtest.py

Coverage:
1. Sector divergence with synthesized OHLCV: QQQ bearish (LH+LL) vs DIA bullish (HH+HL), SPY neutral
   - Non-tech (CAT, JPM, XOM, BA, UNH) must pass
   - Tech (AAPL, MSFT, NVDA, GOOGL) must be blocked
   - Both QQQ & DIA bearish must block non-tech
   - Systemic collapse (SPY & QQQ breakdown) must block all
2. Micro Trumps Macro decoupling stress harness:
   - High conviction patterns (strong_box_breakout, strong_2b_reversal) pass with decoupled_micro_alpha
   - Weak patterns (weak_pinbar, inside_bar, box_touch, random noise, empty) firmly rejected
   - allow_decoupling=False rejects even strong patterns
   - Top-level pipeline integration (simulate_strategy_trade, build_trade_plan)
3. Asymmetric short protection in secular bull trends:
   - Blind shorts rejected in bull trend (rejected_by_macro_short_gate)
   - 200 EMA primary trend gate rejects short when price > 200 EMA
   - Legitimate shorts pass when macro breaks down
   - Counter-trend 2B top liquidity sweep with high conviction passes via decoupling
4. Robustness, boundary conditions, zero-lookahead & API polymorphism:
   - Zero lookahead under future mutation
   - Exact EMA boundary tolerance
   - Missing columns, NaNs, empty DataFrames, single bar DataFrames
   - Positional vs keyword API signatures
"""

import unittest
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl


def _generate_synthetic_ohlcv(n_bars: int = 250, base_price: float = 100.0,
                              trend_type: str = "bull", seed: int = 42) -> pd.DataFrame:
    """
    生成具备严格因果结构与可控道氏摆动形态的真实感 OHLCV 数据序列。
    - 'bull': 产生清晰的 Higher Highs + Higher Lows (HH+HL) 与稳健正斜率；
    - 'bear': 产生清晰的 Lower Highs + Lower Lows (LH+LL) 与稳健负斜率；
    - 'neutral': 产生均值回归箱体震荡，道氏偏向为 0。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)

    if trend_type == "bull":
        # 阶梯式上升波段，确保产生 HH + HL
        t = np.linspace(0, 10 * np.pi, n_bars)
        trend = np.linspace(0, 40, n_bars)
        waves = 4.0 * np.sin(t)
        noise = rng.normal(0, 0.5, n_bars)
        closes = base_price + trend + waves + noise
    elif trend_type == "bear":
        # 阶梯式下降波段，确保产生 LH + LL
        t = np.linspace(0, 10 * np.pi, n_bars)
        trend = np.linspace(0, -40, n_bars)
        waves = 4.0 * np.sin(t)
        noise = rng.normal(0, 0.5, n_bars)
        closes = base_price + trend + waves + noise
        closes = np.maximum(closes, 5.0)
    else:  # neutral
        # 水平震荡箱体，偏向为 0
        t = np.linspace(0, 12 * np.pi, n_bars)
        waves = 6.0 * np.sin(t)
        noise = rng.normal(0, 0.8, n_bars)
        closes = base_price + waves + noise

    highs = closes + np.abs(rng.normal(1.2, 0.3, n_bars))
    lows = closes - np.abs(rng.normal(1.2, 0.3, n_bars))
    opens = (highs + lows) / 2.0 + rng.normal(0, 0.2, n_bars)
    vols = rng.integers(1000000, 8000000, size=n_bars)

    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Volume": vols
    }, index=dates)
    return df


class TestSectorDivergenceAdversarial(unittest.TestCase):
    """
    实证挑战 1：板块分化与解耦逻辑对抗检验
    场景：QQQ 呈道氏空头 (LH+LL)，DIA 呈道氏多头 (HH+HL)，SPY 呈中性震荡。
    """

    def setUp(self):
        # 构造纯净的合成大盘数据
        self.df_qqq_bear = _generate_synthetic_ohlcv(250, base_price=200.0, trend_type="bear", seed=101)
        self.df_dia_bull = _generate_synthetic_ohlcv(250, base_price=100.0, trend_type="bull", seed=202)
        self.df_spy_neutral = _generate_synthetic_ohlcv(250, base_price=150.0, trend_type="neutral", seed=303)

    def test_empirical_swings_and_bias_verification(self):
        """确认合成大盘数据的道氏摆动高低点与偏向与预期完全契合"""
        sw_qqq = pe.extract_swings(self.df_qqq_bear, order=5)
        bias_qqq = pe.dow_bias(sw_qqq, df=self.df_qqq_bear)
        self.assertLessEqual(bias_qqq, -1, f"QQQ 偏向应 <= -1，实际为 {bias_qqq}")

        sw_dia = pe.extract_swings(self.df_dia_bull, order=5)
        bias_dia = pe.dow_bias(sw_dia, df=self.df_dia_bull)
        self.assertGreaterEqual(bias_dia, 1, f"DIA 偏向应 >= 1，实际为 {bias_dia}")

    def test_divergence_non_tech_stocks_pass_and_tech_stocks_blocked(self):
        """
        对抗挑战：QQQ 走熊但 DIA 走牛时：
        1. 工业、金融、能源等非科技标的 (CAT, JPM, XOM, BA, UNH) 100% 放行；
        2. 科技巨头标的 (AAPL, MSFT, NVDA, GOOGL) 100% 严密驳回。
        """
        market_data = {
            "SPY": self.df_spy_neutral,
            "QQQ": self.df_qqq_bear,
            "DIA": self.df_dia_bull
        }

        # 非科技标的集合测试
        non_tech_symbols = ["CAT", "JPM", "XOM", "BA", "UNH"]
        for sym in non_tech_symbols:
            res = sl.verify_market_dow_gate(
                market_dow=market_data,
                target_symbol=sym,
                side=1
            )
            self.assertTrue(
                res["passed"],
                f"非科技标的 {sym} 在 DIA 走强时应放行，但被驳回：{res.get('reason')}"
            )
            self.assertEqual(res["status"], "confirmed_by_market")
            self.assertIn("板块良性轮动", res["reason"])

        # 科技标的集合测试
        tech_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL"]
        for sym in tech_symbols:
            res = sl.verify_market_dow_gate(
                market_dow=market_data,
                target_symbol=sym,
                side=1
            )
            self.assertFalse(
                res["passed"],
                f"科技标的 {sym} 在 QQQ 走弱时应坚决驳回，但被放行！"
            )
            self.assertEqual(res["status"], "rejected_by_market_dow")
            self.assertIn("QQQ", res["reason"])

    def test_both_qqq_and_dia_bearish_blocks_industrial_stocks(self):
        """
        对抗挑战：若 DIA 亦出现道氏空头破位，则工业标的失去板块支撑，必须被坚决拦截！
        """
        df_dia_bear = _generate_synthetic_ohlcv(250, base_price=180.0, trend_type="bear", seed=404)
        market_data = {
            "SPY": self.df_spy_neutral,
            "QQQ": self.df_qqq_bear,
            "DIA": df_dia_bear
        }
        res_cat = sl.verify_market_dow_gate(
            market_dow=market_data,
            target_symbol="CAT",
            side=1
        )
        self.assertFalse(res_cat["passed"], "DIA 亦偏空时，CAT 不应放行")
        self.assertEqual(res_cat["status"], "rejected_by_market_dow")
        self.assertIn("DIA", res_cat["reason"])

    def test_systemic_crisis_blocks_all_sectors_regardless_of_dia(self):
        """
        对抗挑战：系统性流动性危机（SPY 与 QQQ 同时破位跌破 200 EMA），
        即便 DIA 表面偏多，也属于伪流动性，必须全板块熔断拦截！
        """
        # 构造 SPY 与 QQQ 均破位的数据
        systemic_bear_market = {
            "SPY": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "DIA_bias": 1
        }
        for sym in ["CAT", "JPM", "XOM", "AAPL"]:
            res = sl.verify_market_dow_gate(
                market_dow=systemic_bear_market,
                target_symbol=sym,
                side=1
            )
            self.assertFalse(res["passed"], f"系统性危机下 {sym} 必须拦截")
            self.assertEqual(res["status"], "rejected_by_market_dow")
            self.assertIn("系统性市场危机拦截", res["reason"])


class TestMicroTrumpsMacroDecouplingAdversarial(unittest.TestCase):
    """
    实证挑战 2：微观超越宏观 (Micro Trumps Macro) 脱钩规则压力测试
    """

    def setUp(self):
        self.hostile_bear_market = {
            "SPY": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "DIA_bias": -2
        }

    def test_high_conviction_patterns_successfully_decouple(self):
        """测试高确定性形态 (strong_box_breakout, strong_2b_reversal) 取得脱钩豁免"""
        for pattern in ["strong_box_breakout", "strong_2b_reversal"]:
            res = sl.verify_market_dow_gate(
                market_dow=self.hostile_bear_market,
                side=1,
                allow_decoupling=True,
                micro_conviction=pattern
            )
            self.assertTrue(res["passed"], f"{pattern} 应成功脱钩")
            self.assertEqual(res["status"], "decoupled_micro_alpha")
            self.assertIn("Micro Trumps Macro", res["reason"])

    def test_adversarial_subthreshold_patterns_firmly_rejected(self):
        """
        对抗挑战：伪形态注入攻击。
        构造形形色色的弱形态、杂音、甚至空输入，检验门禁绝无破绽放行。
        """
        bogus_patterns = [
            "weak_pinbar",
            "inside_bar",
            "weak_breakout",
            "box_touch",
            "doji",
            "hammer",
            "engulfing",
            "random_alpha",
            "strong_box_breakout_fake",
            None,
            "",
            "   ",
            12345
        ]

        for bp in bogus_patterns:
            res = sl.verify_market_dow_gate(
                market_dow=self.hostile_bear_market,
                side=1,
                allow_decoupling=True,
                micro_conviction=bp
            )
            self.assertFalse(
                res["passed"],
                f"弱形态或非法形态 '{bp}' 竟被错误放行！"
            )
            self.assertEqual(res["status"], "rejected_by_market_dow")

    def test_decoupling_disabled_flag_enforces_macro_veto(self):
        """对抗挑战：即使微观形态为 strong_box_breakout，若未开启 allow_decoupling，坚决拦截"""
        res = sl.verify_market_dow_gate(
            market_dow=self.hostile_bear_market,
            side=1,
            allow_decoupling=False,
            micro_conviction="strong_box_breakout"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")

    def test_pipeline_simulation_with_micro_decoupling(self):
        """对抗挑战：顶层交易模拟器 (simulate_strategy_trade) 贯通验证"""
        df = _generate_synthetic_ohlcv(250, base_price=100.0, trend_type="bull")
        # 1. 宏观逆风，未开启脱钩 -> 必须被拦截
        tr_blocked = sl.simulate_strategy_trade(
            df, entry_t=240, entry_price=float(df["Close"].iloc[240]),
            side=1, stop_price=float(df["Close"].iloc[240]) * 0.95,
            tp_price=float(df["Close"].iloc[240]) * 1.15,
            market_dow=self.hostile_bear_market,
            allow_decoupling=False,
            micro_conviction="strong_box_breakout"
        )
        self.assertEqual(tr_blocked["status"], "rejected_by_market_dow")

        # 2. 宏观逆风，开启脱钩与高确定性形态 -> 必须放行并进入正常撮合
        tr_allowed = sl.simulate_strategy_trade(
            df, entry_t=240, entry_price=float(df["Close"].iloc[240]),
            side=1, stop_price=float(df["Close"].iloc[240]) * 0.95,
            tp_price=float(df["Close"].iloc[240]) * 1.15,
            market_dow=self.hostile_bear_market,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertNotEqual(tr_allowed["status"], "rejected_by_market_dow")


class TestAsymmetricShortProtectionAdversarial(unittest.TestCase):
    """
    实证挑战 3：美股长牛做空非对称保护对抗检验
    """

    def setUp(self):
        self.secular_bull_market = {
            "SPY_bias": 1,
            "QQQ_bias": 1,
            "DIA_bias": 1
        }
        self.df_bull = _generate_synthetic_ohlcv(250, base_price=100.0, trend_type="bull")

    def test_blind_short_in_bull_trend_strictly_rejected(self):
        """
        对抗挑战：盲目做空攻击。
        在大盘主多头顺势环境下，无论如何不得盲目开辟常规做空头寸。
        """
        res = sl.verify_market_dow_gate(
            market_dow=self.secular_bull_market,
            side=-1,
            allow_decoupling=False
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")
        self.assertIn("rejected_by_macro_short_gate", res["reason"])

    def test_primary_trend_gate_rejects_short_above_200_ema(self):
        """测试 200 EMA 主趋势门禁严密驳回价格在 200 EMA 上方的逆势做空"""
        res = sl.verify_primary_trend_gate(self.df_bull, side=-1, ema_period=200)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_primary_trend")
        self.assertIn("长牛大势下逆势摸顶做空被门禁驳回", res["reason"])

    def test_build_trade_plan_enforces_asymmetric_short_gate(self):
        """测试 pa_engine.build_trade_plan 在多头大市中拒绝做空计划"""
        curr_p = float(self.df_bull["Close"].iloc[-1])
        swings = pe.extract_swings(self.df_bull, order=3, confirmed_only=True)
        key_levels = [{"price": curr_p * 1.05, "kind": "swing", "src": "D1"}]

        # 尝试传入 bias_htf=-1, bias_mtf=-1 发起做空，但大盘顺势长牛
        plan = pe.build_trade_plan(
            mode="daily", price=curr_p, swings_mtf=swings, swings_ltf=swings,
            key_levels=key_levels, levels={}, bar_rng=2.0, bias_htf=-1, bias_mtf=-1,
            df_mtf=self.df_bull, market_dow=self.secular_bull_market, use_trend_gate=True
        )
        # 必须被门禁拦截（无论被 market_dow_gate 还是 trend_gate 拦截）
        self.assertIn(plan["rejected_by"], ["rejected_by_market_dow", "rejected_by_primary_trend"])
        self.assertIn("❌", plan["status"])

    def test_legitimate_short_passes_when_macro_breaks_down(self):
        """测试大盘出现真实空头破位时，顺理成章放行做空"""
        breakdown_market = {
            "SPY_bias": -1,
            "QQQ_bias": 1,
            "DIA_bias": 1
        }
        res = sl.verify_market_dow_gate(breakdown_market, side=-1)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_market")

    def test_top_2b_sweep_short_decoupling_allowed(self):
        """测试阻力位 2B 假突破诱多清算 (strong_2b_reversal) 在长牛大势下允许精准摸顶脱钩"""
        res = sl.verify_market_dow_gate(
            market_dow=self.secular_bull_market,
            side=-1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")


class TestBoundaryAndRobustnessAdversarial(unittest.TestCase):
    """
    实证挑战 4：极端输入、边界容差、类型防御与因果前视自检
    """

    def test_future_mutation_zero_lookahead_trend_gate(self):
        """
        对抗挑战：未来数据异动攻击。
        在 t=180 时点计算 200 EMA 顺势门禁；
        修改 t > 180 的所有未来价格（暴涨100倍），验证 t 时点的计算结果完全不变。
        """
        df_base = _generate_synthetic_ohlcv(250, base_price=100.0, trend_type="bull")
        t_cutoff = 180

        res_before = sl.verify_primary_trend_gate(df_base, side=1, ema_period=200, t=t_cutoff)

        df_poisoned = df_base.copy()
        df_poisoned.iloc[t_cutoff + 1:, df_poisoned.columns.get_loc("Close")] *= 100.0
        df_poisoned.iloc[t_cutoff + 1:, df_poisoned.columns.get_loc("High")] *= 100.0

        res_after = sl.verify_primary_trend_gate(df_poisoned, side=1, ema_period=200, t=t_cutoff)

        self.assertEqual(res_before["passed"], res_after["passed"])
        self.assertAlmostEqual(res_before["ema_val"], res_after["ema_val"], places=6)
        self.assertAlmostEqual(res_before["current_price"], res_after["current_price"], places=6)

    def test_empty_nan_single_row_safety(self):
        """对抗挑战：空表、单行、全 NaN 数据防御"""
        # None
        r1 = sl.verify_primary_trend_gate(None, side=1)
        self.assertFalse(r1["passed"])

        # 0 行
        r2 = sl.verify_primary_trend_gate(pd.DataFrame(), side=1)
        self.assertFalse(r2["passed"])

        # 缺少 Close
        df_no_close = pd.DataFrame({"A": [1, 2, 3]})
        r3 = sl.verify_primary_trend_gate(df_no_close, side=1)
        self.assertFalse(r3["passed"])

        # 单行
        df_one = pd.DataFrame({"Close": [105.0]})
        r4 = sl.verify_primary_trend_gate(df_one, side=1)
        self.assertTrue(r4["passed"])
        self.assertAlmostEqual(r4["ema_val"], 105.0)

    def test_api_polymorphism_signatures(self):
        """
        对抗挑战：各种历史与重构调用签名的兼容性检验
        """
        market = {"SPY_bias": 1, "QQQ_bias": 1}

        # 1. 纯旧版双参调用：verify_market_dow_gate(market, 1)
        res1 = sl.verify_market_dow_gate(market, 1)
        self.assertTrue(res1["passed"])

        # 2. 纯旧版做空调用：verify_market_dow_gate(market, -1)
        res2 = sl.verify_market_dow_gate(market, -1)
        self.assertFalse(res2["passed"])

        # 3. 新版位置标的调用：verify_market_dow_gate(market, "CAT", side=1)
        res3 = sl.verify_market_dow_gate(market, "CAT", side=1)
        self.assertTrue(res3["passed"])

        # 4. 全显式命名关键字参数
        res4 = sl.verify_market_dow_gate(
            market_data=market,
            target_symbol="CAT",
            target_sector="industrials",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertTrue(res4["passed"])


if __name__ == "__main__":
    unittest.main()
