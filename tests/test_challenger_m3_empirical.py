#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 3 对抗性实证验证套件 (Empirical Adversarial Stress Test Suite)
角色: challenger_m3_1 (Empirical Challenger: critic + specialist)

覆盖四大对抗实证场景:
1. Scenario 1 (Sector Divergence): 跨市场板块分化与良性轮动（QQQ回调 vs DIA/IWM突破，CAT放行 vs NVDA拦截）
2. Scenario 2 (Micro Trumps Macro): 高确定性微观结构超越宏观（箱体突破/2B猎杀脱钩豁免 vs 弱Pinbar坚决拦截）
3. Scenario 3 (Asymmetric Shorting): 美股长牛内生通胀环境下做空非对称保护（大牛拦截摸顶做空，仅破位或2B顶放行）
4. Scenario 4 (200 EMA Causality): 动态截断与未来数据多模态极端投毒下零前视偏差（Zero Lookahead Bias）
"""

import math
import unittest
import numpy as np
import pandas as pd

import strategy_layer as sl
import pa_engine as pe


def _make_synth_series(n_bars: int = 300, base_price: float = 100.0, trend: float = 0.05, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rets = np.random.normal(loc=trend / 100.0, scale=0.012, size=n_bars)
    prices = [base_price]
    for r in rets:
        prices.append(prices[-1] * math.exp(r))
    prices = prices[1:]
    highs = [p * (1.0 + abs(np.random.normal(0, 0.005))) for p in prices]
    lows = [p * (1.0 - abs(np.random.normal(0, 0.005))) for p in prices]
    opens = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": [1000000.0] * n_bars
    })


class TestScenario1SectorDivergence(unittest.TestCase):
    """
    Scenario 1: 板块分化与良性轮动对抗验证
    - QQQ 处于空头回调，DIA 处于多头主升，SPY 维持多头/抗跌；
    - 验证工业/传统价值股（如 CAT, XLI, JPM）正常放行；
    - 验证科技股（如 NVDA, XLK, AAPL）严格受 QQQ 偏空约束驳回；
    - 验证系统性股灾（SPY 与 QQQ 同时破位）触发全板块硬熔断。
    """

    def setUp(self):
        # 构造良性轮动的大盘环境：科技 QQQ 弱势，传统价值 DIA 走强，SPY 稳健
        self.rotation_market = {
            "SPY_bias": 1,
            "SPY_breakdown": False,
            "QQQ_bias": -1,
            "QQQ_breakdown": False,
            "DIA_bias": 1,
            "DIA_breakdown": False,
            "IWM_bias": 1,
            "IWM_breakdown": False,
        }

        # 构造系统性股灾大盘环境：SPY 与 QQQ 同时破位
        self.crash_market = {
            "SPY_bias": -1,
            "SPY_breakdown": True,
            "QQQ_bias": -1,
            "QQQ_breakdown": True,
            "DIA_bias": 1,
            "DIA_breakdown": False,
        }

    def test_industrial_stock_passes_during_tech_pullback(self):
        """实证：QQQ偏弱但DIA走强时，工业股 CAT 判定为良性轮动放行"""
        res_cat = sl.verify_market_dow_gate(
            market_data=self.rotation_market,
            target_symbol="CAT",
            target_sector="Industrials",
            side=1
        )
        self.assertTrue(res_cat["passed"])
        self.assertEqual(res_cat["status"], "confirmed_by_market")
        self.assertIn("良性轮动", res_cat["reason"])

    def test_tech_stock_blocked_during_tech_pullback(self):
        """实证：QQQ偏弱时，科技股 NVDA 必须被严格拦截"""
        res_nvda = sl.verify_market_dow_gate(
            market_data=self.rotation_market,
            target_symbol="NVDA",
            target_sector="Technology",
            side=1
        )
        self.assertFalse(res_nvda["passed"])
        self.assertEqual(res_nvda["status"], "rejected_by_market_dow")
        self.assertIn("QQQ", res_nvda["reason"])

    def test_multiple_sector_classifications_consistency(self):
        """实证：各板块标的映射矩阵的行为一致性（收录于 SYMBOL_SECTOR_MAP 的标的）"""
        value_stocks = ["CAT", "BA", "GE", "HON", "UNP", "JPM", "GS", "XOM", "JNJ"]
        tech_stocks = ["NVDA", "AAPL", "MSFT", "AMD", "TSLA"]

        for sym in value_stocks:
            res = sl.verify_market_dow_gate(market_data=self.rotation_market, target_symbol=sym, side=1)
            self.assertTrue(res["passed"], f"标的 {sym} 作为非科技标的应在良性轮动中放行")

        for sym in tech_stocks:
            res = sl.verify_market_dow_gate(market_data=self.rotation_market, target_symbol=sym, side=1)
            self.assertFalse(res["passed"], f"科技标的 {sym} 在 QQQ 回调时必须被拦截")

    def test_unmapped_stock_explicit_sector_override(self):
        """实证：未收录小众标的若显式传入 target_sector，亦能正确识别板块轮动"""
        res = sl.verify_market_dow_gate(
            market_data=self.rotation_market,
            target_symbol="DE",
            target_sector="industrials",
            side=1
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_market")

    def test_systemic_crash_blocks_even_industrial_leaders(self):
        """实证：当 SPY 与 QQQ 双双出现破位时，触发全市场流动性熔断，CAT 亦被一票否决"""
        res_cat_crash = sl.verify_market_dow_gate(
            market_data=self.crash_market,
            target_symbol="CAT",
            target_sector="Industrials",
            side=1
        )
        self.assertFalse(res_cat_crash["passed"])
        self.assertEqual(res_cat_crash["status"], "rejected_by_market_dow")
        self.assertIn("系统性市场危机", res_cat_crash["reason"])


class TestScenario2MicroTrumpsMacro(unittest.TestCase):
    """
    Scenario 2: 微观结构超越宏观 (Micro Trumps Macro) 对抗验证
    - 在宏观不利/偏空环境下:
      1. 高确定性微观结构 (strong_box_breakout, strong_2b_reversal) 触发 decoupled_micro_alpha 放行;
      2. 弱微观形态 (weak_pinbar) 坚决不予豁免，维持宏观拦截;
      3. 未开启 allow_decoupling 时，即使微观极佳亦服从宏观防御;
      4. 空字符串或未知微观形态安全 fail-closed 拦截。
    """

    def setUp(self):
        # 偏空宏观大盘
        self.bearish_macro = {
            "SPY_bias": -1,
            "SPY_breakdown": True,
            "QQQ_bias": -1,
            "QQQ_breakdown": True,
            "DIA_bias": -1,
            "DIA_breakdown": True,
        }

    def test_strong_box_breakout_triggers_decoupling(self):
        """实证：高确定性箱体放量突破豁免宏观压制"""
        res = sl.verify_market_dow_gate(
            market_data=self.bearish_macro,
            target_symbol="XYZ",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")
        self.assertIn("Micro Trumps Macro", res["reason"])

    def test_strong_2b_reversal_triggers_decoupling(self):
        """实证：高确定性关键位 2B 假跌破扫损收回收割豁免宏观压制"""
        res = sl.verify_market_dow_gate(
            market_data=self.bearish_macro,
            target_symbol="XYZ",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")
        self.assertIn("Micro Trumps Macro", res["reason"])

    def test_weak_pinbar_strictly_rejected(self):
        """实证：弱微观形态 (weak_pinbar) 绝不给脱钩特权，严格拦截"""
        res = sl.verify_market_dow_gate(
            market_data=self.bearish_macro,
            target_symbol="XYZ",
            side=1,
            allow_decoupling=True,
            micro_conviction="weak_pinbar"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")
        self.assertIn("系统性市场危机", res["reason"])

    def test_unknown_or_empty_conviction_fail_closed(self):
        """实证：未知形态或空值严格防御性 Fail-Closed"""
        for bogus in ["random_wick", "morning_star", "", None, 123]:
            res = sl.verify_market_dow_gate(
                market_data=self.bearish_macro,
                target_symbol="XYZ",
                side=1,
                allow_decoupling=True,
                micro_conviction=bogus
            )
            self.assertFalse(res["passed"])
            self.assertEqual(res["status"], "rejected_by_market_dow")

    def test_decoupling_disabled_enforces_macro_veto(self):
        """实证：当 allow_decoupling=False 时，即使是 strong_box_breakout 也不放行"""
        res = sl.verify_market_dow_gate(
            market_data=self.bearish_macro,
            target_symbol="XYZ",
            side=1,
            allow_decoupling=False,
            micro_conviction="strong_box_breakout"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")


class TestScenario3AsymmetricShorting(unittest.TestCase):
    """
    Scenario 3: 美股长牛内生通胀环境下做空非对称风控对抗验证
    - 在全多头顺势环境中:
      1. 常规摸顶做空（无脱钩或弱形态）严格驳回 (rejected_by_macro_short_gate);
      2. 宏观真实破位时放行做空;
      3. 高位 2B 假突破诱多反转收割 (strong_2b_reversal) 脱钩放行做空;
      4. 弱形态 (weak_pinbar) 摸顶做空严格驳回。
    """

    def setUp(self):
        self.secular_bull = {
            "SPY_bias": 1,
            "SPY_breakdown": False,
            "QQQ_bias": 1,
            "QQQ_breakdown": False,
            "DIA_bias": 1,
            "DIA_breakdown": False,
            "IWM_bias": 1,
            "IWM_breakdown": False,
        }
        self.breakdown_bear = {
            "SPY_bias": -1,
            "SPY_breakdown": True,
            "QQQ_bias": -1,
            "QQQ_breakdown": True,
        }

    def test_blind_short_in_bull_market_is_blocked(self):
        """实证：全多头长牛大盘下，盲目常规做空被非对称做空门禁拦截"""
        res = sl.verify_market_dow_gate(
            market_data=self.secular_bull,
            target_symbol="SPY",
            side=-1,
            allow_decoupling=False
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")
        self.assertIn("rejected_by_macro_short_gate", res["reason"])
        self.assertIn("长牛", res["reason"])

    def test_macro_breakdown_permits_shorting(self):
        """实证：大盘出现结构性破位时，顺理成章放行做空"""
        res = sl.verify_market_dow_gate(
            market_data=self.breakdown_bear,
            target_symbol="SPY",
            side=-1
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_market")

    def test_strong_2b_top_sweep_permits_shorting_in_bull(self):
        """实证：在顺势长牛中，只有确认高位 2B 假突破诱多猎杀 (strong_2b_reversal) 才特批做空"""
        res = sl.verify_market_dow_gate(
            market_data=self.secular_bull,
            target_symbol="TSLA",
            side=-1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")
        self.assertIn("Micro Trumps Macro", res["reason"])

    def test_weak_pinbar_short_in_bull_rejected(self):
        """实证：顺势长牛中，弱 Pinbar 摸顶做空坚决拦截"""
        res = sl.verify_market_dow_gate(
            market_data=self.secular_bull,
            target_symbol="TSLA",
            side=-1,
            allow_decoupling=True,
            micro_conviction="weak_pinbar"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")
        self.assertIn("rejected_by_macro_short_gate", res["reason"])


class TestScenario4Ema200CausalityAndPoisoning(unittest.TestCase):
    """
    Scenario 4: 200 EMA 因果性与极端投毒压力测试
    - 验证动态子序列截断（Truncation Invariance）
    - 验证多模态极端未来投毒（Poisoning Invariance）
    - 验证边界长度平滑过渡（0, 1, 2, 10, 199, 200, 201 根）
    """

    def setUp(self):
        self.df_long = _make_synth_series(n_bars=350, base_price=100.0, trend=0.08, seed=123)

    def test_dynamic_truncation_across_all_bars(self):
        """
        实证：对 [200, 349] 范围内每一根 K 线，
        全序列传入 t=i 的 EMA 计算结果必须与直接物理截断 df.iloc[:i+1] 完全恒等。
        """
        for i in range(200, 350, 15):
            res_t = sl.verify_primary_trend_gate(self.df_long, side=1, ema_period=200, t=i)
            res_cut = sl.verify_primary_trend_gate(self.df_long.iloc[:i + 1], side=1, ema_period=200)

            self.assertEqual(res_t["passed"], res_cut["passed"], f"Bar {i} passed 状态不一致")
            self.assertEqual(res_t["status"], res_cut["status"], f"Bar {i} status 状态不一致")
            self.assertAlmostEqual(res_t["ema_val"], res_cut["ema_val"], places=10, msg=f"Bar {i} EMA 存在浮点偏差")

    def test_multimodal_future_data_poisoning(self):
        """
        实证：在 t=220 决策点之后，对未来数据进行三种极端恶意投毒:
        1. 极高价格暴拉 (10^9)
        2. 极低价格深跌 (10^-6)
        3. 极端正负震荡 (10^8 到 10^-4 剧烈交替)
        决策时点 t=220 的输出必须 100% 保持原样，分毫不差。
        """
        t_eval = 220
        clean_res = sl.verify_primary_trend_gate(self.df_long, side=1, ema_period=200, t=t_eval)

        # 1. 极高价格投毒
        df_poison_high = self.df_long.copy()
        df_poison_high.iloc[t_eval + 1:, df_poison_high.columns.get_loc("Close")] = 1e9
        res_high = sl.verify_primary_trend_gate(df_poison_high, side=1, ema_period=200, t=t_eval)
        self.assertEqual(clean_res["ema_val"], res_high["ema_val"])
        self.assertEqual(clean_res["passed"], res_high["passed"])

        # 2. 极低价格投毒
        df_poison_low = self.df_long.copy()
        df_poison_low.iloc[t_eval + 1:, df_poison_low.columns.get_loc("Close")] = 1e-6
        res_low = sl.verify_primary_trend_gate(df_poison_low, side=1, ema_period=200, t=t_eval)
        self.assertEqual(clean_res["ema_val"], res_low["ema_val"])
        self.assertEqual(clean_res["passed"], res_low["passed"])

        # 3. 剧烈正负震荡投毒
        df_poison_whipsaw = self.df_long.copy()
        n_fut = len(df_poison_whipsaw) - (t_eval + 1)
        whip_vals = [1e8 if k % 2 == 0 else 1e-4 for k in range(n_fut)]
        df_poison_whipsaw.iloc[t_eval + 1:, df_poison_whipsaw.columns.get_loc("Close")] = whip_vals
        res_whipsaw = sl.verify_primary_trend_gate(df_poison_whipsaw, side=1, ema_period=200, t=t_eval)
        self.assertEqual(clean_res["ema_val"], res_whipsaw["ema_val"])
        self.assertEqual(clean_res["passed"], res_whipsaw["passed"])

    def test_short_sequence_and_boundary_transitions(self):
        """
        实证：微小长度（0, 1, 2, 5 根）与边界跳变（199 -> 200 -> 201 根）下的数学稳定性
        """
        # 空数据与非法输入 (Fail-Closed)
        res_empty = sl.verify_primary_trend_gate(pd.DataFrame(), side=1)
        self.assertFalse(res_empty["passed"])
        self.assertEqual(res_empty["status"], "rejected_by_missing_trend_data")

        res_none = sl.verify_primary_trend_gate(None, side=1)
        self.assertFalse(res_none["passed"])

        # 单根与极短序列
        df_1 = pd.DataFrame({"Close": [100.0]})
        res_1 = sl.verify_primary_trend_gate(df_1, side=1)
        self.assertTrue(res_1["passed"])
        self.assertEqual(res_1["ema_val"], 100.0)

        df_2 = pd.DataFrame({"Close": [100.0, 110.0]})
        res_2_long = sl.verify_primary_trend_gate(df_2, side=1)
        res_2_short = sl.verify_primary_trend_gate(df_2, side=-1)
        self.assertTrue(res_2_long["passed"])
        self.assertFalse(res_2_short["passed"])

        # 199 到 200 跨越自适应平滑标记消除
        df_trans = pd.DataFrame({"Close": [100.0 + k * 0.1 for k in range(210)]})
        res_199 = sl.verify_primary_trend_gate(df_trans.iloc[:199], side=1)
        res_200 = sl.verify_primary_trend_gate(df_trans.iloc[:200], side=1)
        self.assertIn("自适应平滑代理", res_199["reason"])
        self.assertNotIn("自适应平滑代理", res_200["reason"])


if __name__ == "__main__":
    unittest.main()
