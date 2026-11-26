#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_trend_gate_and_sector_decoupling.py:
Milestone 3 专项测试套件 —— 200 EMA 顺势门禁与板块解耦 (Requirements R2 & R3)

涵盖核心测试维度：
  1. 200 EMA 大级别主趋势顺势门禁 (verify_primary_trend_gate):
     - 严格因果性与零未来函数截断一致性
     - 多头在 200 EMA 上方放行、下方驳回
     - 空头在 200 EMA 下方放行、上方驳回
     - 价格精确接触 200 EMA 边界容差
     - 历史数据不足 200 根时自适应平滑降级
     - 空数据与异常输入安全防御
  2. 跨市场多指数印证与板块独立性 (verify_market_dow_gate):
     - 消除 SPY/QQQ 科技同义反复死锁
     - 科技回调 (QQQ弱) 但工业/价值走强 (DIA强) 时，非科技标的多头放行
     - 纳指弱但标的为科技股时严格驳回
     - 小盘股独立印证 (IWM) 与市场广度表征
     - 系统性大盘崩溃 (SPY与QQQ均破位) 拦截
     - 部分大盘数据缺失时安全放行
  3. 微观结构超越宏观脱钩规则 (Micro Trumps Macro):
     - 高确定性箱体突破 (strong_box_breakout) 触发脱钩放行
     - 高确定性 2B 裸K反转 (strong_2b_reversal) 触发脱钩放行
     - 弱微观形态 (如 weak_pinbar) 严禁滥用脱钩豁免
     - allow_decoupling 未激活时遵循常规宏观拦截
  4. 美股做空非对称风险门禁 (Asymmetric Short Gate):
     - 处于长牛大势 (大盘均为多头顺势) 时，严格禁止盲目做空
     - 大盘出现明确空头破位/偏空结构时，放行做空计划
     - 触顶 2B 假突破 (strong_2b_reversal) 触发空头脱钩放行
  5. 顶层集成与撮合协同:
     - simulate_strategy_trade 联动 trend_gate 与板块脱钩
     - build_trade_plan 联动 200 EMA 趋势门禁与多指数印证
"""

import math
import unittest
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl


def _create_synthetic_series(n_bars: int = 250, base_price: float = 100.0,
                             trend_slope: float = 0.002, seed: int = 42) -> pd.DataFrame:
    """生成具备明确宏观趋势的合成日线数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    closes = [base_price]
    for i in range(1, n_bars):
        ret = rng.normal(trend_slope, 0.012)
        closes.append(max(closes[-1] * (1.0 + ret), 1.0))
    closes = np.array(closes)
    highs = closes * (1.0 + rng.uniform(0.002, 0.015, size=n_bars))
    lows = closes * (1.0 - rng.uniform(0.002, 0.015, size=n_bars))
    opens = (highs + lows) / 2.0
    vols = rng.integers(100000, 5000000, size=n_bars)

    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Volume": vols
    }, index=dates)
    return df


class TestPrimaryTrendGate(unittest.TestCase):
    """测试 200 EMA 大级别主趋势顺势门禁 (verify_primary_trend_gate)"""

    def test_long_trade_passed_when_price_above_ema200(self):
        """测试多头交易：现价位于 200 EMA 上方时顺势放行"""
        df_bull = _create_synthetic_series(n_bars=250, base_price=100.0, trend_slope=0.003)
        res = sl.verify_primary_trend_gate(df_bull, side=1, ema_period=200)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_primary_trend")
        self.assertGreaterEqual(res["current_price"], res["ema_val"])
        self.assertIn("顺应道氏主要多头趋势", res["reason"])

    def test_long_trade_rejected_when_price_below_ema200(self):
        """测试多头交易：现价位于 200 EMA 下方时逆势驳回"""
        df_bear = _create_synthetic_series(n_bars=250, base_price=200.0, trend_slope=-0.003)
        res = sl.verify_primary_trend_gate(df_bear, side=1, ema_period=200)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_primary_trend")
        self.assertLess(res["current_price"], res["ema_val"])
        self.assertIn("逆势做多被大级别主趋势门禁驳回", res["reason"])

    def test_short_trade_passed_when_price_below_ema200(self):
        """测试空头交易：现价位于 200 EMA 下方时顺势放行"""
        df_bear = _create_synthetic_series(n_bars=250, base_price=200.0, trend_slope=-0.003)
        res = sl.verify_primary_trend_gate(df_bear, side=-1, ema_period=200)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_primary_trend")
        self.assertLessEqual(res["current_price"], res["ema_val"])
        self.assertIn("顺应道氏主要空头趋势", res["reason"])

    def test_short_trade_rejected_when_price_above_ema200(self):
        """测试空头交易：现价位于 200 EMA 上方时长牛大势下禁止摸顶做空"""
        df_bull = _create_synthetic_series(n_bars=250, base_price=100.0, trend_slope=0.003)
        res = sl.verify_primary_trend_gate(df_bull, side=-1, ema_period=200)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_primary_trend")
        self.assertGreater(res["current_price"], res["ema_val"])
        self.assertIn("长牛大势下逆势摸顶做空被门禁驳回", res["reason"])

    def test_neutral_side_handling(self):
        """测试中性 side == 0 时安全放行"""
        df = _create_synthetic_series(n_bars=220, base_price=100.0)
        res = sl.verify_primary_trend_gate(df, side=0, ema_period=200)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "neutral")

    def test_exact_price_on_ema_boundary(self):
        """测试价格精确等于 200 EMA 临界值时的容差与边界放行"""
        df = _create_synthetic_series(n_bars=220, base_price=100.0)
        # 若 P_t = EMA_{t-1}，则 EMA_t = alpha * P_t + (1-alpha) * EMA_{t-1} = P_t，实现精确相等
        ema_prev = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-2])
        df.iloc[-1, df.columns.get_loc("Close")] = ema_prev

        # 多头：Close >= EMA 判定通过
        res_long = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        self.assertTrue(res_long["passed"])

        # 空头：Close <= EMA 判定通过
        res_short = sl.verify_primary_trend_gate(df, side=-1, ema_period=200)
        self.assertTrue(res_short["passed"])

    def test_insufficient_history_adaptive_fallback(self):
        """测试历史数据少于 200 根（如新股 35 根）时的自适应平滑降级"""
        df_short = _create_synthetic_series(n_bars=35, base_price=50.0, trend_slope=0.004)
        res = sl.verify_primary_trend_gate(df_short, side=1, ema_period=200)
        self.assertIsInstance(res, dict)
        self.assertTrue(res["passed"])
        self.assertIn("采用可用周期自适应平滑代理", res["reason"])
        self.assertGreater(res["ema_val"], 0.0)

    def test_empty_or_none_data_graceful_handling(self):
        """测试空数据或 None 输入时的安全防御"""
        res_none = sl.verify_primary_trend_gate(None, side=1)
        self.assertFalse(res_none["passed"])
        self.assertEqual(res_none["status"], "rejected_by_missing_trend_data")

        res_empty = sl.verify_primary_trend_gate(pd.DataFrame(), side=1)
        self.assertFalse(res_empty["passed"])
        self.assertEqual(res_empty["status"], "rejected_by_missing_trend_data")

    def test_pure_causal_truncation_zero_lookahead(self):
        """测试 200 EMA 纯因果计算：截断重算与全量序列完全相等，零未来函数"""
        df_full = _create_synthetic_series(n_bars=300, base_price=100.0, trend_slope=0.002)
        cutoff = 220
        df_sub = df_full.iloc[:cutoff + 1].copy()

        res_at_t = sl.verify_primary_trend_gate(df_full, side=1, ema_period=200, t=cutoff)
        res_sub = sl.verify_primary_trend_gate(df_sub, side=1, ema_period=200)

        self.assertAlmostEqual(res_at_t["ema_val"], res_sub["ema_val"], places=4)
        self.assertEqual(res_at_t["passed"], res_sub["passed"])


class TestMarketDowSectorDecoupling(unittest.TestCase):
    """测试大盘多指数印证与板块解耦 (verify_market_dow_gate)"""

    def test_tautology_break_spy_and_qqq_divergence_for_non_tech(self):
        """
        测试解耦 SPY/QQQ 同义反复：
        当科技基准 (QQQ) 回调转弱，但工业基准 (DIA) 走强且 SPY 稳定时，
        传统工业标的 (CAT) 多头计划应当放行，不受 QQQ 科技回调绑架。
        """
        market_data = {
            "SPY_bias": 0,
            "QQQ_bias": -1,  # 纳指走弱
            "DIA_bias": 1    # 道指工业走强
        }
        res_cat = sl.verify_market_dow_gate(
            market_dow=market_data,
            target_symbol="CAT",
            target_sector="industrials",
            side=1
        )
        self.assertTrue(res_cat["passed"])
        self.assertEqual(res_cat["status"], "confirmed_by_market")
        self.assertIn("判定为板块良性轮动", res_cat["reason"])

    def test_tech_stock_blocked_when_qqq_is_bearish(self):
        """
        测试板块特异性：
        当科技基准 (QQQ) 走弱时，科技标的 (NVDA) 仍应严格受 QQQ 空头门禁约束驳回。
        """
        market_data = {
            "SPY_bias": 0,
            "QQQ_bias": -1,
            "DIA_bias": 1
        }
        res_nvda = sl.verify_market_dow_gate(
            market_dow=market_data,
            target_symbol="NVDA",
            target_sector="technology",
            side=1
        )
        self.assertFalse(res_nvda["passed"])
        self.assertEqual(res_nvda["status"], "rejected_by_market_dow")
        self.assertIn("QQQ", res_nvda["reason"])

    def test_small_cap_iwm_market_breadth_corroboration(self):
        """
        测试小盘基准 (IWM) 市场广度印证：
        纳指回调但 IWM 走强时，小盘标的放行。
        """
        market_data = {
            "SPY_bias": 0,
            "QQQ_bias": -1,
            "IWM_bias": 1
        }
        res_small = sl.verify_market_dow_gate(
            market_dow=market_data,
            target_sector="small_cap",
            side=1
        )
        self.assertTrue(res_small["passed"])
        self.assertEqual(res_small["status"], "confirmed_by_market")
        self.assertIn("IWM", res_small["reason"])

    def test_systemic_bear_market_crash_blocks_all_sectors(self):
        """
        测试系统性大盘崩盘：
        当 SPY 与 QQQ 双双破位时，属于全市场流动性危机，无论哪个板块一律驳回。
        """
        systemic_bear = {
            "SPY": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "DIA_bias": 0
        }
        res_ind = sl.verify_market_dow_gate(
            market_dow=systemic_bear,
            target_sector="industrials",
            side=1
        )
        self.assertFalse(res_ind["passed"])
        self.assertEqual(res_ind["status"], "rejected_by_market_dow")
        self.assertIn("系统性市场危机拦截", res_ind["reason"])


class TestMicroTrumpsMacroDecoupling(unittest.TestCase):
    """测试微观结构超越宏观 (Micro Trumps Macro) 脱钩规则"""

    def test_strong_box_breakout_decouples_from_bear_market(self):
        """测试高确定性箱体突破 (strong_box_breakout) 触发脱钩放行"""
        market_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1
        }
        res = sl.verify_market_dow_gate(
            market_dow=market_bear,
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")
        self.assertIn("Micro Trumps Macro", res["reason"])

    def test_strong_2b_reversal_decouples_from_bear_market(self):
        """测试高确定性 2B 假跌破反转 (strong_2b_reversal) 触发脱钩放行"""
        market_bear = {
            "SPY": {"mtf": {"phase": "空头"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头"}, "is_breakdown": True}
        }
        res = sl.verify_market_dow_gate(
            market_dow=market_bear,
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")

    def test_subthreshold_weak_conviction_not_exempt(self):
        """测试弱微观形态 (如 weak_pinbar) 不得滥用脱钩豁免权"""
        market_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1
        }
        res = sl.verify_market_dow_gate(
            market_dow=market_bear,
            side=1,
            allow_decoupling=True,
            micro_conviction="weak_pinbar"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")

    def test_decoupling_disabled_flag_enforces_macro_gate(self):
        """测试未开启 allow_decoupling 时，即便形态强劲也坚守宏观底线"""
        market_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1
        }
        res = sl.verify_market_dow_gate(
            market_dow=market_bear,
            side=1,
            allow_decoupling=False,
            micro_conviction="strong_box_breakout"
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")


class TestAsymmetricShortRiskGate(unittest.TestCase):
    """测试美股长牛做空非对称风控门禁"""

    def test_blind_short_rejected_in_bull_market(self):
        """测试大盘处于多头顺势环境时，严禁盲目逆势摸顶做空"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        res = sl.verify_market_dow_gate(market_bull, side=-1)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_market_dow")
        self.assertIn("rejected_by_macro_short_gate", res["reason"])

    def test_short_allowed_when_market_breaks_down(self):
        """测试大盘出现空头破位或负偏向时，顺势放行做空"""
        market_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1
        }
        res = sl.verify_market_dow_gate(market_bear, side=-1)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "confirmed_by_market")
        self.assertIn("放行做空计划", res["reason"])

    def test_short_allowed_via_2b_top_liquidity_sweep(self):
        """测试大盘多头环境下，高位 2B 假突破诱多 (strong_2b_reversal) 脱钩放行做空"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        res = sl.verify_market_dow_gate(
            market_bull, side=-1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "decoupled_micro_alpha")


class TestTopLevelStrategyIntegration(unittest.TestCase):
    """测试顶层策略闭环与计划构建中的 M3 门禁协同"""

    def test_simulate_strategy_trade_with_primary_trend_gate(self):
        """测试 simulate_strategy_trade 在开启 use_trend_gate 时的顺势把关"""
        # 构造收盘价低于 200 EMA 的空头形态序列
        df_bear = _create_synthetic_series(n_bars=250, base_price=200.0, trend_slope=-0.003)
        # 尝试顺势做多（但价格在 200 EMA 之下）
        tr_long = sl.simulate_strategy_trade(
            df_bear, entry_t=240, entry_price=float(df_bear["Close"].iloc[240]),
            side=1, stop_price=float(df_bear["Close"].iloc[240]) * 0.95,
            tp_price=float(df_bear["Close"].iloc[240]) * 1.15,
            use_trend_gate=True
        )
        self.assertEqual(tr_long["status"], "rejected_by_primary_trend")

    def test_simulate_strategy_trade_with_micro_decoupling(self):
        """测试 simulate_strategy_trade 接入微观脱钩豁免顺利进入撮合"""
        df_bull = _create_synthetic_series(n_bars=250, base_price=100.0, trend_slope=0.003)
        market_bear = {"SPY_bias": -1, "QQQ_bias": -1}

        # 逆风环境下开启微观脱钩
        tr_decoupled = sl.simulate_strategy_trade(
            df_bull, entry_t=240, entry_price=float(df_bull["Close"].iloc[240]),
            side=1, stop_price=float(df_bull["Close"].iloc[240]) * 0.95,
            tp_price=float(df_bull["Close"].iloc[240]) * 1.15,
            market_dow=market_bear,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        # 不应被 market_dow 驳回，而是完成持仓模拟
        self.assertNotEqual(tr_decoupled.get("status"), "rejected_by_market_dow")

    def test_build_trade_plan_macro_and_trend_gate_synergy(self):
        """测试 pa_engine.build_trade_plan 整合 M3 宏观与大势门禁"""
        df = _create_synthetic_series(n_bars=250, base_price=120.0, trend_slope=0.003)
        curr_p = float(df["Close"].iloc[-1])
        swings = pe.extract_swings(df, order=3, confirmed_only=True)
        key_levels = [{"price": curr_p * 0.95, "kind": "swing", "src": "D1"}]

        # 1. 正常多头在 200 EMA 上方 + 大盘健康
        market_bull = {"SPY_bias": 1, "QQQ_bias": 1}
        plan_pass = pe.build_trade_plan(
            mode="daily", price=curr_p, swings_mtf=swings, swings_ltf=swings,
            key_levels=key_levels, levels={}, bar_rng=2.0, bias_htf=1, bias_mtf=1,
            df_mtf=df, market_dow=market_bull, use_trend_gate=True
        )
        self.assertIn("trend_gate", plan_pass)
        self.assertTrue(plan_pass["trend_gate"]["passed"])

        # 2. 大盘偏空但微观强信念脱钩
        market_bear = {"SPY_bias": -1, "QQQ_bias": -1}
        plan_decoupled = pe.build_trade_plan(
            mode="daily", price=curr_p, swings_mtf=swings, swings_ltf=swings,
            key_levels=key_levels, levels={}, bar_rng=2.0, bias_htf=1, bias_mtf=1,
            df_mtf=df, market_dow=market_bear,
            allow_decoupling=True, micro_conviction="strong_box_breakout"
        )
        self.assertIn("market_dow_gate", plan_decoupled)
        self.assertEqual(plan_decoupled["market_dow_gate"]["status"], "decoupled_micro_alpha")


if __name__ == "__main__":
    unittest.main()
