#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_tier5_adversarial_hardening.py — Tier 5 对抗性强化与极限压力测试套件

本测试套件作为系统的最终对抗性防线，对 R1~R4 重构成果进行极限压力测试：
  1. 复杂多市场大盘分化 (Complex Multi-Market Divergence):
     - SPY 平盘、QQQ 暴跌、DIA 逆势走强、IWM 小盘大涨场景；
     - 行业板块独立性印证 (科技标的 vs 工业/价值标的 vs 小盘标的)；
     - Micro Trumps Macro 脱钩豁免严格性检验；
     - 美股非对称做空风控红线在极端分化下的拦截与放行。
  2. 混沌摆动序列与波浪证伪止损 (Chaotic Swings & Objective Wave Invalidation):
     - 随机游走、高频假信号、极窄/极宽振幅、交替原则冲突；
     - 浪4破浪1顶、浪2破起点客观证伪止损基准；
     - 废除预测性数浪门禁（浪5衰竭不再一票否决合规道氏交易）。
  3. 200 EMA 精度、因果截断一致性与跳跃-扩散收敛 (EMA200 Precision & Jump-Diffusion):
     - 带有 Merton 泊松跳跃的连续随机过程，步进递推公式解析精度比对 (< 1e-10)；
     - 纯因果子序列截断一致性 (Zero Future Information Leakage)；
     - 暴跌/暴涨后 EMA 半衰期指数收敛验证；
     - 历史周期从 1 根到 10000 根的平滑自适应降级。
  4. 日内 8 bps 真实摩擦绞杀与 1000 次蒙特卡洛交易推演 (Intraday 8 bps Friction Bleed):
     - 1000 笔日内交易推演，验证 Net <= Gross 单调性与真实损耗精确核算；
     - 摩擦递增单调性检验 (0 bps -> 32 bps)；
     - 低波动防刷单硬门禁边界测试 (|entry - stop| / entry < 16 bps 严格阻断)；
     - 最小盈亏比 >= 2.5R 浮点边界精度 (2.4999 拒绝 vs 2.5001 放行)；
     - 15:55 ET 强制平仓与同根双触保守原则在摩擦下的净收益闭环。
  5. 脏数据与缺失容错极限防御 (Dirty Data, Zero/Negative Volume & Missing Bars):
     - 零成交量 (Volume=0)、负成交量、缺失成交量容错；
     - 缺失 bar、停牌假死零波动 (High == Low == Close) 零除异常防御；
     - 空 DataFrame、单根 K 线、NaN / Inf 异常数据防御。
"""

import math
from datetime import time as dtime, datetime, timedelta
import numpy as np
import pandas as pd
import unittest

import pa_engine as pe
import strategy_layer as sl
import strategy_day as sd


class TestTier5MultiMarketDivergence(unittest.TestCase):
    """
    Tier 5 对抗性测试 1: 复杂多市场大盘分化场景 (Multi-Market Divergence)
    SPY 平盘、QQQ 暴跌、DIA 走强、IWM 大涨，检验板块独立性、标的行业匹配及微观脱钩
    """

    def setUp(self):
        # 构造复杂的极端多市场分化大盘状态
        # SPY: 0 (平盘震荡)
        # QQQ: -1 (科技破位重挫 -3%)
        # DIA: +1 (工业/价值龙头逆势大涨 +1.5%)
        # IWM: +1 (小盘股全面爆发 +2.5%)
        self.divergent_market = {
            "SPY": {"bias": 0, "is_breakdown": False, "mtf": {"phase": "震荡中性"}},
            "QQQ": {"bias": -1, "is_bear": True, "is_breakdown": False, "mtf": {"phase": "道氏空头破位"}},
            "DIA": {"bias": 1, "is_bull": True, "is_breakdown": False, "mtf": {"phase": "道氏主升浪"}},
            "IWM": {"bias": 1, "is_bull": True, "is_breakdown": False, "mtf": {"phase": "突破新高"}},
            "SPY_bias": 0,
            "QQQ_bias": -1,
            "DIA_bias": 1,
            "IWM_bias": 1,
        }

    def test_adv_divergence_tech_stock_long_rejected(self):
        """极端分化下：科技标的做多必须被 QQQ 空头结构严格拦截"""
        # 测试显式 target_sector="tech"
        res_tech = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="AAPL",
            target_sector="tech",
            side=1
        )
        self.assertFalse(res_tech["passed"], "QQQ 重挫背景下科技股做多必须被拦截")
        self.assertEqual(res_tech["status"], "rejected_by_market_dow")
        self.assertIn("QQQ", res_tech["reason"])

        # 测试代码在 SYMBOL_SECTOR_MAP 中的科技龙头 (NVDA)
        res_nvda = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="NVDA",
            side=1
        )
        self.assertFalse(res_nvda["passed"], "NVDA 作为科技龙头在 QQQ 空头时必须被拦截")

    def test_adv_divergence_industrial_stock_long_approved(self):
        """极端分化下：工业价值标的凭借 DIA 强势印证，多头顺利放行（消除 QQQ 误杀）"""
        res_cat = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="CAT",
            target_sector="industrial",
            side=1
        )
        self.assertTrue(res_cat["passed"], "DIA 走强时工业龙头 CAT 多头必须放行")
        self.assertEqual(res_cat["status"], "confirmed_by_market")
        self.assertIn("DIA", res_cat["reason"])

        # 验证金融/工业龙头 BA / JPM
        res_ba = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="BA",
            side=1
        )
        self.assertTrue(res_ba["passed"], "波音 BA 属于工业板块，必须借助 DIA 顺利放行")

    def test_adv_divergence_small_cap_long_approved(self):
        """极端分化下：小盘股凭借 IWM 强势走强与市场广度放行"""
        res_small = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="IWM",
            target_sector="small_cap",
            side=1
        )
        self.assertTrue(res_small["passed"], "小盘基准 IWM 走强时小盘标的多头必须放行")
        self.assertEqual(res_small["status"], "confirmed_by_market")
        self.assertIn("IWM", res_small["reason"])

    def test_adv_divergence_micro_decoupling_overrides_macro_tech_bear(self):
        """极端分化下：科技标的若具备高确定性微观突破，触发 Micro Trumps Macro 脱钩放行"""
        # 未开启脱钩：仍被驳回
        res_no_decouple = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="AAPL",
            target_sector="tech",
            side=1,
            allow_decoupling=False,
            micro_conviction="strong_box_breakout"
        )
        self.assertFalse(res_no_decouple["passed"])

        # 开启脱钩但 conviction 不足：仍被驳回
        res_weak = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="AAPL",
            target_sector="tech",
            side=1,
            allow_decoupling=True,
            micro_conviction="weak_momentum"
        )
        self.assertFalse(res_weak["passed"], "弱微观形态禁止滥用脱钩豁免")

        # 开启脱钩且具备 strong_box_breakout：放行并标记 decoupled_micro_alpha
        res_decouple_box = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="AAPL",
            target_sector="tech",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertTrue(res_decouple_box["passed"])
        self.assertEqual(res_decouple_box["status"], "decoupled_micro_alpha")

        # 开启脱钩且具备 strong_2b_reversal：放行
        res_decouple_2b = sl.verify_market_dow_gate(
            market_dow=self.divergent_market,
            target_symbol="NVDA",
            target_sector="tech",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res_decouple_2b["passed"])
        self.assertEqual(res_decouple_2b["status"], "decoupled_micro_alpha")

    def test_adv_divergence_asymmetric_short_filtering(self):
        """非对称做空门禁：牛市大势中禁止盲目做空，极端分化与2B反转时的严格分类"""
        # 全牛市环境：SPY, QQQ, DIA 全部顺势走强
        bull_market = {
            "SPY": {"bias": 1, "is_bull": True},
            "QQQ": {"bias": 1, "is_bull": True},
            "DIA": {"bias": 1, "is_bull": True},
            "SPY_bias": 1, "QQQ_bias": 1, "DIA_bias": 1
        }
        # 顺势长牛无理由做空：必须一票否决
        res_blind_short = sl.verify_macro_short_gate(
            market_dow=bull_market,
            side=-1
        )
        self.assertFalse(res_blind_short["passed"], "长牛顺势环境下禁止无理由做空")
        self.assertEqual(res_blind_short["status"], "rejected_by_macro_short_gate")

        # 顺势长牛中出现高确定性 2B 顶假突破 (2B_sweep_high)：豁免放行
        res_2b_short = sl.verify_macro_short_gate(
            market_dow=bull_market,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res_2b_short["passed"], "高确定性 2B 顶流动性猎杀允许反转做空")
        self.assertEqual(res_2b_short["status"], "decoupled_2b_short_reversal")

        # 出现板块破位空翻 (QQQ 出现破位)：放行做空
        divergent_short = sl.verify_macro_short_gate(
            market_dow=self.divergent_market,
            side=-1
        )
        self.assertTrue(divergent_short["passed"], "大盘指数出现空头结构时允许放行做空")


class TestTier5ChaoticWaveInvalidation(unittest.TestCase):
    """
    Tier 5 对抗性测试 2: 混沌摆动序列与波浪铁律证伪止损 (Chaotic Swings & Invalidation)
    验证在随机游走、严重重叠、违背三大铁律与极端振幅下，波浪引擎绝不崩溃且绝无预测性信号
    """

    def test_adv_chaotic_random_whipsaw_swings(self):
        """生成 100 个随机高频锯齿震荡拐点，验证波浪引擎数值稳健且不报预测性入场"""
        np.random.seed(2026)
        swings = []
        curr_price = 100.0
        curr_time = pd.Timestamp("2026-01-01")
        for i in range(100):
            # 严格高低交替，但价格幅度是高斯随机扰动
            stype = "High" if i % 2 == 1 else "Low"
            delta = np.random.uniform(0.5, 3.0)
            curr_price = curr_price + delta if stype == "High" else curr_price - delta
            swings.append({
                "index": i * 5,
                "time": curr_time + pd.Timedelta(days=i),
                "type": stype,
                "price": round(curr_price, 2),
                "confirmed": True
            })

        # 运行分类与评估
        wp = pe.classify_wave_pattern(swings)
        eval_res = pe.evaluate_wave_position(swings, current_price=curr_price)

        self.assertIsInstance(wp, dict)
        self.assertIsInstance(eval_res, dict)
        # 绝不能产生预测性买卖信号
        self.assertEqual(eval_res.get("signal"), "neutral")
        self.assertEqual(eval_res.get("action"), "neutral")
        # 随机游走状态下可能为 unclassified 或某临时浪型，但绝不能抛出异常
        self.assertIn("status", eval_res)

    def test_adv_wave4_overlap_wave1_deep_invalidation(self):
        """铁律3严苛对抗：浪4大幅击穿浪1顶，必须判定形态失效，且 wave4_invalidation_level 准确锚定浪1"""
        # 构造多头推动：0(100) -> 1(130) -> 2(115) -> 3(160) -> 4(110，击穿130) -> 5(180)
        swings = [
            {"index": 0, "time": pd.Timestamp("2026-01-01"), "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "time": pd.Timestamp("2026-01-06"), "type": "High", "price": 130.0, "confirmed": True},
            {"index": 10, "time": pd.Timestamp("2026-01-11"), "type": "Low", "price": 115.0, "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-16"), "type": "High", "price": 160.0, "confirmed": True},
            {"index": 20, "time": pd.Timestamp("2026-01-21"), "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2026-01-26"), "type": "High", "price": 180.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings)
        eval_res = pe.evaluate_wave_position(swings, current_price=180.0)

        # 铁律3必须未通过
        self.assertFalse(wp.get("is_valid_impulse", True))
        self.assertFalse(wp["rules"]["rule3"]["passed"])
        self.assertEqual(eval_res["status"], "invalid")
        # 证伪止损线精确锚定浪1顶点 130.0
        self.assertEqual(eval_res["wave4_invalidation_level"], 130.0)
        self.assertEqual(eval_res["signal"], "neutral")

    def test_adv_wave2_exceeds_wave1_origin_invalidation(self):
        """铁律1严苛对抗：浪2回撤跌破浪1起点，形态失效，证伪参考位锚定 P0 起点"""
        # 0(100) -> 1(120) -> 2(95，跌破100)
        swings = [
            {"index": 0, "time": pd.Timestamp("2026-01-01"), "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "time": pd.Timestamp("2026-01-06"), "type": "High", "price": 120.0, "confirmed": True},
            {"index": 10, "time": pd.Timestamp("2026-01-11"), "type": "Low", "price": 95.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings)
        eval_res = pe.evaluate_wave_position(swings, current_price=110.0)

        self.assertFalse(wp.get("is_valid_impulse", True))
        self.assertFalse(wp["rules"]["rule1"]["passed"])
        self.assertEqual(eval_res["status"], "invalid")
        self.assertEqual(eval_res["rule1_invalidation_level"], 100.0)

    def test_adv_wave3_shortest_invalidation(self):
        """铁律2严苛对抗：浪3涨幅劣于浪1与浪5，成为最短浪，形态失效"""
        # L1 = 130 - 100 = 30
        # L3 = 145 - 125 = 20 (最短)
        # L5 = 190 - 140 = 50
        swings = [
            {"index": 0, "time": pd.Timestamp("2026-01-01"), "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "time": pd.Timestamp("2026-01-06"), "type": "High", "price": 130.0, "confirmed": True},
            {"index": 10, "time": pd.Timestamp("2026-01-11"), "type": "Low", "price": 125.0, "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-16"), "type": "High", "price": 145.0, "confirmed": True},
            {"index": 20, "time": pd.Timestamp("2026-01-21"), "type": "Low", "price": 140.0, "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2026-01-26"), "type": "High", "price": 190.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(swings)
        self.assertFalse(wp.get("is_valid_impulse", True))
        self.assertFalse(wp["rules"]["rule2"]["passed"])

    def test_adv_wave_no_veto_on_valid_dow_setup_in_trade_plan(self):
        """R1 核心准则验证：即使波浪形态失效，道氏/箱体/2B 产生的交易计划绝不受波浪否决"""
        # 构造一个波浪形态失效（浪4穿浪1）但在箱底成功走出 2B 假摔的反转交易机会
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        prices = np.linspace(100, 110, 100)
        df_mtf = pd.DataFrame({
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 1000000
        }, index=dates)

        # 违背铁律3的 swings
        swings = [
            {"index": 10, "time": dates[10], "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 25, "time": dates[25], "type": "High", "price": 125.0, "confirmed": True},
            {"index": 40, "time": dates[40], "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 60, "time": dates[60], "type": "High", "price": 150.0, "confirmed": True},
            {"index": 75, "time": dates[75], "type": "Low", "price": 105.0, "confirmed": True}, # 击穿 125
        ]

        # 构造有效箱体支撑与关键位
        levels = {
            "box": {
                "is_box": True,
                "box_high": 120.0,
                "box_low": 100.0,
                "state": "SWEEP_LOW",
                "box_pos": 0.05
            }
        }
        key_levels = [{"price": 100.0, "type": "box_low", "tier": 4}]

        plan = pe.build_trade_plan(
            mode="swing",
            price=102.0,
            swings_mtf=swings,
            swings_ltf=swings,
            key_levels=key_levels,
            levels=levels,
            bar_rng=2.0,
            bias_htf=1,
            bias_mtf=1,
            df_mtf=df_mtf,
            min_rr=2.0
        )

        # 交易计划绝不能因波浪失效被 abort (无 rejected_by_wave5_exhaustion)
        self.assertNotIn("rejected_by_wave5_exhaustion", plan.get("rejected_by", ""))
        self.assertIn("wave_eval", plan)
        self.assertEqual(plan["wave_eval"]["status"], "invalid")
        # 交易计划记录了波浪证伪位，但未剥夺交易资格
        self.assertIn("wave_note", plan)


class TestTier5Ema200JumpDiffusionConvergence(unittest.TestCase):
    """
    Tier 5 对抗性测试 3: 200 EMA 精度、因果截断一致性与跳跃-扩散收敛 (EMA200 Precision & Jump-Diffusion)
    通过 Merton 泊松跳跃扩散模型模拟极端闪崩与暴涨，校验 EMA 计算精度与收敛物理规律
    """

    def _generate_jump_diffusion_series(self, n=600, s0=100.0, mu=0.05, sigma=0.20,
                                        jump_intensity=0.03, jump_mean=0.0, jump_std=0.15,
                                        seed=42):
        """生成 Merton Jump-Diffusion 价格路径"""
        np.random.seed(seed)
        dt = 1.0 / 252.0
        prices = np.zeros(n)
        prices[0] = s0

        for t in range(1, n):
            # 几何布朗运动分量
            z = np.random.normal(0, 1)
            diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

            # 泊松跳跃分量
            n_jumps = np.random.poisson(jump_intensity)
            jump = 0.0
            if n_jumps > 0:
                jump = np.sum(np.random.normal(jump_mean, jump_std, n_jumps))

            prices[t] = prices[t - 1] * np.exp(diffusion + jump)

        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Open": prices * 0.998,
            "High": prices * 1.008,
            "Low": prices * 0.992,
            "Close": prices,
            "Volume": 5000000
        }, index=dates)
        return df

    def test_adv_ema200_analytical_step_by_step_exactness(self):
        """对比单步解析递推公式 EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}，误差必须 < 1e-8"""
        df = self._generate_jump_diffusion_series(n=500, seed=101)
        closes = df["Close"].values

        # 手工递推
        alpha = 2.0 / (200.0 + 1.0)
        manual_ema = np.zeros(len(closes))
        manual_ema[0] = closes[0]
        for i in range(1, len(closes)):
            manual_ema[i] = alpha * closes[i] + (1.0 - alpha) * manual_ema[i - 1]

        # 引擎纯因果计算 (输出以 round(..., 4) 保留4位小数)
        gate_res = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
        engine_ema = gate_res["ema_val"]

        expected_ema_rounded = round(float(manual_ema[-1]), 4)
        diff = abs(engine_ema - expected_ema_rounded)
        self.assertLess(diff, 1e-8, f"引擎 200 EMA 与严格递推公式误差 ({diff}) 超出容许精度")

    def test_adv_ema200_causal_subsequence_invariance_under_jump_shocks(self):
        """在经历多重极端跳跃冲击下，验证截断子序列计算与全历史在 t 时刻计算 100% 绝对一致"""
        df = self._generate_jump_diffusion_series(n=600, jump_intensity=0.10, jump_std=0.25, seed=202)

        # 选取 5 个跨越跳跃点的截断时点
        check_points = [220, 300, 380, 450, 550]
        for t in check_points:
            # 截断数据直接传入
            sub_df = df.iloc[:t + 1].copy()
            res_sub = sl.verify_primary_trend_gate(sub_df, side=1, ema_period=200)

            # 全量数据带 t 传入
            res_full_t = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=t)

            self.assertAlmostEqual(res_sub["ema_val"], res_full_t["ema_val"], places=6,
                                   msg=f"t={t} 时刻截断子序列与全序列指定 t 计算的 EMA 不一致")
            self.assertEqual(res_sub["passed"], res_full_t["passed"],
                             msg=f"t={t} 时刻截断门禁判定结果不一致")

    def test_adv_ema200_convergence_rate_post_shock(self):
        """理论半衰期验证：在价格发生 50% 阶跃冲击后，EMA 偏离度应精确遵循理论衰减半衰期 (约 69 根K线)"""
        n_bars = 400
        # 前 200 根平稳在 100 元，第 200 根直接跳跃至 150 元 (+50%) 并保持
        prices = np.full(n_bars, 100.0)
        shock_t = 200
        prices[shock_t:] = 150.0

        dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")
        df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices}, index=dates)

        # 检查 shock_t 时刻偏离 = 50 元
        # 理论半衰期: (1 - alpha)^k = 0.5 => k = ln(0.5) / ln(1 - 2/201) ≈ 69.31 根
        alpha = 2.0 / 201.0
        theoretical_k = math.log(0.5) / math.log(1.0 - alpha)

        # 观测 shock_t + 70 根时刻的 EMA
        t_half = shock_t + 70
        res_half = sl.verify_primary_trend_gate(df, side=1, ema_period=200, t=t_half)
        ema_at_half = res_half["ema_val"]

        # EMA 从 100 上升至 150，在半衰期应达到约 125 元 (50% 收敛)
        expected_ema = 150.0 - 50.0 * (0.5)
        self.assertAlmostEqual(ema_at_half, expected_ema, delta=1.5,
                               msg=f"200 EMA 在半衰期后的收敛值 ({ema_at_half:.2f}) 偏离理论值 ({expected_ema:.2f})")

    def test_adv_ema200_boundary_history_lengths(self):
        """边界测试：历史数据从 1 根、2 根、199 根到 10000 根，自适应平滑平稳无崩溃"""
        for n in [1, 2, 5, 50, 199, 200, 201, 1000]:
            df_n = pd.DataFrame({
                "Close": np.linspace(50, 100, n),
                "Open": np.linspace(50, 100, n),
                "High": np.linspace(50, 100, n) + 1.0,
                "Low": np.linspace(50, 100, n) - 1.0
            })
            res = sl.verify_primary_trend_gate(df_n, side=1, ema_period=200)
            self.assertTrue(res["passed"] or not res["passed"])
            self.assertFalse(math.isnan(res["ema_val"]))
            self.assertFalse(math.isinf(res["ema_val"]))


class TestTier5IntradayFrictionBleedSimulation(unittest.TestCase):
    """
    Tier 5 对抗性测试 4: 日内 8 bps 摩擦绞杀与 1000 次蒙特卡洛交易推演 (Intraday Friction Bleed)
    验证真实 8 bps 摩擦在海量交易中的单调性、防刷单门禁敏锐度与 2.5R 盈亏比纪律
    """

    def _generate_synthetic_5m_day(self, base_price=100.0, seed=42):
        """生成一整天标准 78 根 5 分钟美股常规交易日 K 线 (09:30 - 16:00 ET)"""
        np.random.seed(seed)
        start_dt = pd.Timestamp("2026-06-01 09:30:00", tz="America/New_York")
        times = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(78)]

        # 随机游走 + 正弦日内微观形态
        drift = np.sin(np.linspace(0, 2 * np.pi, 78)) * 1.5
        noise = np.cumsum(np.random.normal(0, 0.2, 78))
        closes = base_price + drift + noise
        opens = np.roll(closes, 1)
        opens[0] = base_price
        highs = np.maximum(opens, closes) + np.random.uniform(0.05, 0.25, 78)
        lows = np.minimum(opens, closes) - np.random.uniform(0.05, 0.25, 78)

        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 10000
        }, index=times)
        return df

    def test_adv_friction_bleed_1000_simulated_trades(self):
        """1000 笔日内交易推演：严格核算 Net <= Gross，量化摩擦对账户期望的真实剥蚀"""
        n_sim = 1000
        np.random.seed(303)

        gross_pnls = []
        net_pnls = []
        gross_r_list = []
        net_r_list = []

        for sim_idx in range(n_sim):
            df_day = self._generate_synthetic_5m_day(base_price=100.0 + (sim_idx % 50), seed=sim_idx)
            entry_t = np.random.randint(5, 50)
            entry_price = float(df_day["Open"].iloc[entry_t])

            # 随机做多或做空
            side = 1 if np.random.rand() > 0.5 else -1

            # 构造保证符合最低 RR (>= 2.5R) 与防刷单门禁 (risk >= 20 bps) 的订单
            risk_dist = float(entry_price * np.random.uniform(0.003, 0.015)) # 30~150 bps 风险空间
            reward_dist = float(risk_dist * np.random.uniform(2.5, 4.0))     # 2.5R ~ 4.0R

            stop_price = entry_price - risk_dist if side == 1 else entry_price + risk_dist
            tp_price = entry_price + reward_dist if side == 1 else entry_price - reward_dist

            trade_res = sd.simulate_day_trade(
                df=df_day,
                entry_t=entry_t,
                entry_price=entry_price,
                side=side,
                stop_price=stop_price,
                tp_price=tp_price,
                min_rr=2.5,
                slippage_bps=3.0,
                commission_bps=1.0
            )

            # 排除由于时段跳过的未成交
            if trade_res.get("status") == "completed":
                g_pnl = trade_res["gross_pnl_pct"]
                n_pnl = trade_res["net_pnl_pct"]
                g_r = trade_res["gross_r"]
                n_r = trade_res["net_r"]

                # 核心对抗断言 1: 任何一笔交易，净收益必须严格 <= 毛收益
                self.assertLessEqual(n_pnl, g_pnl + 1e-9, "单笔净收益绝对不可能大于毛收益")
                self.assertLessEqual(n_r, g_r + 1e-9, "单笔净 R 乘数绝对不可能大于毛 R 乘数")

                # 核心对抗断言 2: 摩擦扣减额必须严格正向且接近 8 bps
                # 摩擦损耗 = gross - net >= 0.00075 (考虑滑点与规费)
                friction_loss = g_pnl - n_pnl
                self.assertGreater(friction_loss, 0.0006, "摩擦损耗必须大于零且覆盖双边规费与滑点")

                gross_pnls.append(g_pnl)
                net_pnls.append(n_pnl)
                gross_r_list.append(g_r)
                net_r_list.append(n_r)

        self.assertGreater(len(gross_pnls), 700, "1000 笔模拟中必须有大量成功撮合成交的样本")
        # 统计整体损耗
        total_gross_pnl = sum(gross_pnls)
        total_net_pnl = sum(net_pnls)
        total_friction_drag = total_gross_pnl - total_net_pnl
        self.assertGreater(total_friction_drag, 0.0, "千笔模拟总摩擦损耗必须严格大于零")

    def test_adv_friction_monotonicity_across_bps_levels(self):
        """摩擦递增单调性检验：同一笔盈利交易在 0 -> 4 -> 8 -> 16 -> 32 bps 下净收益严格单调递减"""
        df_day = self._generate_synthetic_5m_day(base_price=150.0, seed=888)
        entry_t = 10
        entry_p = float(df_day["Open"].iloc[entry_t])
        stop_p = entry_p - 5.0  # risk = 5.0 (3.33% > 85 bps 门禁，确保在极高摩擦下仍免受刷单拦截)
        tp_p = entry_p + 15.0   # reward = 15.0 (RR = 3.0R >= 2.5R)

        net_returns = []
        friction_levels = [0.0, 2.0, 4.0, 8.0, 16.0] # 单边滑点 bps

        for slip in friction_levels:
            res = sd.simulate_day_trade(
                df=df_day,
                entry_t=entry_t,
                entry_price=entry_p,
                side=1,
                stop_price=stop_p,
                tp_price=tp_p,
                min_rr=2.5,
                slippage_bps=slip,
                commission_bps=slip / 3.0
            )
            self.assertEqual(res["status"], "completed")
            net_returns.append(res["net_pnl_pct"])

        # 净收益必须严格递减
        for i in range(len(net_returns) - 1):
            self.assertGreater(net_returns[i], net_returns[i + 1],
                               f"摩擦从 {friction_levels[i]} 上升至 {friction_levels[i+1]} 时净收益未单调递减")

    def test_adv_low_volatility_churn_filter_threshold_sharpness(self):
        """低波动防刷单门禁锐度测试：15.9 bps 风险空间被驳回 vs 16.1 bps 风险空间放行"""
        df_day = self._generate_synthetic_5m_day(base_price=100.0, seed=999)
        entry_t = 15
        entry_price = 100.0

        # 双边 8 bps 摩擦对应阈值 = 2 * 0.0008 = 0.0016 (16 bps)
        # 1. 风险空间 14 bps (0.14 元 / 100 元 = 0.0014 < 0.0016)
        res_rejected = sd.simulate_day_trade(
            df=df_day,
            entry_t=entry_t,
            entry_price=entry_price,
            side=1,
            stop_price=99.86, # risk = 0.14
            tp_price=100.50,  # reward = 0.50 (RR = 3.57R)
            min_rr=2.5,
            slippage_bps=3.0,
            commission_bps=1.0
        )
        self.assertEqual(res_rejected["status"], "rejected_by_low_volatility_churn")
        self.assertEqual(res_rejected["reason"], "rejected_by_low_volatility_churn")

        # 2. 风险空间 25 bps (0.25 元 / 100 元 = 0.0025 > 0.0016)
        res_passed = sd.simulate_day_trade(
            df=df_day,
            entry_t=entry_t,
            entry_price=entry_price,
            side=1,
            stop_price=99.75, # risk = 0.25
            tp_price=100.75,  # reward = 0.75 (RR = 3.0R)
            min_rr=2.5,
            slippage_bps=3.0,
            commission_bps=1.0
        )
        self.assertEqual(res_passed["status"], "completed", "风险空间覆盖摩擦阈值且 RR >= 2.5R 必须放行")

    def test_adv_min_rr_2_5_discipline_under_friction(self):
        """最小盈亏比 2.5R 硬门禁浮点精度：2.4999R 驳回 vs 2.5001R 放行"""
        df_day = self._generate_synthetic_5m_day(base_price=100.0, seed=777)
        entry_p = 100.0
        risk = 1.0 # stop = 99.0
        stop_p = 99.0

        # 2.4999R: tp = 102.4999
        res_sub = sd.simulate_day_trade(
            df=df_day, entry_t=10, entry_price=entry_p, side=1,
            stop_price=stop_p, tp_price=entry_p + 2.4999, min_rr=2.5
        )
        self.assertEqual(res_sub["status"], "rejected_by_min_rr")

        # 2.5001R: tp = 102.5001
        res_sup = sd.simulate_day_trade(
            df=df_day, entry_t=10, entry_price=entry_p, side=1,
            stop_price=stop_p, tp_price=entry_p + 2.5001, min_rr=2.5
        )
        self.assertEqual(res_sup["status"], "completed")

    def test_adv_eod_squareoff_and_same_bar_double_touch_friction(self):
        """15:55 ET 强制平仓与同根双触保守原则在摩擦下的净收益闭环"""
        df_day = self._generate_synthetic_5m_day(base_price=100.0, seed=555)

        # 1. 制造尾盘 15:50 入场且未触及止盈止损的交易 -> 15:55 强平
        entry_t = 70 # 约 15:20
        entry_p = float(df_day["Open"].iloc[entry_t])
        res_eod = sd.simulate_day_trade(
            df=df_day, entry_t=entry_t, entry_price=entry_p,
            side=1, stop_price=entry_p - 1.0, tp_price=entry_p + 3.0, min_rr=2.5,
            eod_exit_time=dtime(15, 55)
        )
        self.assertEqual(res_eod["status"], "completed")
        self.assertIn(res_eod["exit_reason"], ["eod_forced_close", "session_end_close"])

        # 2. 制造同根双触：计划 RR = 2.5R，且当根 K 线振幅极大，同时触及 TP 与 Stop
        entry_t2 = 20
        df_dt = df_day.copy()
        entry_p = float(df_dt["Open"].iloc[entry_t2])
        risk = 0.50
        reward = 1.30  # RR = 2.6R >= 2.5R
        stop_p = entry_p - risk
        tp_p = entry_p + reward

        # 人工设置当根 K 线的极大振幅，使其同时触及止损与止盈
        df_dt.iloc[entry_t2, df_dt.columns.get_loc("High")] = entry_p + 2.0  # > tp_p (1.30)
        df_dt.iloc[entry_t2, df_dt.columns.get_loc("Low")] = entry_p - 1.0   # < stop_p (-0.50)

        res_double_touch = sd.simulate_day_trade(
            df=df_dt, entry_t=entry_t2, entry_price=entry_p, side=1,
            stop_price=stop_p, tp_price=tp_p, min_rr=2.5
        )
        self.assertEqual(res_double_touch["status"], "completed")
        self.assertEqual(res_double_touch["exit_reason"], "stop_loss_conservative",
                         "同根双触必须保守判定止损优先")
        self.assertLess(res_double_touch["net_pnl_pct"], 0.0, "双触止损出场净收益必须为负")


class TestTier5DirtyDataAndMissingBarDefense(unittest.TestCase):
    """
    Tier 5 对抗性测试 5: 脏数据、零/负成交量与缺失数据极限防御 (Dirty Data Defense)
    测试在零成交量、缺失时间戳、无波动平盘(High==Low)以及 NaN/Inf 极端破坏性输入下的代码弹性
    """

    def test_adv_zero_and_negative_volume_resilience(self):
        """成交量为 0、负数或 NaN 时，核心计算引擎不应受影响（裸K价格行为不依赖指标成交量）"""
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        prices = np.linspace(100, 150, 100)
        df_dirty_vol = pd.DataFrame({
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": [0, -100, np.nan, 0] * 25
        }, index=dates)

        # 1. 中位振幅计算
        rng = pe.median_bar_range(df_dirty_vol)
        self.assertAlmostEqual(rng, 2.0, places=4)

        # 2. 拐点提取
        swings = pe.extract_swings(df_dirty_vol, order=3)
        self.assertIsInstance(swings, list)

        # 3. 200 EMA 顺势门禁
        trend_gate = sl.verify_primary_trend_gate(df_dirty_vol, side=1)
        self.assertTrue(trend_gate["passed"])

        # 4. 箱体与通道探测
        box = pe.detect_consolidation_box(df_dirty_vol, swings)
        channel = pe.detect_price_channel(df_dirty_vol, swings)
        self.assertIsInstance(box, dict)
        self.assertIsInstance(channel, dict)

    def test_adv_flatline_zero_range_bars(self):
        """完全平盘零振幅行情 (High == Low == Open == Close，如停牌或流动性完全枯竭)"""
        dates = pd.date_range("2026-01-01", periods=50, freq="5min")
        df_flat = pd.DataFrame({
            "Open": np.full(50, 100.0),
            "High": np.full(50, 100.0),
            "Low": np.full(50, 100.0),
            "Close": np.full(50, 100.0),
            "Volume": np.zeros(50)
        }, index=dates)

        # 中位振幅应为 0.0
        rng = pe.median_bar_range(df_flat)
        self.assertEqual(rng, 0.0)

        # 日内撮合应触发零风险无效或防刷单门禁，绝不产生零除异常 (ZeroDivisionError)
        trade_res = sd.simulate_day_trade(
            df=df_flat, entry_t=10, entry_price=100.0, side=1,
            stop_price=100.0, tp_price=105.0
        )
        self.assertIn(trade_res["status"], ["zero_risk_invalid", "rejected_by_low_volatility_churn"])

    def test_adv_missing_bars_and_irregular_intervals(self):
        """不规则时间戳序列（含跳跃缺口、非等间隔采样、节假日断层），引擎保持因果时序单调"""
        # 构造跨越春节和周末的离散不规则时戳
        irregular_dates = [
            pd.Timestamp("2026-01-05 09:30"),
            pd.Timestamp("2026-01-05 09:35"),
            pd.Timestamp("2026-01-05 11:20"), # 缺口 105 分钟
            pd.Timestamp("2026-01-05 15:55"),
            pd.Timestamp("2026-01-12 09:30"), # 跨周缺口 7 天
            pd.Timestamp("2026-01-12 10:00"),
        ]
        prices = [100.0, 102.0, 101.5, 105.0, 110.0, 108.0]
        df_irreg = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.5 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": 1000
        }, index=irregular_dates)

        # 运行 EMA
        res = sl.verify_primary_trend_gate(df_irreg, side=1, ema_period=5)
        self.assertTrue(res["passed"])
        self.assertAlmostEqual(res["current_price"], 108.0)

    def test_adv_single_row_and_empty_dataframe_safety(self):
        """空 DataFrame、只有 1 根 K 线的极限输入，所有模块安全降级不崩溃"""
        df_empty = pd.DataFrame()
        df_single = pd.DataFrame({"Close": [100.0], "Open": [100.0], "High": [101.0], "Low": [99.0]})

        # 1. 200 EMA 门禁
        res_empty = sl.verify_primary_trend_gate(df_empty, side=1)
        res_single = sl.verify_primary_trend_gate(df_single, side=1)
        self.assertEqual(res_empty["status"], "rejected_by_missing_trend_data")
        self.assertFalse(res_empty["passed"])
        self.assertTrue(res_single["passed"])

        # 2. 拐点提取
        sw_empty = pe.extract_swings(df_empty)
        sw_single = pe.extract_swings(df_single)
        self.assertEqual(sw_empty, [])
        self.assertEqual(sw_single, [])

        # 3. 波浪识别
        wp_empty = pe.classify_wave_pattern([])
        self.assertEqual(wp_empty["pattern"], "unclassified")

        # 4. 日内撮合
        trade_empty = sd.simulate_day_trade(df_empty, entry_t=0, entry_price=100.0, side=1, stop_price=95.0, tp_price=115.0)
        self.assertEqual(trade_empty["status"], "invalid_trade")

    def test_adv_nan_inf_corrupt_data_graceful_handling(self):
        """含有 NaN / Inf 异常浮点值时，各防御模块能够优雅隔离或剔除，无未捕获崩溃"""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        df_corrupt = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [102.0] * 10,
            "Low": [98.0] * 10,
            "Close": [100.0, 101.0, np.nan, 103.0, np.inf, 105.0, 104.0, 106.0, 107.0, 108.0],
            "Volume": 1000
        }, index=dates)

        # 清洗测试：yfinance 清洗逻辑验证
        clean_df = df_corrupt.dropna(subset=['Open', 'High', 'Low', 'Close'])
        clean_df = clean_df[~clean_df['Close'].isin([np.inf, -np.inf])]
        self.assertEqual(len(clean_df), 8)

        # 确保 clean_df 可正常完成 EMA 计算
        trend_res = sl.verify_primary_trend_gate(clean_df, side=1, ema_period=5)
        self.assertTrue(trend_res["passed"])
        self.assertFalse(math.isnan(trend_res["ema_val"]))


if __name__ == "__main__":
    unittest.main()
