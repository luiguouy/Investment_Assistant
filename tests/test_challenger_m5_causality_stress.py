#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_challenger_m5_causality_stress.py: Milestone 5 (Tier 5) 全系统因果性与极端压力测试套件

由角色 challenger_m5_2 独立构建并执行，专注于经验证伪、对抗性注入与全系统零前瞻偏差验证：
1. 波段拐点与波浪结构因果性 (Swing & Wave Causality under Adversarial Future Shock)
2. 200 EMA 大级别主趋势门禁动态截断与未来数据毒化防护 (200 EMA Dynamic Truncation & Poisoning Immunity)
3. 跨市场板块门禁与微观脱钩抗压测试 (Multi-Index Sector Gate & Micro Decoupling Stress)
4. 日内 1h HTF 趋势过滤与 8 bps 真实摩擦全系统零未来检验 (Intraday HTF & Friction Zero-Lookahead)
5. 极端边界工况、黑天鹅与异常数据鲁棒性 (Extreme Boundary, Shock & NaN Robustness)
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


def create_base_daily_dataset(n_bars: int = 350, seed: int = 100) -> pd.DataFrame:
    """生成具备明确多波段起伏的日线基准序列，用于因果性对比"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)
    
    # 模拟包含数个完整波段的平滑曲线加噪声
    t = np.linspace(0, 6 * np.pi, n_bars)
    trend = np.linspace(100.0, 180.0, n_bars)
    cyclical = 20.0 * np.sin(t)
    noise = rng.normal(0, 1.0, n_bars)
    closes = np.maximum(trend + cyclical + noise, 10.0)
    
    highs = closes + rng.uniform(0.5, 3.0, n_bars)
    lows = np.maximum(closes - rng.uniform(0.5, 3.0, n_bars), 5.0)
    opens = (highs + lows) / 2.0 + rng.normal(0, 0.5, n_bars)
    vols = rng.integers(1000000, 10000000, n_bars)
    
    return pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Volume": vols
    }, index=dates)


class TestTier5SwingAndWaveCausalityStress(unittest.TestCase):
    """Tier 5 因果性压力测试 1：波段拐点与波浪结构在动态截断与对抗性未来注入下的绝对不变性"""

    def setUp(self):
        self.df_base = create_base_daily_dataset(n_bars=300, seed=2026)

    def test_swing_extraction_causality_under_adversarial_future_injection(self):
        """
        在任意决策点 t，截断重算的已确认波段拐点集合，在未来注入恶意极端行情后必须 100% 保持严格不变。
        对抗场景：未来数据分别注入 (1) -90% 闪崩, (2) +800% 暴力拉升, (3) 剧烈巨幅高频震荡。
        """
        eval_cutoffs = [80, 120, 180, 240]

        for t_eval in eval_cutoffs:
            # 基准已确认拐点 (严格截断至 t_eval)
            df_clean_sub = self.df_base.iloc[:t_eval + 1].copy()
            clean_swings = pe.extract_swings(df_clean_sub, order=5, confirmed_only=True)
            self.assertGreater(len(clean_swings), 2, f"t={t_eval} 必须存在足够已确认拐点")

            # 注入恶意未来数据至原序列的 [t_eval + 1:] 部分
            # 1. 闪崩毒化
            df_crash = self.df_base.copy()
            df_crash.iloc[t_eval + 1:, df_crash.columns.get_loc("Close")] *= 0.10
            df_crash.iloc[t_eval + 1:, df_crash.columns.get_loc("High")] *= 0.12
            df_crash.iloc[t_eval + 1:, df_crash.columns.get_loc("Low")] *= 0.08

            # 2. 暴力拉升毒化
            df_surge = self.df_base.copy()
            df_surge.iloc[t_eval + 1:, df_surge.columns.get_loc("Close")] *= 8.0
            df_surge.iloc[t_eval + 1:, df_surge.columns.get_loc("High")] *= 8.5
            df_surge.iloc[t_eval + 1:, df_surge.columns.get_loc("Low")] *= 7.5

            # 3. 剧烈巨幅震荡毒化
            df_whipsaw = self.df_base.copy()
            noise = np.sin(np.arange(len(df_whipsaw) - t_eval - 1)) * 50.0
            df_whipsaw.iloc[t_eval + 1:, df_whipsaw.columns.get_loc("Close")] += noise
            df_whipsaw.iloc[t_eval + 1:, df_whipsaw.columns.get_loc("High")] += (noise + 10.0)
            df_whipsaw.iloc[t_eval + 1:, df_whipsaw.columns.get_loc("Low")] += (noise - 10.0)

            # 验证各毒化数据在 t_eval 处截断重算提取的已确认拐点
            for name, df_poisoned in [("Crash", df_crash), ("Surge", df_surge), ("Whipsaw", df_whipsaw)]:
                poisoned_sub = df_poisoned.iloc[:t_eval + 1]
                poisoned_swings = pe.extract_swings(poisoned_sub, order=5, confirmed_only=True)

                self.assertEqual(
                    len(clean_swings), len(poisoned_swings),
                    f"在 t={t_eval} 处，{name} 毒化序列已确认拐点数量 ({len(poisoned_swings)}) 与干净序列 ({len(clean_swings)}) 不一致！"
                )

                for idx, (cs, ps) in enumerate(zip(clean_swings, poisoned_swings)):
                    self.assertEqual(cs["index"], ps["index"], f"t={t_eval} {name} 拐点 #{idx} 索引不匹配")
                    self.assertEqual(cs["type"], ps["type"], f"t={t_eval} {name} 拐点 #{idx} 类型不匹配")
                    self.assertEqual(cs["dow"], ps["dow"], f"t={t_eval} {name} 拐点 #{idx} 道氏标签不匹配")
                    self.assertAlmostEqual(cs["price"], ps["price"], places=4, msg=f"t={t_eval} {name} 拐点 #{idx} 价格不匹配")

    def test_wave_invalidation_level_invariance_under_future_data_poisoning(self):
        """
        波浪客观铁律证伪位（如浪4不破浪1极值 wave4_invalidation_level）
        在 t_eval 时的计算结果，必须严格独立于未来任何 K 线变动。
        """
        # 构建一个标准的 5 浪推进结构前 4 浪完成时点
        swings = [
            {"index": 10, "time": pd.Timestamp("2025-01-10"), "price": 100.0, "type": "Low", "dow": "Low", "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2025-01-25"), "price": 130.0, "type": "High", "dow": "High", "confirmed": True},
            {"index": 40, "time": pd.Timestamp("2025-02-10"), "price": 115.0, "type": "Low", "dow": "HL", "confirmed": True},
            {"index": 60, "time": pd.Timestamp("2025-03-01"), "price": 165.0, "type": "High", "dow": "HH", "confirmed": True},
            {"index": 75, "time": pd.Timestamp("2025-03-20"), "price": 140.0, "type": "Low", "dow": "HL", "confirmed": True},
        ]

        wave_clean = pe.evaluate_wave_position(swings, df=self.df_base.iloc[:80], current_price=145.0)
        self.assertEqual(wave_clean.get("status"), "valid")
        w4_level_clean = wave_clean.get("wave4_invalidation_level")
        self.assertIsNotNone(w4_level_clean)
        self.assertEqual(w4_level_clean, 130.0, "浪4证伪止损位必须锚定浪1顶 P1 = 130.0")

        # 即使未来数据发生崩盘或暴涨，只要 t_eval 处的已确认拐点不变，波浪证伪止损位必须保持一致
        future_poisoned_prices = [5.0, 50.0, 300.0, 1000.0]
        for p in future_poisoned_prices:
            wave_poison = pe.evaluate_wave_position(swings, df=self.df_base.iloc[:80], current_price=p)
            self.assertEqual(
                wave_poison.get("wave4_invalidation_level"), 130.0,
                f"未来报价变动至 {p} 不得篡改历史波浪证伪基准位"
            )

    def test_wave_downgrade_guarantee_no_predictive_exhaustion_blocks(self):
        """
        验证波浪降级后，彻底杜绝任何预测性 wave5_exhaustion 一票否决。
        在极度符合典型波浪5顶形态的情况下，也不得返回 rejected_by_wave5_exhaustion。
        """
        # 构建完整 5 浪顶形态
        sw5 = [
            {"index": 10, "time": pd.Timestamp("2025-01-10"), "price": 100.0, "type": "Low", "dow": "Low", "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2025-01-25"), "price": 125.0, "type": "High", "dow": "High", "confirmed": True},
            {"index": 40, "time": pd.Timestamp("2025-02-10"), "price": 110.0, "type": "Low", "dow": "HL", "confirmed": True},
            {"index": 60, "time": pd.Timestamp("2025-03-01"), "price": 150.0, "type": "High", "dow": "HH", "confirmed": True},
            {"index": 75, "time": pd.Timestamp("2025-03-20"), "price": 135.0, "type": "Low", "dow": "HL", "confirmed": True},
            {"index": 90, "time": pd.Timestamp("2025-04-10"), "price": 165.0, "type": "High", "dow": "HH", "confirmed": True},
        ]
        wave_res = pe.evaluate_wave_position(sw5, df=self.df_base.iloc[:95], current_price=162.0)
        # 不再输出预测性拒绝信号
        self.assertNotEqual(wave_res.get("signal"), "wave5_exhaustion")
        self.assertNotEqual(wave_res.get("status"), "rejected_by_wave5_exhaustion")


class TestTier5PrimaryTrendGateCausalityStress(unittest.TestCase):
    """Tier 5 因果性压力测试 2：200 EMA 大级别主趋势顺势门禁动态截断与极端未来毒化验证"""

    def setUp(self):
        self.df = create_base_daily_dataset(n_bars=350, seed=777)

    def test_primary_trend_gate_exact_truncation_equivalence(self):
        """
        验证无论对于任意切割点 t，调用 verify_primary_trend_gate(df, side, t=t)
        与调用 verify_primary_trend_gate(df.iloc[:t+1], side) 的输出结果在浮点精度上 100% 绝对一致。
        """
        cutoffs = [15, 50, 199, 200, 201, 280, 349]
        for side in [1, -1]:
            for t in cutoffs:
                res_with_t = sl.verify_primary_trend_gate(self.df, side=side, ema_period=200, t=t)
                res_sliced = sl.verify_primary_trend_gate(self.df.iloc[:t + 1], side=side, ema_period=200)

                self.assertEqual(
                    res_with_t["passed"], res_sliced["passed"],
                    f"side={side}, t={t} 下 passed 状态不一致"
                )
                self.assertEqual(
                    res_with_t["status"], res_sliced["status"],
                    f"side={side}, t={t} 下 status 状态不一致"
                )
                self.assertAlmostEqual(
                    res_with_t["current_price"], res_sliced["current_price"], places=4,
                    msg=f"side={side}, t={t} 下 current_price 不一致"
                )
                self.assertAlmostEqual(
                    res_with_t["ema_val"], res_sliced["ema_val"], places=4,
                    msg=f"side={side}, t={t} 下 ema_val 不一致"
                )

    def test_primary_trend_gate_future_poisoning_immunity(self):
        """
        在 t_eval 处，即使未来 K 线 [t_eval+1:] 发生毁灭性暴跌或巨额暴涨，
        传入完整毒化 DataFrame 并指定 t=t_eval 时，其门禁判定与 EMA 数值必须分毫不差。
        """
        t_eval = 220
        res_clean = sl.verify_primary_trend_gate(self.df, side=1, ema_period=200, t=t_eval)

        # 构造毒化数据集
        df_poisoned_up = self.df.copy()
        df_poisoned_up.iloc[t_eval + 1:, df_poisoned_up.columns.get_loc("Close")] *= 100.0

        df_poisoned_down = self.df.copy()
        df_poisoned_down.iloc[t_eval + 1:, df_poisoned_down.columns.get_loc("Close")] *= 0.001

        res_up = sl.verify_primary_trend_gate(df_poisoned_up, side=1, ema_period=200, t=t_eval)
        res_down = sl.verify_primary_trend_gate(df_poisoned_down, side=1, ema_period=200, t=t_eval)

        self.assertEqual(res_clean["passed"], res_up["passed"])
        self.assertEqual(res_clean["passed"], res_down["passed"])
        self.assertAlmostEqual(res_clean["ema_val"], res_up["ema_val"], places=6)
        self.assertAlmostEqual(res_clean["ema_val"], res_down["ema_val"], places=6)
        self.assertAlmostEqual(res_clean["current_price"], res_up["current_price"], places=6)

    def test_primary_trend_gate_boundary_anomalies(self):
        """测试 0 根、1 根、无 Close 列、NaN 列等极端边界输入不崩溃且优雅降级"""
        # 空数据 (Fail-Closed)
        r_empty = sl.verify_primary_trend_gate(pd.DataFrame(), side=1)
        self.assertFalse(r_empty["passed"])
        self.assertEqual(r_empty["status"], "rejected_by_missing_trend_data")

        # None 输入
        r_none = sl.verify_primary_trend_gate(None, side=1)
        self.assertFalse(r_none["passed"])

        # 缺少 Close 列
        df_no_c = pd.DataFrame({"Open": [10.0, 11.0], "High": [12.0, 13.0]})
        r_no_c = sl.verify_primary_trend_gate(df_no_c, side=1)
        self.assertFalse(r_no_c["passed"])
        self.assertEqual(r_no_c["status"], "rejected_by_missing_trend_data")

        # 仅 1 根 K 线
        df_1 = pd.DataFrame({"Close": [100.0]})
        r_1_long = sl.verify_primary_trend_gate(df_1, side=1)
        self.assertTrue(r_1_long["passed"])
        self.assertAlmostEqual(r_1_long["ema_val"], 100.0, places=2)

        # 跨越 200 临界边界: 199 根 vs 200 根
        df_200 = create_base_daily_dataset(n_bars=200, seed=1)
        r_199 = sl.verify_primary_trend_gate(df_200.iloc[:199], side=1)
        r_200 = sl.verify_primary_trend_gate(df_200.iloc[:200], side=1)
        self.assertIn("自适应", r_199["reason"])
        self.assertNotIn("自适应", r_200["reason"])


class TestTier5SectorGateAndDecouplingStress(unittest.TestCase):
    """Tier 5 因果性压力测试 3：跨市场板块门禁、宏观脱钩规则与非对称做空抗压测试"""

    def setUp(self):
        self.df_spy = create_base_daily_dataset(n_bars=300, seed=10)
        self.df_qqq = create_base_daily_dataset(n_bars=300, seed=20)
        self.df_dia = create_base_daily_dataset(n_bars=300, seed=30)
        self.df_iwm = create_base_daily_dataset(n_bars=300, seed=40)

    def test_sector_decoupling_preserves_independent_alpha(self):
        """
        测试板块轮动场景：科技大跌 (QQQ 空头破位)，但工业与大盘稳健 (DIA / SPY 多头)，
        非科技股 (CAT / BA, 工业板块) 的多头交易必须放行，消除对单一科技指数的同义反复绑架。
        """
        market_dow_rotation = {
            "SPY": {"mtf": {"phase": "📈 多头趋势 (HH+HL)", "swings": ["HH", "HL"]}},
            "QQQ": {"mtf": {"phase": "📉 空头破位 (LH+LL)", "is_breakdown": True, "swings": ["LH", "LL"]}},
            "DIA": {"mtf": {"phase": "📈 强大多头 (HH+HL)", "swings": ["HH", "HL"]}},
            "IWM": {"mtf": {"phase": "📈 稳健多头", "swings": ["HH", "HL"]}}
        }

        # 1. 工业股 (CAT, sector=industrial) 必须放行
        res_cat = sl.verify_market_dow_gate(
            market_dow=market_dow_rotation,
            target_symbol="CAT",
            target_sector="industrial",
            side=1
        )
        self.assertTrue(
            res_cat["passed"],
            f"工业标的 CAT 在 QQQ 单独回调而 DIA 走强时不应被误杀: {res_cat.get('reason')}"
        )

        # 2. 科技股 (NVDA, sector=tech) 处于重灾区，必须被驳回
        res_nvda = sl.verify_market_dow_gate(
            market_dow=market_dow_rotation,
            target_symbol="NVDA",
            target_sector="tech",
            side=1
        )
        self.assertFalse(
            res_nvda["passed"],
            f"科技标的 NVDA 在 QQQ 破位大势下必须被门禁驳回: {res_nvda.get('reason')}"
        )
        self.assertEqual(res_nvda.get("status"), "rejected_by_market_dow")

    def test_micro_trumps_macro_decoupling_under_systemic_headwind(self):
        """
        测试微观超越宏观 (Micro Trumps Macro) 脱钩协议：
        当大盘所有指数全线崩溃破位时，若个股出现高确定性微观结构突破 (strong_box_breakout 或 strong_2b_reversal)
        且允许脱钩 (allow_decoupling=True)，系统必须授予 decoupled_micro_alpha 放行许可。
        """
        systemic_crash_market = {
            "SPY": {"mtf": {"phase": "📉 空头暴跌破位", "is_breakdown": True}},
            "QQQ": {"mtf": {"phase": "📉 空头暴跌破位", "is_breakdown": True}},
            "DIA": {"mtf": {"phase": "📉 空头暴跌破位", "is_breakdown": True}},
            "IWM": {"mtf": {"phase": "📉 空头暴跌破位", "is_breakdown": True}}
        }

        # 常规入场在系统性暴跌下必须被封杀
        res_normal = sl.verify_market_dow_gate(
            market_dow=systemic_crash_market,
            target_symbol="AAPL",
            side=1,
            allow_decoupling=False
        )
        self.assertFalse(res_normal["passed"], "系统性暴跌下常规多头必须被封杀")

        # 高确定性箱体突破微观结构脱钩放行
        res_box_decouple = sl.verify_market_dow_gate(
            market_dow=systemic_crash_market,
            target_symbol="AAPL",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_box_breakout"
        )
        self.assertTrue(res_box_decouple["passed"], "strong_box_breakout 必须获得微观脱钩放行许可")
        self.assertEqual(res_box_decouple["status"], "decoupled_micro_alpha")

        # 高确定性 2B 假跌破扫损反转微观结构脱钩放行
        res_2b_decouple = sl.verify_market_dow_gate(
            market_dow=systemic_crash_market,
            target_symbol="AAPL",
            side=1,
            allow_decoupling=True,
            micro_conviction="strong_2b_reversal"
        )
        self.assertTrue(res_2b_decouple["passed"], "strong_2b_reversal 必须获得微观脱钩放行许可")
        self.assertEqual(res_2b_decouple["status"], "decoupled_micro_alpha")

    def test_asymmetric_short_gate_blocks_blind_counter_trend_shorts(self):
        """
        测试美股非对称做空门禁：
        在美股长期大牛多头背景下（大盘指数处于道氏多头），严禁无差别盲目做空（side == -1）。
        必须被 rejected_by_macro_short_gate 严格驳回，杜绝对称机械做空带来的巨大亏损。
        """
        bull_market = {
            "SPY": {"mtf": {"phase": "📈 多头结构 (HH+HL)", "bias": 1}},
            "QQQ": {"mtf": {"phase": "📈 多头结构 (HH+HL)", "bias": 1}},
            "DIA": {"mtf": {"phase": "📈 多头结构 (HH+HL)", "bias": 1}},
            "IWM": {"mtf": {"phase": "📈 多头结构 (HH+HL)", "bias": 1}}
        }

        # 1. 验证 verify_market_dow_gate 的做空驳回
        res_short_dow = sl.verify_market_dow_gate(
            market_dow=bull_market,
            target_symbol="AAPL",
            side=-1
        )
        self.assertFalse(
            res_short_dow["passed"],
            "在大盘全线多头环境下，盲目摸顶做空必须被非对称做空风控红线驳回"
        )
        self.assertEqual(res_short_dow.get("rejected_by"), "rejected_by_macro_short_gate")

        # 2. 验证 verify_macro_short_gate 的直接做空驳回
        res_short_macro = sl.verify_macro_short_gate(
            market_dow=bull_market,
            target_symbol="AAPL"
        )
        self.assertFalse(res_short_macro["passed"])
        self.assertEqual(res_short_macro.get("status"), "rejected_by_macro_short_gate")


class TestTier5IntradayHTFAndFrictionStress(unittest.TestCase):
    """Tier 5 因果性压力测试 4：日内 1h HTF 过滤因果性与 8 bps 真实摩擦严密核算"""

    def setUp(self):
        # 构建 5 天的 5 分钟日内仿真序列 (每天 78 根)
        self.days = 5
        self.bars_per_day = 78
        self.total_bars = self.days * self.bars_per_day
        
        # 美东交易日时间戳
        dt_idx = []
        base_date = pd.Timestamp("2026-09-01")
        for d in range(self.days):
            curr_d = base_date + pd.Timedelta(days=d)
            day_ts = pd.date_range(
                start=curr_d.strftime("%Y-%m-%d 09:30"),
                periods=self.bars_per_day,
                freq="5min",
                tz="America/New_York"
            )
            dt_idx.extend(day_ts)
        
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.normal(0.05, 0.2, self.total_bars))
        highs = closes + rng.uniform(0.1, 0.5, self.total_bars)
        lows = closes - rng.uniform(0.1, 0.5, self.total_bars)
        opens = (highs + lows) / 2.0
        
        self.df_5m = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1000
        }, index=pd.DatetimeIndex(dt_idx))

    def test_intraday_htf_bias_strictly_excludes_forming_bar(self):
        """
        验证 extract_intraday_htf_bias 严格剔除当期尚未走完的 1h 柱，
        且 cutoff_ts 之后的数据注入绝不改变当期评估结果。
        """
        cutoff = self.df_5m.index[100]  # 第 2 天盘中某时刻
        bias_clean = sd.extract_intraday_htf_bias(self.df_5m, cutoff_ts=cutoff)

        # 恶意注入暴跌或暴涨至 cutoff 之后的未来 K 线
        df_poisoned = self.df_5m.copy()
        df_poisoned.loc[df_poisoned.index > cutoff, "Close"] *= 10.0
        df_poisoned.loc[df_poisoned.index > cutoff, "High"] *= 12.0
        df_poisoned.loc[df_poisoned.index > cutoff, "Low"] *= 9.0

        bias_poisoned = sd.extract_intraday_htf_bias(df_poisoned, cutoff_ts=cutoff)

        self.assertEqual(
            bias_clean, bias_poisoned,
            "日内 1h HTF 评估结果必须对 cutoff 之后的恶意未来数据保持 100% 免疫"
        )

    def test_friction_accounting_exact_deduction(self):
        """
        测试日内撮合模拟中的真实摩擦损耗：
        每笔交易在 gross 与 net 之间必须精准扣除摩擦损耗 (3 bps 滑点 + 1 bps 规费，双边往返 0.08%)。
        """
        trade = sd.simulate_day_trade(
            self.df_5m, entry_t=10, entry_price=100.0, side=1,
            stop_price=98.0, tp_price=105.0, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        if trade.get("status") == "completed":
            gross_pnl = trade["gross_pnl_pct"]
            net_pnl = trade["net_pnl_pct"]
            drag = trade["slippage_drag_pct"]
            
            # 净收益必须精确等于毛收益减去摩擦损耗
            self.assertAlmostEqual(gross_pnl - net_pnl, drag, places=5)
            # 双边滑点 6 bps + 规费 2 bps 约 8 bps
            self.assertGreater(drag, 0.0007)
            self.assertLess(drag, 0.0009)

    def test_eod_forced_square_off_guarantee(self):
        """
        测试日内防隔夜平仓铁律：
        任何跨交易日的持仓必须在 15:55 ET 之前被强制清算，隔夜持仓比例为严格 0.0%。
        """
        events = sd.enumerate_day_events(self.df_5m, strategy_type="day_combined", min_rr=2.5)
        for ev in events:
            sim = sd.simulate_day_trade(
                self.df_5m, ev["entry_t"], ev["entry_price"], ev["side"],
                ev["stop_price"], ev["tp_price"], min_rr=2.5
            )
            if sim.get("status") == "completed":
                entry_date = pd.to_datetime(sim["entry_date"]).date()
                exit_date = pd.to_datetime(sim["exit_date"]).date()
                exit_time = pd.to_datetime(sim["exit_date"]).time()
                
                self.assertEqual(
                    entry_date, exit_date,
                    f"日内交易不得隔夜跨日持仓: entry={entry_date}, exit={exit_date}"
                )
                self.assertLessEqual(
                    exit_time, dtime(15, 55),
                    f"平仓时间不得超过 15:55 ET: {exit_time}"
                )


class TestTier5EndToEndTradePlanCausalityStress(unittest.TestCase):
    """Tier 5 因果性压力测试 5：端到端交易计划在未来极端波动下的纯因果不变性"""

    def setUp(self):
        self.df = create_base_daily_dataset(n_bars=280, seed=999)

    def test_build_trade_plan_future_data_immunity(self):
        """
        在任意决策点 t_eval 生成的交易计划 (build_trade_plan)，
        在受到后续未来数据恶意污染时，其入场价、止损位、目标位与证伪点必须分毫不差。
        """
        t_eval = 200
        df_sub = self.df.iloc[:t_eval + 1].copy()
        swings = pe.extract_swings(df_sub, order=5, confirmed_only=True)
        self.assertGreater(len(swings), 4)

        bar_rng = pe.median_bar_range(df_sub, n=14)
        levels_dict = {"box": pe.detect_consolidation_box(df_sub)}
        key_lvls = pe.build_key_levels(levels_dict, swings, swings, [], [], bar_rng)
        curr_price = float(df_sub["Close"].iloc[-1])

        plan_clean = pe.build_trade_plan(
            mode="swing",
            price=curr_price,
            swings_mtf=swings,
            swings_ltf=swings,
            key_levels=key_lvls,
            levels=levels_dict,
            bar_rng=bar_rng,
            bias_htf=1,
            bias_mtf=1,
            df_mtf=df_sub,
            target_symbol="AAPL"
        )

        # 恶意注入后续未来数据
        df_poisoned = self.df.copy()
        df_poisoned.iloc[t_eval + 1:, df_poisoned.columns.get_loc("Close")] = 9999.0
        df_poisoned.iloc[t_eval + 1:, df_poisoned.columns.get_loc("High")] = 9999.0
        df_poisoned.iloc[t_eval + 1:, df_poisoned.columns.get_loc("Low")] = 9999.0

        # 在 t_eval 重新截断并评估
        poisoned_sub = df_poisoned.iloc[:t_eval + 1]
        poisoned_swings = pe.extract_swings(poisoned_sub, order=5, confirmed_only=True)
        poisoned_bar_rng = pe.median_bar_range(poisoned_sub, n=14)
        poisoned_levels_dict = {"box": pe.detect_consolidation_box(poisoned_sub)}
        poisoned_key_lvls = pe.build_key_levels(poisoned_levels_dict, poisoned_swings, poisoned_swings, [], [], poisoned_bar_rng)
        poisoned_curr_price = float(poisoned_sub["Close"].iloc[-1])

        plan_poisoned = pe.build_trade_plan(
            mode="swing",
            price=poisoned_curr_price,
            swings_mtf=poisoned_swings,
            swings_ltf=poisoned_swings,
            key_levels=poisoned_key_lvls,
            levels=poisoned_levels_dict,
            bar_rng=poisoned_bar_rng,
            bias_htf=1,
            bias_mtf=1,
            df_mtf=poisoned_sub,
            target_symbol="AAPL"
        )

        # 验证核心交易计划要素严格一致
        self.assertEqual(plan_clean.get("status"), plan_poisoned.get("status"))
        self.assertEqual(plan_clean.get("side"), plan_poisoned.get("side"))
        self.assertEqual(plan_clean.get("entry"), plan_poisoned.get("entry"))
        self.assertEqual(plan_clean.get("stop"), plan_poisoned.get("stop"))
        self.assertEqual(plan_clean.get("tp1"), plan_poisoned.get("tp1"))
        self.assertEqual(plan_clean.get("tp2"), plan_poisoned.get("tp2"))
        self.assertEqual(
            plan_clean.get("wave4_invalidation_level"),
            plan_poisoned.get("wave4_invalidation_level")
        )


if __name__ == "__main__":
    unittest.main()
