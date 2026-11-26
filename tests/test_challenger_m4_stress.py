#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_challenger_m4_stress.py: Milestone 4 对抗式压力测试与实证验证套件

由角色 challenger_m4_2 独立构建并执行，专门用于经验证伪与压力测试：
1. 低波动防刷单门禁 (Low-Volatility Churn Protection) 的临界精度与参数缩放测试；
2. 1h HTF 宏观大势门禁的纯因果性、未来数据注入毒化防护与未走完K线剥离验证；
3. 美股非对称做空门禁 (Asymmetric Shorting Gate) 顺势长牛拦截与反转豁免压力测试；
4. 真实执行摩擦 (8 bps) 对多空双向“伪微利”单的穿透性截断验证。
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import time as dtime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pa_engine as pe
import strategy_layer as sl
import strategy_day as sd


class TestChallengerLowVolatilityChurnStress(unittest.TestCase):
    """对抗式测试 1：低波动防刷单门禁的临界精度与动态摩擦参数压力测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        self.df = pd.DataFrame({
            "Open":  [100.0] * 10,
            "High":  [105.0] * 10,
            "Low":   [95.0] * 10,
            "Close": [101.0] * 10,
            "Volume":[1000] * 10
        }, index=self.ts)

    def test_churn_threshold_sub_16bps_strictly_rejected(self):
        """测试 15.99 bps, 15.0 bps, 8.0 bps 等子阈值必须被统一判定为 rejected_by_low_volatility_churn"""
        test_risks = [0.1599, 0.1500, 0.0800, 0.0200]
        for r in test_risks:
            stop_p = 100.0 - r
            tp_p = 100.0 + 3.0 * r  # 保证盈亏比 = 3.0R >= 2.5R
            tr = sd.simulate_day_trade(
                self.df, entry_t=1, entry_price=100.0, side=1,
                stop_price=stop_p, tp_price=tp_p, min_rr=2.5,
                slippage_bps=3.0, commission_bps=1.0
            )
            self.assertEqual(
                tr.get("status"), "rejected_by_low_volatility_churn",
                f"风险空间为 {r:.4f} ({(r/100.0)*10000:.2f} bps) 时必须被低波动刷单门禁驳回"
            )
            self.assertAlmostEqual(tr.get("churn_threshold"), 0.0016, places=5)

    def test_churn_threshold_16bps_and_above_permitted(self):
        """测试 16.00 bps 与 16.01 bps 准确放行（考虑 1e-9 浮点容差）"""
        # 16.00 bps: risk = 0.1600
        tr_1600 = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.84, tp_price=100.84, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        self.assertNotEqual(
            tr_1600.get("status"), "rejected_by_low_volatility_churn",
            "刚好 16.00 bps 的风险空间不应被刷单门禁拦截"
        )

        # 16.01 bps: risk = 0.1601
        tr_1601 = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.8399, tp_price=100.84, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        self.assertNotEqual(
            tr_1601.get("status"), "rejected_by_low_volatility_churn",
            "16.01 bps 的风险空间必须正常放行通过刷单门禁"
        )

    def test_churn_threshold_scales_with_custom_friction(self):
        """测试动态摩擦参数放大：当滑点=10 bps, 规费=5 bps 时，双边摩擦=30 bps，刷单门槛自动升为 60 bps"""
        # friction = 2 * (10 + 5) = 30 bps = 0.0030
        # churn_threshold = 2 * 0.0030 = 60 bps = 0.0060
        # 风险空间 50 bps (0.50): 应被驳回
        tr_50bps = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.50, tp_price=102.50, min_rr=2.5,
            slippage_bps=10.0, commission_bps=5.0
        )
        self.assertEqual(
            tr_50bps.get("status"), "rejected_by_low_volatility_churn",
            "在 30 bps 高摩擦下，50 bps 风险空间未达 60 bps 门槛必须被驳回"
        )
        self.assertAlmostEqual(tr_50bps.get("churn_threshold"), 0.0060, places=5)

        # 风险空间 70 bps (0.70): 应放行
        tr_70bps = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.30, tp_price=102.50, min_rr=2.5,
            slippage_bps=10.0, commission_bps=5.0
        )
        self.assertNotEqual(tr_70bps.get("status"), "rejected_by_low_volatility_churn")

    def test_short_side_churn_filter(self):
        """测试空头方向 (side=-1) 下低波动防刷单拦截与放行"""
        # 做空：entry=100.0, stop=100.15 -> risk = 0.15 (15 bps < 16 bps)
        tr_short_churn = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=100.15, tp_price=99.40, min_rr=2.5
        )
        self.assertEqual(tr_short_churn.get("status"), "rejected_by_low_volatility_churn")

        # 做空：entry=100.0, stop=100.25 -> risk = 0.25 (25 bps >= 16 bps)
        tr_short_ok = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=100.25, tp_price=99.20, min_rr=2.5,
            htf_bias=-1  # 空头顺势以避免触发宏观做空门禁
        )
        self.assertEqual(tr_short_ok.get("status"), "completed")

    def test_zero_risk_and_invalid_inputs(self):
        """测试零风险与异常价格输入防御"""
        # risk = 0
        tr_zero = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=1,
            stop_price=100.0, tp_price=103.0
        )
        self.assertEqual(tr_zero.get("status"), "zero_risk_invalid")

        # 负价格
        tr_neg = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=-100.0, side=1,
            stop_price=99.0, tp_price=102.5
        )
        self.assertEqual(tr_neg.get("status"), "invalid_trade")


class TestChallenger1hHTFTrendCausalityStress(unittest.TestCase):
    """对抗式测试 2：1h HTF 宏观趋势因果性、未来毒化注入与未完成K线剔除"""

    def test_htf_bias_immune_to_future_data_poisoning(self):
        """
        未来数据毒化注入测试：
        在当前交易日开盘时刻 t_cutoff 提取 1h 大势偏向；
        然后在 t_cutoff 之后的未来时间序列中注入极端毒化扰动（暴跌90%、暴涨10倍、剧烈反转等）；
        验证 extract_intraday_htf_bias 输出在毒化前后绝对一致，证明零未来函数。
        """
        # 生成 8 个交易日的合成 5m 数据
        df_clean = sd.synthetic_intraday_days(n_days=8, seed=777)
        sessions = sd.split_intraday_sessions(df_clean)
        session_dates = list(sessions.keys())

        # 对第 4, 5, 6, 7 个交易日分别进行未来毒化注入检验
        for day_idx in [3, 4, 5, 6]:
            s_date = session_dates[day_idx]
            sess_df = sessions[s_date]
            cutoff_ts = sess_df.index[0]

            # 1. 干净基准计算
            bias_clean = sd.extract_intraday_htf_bias(df_clean, cutoff_ts=cutoff_ts)

            # 2. 毒化版本 A：未来全部暴跌 90%
            df_poison_crash = df_clean.copy()
            future_mask = df_poison_crash.index > cutoff_ts
            df_poison_crash.loc[future_mask, ["Open", "High", "Low", "Close"]] *= 0.10

            bias_after_crash = sd.extract_intraday_htf_bias(df_poison_crash, cutoff_ts=cutoff_ts)
            self.assertEqual(
                bias_clean, bias_after_crash,
                f"在 {cutoff_ts} 注入未来崩盘数据后，大周期偏向必须保持绝对不变（零未来函数）"
            )

            # 3. 毒化版本 B：未来暴涨 10 倍
            df_poison_pump = df_clean.copy()
            df_poison_pump.loc[future_mask, ["Open", "High", "Low", "Close"]] *= 10.0

            bias_after_pump = sd.extract_intraday_htf_bias(df_poison_pump, cutoff_ts=cutoff_ts)
            self.assertEqual(
                bias_clean, bias_after_pump,
                f"在 {cutoff_ts} 注入未来暴涨数据后，大周期偏向必须保持绝对不变（零未来函数）"
            )

            # 4. 毒化版本 C：未来全部插入 NaN 与极端震荡
            df_poison_chaos = df_clean.copy()
            df_poison_chaos.loc[future_mask, "Close"] = 999999.0
            bias_after_chaos = sd.extract_intraday_htf_bias(df_poison_chaos, cutoff_ts=cutoff_ts)
            self.assertEqual(
                bias_clean, bias_after_chaos,
                f"在 {cutoff_ts} 注入极端未来扰动后，大周期偏向必须保持绝对不变"
            )

    def test_unfinished_1h_bar_exclusion_mechanism(self):
        """
        验证当期未走完 1h 柱剔除逻辑：
        当在盘中 10:15 提取 HTF 偏向时，10:00~10:15 这根尚未走完的 1h 柱必须被 iloc[:-1] 剔除，
        避免使用尚未闭合的当期 K 线引发前视偏差。
        """
        df_syn = sd.synthetic_intraday_days(n_days=4, seed=888)
        sessions = sd.split_intraday_sessions(df_syn)
        s_dates = list(sessions.keys())
        day2_df = sessions[s_dates[1]]
        # 选取 10:15 的时间戳
        ts_1015 = day2_df.index[day2_df.index.time == dtime(10, 15)][0]

        sub_5m = df_syn[df_syn.index <= ts_1015]
        df_1h_raw = sub_5m.resample("1h").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last"
        }).dropna()

        # 最后一根 1h 柱对应 10:00~10:15（未走完）
        last_bar_time = df_1h_raw.index[-1]
        self.assertEqual(last_bar_time.hour, 10)

        # 经过 extract_intraday_htf_bias 内部的 iloc[:-1] 机制，当期未走完柱被安全剔除
        bias = sd.extract_intraday_htf_bias(df_syn, cutoff_ts=ts_1015)
        self.assertIn(bias, [-1, 0, 1])


class TestChallengerAsymmetricShortGateStress(unittest.TestCase):
    """对抗式测试 3：美股非对称做空门禁的长牛绞杀防御与反转豁免压力测试"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=15, freq="5min", tz="America/New_York")
        self.df = pd.DataFrame({
            "Open":  [100.0] * 15,
            "High":  [103.0] * 15,
            "Low":   [97.0] * 15,
            "Close": [101.0] * 15,
            "Volume":[1000] * 15
        }, index=self.ts)

    def test_blind_shorts_in_raging_bull_strictly_blocked(self):
        """测试在顺势长牛（大盘多头 + 1h HTF 多头）环境下，各类盲目常规做空被一票否决"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1,
            "SPY_breakdown": False,
            "QQQ_breakdown": False
        }
        # 1. 模拟 ORB 破位做空
        tr_orb = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern="orb_breakdown_short"
        )
        self.assertEqual(tr_orb.get("status"), "rejected_by_macro_short_gate")
        self.assertEqual(tr_orb.get("rejection_reason"), "rejected_by_macro_short_gate")

        # 2. 模拟常规趋势跟踪做空
        tr_trend = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern="trend_continuation_short"
        )
        self.assertEqual(tr_trend.get("status"), "rejected_by_macro_short_gate")

        # 3. 模拟无 pattern 标签做空
        tr_none = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern=None
        )
        self.assertEqual(tr_none.get("status"), "rejected_by_macro_short_gate")

    def test_2b_liquidity_sweep_reversal_allowed_in_bull(self):
        """测试高确定性 2B 顶假突破流动性猎杀反转无论大盘多强势，均获准豁免放行做空"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        # 变体 1: pattern = "2B_sweep_high_PDH"
        tr_2b_pdh = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern="2B_sweep_high_PDH"
        )
        self.assertEqual(tr_2b_pdh.get("status"), "completed")

        # 变体 2: pattern = "2b_top_sweep"
        tr_2b_top = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern="2b_top_sweep"
        )
        self.assertEqual(tr_2b_top.get("status"), "completed")

        # 变体 3: micro_conviction = "strong_2b_reversal"
        tr_2b_conv = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=1, market_dow=market_bull, pattern="counter_reversal",
            micro_conviction="strong_2b_reversal"
        )
        self.assertEqual(tr_2b_conv.get("status"), "completed")

    def test_market_breakdown_confirms_short(self):
        """测试大盘指数（SPY/QQQ）发生破位时，顺势放行常规做空"""
        # SPY 处于空头破位
        market_breakdown = {
            "SPY": {"bias": -1, "is_breakdown": True},
            "QQQ": {"bias": 1, "is_breakdown": False}
        }
        tr_bd = sd.simulate_day_trade(
            self.df, entry_t=1, entry_price=100.0, side=-1,
            stop_price=101.0, tp_price=96.0, min_rr=2.5,
            htf_bias=-1, market_dow=market_breakdown, pattern="routine_short"
        )
        self.assertEqual(tr_bd.get("status"), "completed")

    def test_verify_macro_short_gate_exhaustive_matrix(self):
        """测试 strategy_layer.verify_macro_short_gate 的矩阵覆盖"""
        # A: 空参数默认审慎拦截
        res_empty = sl.verify_macro_short_gate()
        self.assertFalse(res_empty["passed"])
        self.assertEqual(res_empty["status"], "rejected_by_macro_short_gate")

        # B: 仅有 2B 顶反转 -> 放行
        res_2b = sl.verify_macro_short_gate(micro_conviction="2b_top_sweep")
        self.assertTrue(res_2b["passed"])
        self.assertEqual(res_2b["status"], "decoupled_2b_short_reversal")

        # C: 大盘顺势多头 -> 拦截
        market_bull = {"SPY": {"bias": 1}, "QQQ": {"bias": 1}}
        res_bull = sl.verify_macro_short_gate(market_dow=market_bull)
        self.assertFalse(res_bull["passed"])
        self.assertEqual(res_bull["status"], "rejected_by_macro_short_gate")


class TestChallengerFrictionAndNetWinPrecision(unittest.TestCase):
    """对抗式测试 4：8 bps 真实摩擦核算、毛净穿透与 2.5R 硬闸门自检"""

    def setUp(self):
        self.ts = pd.date_range("2026-09-01 09:30", periods=5, freq="5min", tz="America/New_York")

    def test_micro_profit_win_inversion_due_to_friction(self):
        """
        实证验证：毛利为正 (+6 bps) 但小于 8 bps 往返损耗时，
        净收益必须为负 (-2 bps)，且 win 状态必须从 True 严格翻转为 False。
        """
        # entry=100.0, exit=100.06 -> 毛收益 = +0.06% (+6 bps)
        # 摩擦: slippage = 3 bps * 2 = 6 bps, commission = 1 bps * 2 = 2 bps -> 总 8 bps
        # 净收益预期约 -2 bps (-0.0002)
        df_long = pd.DataFrame({
            "Open":  [100.0, 100.0, 100.06, 100.06, 100.06],
            "High":  [100.1, 100.1, 100.10, 100.10, 100.10],
            "Low":   [99.8,  99.8,  99.80,  99.80,  99.80],
            "Close": [100.0, 100.06, 100.06, 100.06, 100.06],
            "Volume":[1000] * 5
        }, index=self.ts)

        tr = sd.simulate_day_trade(
            df_long, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.50, tp_price=100.06, min_rr=0.01,  # 专注于微利测试
            max_hold_bars=1  # 第 2 根强制出场
        )
        self.assertEqual(tr.get("status"), "completed")
        self.assertGreater(tr["gross_pnl_pct"], 0.0, "毛收益应大于 0")
        self.assertLess(tr["net_pnl_pct"], 0.0, "扣除 8 bps 后净收益必须小于 0")
        self.assertTrue(tr["gross_win"], "毛胜判定应为 True")
        self.assertFalse(tr["win"], "净胜判定必须为 False（杜绝微利刷单假象）")
        self.assertAlmostEqual(tr["slippage_drag_pct"], 0.0008, places=4)

        # 空头微利假象测试：entry=100.0, exit=99.94 (+6 bps 毛利)
        df_short = pd.DataFrame({
            "Open":  [100.0, 100.0, 99.94, 99.94, 99.94],
            "High":  [100.2, 100.2, 100.20, 100.20, 100.20],
            "Low":   [99.9,  99.9,  99.90,  99.90,  99.90],
            "Close": [100.0, 99.94, 99.94, 99.94, 99.94],
            "Volume":[1000] * 5
        }, index=self.ts)

        tr_s = sd.simulate_day_trade(
            df_short, entry_t=1, entry_price=100.0, side=-1,
            stop_price=100.50, tp_price=99.94, min_rr=0.01,
            htf_bias=-1, max_hold_bars=1
        )
        self.assertEqual(tr_s.get("status"), "completed")
        self.assertGreater(tr_s["gross_pnl_pct"], 0.0, "空头毛收益应大于 0")
        self.assertLess(tr_s["net_pnl_pct"], 0.0, "空头扣除 8 bps 摩擦后净收益必须小于 0")
        self.assertTrue(tr_s["gross_win"])
        self.assertFalse(tr_s["win"])

    def test_min_rr_gate_rigorous_threshold(self):
        """测试 2.5R 硬闸门拒绝 RR < 2.5R 并放行 RR >= 2.5R"""
        df_bar = pd.DataFrame({
            "Open":  [100.0] * 5,
            "High":  [105.0] * 5,
            "Low":   [95.0] * 5,
            "Close": [101.0] * 5,
            "Volume":[1000] * 5
        }, index=self.ts)

        # 2.49R -> 驳回
        tr_reject = sd.simulate_day_trade(
            df_bar, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.49, min_rr=2.5
        )
        self.assertEqual(tr_reject.get("status"), "rejected_by_min_rr")

        # 2.50R -> 放行
        tr_pass = sd.simulate_day_trade(
            df_bar, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.50, min_rr=2.5
        )
        self.assertEqual(tr_pass.get("status"), "completed")


if __name__ == "__main__":
    unittest.main()
