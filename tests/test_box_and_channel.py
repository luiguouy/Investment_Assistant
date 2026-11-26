#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
箱体震荡（Consolidation Box）与动态通道（Channel）单元测试集 (Unit Tests for Box & Channel)

测试覆盖：
  1. detect_consolidation_box: 箱体边界、包含率、多点触碰、纯因果识别
  2. 震荡与趋势鉴别: 强趋势下不误判箱体；横盘震荡稳定识别
  3. 2B 流动性猎杀与假突破状态机: IN_BOX, SWEEP_HIGH, SWEEP_LOW, BREAKOUT_UP, BREAKOUT_DOWN
  4. verify_box_gate: 箱体中上部禁多/中下部禁空，假突破顺势反打
  5. detect_price_channel: 线性回归中轴、残差平行带、斜率与决定系数 R²
  6. verify_channel_gate: 通道极值超买禁多、超卖禁空、中轴健康区间放行
  7. 纯因果无未来函数: 历史截断子序列计算严格无未来泄漏
  8. pa_engine 顶层集成: build_key_levels 与 build_trade_plan 门禁驳回自检
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import strategy_layer as sl
import pa_engine as pe


class TestBoxAndChannel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.bdate_range("2026-01-01", periods=120)
        # 基础平稳序列 (95 ~ 105 之间震荡)
        base = 100.0 + 4.0 * np.sin(np.linspace(0, 15, 120))
        noise = np.random.normal(0, 0.5, 120)
        close = base + noise
        high = close + np.random.uniform(0.5, 1.5, 120)
        low = close - np.random.uniform(0.5, 1.5, 120)
        open_ = close + np.random.normal(0, 0.5, 120)
        self.box_df = pd.DataFrame({
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(120, 1000000.0)
        }, index=dates)

    def test_detect_consolidation_box_flat_range(self):
        """测试横盘震荡中箱体边界识别、包含率与触碰计数"""
        box = pe.detect_consolidation_box(self.box_df, min_bars=15, lookback_bars=60)
        self.assertTrue(box.get("is_box", False), "平稳震荡序列应识别为有效箱体")
        self.assertGreaterEqual(box.get("containment_rate", 0.0), 0.70, "包含率应 >= 70%")
        self.assertGreater(box["box_high"], box["box_low"])
        self.assertIn(box["state"], ["IN_BOX", "SWEEP_HIGH", "SWEEP_LOW"])
        self.assertGreaterEqual(box.get("tests_high", 0), 1)
        self.assertGreaterEqual(box.get("tests_low", 0), 1)

    def test_detect_consolidation_box_trending_reject(self):
        """测试单边强趋势不被误判为箱体"""
        dates = pd.bdate_range("2026-01-01", periods=60)
        close = np.linspace(100.0, 200.0, 60)
        trend_df = pd.DataFrame({
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(60, 1000000.0)
        }, index=dates)

        box = pe.detect_consolidation_box(trend_df, min_bars=15, lookback_bars=50)
        self.assertFalse(box.get("is_box", False), "单边强趋势序列不应被判定为箱体")

    def test_box_sweep_2b_fakeout_states(self):
        """测试箱体 2B 假突破流动性猎杀状态机 (SWEEP_HIGH / SWEEP_LOW)"""
        df_sweep_h = self.box_df.iloc[:50].copy()
        box_prev = pe.detect_consolidation_box(df_sweep_h.iloc[:-1])
        if box_prev.get("is_box"):
            b_high = box_prev["box_high"]
            b_low = box_prev["box_low"]
            df_sweep_h.iloc[-1, df_sweep_h.columns.get_loc("Open")] = b_high - 0.2
            df_sweep_h.iloc[-1, df_sweep_h.columns.get_loc("High")] = b_high + 2.0
            df_sweep_h.iloc[-1, df_sweep_h.columns.get_loc("Close")] = b_high - 0.5
            box_res = pe.detect_consolidation_box(df_sweep_h)
            self.assertEqual(box_res.get("state"), "SWEEP_HIGH", "冲高刺破上沿回落应判定为 SWEEP_HIGH")

            df_sweep_l = self.box_df.iloc[:50].copy()
            df_sweep_l.iloc[-1, df_sweep_l.columns.get_loc("Open")] = b_low + 0.2
            df_sweep_l.iloc[-1, df_sweep_l.columns.get_loc("Low")] = b_low - 2.0
            df_sweep_l.iloc[-1, df_sweep_l.columns.get_loc("Close")] = b_low + 0.5
            box_res_l = pe.detect_consolidation_box(df_sweep_l)
            self.assertEqual(box_res_l.get("state"), "SWEEP_LOW", "探底刺破下沿收阳应判定为 SWEEP_LOW")

    def test_verify_box_gate_rules(self):
        """测试箱体风控门禁 (中上部阻力区禁多、中下部支撑区禁空、2B假突破顺势反打)"""
        df = self.box_df.iloc[:60].copy()
        box_info = pe.detect_consolidation_box(df)
        if not box_info.get("is_box"):
            self.skipTest("未构造出基准箱体")

        b_h = box_info["box_high"]
        b_l = box_info["box_low"]
        df.iloc[-1, df.columns.get_loc("Close")] = b_l + (b_h - b_l) * 0.80
        df.iloc[-1, df.columns.get_loc("High")] = b_h - 0.1
        df.iloc[-1, df.columns.get_loc("Low")] = b_l + (b_h - b_l) * 0.75

        # 在箱体中上部做多 -> 必须被硬门禁拦截
        gate_long = sl.verify_box_gate(df, 59, 1)
        self.assertFalse(gate_long["passed"], "箱体高位做多必须被拦截")
        self.assertEqual(gate_long["status"], "rejected_by_box_range")

        # 在箱体中上部做空 -> 允许通过
        gate_short = sl.verify_box_gate(df, 59, -1)
        self.assertTrue(gate_short["passed"], "箱体高位做空应予以放行")

        # 人工设置最后一根收盘价位于箱体 20% 低度
        df.iloc[-1, df.columns.get_loc("Close")] = b_l + (b_h - b_l) * 0.20
        df.iloc[-1, df.columns.get_loc("Low")] = b_l + 0.1
        df.iloc[-1, df.columns.get_loc("High")] = b_l + (b_h - b_l) * 0.25

        # 在箱体中下部做空 -> 必须被硬门禁拦截
        gate_short2 = sl.verify_box_gate(df, 59, -1)
        self.assertFalse(gate_short2["passed"], "箱体低位追空必须被拦截")
        self.assertEqual(gate_short2["status"], "rejected_by_box_range")

        # 在箱体中下部做多 -> 允许通过
        gate_long2 = sl.verify_box_gate(df, 59, 1)
        self.assertTrue(gate_long2["passed"], "箱体低位做多应予以放行")

    def test_detect_price_channel_uptrend(self):
        """测试上升线性回归通道与几何平行通道计算"""
        dates = pd.bdate_range("2026-01-01", periods=60)
        x = np.arange(60)
        close = 100.0 + 0.5 * x + np.sin(x * 0.3) * 1.5
        chan_df = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(60, 1000000.0)
        }, index=dates)

        chan = pe.detect_price_channel(chan_df, lookback_bars=50)
        self.assertEqual(chan.get("channel_type"), "ascending", "规则上升走势应识别为上升通道")
        self.assertGreater(chan["slope"], 0.0, "上升通道斜率必须 > 0")
        self.assertGreater(chan["r_squared"], 0.70, "通道 R² 必须保持高线性拟合度")
        self.assertIsNotNone(chan["upper_rail"])
        self.assertIsNotNone(chan["lower_rail"])
        self.assertIn("channel_pos", chan)

    def test_verify_channel_gate(self):
        """测试通道超买超卖门禁规则（水平通道与上升通道多维度校验）"""
        dates = pd.bdate_range("2026-01-01", periods=60)

        # 1. 水平通道超买超卖
        horiz_close = np.full(60, 100.0) + np.sin(np.arange(60) * 0.5) * 1.0
        horiz_df = pd.DataFrame({
            "Open": horiz_close,
            "High": horiz_close + 1.5,
            "Low": horiz_close - 1.5,
            "Close": horiz_close,
            "Volume": np.full(60, 1000000.0)
        }, index=dates)

        # 最后一根价格拉至水平通道上轨超买区 (pos >= 0.85)
        horiz_df.iloc[-1, horiz_df.columns.get_loc("Close")] = 100.0 + 3.0
        gate_buy_top = sl.verify_channel_gate(horiz_df, 59, 1)
        self.assertFalse(gate_buy_top["passed"], "水平通道极值超买区严禁追多")
        self.assertEqual(gate_buy_top["status"], "rejected_by_channel_overbought")

        # 最后一根价格砸至水平通道下轨超卖区 (pos <= 0.15)
        horiz_df.iloc[-1, horiz_df.columns.get_loc("Close")] = 100.0 - 3.0
        gate_short_bot = sl.verify_channel_gate(horiz_df, 59, -1)
        self.assertFalse(gate_short_bot["passed"], "水平通道极值超卖区严禁杀跌追空")
        self.assertEqual(gate_short_bot["status"], "rejected_by_channel_oversold")

        # 2. 上升通道：下轨支撑区 (0.0 <= pos <= 0.25) 拦截做空；极端脱轨冲顶 (pos > 1.25) 拦截做多；击穿下轨破位放行
        x = np.arange(60)
        osc = 1.0 * np.sin(x * 0.4)
        close = 100.0 + 0.5 * x + osc
        chan_df = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(60, 1000000.0)
        }, index=dates)

        chan_df.iloc[-1, chan_df.columns.get_loc("Close")] = close[-1] + 4.0
        gate_up_top = sl.verify_channel_gate(chan_df, 59, 1)
        self.assertFalse(gate_up_top["passed"], "上升通道极度脱轨冲顶严禁追多")
        self.assertEqual(gate_up_top["status"], "rejected_by_channel_overbought")

        chan_df.iloc[-1, chan_df.columns.get_loc("Close")] = close[-1] - 0.5
        gate_up_short = sl.verify_channel_gate(chan_df, 59, -1)
        self.assertFalse(gate_up_short["passed"], "上升通道下轨动态支撑区严禁逆势杀跌")
        self.assertEqual(gate_up_short["status"], "rejected_by_channel_support")

        chan_df.iloc[-1, chan_df.columns.get_loc("Close")] = close[-1] - 2.5
        gate_breakdown = sl.verify_channel_gate(chan_df, 59, -1)
        self.assertTrue(gate_breakdown["passed"], "上升通道击穿下轨破位杀跌放行")

    def test_causal_no_future_leak(self):
        """测试截断历史重算无未来数据泄漏"""
        df_sub50 = self.box_df.iloc[:50].copy()
        df_sub80 = self.box_df.iloc[:80].copy()

        box_50 = pe.detect_consolidation_box(df_sub50)
        box_from_sub80 = pe.detect_consolidation_box(df_sub80.iloc[:50])

        self.assertEqual(box_50["is_box"], box_from_sub80["is_box"])
        if box_50["is_box"]:
            self.assertAlmostEqual(box_50["box_high"], box_from_sub80["box_high"], places=4)
            self.assertAlmostEqual(box_50["box_low"], box_from_sub80["box_low"], places=4)
            self.assertEqual(box_50["state"], box_from_sub80["state"])

        chan_50 = pe.detect_price_channel(df_sub50)
        chan_from_sub80 = pe.detect_price_channel(df_sub80.iloc[:50])

        self.assertEqual(chan_50["channel_type"], chan_from_sub80["channel_type"])
        self.assertAlmostEqual(chan_50["slope"], chan_from_sub80["slope"], places=4)
        self.assertAlmostEqual(chan_50["r_squared"], chan_from_sub80["r_squared"], places=4)

    def test_pa_engine_build_key_levels_integration(self):
        """测试 pa_engine.build_key_levels 对箱体与通道关键位的聚合输出"""
        box = pe.detect_consolidation_box(self.box_df)
        chan = pe.detect_price_channel(self.box_df)
        levels_dict = {"box": box, "channel": chan}
        swings = pe.extract_swings(self.box_df, order=3, confirmed_only=True)
        bar_rng = pe.median_bar_range(self.box_df)

        merged_levels = pe.build_key_levels(levels_dict, swings, swings, [], [], bar_rng)
        level_sources = []
        for lvl in merged_levels:
            level_sources.append(lvl.get("src", ""))
            level_sources.extend(lvl.get("confluence", []))

        if box.get("is_box"):
            self.assertTrue(any("Box" in s for s in level_sources), "应输出 Box 关键位")
        if chan.get("upper_rail"):
            self.assertTrue(any("Channel" in s for s in level_sources), "应输出 Channel 关键位")

    def test_pa_engine_build_trade_plan_rejection(self):
        """测试 build_trade_plan 在箱体中上部或通道超买时的门禁防御"""
        df = self.box_df.iloc[:60].copy()
        box_info = pe.detect_consolidation_box(df)
        if not box_info.get("is_box"):
            self.skipTest("未构造出基准箱体")

        # 调高末端收盘价至箱体顶部 (85%)
        b_h = box_info["box_high"]
        b_l = box_info["box_low"]
        df.iloc[-1, df.columns.get_loc("Close")] = b_l + (b_h - b_l) * 0.85
        df.iloc[-1, df.columns.get_loc("High")] = b_h - 0.1

        swings = pe.extract_swings(df, order=3, confirmed_only=True)
        bar_rng = pe.median_bar_range(df)
        levels_dict = {"box": pe.detect_consolidation_box(df)}
        key_lvls = pe.build_key_levels(levels_dict, swings, swings, [], [], bar_rng)

        curr_price = float(df["Close"].iloc[-1])
        plan = pe.build_trade_plan(mode="swing", price=curr_price, swings_mtf=swings, swings_ltf=swings,
                                   key_levels=key_lvls, levels=levels_dict, bar_rng=bar_rng,
                                   bias_htf=1, bias_mtf=1, df_mtf=df)

        self.assertEqual(plan.get("rejected_by"), "rejected_by_box_range",
                         "箱体顶部构建多头交易计划应被 rejected_by_box_range 驳回")
        self.assertIsNone(plan.get("entry"), "被驳回的计划 entry 应为 None")


if __name__ == "__main__":
    unittest.main()
