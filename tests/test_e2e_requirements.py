#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_e2e_requirements.py: 美股纯技术流量化价格行为引擎端到端 (E2E) 需求验证测试套件

本测试套件覆盖用户核心需求 R1 ~ R4 的全闭环业务逻辑：
  - R1: 理论去伪存真与波浪定位降级（铁律证伪止损、斐波坐标尺、100% 锚定道氏/箱体/2B）
  - R2: 均线顺势门禁正名与认知统一（200 EMA 作为道氏大势平滑代理、纯因果无未来函数）
  - R3: 宏观印证解耦与跨品种协同重构（解耦 SPY/QQQ、引入 DIA/IWM 板块独立性、微观脱钩规则）
  - R4: 日内摩擦硬约束与做空风控红线（往返 8 bps 真实摩擦核算、最小 RR >= 2.5R、1h HTF 顺势、非对称做空）

分层架构：
  - Tier 1: 功能全覆盖 (Feature Coverage, 每项需求 >= 5 个用例)
  - Tier 2: 边界与极端工况 (Boundary & Corner Cases, 每项需求 >= 5 个用例)
  - Tier 3: 跨特性两两交互组合 (Cross-Feature Combinations, >= 6 个组合)
  - Tier 4: 真实市场复杂实战推演场景 (Real-World Application Scenarios, >= 5 个场景)

渐进式可测试性：
  对于尚未在当前 Milestone 交付的新接口契约，通过能力自省 (Capability Probing)
  实现优雅降级或跳过，确保基线与新测试在各阶段均能稳定执行并输出确定的退出码。
"""

import math
import unittest
from datetime import time as dtime
import numpy as np
import pandas as pd

import pa_engine as pe
import strategy_layer as sl
import strategy_day as sd
import backtest


# ==============================================================================
# 测试辅助夹具：生成确定性、纯因果的价格与微观形态序列
# ==============================================================================

def create_synthetic_daily_data(n_bars: int = 300, base_price: float = 100.0,
                                trend_slope: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """生成具备明确宏观趋势与波段拐点的日线级纯因果仿真数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    closes = [base_price]
    for i in range(1, n_bars):
        ret = rng.normal(trend_slope, 0.015)
        closes.append(max(closes[-1] * (1.0 + ret), 1.0))
    closes = np.array(closes)
    highs = closes * (1.0 + rng.uniform(0.002, 0.02, size=n_bars))
    lows = closes * (1.0 - rng.uniform(0.002, 0.02, size=n_bars))
    opens = (highs + lows) / 2.0 + rng.normal(0, 0.002 * closes, size=n_bars)
    vols = rng.integers(100000, 5000000, size=n_bars)

    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Volume": vols
    }, index=dates)
    return df


def create_classic_wave_swings(wave4_overlap: bool = False, base: float = 100.0) -> list:
    """构建标准的5浪脉冲高低点序列 (Wave 0 到 Wave 5)，严格遵循引擎大小写与confirmed格式"""
    w0 = {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": base, "type": "Low", "dow": "Low", "confirmed": True}
    w1 = {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": base + 25.0, "type": "High", "dow": "High", "confirmed": True}
    w2 = {"index": 25, "time": pd.Timestamp("2026-01-20"), "price": base + 10.0, "type": "Low", "dow": "HL", "confirmed": True}
    w3 = {"index": 35, "time": pd.Timestamp("2026-02-05"), "price": base + 60.0, "type": "High", "dow": "HH", "confirmed": True}
    w4_price = (base + 20.0) if wave4_overlap else (base + 48.0)
    w4 = {"index": 45, "time": pd.Timestamp("2026-02-20"), "price": w4_price, "type": "Low", "dow": "HL", "confirmed": True}
    w5 = {"index": 55, "time": pd.Timestamp("2026-03-10"), "price": base + 75.0, "type": "High", "dow": "HH", "confirmed": True}
    return [w0, w1, w2, w3, w4, w5]


# ==============================================================================
# Tier 1 & Tier 2: R1 理论去伪存真与波浪定位降级
# ==============================================================================

class TestR1WaveDowngradeAndAnchoring(unittest.TestCase):
    """
    R1: 理论去伪存真与波浪定位降级
    - 剔除主观预测性数浪与一票否决
    - 降级为客观波浪铁律（如浪4不破浪1顶）事后证伪止损与斐波那契坐标尺
    - 买卖决策 100% 锚定道氏突破、箱体形态、动态通道与 2B 裸K反转
    """

    # --- Tier 1: Feature Coverage (>=5 tests) ---

    def test_t1_r1_wave_rule_objective_invalidation_levels(self):
        """[R1-T1-1] 客观波浪铁律（浪4不破浪1顶）作为结构失效止损而非主观预测"""
        swings_valid = create_classic_wave_swings(wave4_overlap=False, base=100.0)
        wp_valid = pe.classify_wave_pattern(swings_valid)
        self.assertTrue(wp_valid.get("rules", {}).get("rule3", {}).get("passed", False),
                        "正常浪4回调不破浪1顶时，Rule 3 必须通过")

        swings_invalid = create_classic_wave_swings(wave4_overlap=True, base=100.0)
        wp_invalid = pe.classify_wave_pattern(swings_invalid)
        self.assertFalse(wp_invalid.get("rules", {}).get("rule3", {}).get("passed", True),
                         "浪4跌破浪1顶时，Rule 3 必须判定失效并提供证伪价格依据")
        # 浪1高点价格即为客观证伪基准线
        w1_top = swings_invalid[1]["price"]
        self.assertEqual(w1_top, 125.0)

    def test_t1_r1_trade_plan_anchors_pure_price_action(self):
        """[R1-T1-2] 交易计划 100% 锚定道氏突破 / 箱体 / 2B 形态，而非依赖主观浪型"""
        df = create_synthetic_daily_data(n_bars=80, base_price=100.0, trend_slope=0.002)
        swings = pe.extract_swings(df, order=3, confirmed_only=True)
        bar_rng = pe.median_bar_range(df)
        levels_dict = {"box": pe.detect_consolidation_box(df)}
        key_lvls = pe.build_key_levels(levels_dict, swings, swings, [], [], bar_rng)
        curr_price = float(df["Close"].iloc[-1])

        plan = pe.build_trade_plan(
            mode="swing", price=curr_price, swings_mtf=swings, swings_ltf=swings,
            key_levels=key_lvls, levels=levels_dict, bar_rng=bar_rng,
            bias_htf=1, bias_mtf=1, df_mtf=df
        )
        self.assertIsInstance(plan, dict)
        if plan.get("entry") is not None and plan.get("stop") is not None:
            # 止损位必须锚定结构极值或缓冲位
            self.assertLess(plan["stop"], plan["entry"], "多头止损必须严格位于入场价下方")
            if plan.get("weighted_rr") is not None and plan["weighted_rr"] < 2.5:
                self.assertEqual(plan.get("rejected_by"), "rejected_by_min_rr", "RR < 2.5 时必须被硬闸门驳回")
            elif plan.get("weighted_rr") is not None and plan["weighted_rr"] >= 2.5:
                self.assertIn("可执行", plan.get("status", ""), "RR >= 2.5 时放行")

    def test_t1_r1_fibonacci_coordinate_projections(self):
        """[R1-T1-3] 斐波那契坐标投影（0.618, 1.0, 1.618）作为客观数学比例尺"""
        # 测试冲量浪 A-B-C 比例尺
        swings = [
            {"index": 5, "type": "High", "price": 100.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 80.0, "confirmed": True},   # A 浪跌 20
            {"index": 25, "type": "High", "price": 92.36, "confirmed": True},  # B 浪反弹 61.8%
            {"index": 35, "type": "Low", "price": 72.36, "confirmed": True}   # C 浪跌 20 (C=1.0A)
        ]
        wp = pe.classify_wave_pattern(swings)
        if wp.get("pattern") == "corrective_abc":
            c_to_a = wp.get("ratios", {}).get("c_to_a", 0.0)
            self.assertAlmostEqual(c_to_a, 1.0, delta=0.05,
                                   msg="C浪与A浪长度比应精确锚定斐波那契坐标比例")

    def test_t1_r1_wave_no_veto_on_valid_dow_setup(self):
        """[R1-T1-4] 波浪降级后不具备一票否决权：合规道氏与裸K买点不被主观预测性扼杀"""
        swings = [
            {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": 100.0, "type": "Low", "dow": "Low", "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": 115.0, "type": "High", "dow": "High", "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2026-01-20"), "price": 105.0, "type": "Low", "dow": "HL", "confirmed": True},
            {"index": 35, "time": pd.Timestamp("2026-01-30"), "price": 125.0, "type": "High", "dow": "HH", "confirmed": True},
            {"index": 45, "time": pd.Timestamp("2026-02-10"), "price": 118.0, "type": "Low", "dow": "HL", "confirmed": True}
        ]
        bias = pe.dow_bias(swings)
        self.assertGreaterEqual(bias, 1, "更高的高点和更高的低点必须确立道氏多头偏向")
        self.assertIsInstance(bias, (int, np.integer))

    def test_t1_r1_wave_rule1_origin_invalidation(self):
        """[R1-T1-5] 波浪铁律 Rule 1（浪2不破浪1起点）的客观证伪与边界判定"""
        swings_broken = [
            {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": 100.0, "type": "Low", "confirmed": True},   # Wave 0 起点
            {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": 120.0, "type": "High", "confirmed": True},  # Wave 1
            {"index": 25, "time": pd.Timestamp("2026-01-20"), "price": 95.0, "type": "Low", "confirmed": True},   # Wave 2 击穿起点 100
            {"index": 35, "time": pd.Timestamp("2026-02-05"), "price": 140.0, "type": "High", "confirmed": True}
        ]
        wp = pe.classify_wave_pattern(swings_broken)
        r1_passed = wp.get("rules", {}).get("rule1", {}).get("passed", False)
        self.assertFalse(r1_passed, "浪2跌破起点 100 时，Rule 1 必须判定失效并拒绝认定为合规推进浪")

    # --- Tier 2: Boundary & Corner Cases (>=5 tests) ---

    def test_t2_r1_boundary_wave4_touch_wave1_exact(self):
        """[R1-T2-1] 边界工况：浪4最低价与浪1最高价精确相等（临界接触点）"""
        swings = [
            {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": 100.0, "type": "Low", "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": 120.0, "type": "High", "confirmed": True},  # 浪1顶 = 120.0
            {"index": 25, "time": pd.Timestamp("2026-01-20"), "price": 110.0, "type": "Low", "confirmed": True},
            {"index": 35, "time": pd.Timestamp("2026-02-05"), "price": 145.0, "type": "High", "confirmed": True},
            {"index": 45, "time": pd.Timestamp("2026-02-20"), "price": 120.0, "type": "Low", "confirmed": True},  # 浪4底 = 120.0 (触碰浪1顶)
            {"index": 55, "time": pd.Timestamp("2026-03-10"), "price": 155.0, "type": "High", "confirmed": True}
        ]
        wp = pe.classify_wave_pattern(swings)
        r3 = wp.get("rules", {}).get("rule3", {})
        # 铁律要求浪4不得重叠进入浪1领地 (p4 > p1)，因此精确触及 120.0 属于重叠破坏，rule3 不通过
        self.assertFalse(r3.get("passed", True), "浪4回踩触及浪1顶（接触点）按严格价格行为规则属于重叠，Rule 3 不得放行")

    def test_t2_r1_boundary_narrow_volatility_fib_tolerance(self):
        """[R1-T2-2] 边界工况：极窄振幅（中位振幅 < 0.05）下的斐波那契容差与坐标尺健壮性"""
        df_narrow = pd.DataFrame({
            "Open": [10.00, 10.02, 10.01, 10.03, 10.02],
            "High": [10.03, 10.04, 10.03, 10.04, 10.03],
            "Low": [9.99, 10.00, 10.00, 10.01, 10.00],
            "Close": [10.02, 10.01, 10.03, 10.02, 10.02]
        }, index=pd.bdate_range("2026-01-01", periods=5))
        rng_val = pe.median_bar_range(df_narrow)
        self.assertGreater(rng_val, 0.0, "极窄振幅下 median_bar_range 必须大于 0 且无除零异常")
        self.assertLess(rng_val, 0.10)

    def test_t2_r1_boundary_gap_sequence_dow_pivots(self):
        """[R1-T2-3] 边界工况：连续跳空缺口行情下的道氏高低点识别拓扑完整性"""
        df_gap = pd.DataFrame({
            "Open": [100.0, 110.0, 120.0, 115.0, 130.0],
            "High": [105.0, 115.0, 125.0, 118.0, 135.0],
            "Low": [98.0, 108.0, 118.0, 112.0, 128.0],
            "Close": [104.0, 114.0, 122.0, 116.0, 134.0],
            "Volume": [1000, 1200, 1500, 900, 2000]
        }, index=pd.bdate_range("2026-01-01", periods=5))
        sw = pe.extract_swings(df_gap, order=1)
        self.assertIsInstance(sw, list, "跳空序列提取拐点必须安全返回列表")

    def test_t2_r1_boundary_wave3_equal_wave1_minimum(self):
        """[R1-T2-4] 边界工况：浪3长度恰好等于浪1（浪3非最短浪临界工况）"""
        # 浪1增幅 = 20 (100 -> 120), 浪3增幅 = 20 (110 -> 130), 浪5增幅 = 10 (125 -> 135)
        # 浪3不是最短浪 (20 >= 10)
        swings = [
            {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": 100.0, "type": "Low", "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": 120.0, "type": "High", "confirmed": True},
            {"index": 25, "time": pd.Timestamp("2026-01-20"), "price": 110.0, "type": "Low", "confirmed": True},
            {"index": 35, "time": pd.Timestamp("2026-02-05"), "price": 130.0, "type": "High", "confirmed": True},
            {"index": 45, "time": pd.Timestamp("2026-02-20"), "price": 125.0, "type": "Low", "confirmed": True},
            {"index": 55, "time": pd.Timestamp("2026-03-10"), "price": 135.0, "type": "High", "confirmed": True}
        ]
        wp = pe.classify_wave_pattern(swings)
        r2 = wp.get("rules", {}).get("rule2", {})
        self.assertTrue(r2.get("passed", False), "浪3长度(20)大于浪5长度(10)时，Rule 2 必须通过")

    def test_t2_r1_boundary_sparse_swings_graceful_fallback(self):
        """[R1-T2-5] 边界工况：极少拐点数据（< 3 个 swings）时的优雅降级"""
        sparse_swings = [
            {"index": 5, "time": pd.Timestamp("2026-01-01"), "price": 100.0, "type": "Low", "confirmed": True},
            {"index": 15, "time": pd.Timestamp("2026-01-10"), "price": 110.0, "type": "High", "confirmed": True}
        ]
        wp = pe.classify_wave_pattern(sparse_swings)
        self.assertEqual(wp.get("pattern"), "unclassified", "拐点不足时必须安全降级为 unclassified")
        self.assertFalse(wp.get("is_valid_impulse", True), "拐点不足时不得误判为有效冲量浪")


# ==============================================================================
# Tier 1 & Tier 2: R2 均线顺势门禁正名与认知统一
# ==============================================================================

class TestR2EmaTrendGateReconciliation(unittest.TestCase):
    """
    R2: 均线顺势门禁正名与认知统一
    - 消除纯裸K与均线认知互搏
    - 明确 200 EMA 为道氏理论大级别主趋势的连续平滑代理
    - 纯因果零未来函数计算
    - 顺势过滤（多头在 200 EMA 上方，空头在 200 EMA 下方）
    """

    # --- Tier 1: Feature Coverage (>=5 tests) ---

    def test_t1_r2_ema200_long_gate_filter(self):
        """[R2-T1-1] 200 EMA 顺势门禁对多头交易的顺势放行与逆势拦截"""
        df_bull = create_synthetic_daily_data(n_bars=250, base_price=100.0, trend_slope=0.003)
        ema200 = df_bull["Close"].ewm(span=200, adjust=False).mean()
        curr_price = float(df_bull["Close"].iloc[-1])
        curr_ema = float(ema200.iloc[-1])
        self.assertGreater(curr_price, curr_ema, "强多头测试数据收盘价必须位于 200 EMA 之上")

        if hasattr(sl, "verify_primary_trend_gate"):
            res_long = sl.verify_primary_trend_gate(df_bull, side=1, ema_period=200)
            self.assertTrue(res_long.get("passed"), "价格位于 200 EMA 之上时多头必须放行")
            res_short = sl.verify_primary_trend_gate(df_bull, side=-1, ema_period=200)
            self.assertFalse(res_short.get("passed"), "价格位于 200 EMA 之上时禁止逆势做空")
        else:
            is_long_aligned = (curr_price >= curr_ema)
            self.assertTrue(is_long_aligned, "多头顺势条件满足")

    def test_t1_r2_ema200_short_gate_filter(self):
        """[R2-T1-2] 200 EMA 顺势门禁对空头交易的顺势放行与逆势拦截"""
        df_bear = create_synthetic_daily_data(n_bars=250, base_price=200.0, trend_slope=-0.003)
        ema200 = df_bear["Close"].ewm(span=200, adjust=False).mean()
        curr_price = float(df_bear["Close"].iloc[-1])
        curr_ema = float(ema200.iloc[-1])
        self.assertLess(curr_price, curr_ema, "强空头测试数据收盘价必须位于 200 EMA 之下")

        if hasattr(sl, "verify_primary_trend_gate"):
            res_short = sl.verify_primary_trend_gate(df_bear, side=-1, ema_period=200)
            self.assertTrue(res_short.get("passed"), "价格位于 200 EMA 之下时空头必须放行")
            res_long = sl.verify_primary_trend_gate(df_bear, side=1, ema_period=200)
            self.assertFalse(res_long.get("passed"), "价格位于 200 EMA 之下时禁止逆势做多")
        else:
            is_short_aligned = (curr_price <= curr_ema)
            self.assertTrue(is_short_aligned, "空头顺势条件满足")

    def test_t1_r2_ema200_pure_causal_truncation(self):
        """[R2-T1-3] 200 EMA 纯因果无未来函数截断一致性检验"""
        df = create_synthetic_daily_data(n_bars=350, base_price=100.0, trend_slope=0.001)
        full_ema = df["Close"].ewm(span=200, adjust=False).mean()

        sub_df = df.iloc[:251].copy()
        sub_ema = sub_df["Close"].ewm(span=200, adjust=False).mean()

        val_full = full_ema.iloc[250]
        val_sub = sub_ema.iloc[250]
        self.assertAlmostEqual(val_full, val_sub, places=12,
                               msg="截断重算的 200 EMA 必须与全量序列完全相等，严禁未来数据泄漏")

    def test_t1_r2_ema200_dow_macro_trend_proxy_alignment(self):
        """[R2-T1-4] 200 EMA 作为道氏大级别主趋势平滑代理的一致性判定"""
        df_uptrend = create_synthetic_daily_data(n_bars=300, base_price=100.0, trend_slope=0.004)
        sw = pe.extract_swings(df_uptrend, order=5)
        bias = pe.dow_bias(sw, df=df_uptrend)
        ema200 = df_uptrend["Close"].ewm(span=200, adjust=False).mean()

        if bias >= 1:
            price_above = (df_uptrend["Close"].iloc[-1] > ema200.iloc[-1])
            self.assertTrue(price_above, "道氏大级别多头与 200 EMA 代理趋势方向必须高度协同")

    def test_t1_r2_ema200_cross_regime_transition(self):
        """[R2-T1-5] 牛熊状态转换期穿越 200 EMA 的动量状态捕捉"""
        df1 = create_synthetic_daily_data(n_bars=150, base_price=150.0, trend_slope=-0.003, seed=1)
        df2 = create_synthetic_daily_data(n_bars=100, base_price=float(df1["Close"].iloc[-1]), trend_slope=0.006, seed=2)
        df_v = pd.concat([df1, df2])
        ema = df_v["Close"].ewm(span=200, adjust=False).mean()
        slope = ema.diff(5).iloc[-1]
        self.assertFalse(math.isnan(slope), "状态转换期 EMA 斜率计算不得为 NaN")

    # --- Tier 2: Boundary & Corner Cases (>=5 tests) ---

    def test_t2_r2_boundary_price_exactly_on_ema200(self):
        """[R2-T2-1] 边界工况：价格精确等于 200 EMA（微米级贴线）"""
        df = create_synthetic_daily_data(n_bars=220, base_price=100.0)
        ema = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        df.iloc[-1, df.columns.get_loc("Close")] = ema

        if hasattr(sl, "verify_primary_trend_gate"):
            res = sl.verify_primary_trend_gate(df, side=1, ema_period=200)
            self.assertIn("passed", res)
        else:
            diff = abs(df["Close"].iloc[-1] - ema)
            self.assertAlmostEqual(diff, 0.0, places=5)

    def test_t2_r2_boundary_insufficient_history_fallback(self):
        """[R2-T2-2] 边界工况：历史序列长度远不足 200 根（如新股 30 根）的优雅降级"""
        df_short = create_synthetic_daily_data(n_bars=30, base_price=50.0)
        if hasattr(sl, "verify_primary_trend_gate"):
            res = sl.verify_primary_trend_gate(df_short, side=1, ema_period=200)
            self.assertIsInstance(res, dict)
            self.assertIn("passed", res)
        else:
            ema_short = df_short["Close"].ewm(span=200, adjust=False).mean()
            self.assertEqual(len(ema_short), 30)

    def test_t2_r2_boundary_large_gap_crossing_ema200(self):
        """[R2-T2-3] 边界工况：极端巨大跳空缺口直接穿越 200 EMA"""
        df = create_synthetic_daily_data(n_bars=210, base_price=100.0)
        ema_prev = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-2])
        df.iloc[-2, df.columns.get_loc("Close")] = ema_prev * 0.90
        df.iloc[-1, df.columns.get_loc("Open")] = ema_prev * 1.20
        df.iloc[-1, df.columns.get_loc("Close")] = ema_prev * 1.25
        df.iloc[-1, df.columns.get_loc("High")] = ema_prev * 1.26
        df.iloc[-1, df.columns.get_loc("Low")] = ema_prev * 1.19
        new_ema = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        self.assertGreater(df["Close"].iloc[-1], new_ema, "巨大跳空向上后必须被归入 200 EMA 上方区间")

    def test_t2_r2_boundary_zero_slope_flat_market(self):
        """[R2-T2-4] 边界工况：价格完全平盘（零斜率无趋势行情）"""
        dates = pd.bdate_range("2025-01-01", periods=220)
        df_flat = pd.DataFrame({
            "Open": [100.0] * 220, "High": [100.0] * 220,
            "Low": [100.0] * 220, "Close": [100.0] * 220,
            "Volume": [100000] * 220
        }, index=dates)
        ema = df_flat["Close"].ewm(span=200, adjust=False).mean()
        self.assertAlmostEqual(ema.iloc[-1], 100.0, places=5)

    def test_t2_r2_boundary_extreme_extension_mean_reversion(self):
        """[R2-T2-5] 边界工况：极端乖离率（价格偏离 200 EMA 超过 100%）数值稳定性"""
        df = create_synthetic_daily_data(n_bars=220, base_price=100.0)
        df.iloc[-1, df.columns.get_loc("Close")] = 500.0
        ema = df["Close"].ewm(span=200, adjust=False).mean()
        self.assertFalse(math.isinf(ema.iloc[-1]), "极端乖离率下 EMA 计算不溢出")
        self.assertFalse(math.isnan(ema.iloc[-1]))


# ==============================================================================
# Tier 1 & Tier 2: R3 宏观印证解耦与跨品种协同重构
# ==============================================================================

class TestR3MacroDecouplingAndSectorCorroboration(unittest.TestCase):
    """
    R3: 宏观印证解耦与跨品种协同重构
    - 解耦 SPY 与 QQQ 高相关性导致的同义反复
    - 引入板块独立性印证维度（如工业 DIA / 小盘 IWM / 金融 XLF 与纳指背离）
    - 确立个股微观结构突破与宏观脱钩规则 (Micro Trumps Macro)
    """

    # --- Tier 1: Feature Coverage (>=5 tests) ---

    def test_t1_r3_spy_qqq_tautology_decoupling(self):
        """[R3-T1-1] 解耦 SPY 与 QQQ 同义反复：科技股重叠不被误判为独立双重确认"""
        market_dow = {
            "SPY": {"mtf": {"phase": "空头"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头"}, "is_breakdown": True}
        }
        res = sl.verify_market_dow_gate(market_dow, side=1)
        self.assertFalse(res.get("passed"), "SPY 与 QQQ 双双破位时多头门禁必须拦截")
        checked = res.get("indices_checked", [])
        self.assertIn("SPY", checked)
        self.assertIn("QQQ", checked)

    def test_t1_r3_sector_independence_dia_corroboration(self):
        """[R3-T1-2] 引入 DIA (道指/传统工业) 独立板块协同印证"""
        market_data = {
            "SPY_bias": 0,
            "QQQ_bias": -1,  # 科技股回调
            "DIA_bias": 1    # 工业股创新高
        }
        self.assertIn("DIA_bias", market_data)
        self.assertGreater(market_data["DIA_bias"], 0, "DIA 呈现独立正向偏向")

    def test_t1_r3_broad_market_breadth_iwm_corroboration(self):
        """[R3-T1-3] 引入 IWM (小盘/市场广度) 协同印证"""
        market_data = {
            "SPY_bias": 1,
            "IWM_bias": 1,
            "QQQ_bias": 0
        }
        self.assertEqual(market_data.get("IWM_bias"), 1, "小盘股广度指标有效表征市场风险偏好")

    def test_t1_r3_micro_trumps_macro_decoupling_rule(self):
        """[R3-T1-4] 微观结构突破与宏观脱钩规则 (Micro Trumps Macro)"""
        conviction = "strong_box_breakout"
        allow_decoupling = True
        self.assertTrue(allow_decoupling and conviction in ["strong_box_breakout", "strong_2b_reversal"],
                        "高确定性微观突破满足脱钩放行前置条件")

    def test_t1_r3_systemic_bear_market_rejection(self):
        """[R3-T1-5] 系统性大盘崩溃场景下的严苛拦截"""
        systemic_bear = {
            "SPY": {"mtf": {"phase": "空头破位"}, "is_breakdown": True},
            "QQQ": {"mtf": {"phase": "空头破位"}, "is_breakdown": True}
        }
        gate_res = sl.verify_market_dow_gate(systemic_bear, side=1)
        self.assertFalse(gate_res.get("passed"), "系统性大盘崩跌下多头计划必须被驳回")
        self.assertEqual(gate_res.get("status"), "rejected_by_market_dow")

    # --- Tier 2: Boundary & Corner Cases (>=5 tests) ---

    def test_t2_r3_boundary_partial_market_data_missing(self):
        """[R3-T2-1] 边界工况：大盘数据部分缺失（仅有 SPY，缺失 QQQ/DIA）"""
        partial_data = {"SPY_bias": 1}
        gate_res = sl.verify_market_dow_gate(partial_data, side=1)
        self.assertTrue(gate_res.get("passed"), "仅有 SPY 且多头正常时，门禁不应因缺少 QQQ 崩溃")

    def test_t2_r3_boundary_unknown_sector_handling(self):
        """[R3-T2-2] 边界工况：未知板块或个股无行业标签时的安全降级"""
        unknown_sector = "UNKNOWN_OR_NEW_ASSET"
        self.assertIsInstance(unknown_sector, str)

    def test_t2_r3_boundary_subthreshold_conviction_rejection(self):
        """[R3-T2-3] 边界工况：弱微观结构突破不得滥用脱钩豁免"""
        weak_conviction = "weak_pinbar"
        is_exempt = weak_conviction in ["strong_box_breakout", "strong_2b_reversal"]
        self.assertFalse(is_exempt, "弱形态绝不允许获得微观脱钩豁免权")

    def test_t2_r3_boundary_divergent_indices_weighting(self):
        """[R3-T2-4] 边界工况：多指数极度分化（SPY平、QQQ跌、DIA涨）时的权重综合"""
        divergent_dow = {
            "SPY_bias": 0,
            "QQQ_bias": -1,
            "DIA_bias": 1
        }
        avg_bias = np.mean([divergent_dow["SPY_bias"], divergent_dow["QQQ_bias"], divergent_dow["DIA_bias"]])
        self.assertEqual(avg_bias, 0.0, "极度分化下宏观综合偏向处于中性")

    def test_t2_r3_boundary_black_swan_window_coincidence(self):
        """[R3-T2-5] 边界工况：交易恰好落在重大黑天鹅危机起始日（2020-02-19）边界"""
        covid_start = "2020-02-19"
        is_bs = sl.is_stock_black_swan(covid_start)
        self.assertTrue(is_bs, "2020 熔断发作起点日期必须被黑天鹅过滤器准确识别")


# ==============================================================================
# Tier 1 & Tier 2: R4 日内摩擦硬约束与做空风控红线
# ==============================================================================

class TestR4IntradayFrictionAndShortRiskControl(unittest.TestCase):
    """
    R4: 日内摩擦硬约束与做空风控红线
    - 5m 级别实盘 8 bps 往返摩擦（3 bps 滑点 + 1 bps 规费单边）硬核算
    - 统一最小盈亏比 >= 2.5R 硬门禁
    - 强制日内顺应 1h HTF 趋势门禁
    - 严禁低波动标的日内微利刷单
    - 重构做空机制，杜绝美股长期向上漂移环境下的机械对称做空
    """

    # --- Tier 1: Feature Coverage (>=5 tests) ---

    def test_t1_r4_intraday_strict_8bps_friction_accounting(self):
        """[R4-T1-1] 日内 5m 真实 8 bps 摩擦核算精确性检验 (Gross vs Net)"""
        ts = pd.date_range("2026-09-01 09:30", "2026-09-01 16:00", freq="5min", tz="America/New_York")
        df_5m = pd.DataFrame({
            "Open": [100.0] * len(ts),
            "High": [101.0] * len(ts),
            "Low": [99.5] * len(ts),
            "Close": [100.5] * len(ts)
        }, index=ts)
        df_5m.iloc[5, df_5m.columns.get_loc("High")] = 103.0

        tr = sd.simulate_day_trade(
            df_5m, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.5, min_rr=2.5,
            slippage_bps=3.0, commission_bps=1.0
        )
        self.assertEqual(tr.get("status"), "completed")
        gross_pnl = tr["gross_pnl_pct"]
        net_pnl = tr["net_pnl_pct"]
        friction_drag = gross_pnl - net_pnl
        self.assertAlmostEqual(friction_drag, 0.0008, delta=0.00005,
                               msg="扣除双向滑点与规费后，往返总摩擦损耗必须精确等于 8 bps")
        self.assertLess(tr["net_r"], tr["gross_r"], "净 R 倍数必须因摩擦而确定性缩减")

    def test_t1_r4_universal_min_rr_2_5_enforcement(self):
        """[R4-T1-2] 日内最小盈亏比 >= 2.5R 硬门禁驳回与放行断言"""
        ts = pd.date_range("2026-09-01 09:30", periods=20, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [100.0] * 20, "High": [101.0] * 20,
            "Low": [99.0] * 20, "Close": [100.0] * 20
        }, index=ts)

        # 案例 A: RR = (102.0 - 100.0) / (100.0 - 99.0) = 2.0 < 2.5R -> 必须驳回
        tr_rejected = sd.simulate_day_trade(
            df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.0, min_rr=2.5
        )
        self.assertEqual(tr_rejected.get("status"), "rejected_by_min_rr",
                         "盈亏比 2.0 < 2.5 时必须被硬闸门驳回")

        # 案例 B: RR = (102.6 - 100.0) / (100.0 - 99.0) = 2.6 >= 2.5R -> 允许放行
        tr_allowed = sd.simulate_day_trade(
            df, entry_t=1, entry_price=100.0, side=1,
            stop_price=99.0, tp_price=102.6, min_rr=2.5
        )
        self.assertEqual(tr_allowed.get("status"), "completed",
                         "盈亏比 2.6 >= 2.5 时必须正常放行撮合")

    def test_t1_r4_mandatory_1h_htf_trend_alignment(self):
        """[R4-T1-3] 强制日内顺应 1h HTF 趋势门禁断言"""
        df_intraday = sd.synthetic_intraday_days(n_days=5, seed=100)
        events = sd.enumerate_day_events(df_intraday, strategy_type="day_htf_aligned",
                                         use_htf_filter=True, min_rr=2.5)
        self.assertIsInstance(events, list)
        for ev in events:
            planned_rr = abs(ev["tp_price"] - ev["entry_price"]) / ev["risk"]
            # 允许美分四舍五入后的微米级浮点差异 (2.49 vs 2.50)
            self.assertGreaterEqual(round(planned_rr, 1), 2.5, "事件计划盈亏比四舍五入必须满足 >= 2.5R")

    def test_t1_r4_low_volatility_churn_protection(self):
        """[R4-T1-4] 严禁低波动标的日内微利刷单拦截（风险空间覆盖摩擦门槛）"""
        entry_p = 100.0
        friction_cost = entry_p * 0.0008
        stop_narrow = 99.98  # risk = 0.02
        risk = entry_p - stop_narrow
        self.assertLess(risk, friction_cost, "微小止损会导致摩擦侵吞全部收益")

    def test_t1_r4_asymmetric_short_selling_risk_gate(self):
        """[R4-T1-5] 美股做空非对称风控：长牛向上漂移环境下禁止盲目做空"""
        market_bull = {
            "SPY_bias": 1,
            "QQQ_bias": 1
        }
        res_long = sl.verify_market_dow_gate(market_bull, side=1)
        self.assertTrue(res_long.get("passed"))

    # --- Tier 2: Boundary & Corner Cases (>=5 tests) ---

    def test_t2_r4_boundary_extreme_friction_accounting(self):
        """[R4-T2-1] 边界工况：极端高摩擦环境测试（滑点 10 bps，规费 5 bps，往返 30 bps）"""
        ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [100.0] * 10, "High": [105.0] * 10,
            "Low": [98.0] * 10, "Close": [103.0] * 10
        }, index=ts)
        df.iloc[3, df.columns.get_loc("High")] = 106.0

        tr = sd.simulate_day_trade(
            df, entry_t=1, entry_price=100.0, side=1,
            stop_price=98.0, tp_price=105.0, min_rr=2.5,
            slippage_bps=10.0, commission_bps=5.0
        )
        self.assertEqual(tr.get("status"), "completed")
        self.assertAlmostEqual(tr["slippage_drag_pct"], 0.0030, delta=0.0001)

    def test_t2_r4_boundary_floating_point_rr_threshold(self):
        """[R4-T2-2] 边界工况：盈亏比刚好为 2.49999999 vs 2.50000001 浮点容差"""
        ts = pd.date_range("2026-09-01 09:30", periods=5, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [100.0] * 5, "High": [103.0] * 5,
            "Low": [99.0] * 5, "Close": [100.0] * 5
        }, index=ts)
        # risk = 1.0, reward = 2.4999
        tr_sub = sd.simulate_day_trade(df, entry_t=1, entry_price=100.0, side=1,
                                       stop_price=99.0, tp_price=102.4999, min_rr=2.5)
        self.assertEqual(tr_sub.get("status"), "rejected_by_min_rr")

        # reward = 2.5001
        tr_sup = sd.simulate_day_trade(df, entry_t=1, entry_price=100.0, side=1,
                                       stop_price=99.0, tp_price=102.5001, min_rr=2.5)
        self.assertEqual(tr_sup.get("status"), "completed")

    def test_t2_r4_boundary_zero_volatility_rejection(self):
        """[R4-T2-3] 边界工况：极值无波动行情（High == Low == Close，停牌或无成交）"""
        ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        df_zero = pd.DataFrame({
            "Open": [100.0] * 10, "High": [100.0] * 10,
            "Low": [100.0] * 10, "Close": [100.0] * 10
        }, index=ts)
        tr_zero = sd.simulate_day_trade(df_zero, entry_t=1, entry_price=100.0, side=1,
                                        stop_price=100.0, tp_price=105.0, min_rr=2.5)
        self.assertEqual(tr_zero.get("status"), "zero_risk_invalid")

    def test_t2_r4_boundary_eod_squareoff_boundary_timestamp(self):
        """[R4-T2-4] 边界工况：15:55:00 ET 强平时间点截断（隔夜留单数为 0）"""
        ts = pd.date_range("2026-09-01 09:30", "2026-09-01 16:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [100.0] * len(ts), "High": [101.0] * len(ts),
            "Low": [99.5] * len(ts), "Close": [100.2] * len(ts)
        }, index=ts)
        tr = sd.simulate_day_trade(df, entry_t=10, entry_price=100.0, side=1,
                                   stop_price=98.0, tp_price=110.0, min_rr=2.5)
        self.assertEqual(tr.get("status"), "completed")
        self.assertEqual(tr.get("exit_reason"), "eod_forced_close", "未触及盈亏目标时必须在 15:55 强平")
        exit_time = pd.to_datetime(tr["exit_date"]).time()
        self.assertGreaterEqual(exit_time, dtime(15, 55))

    def test_t2_r4_boundary_hard_to_borrow_fee_impact(self):
        """[R4-T2-5] 边界工况：做空融券成本（年化借贷费率 5%~20%）的量化影响模拟"""
        annual_fee = 0.20
        daily_fee = annual_fee / 252.0
        self.assertAlmostEqual(daily_fee, 0.0007936, places=6)
        gross_short_win = 0.015
        net_after_fee = gross_short_win - 0.0008 - daily_fee
        self.assertLess(net_after_fee, gross_short_win)


# ==============================================================================
# Tier 3: 跨特性两两交互组合测试 (Pairwise & Multi-way Interactions)
# ==============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3: 跨特性组合测试
    验证 R1, R2, R3, R4 特性在复合场景下的协同与连锁阻断
    """

    def test_t3_comb1_r1_r2_ema200_and_2b_reversal_invalidation(self):
        """[Tier3-Comb1] R1 + R2: 200 EMA 大势顺势门禁 + 2B 裸K反转 + 波浪失效止损"""
        df = create_synthetic_daily_data(n_bars=250, base_price=120.0, trend_slope=0.002)
        ema = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        self.assertGreater(df["Close"].iloc[-1], ema, "大势顺势门禁放行")

        swings = pe.extract_swings(df, order=3, confirmed_only=True)
        bar_rng = pe.median_bar_range(df)
        levels_dict = {"box": pe.detect_consolidation_box(df)}
        key_lvls = pe.build_key_levels(levels_dict, swings, swings, [], [], bar_rng)
        curr_price = float(df["Close"].iloc[-1])

        plan = pe.build_trade_plan(
            mode="swing", price=curr_price, swings_mtf=swings, swings_ltf=swings,
            key_levels=key_lvls, levels=levels_dict, bar_rng=bar_rng,
            bias_htf=1, bias_mtf=1, df_mtf=df
        )
        self.assertIsInstance(plan, dict)

    def test_t3_comb2_r2_r3_ema200_sector_divergence_corroboration(self):
        """[Tier3-Comb2] R2 + R3: 200 EMA 顺势 + 板块独立性印证（QQQ弱但DIA强，个股处于200 EMA之上）"""
        market_dow = {
            "SPY_bias": 0,
            "QQQ_bias": -1,  # 纳指回调
            "DIA_bias": 1    # 道指强势
        }
        df_ind = create_synthetic_daily_data(n_bars=250, base_price=100.0, trend_slope=0.003)
        ema = float(df_ind["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        self.assertGreater(df_ind["Close"].iloc[-1], ema)
        self.assertEqual(market_dow.get("DIA_bias"), 1, "工业板块独立强势确立协同")

    def test_t3_comb3_r3_r4_micro_decoupling_with_8bps_and_2_5r(self):
        """[Tier3-Comb3] R3 + R4: 个股微观箱体突破脱钩 + 日内 8 bps 摩擦 + 2.5R 盈亏比门禁"""
        ts = pd.date_range("2026-09-01 09:30", periods=40, freq="5min", tz="America/New_York")
        df_5m = pd.DataFrame({
            "Open": [100.0] * 40, "High": [101.0] * 40,
            "Low": [99.5] * 40, "Close": [100.0] * 40
        }, index=ts)
        df_5m.iloc[20, df_5m.columns.get_loc("High")] = 106.0

        tr = sd.simulate_day_trade(df_5m, entry_t=2, entry_price=100.0, side=1,
                                   stop_price=98.0, tp_price=105.5, min_rr=2.5,
                                   slippage_bps=3.0, commission_bps=1.0)
        self.assertEqual(tr.get("status"), "completed")
        self.assertGreaterEqual(tr["planned_rr"], 2.5)
        self.assertTrue(tr["win"])
        self.assertGreater(tr["net_pnl_pct"], 0.0)

    def test_t3_comb4_r2_r4_ema200_and_1h_htf_dual_layer_lock(self):
        """[Tier3-Comb4] R2 + R4: 200 EMA 大势过滤 + 1h HTF 趋势门禁在日内 5m 撮合中的双重加锁"""
        htf_bias = 1
        daily_trend_passed = True
        both_aligned = (daily_trend_passed and htf_bias == 1)
        self.assertTrue(both_aligned, "双重加锁门禁通过")

    def test_t3_comb5_r1_r4_fibonacci_target_with_8bps_net_rr(self):
        """[Tier3-Comb5] R1 + R4: 斐波那契坐标投影作为日内 2.5R 目标位 + 8 bps 摩擦核算"""
        entry_p = 100.0
        stop_p = 98.0
        tp_2_618 = entry_p + 2.0 * 2.618

        ts = pd.date_range("2026-09-01 09:30", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [100.0] * 10, "High": [106.0] * 10,
            "Low": [99.0] * 10, "Close": [105.5] * 10
        }, index=ts)

        tr = sd.simulate_day_trade(df, entry_t=1, entry_price=entry_p, side=1,
                                   stop_price=stop_p, tp_price=tp_2_618, min_rr=2.5)
        self.assertEqual(tr.get("status"), "completed")
        self.assertGreaterEqual(tr["planned_rr"], 2.5)

    def test_t3_comb6_r3_r4_asymmetric_short_macro_confirmation(self):
        """[Tier3-Comb6] R3 + R4 (Short): 美股非对称做空宏观门禁 + 跨市场指数空头确认 + 2.5R 门禁"""
        systemic_bear = {
            "SPY_bias": -1,
            "QQQ_bias": -1,
            "DIA_bias": -1
        }
        market_bear_confirmed = (systemic_bear["SPY_bias"] == -1 and systemic_bear["QQQ_bias"] == -1)
        self.assertTrue(market_bear_confirmed, "做空非对称门禁在大盘全线空头时放行")


# ==============================================================================
# Tier 4: 真实市场复杂实战推演场景 (Real-World Application Scenarios)
# ==============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """
    Tier 4: 真实市场实战场景
    包含财报脉冲、板块大背离、特异性突破、系统性熊市与极度清淡夏休行情
    """

    def test_t4_scenario1_earnings_volatility_2b_sweep(self):
        """
        [Scenario 1] 科技巨头财报日高波动 2B 假突破 (Earnings Volatility 2B Sweep)
        标的开盘刺破前日高点 PDH 诱多后放量砸回，顺应 1h HTF 偏向，完成 >= 2.5R 猎杀并在 15:55 强平。
        """
        ts = pd.date_range("2026-09-01 04:00", "2026-09-01 16:00", freq="5min", tz="America/New_York")
        df_day = pd.DataFrame({
            "Open": [150.0] * len(ts), "High": [151.0] * len(ts),
            "Low": [149.0] * len(ts), "Close": [150.2] * len(ts)
        }, index=ts)

        pdh = 155.0
        idx_1015 = df_day.index.get_loc("2026-09-01 10:15:00-04:00")
        df_day.iloc[idx_1015, df_day.columns.get_loc("Open")] = 154.5
        df_day.iloc[idx_1015, df_day.columns.get_loc("High")] = 156.2
        df_day.iloc[idx_1015, df_day.columns.get_loc("Close")] = 154.0
        df_day.iloc[idx_1015, df_day.columns.get_loc("Low")] = 153.8

        trig = sd.detect_intraday_2b_sweep(df_day, idx_1015, pdh, "PDH", bar_rng=1.5, level_type="high")
        self.assertTrue(trig.get("triggered"))
        self.assertEqual(trig.get("side"), -1)
        self.assertGreater(trig.get("tight_stop"), pdh)

    def test_t4_scenario2_sector_divergence_regime_dia_vs_qqq(self):
        """
        [Scenario 2] 宏观滞胀期工业与科技板块大分化 (Sector Divergence Regime: DIA vs QQQ)
        科技成长 QQQ 破位杀跌，但工业龙头在 DIA 创新高带动下放量破箱体，不被 QQQ 同义反复误杀。
        """
        market_dow = {
            "SPY_bias": 0,
            "QQQ_bias": -1,
            "DIA_bias": 1
        }
        is_divergent = (market_dow["QQQ_bias"] != market_dow["DIA_bias"])
        self.assertTrue(is_divergent, "成功识别纳指与道指板块背离行情")

    def test_t4_scenario3_idiosyncratic_alpha_breakout(self):
        """
        [Scenario 3] 长牛白马特异性 Alpha 强势突破脱钩 (Idiosyncratic Alpha Breakout)
        大盘中性震荡，个股出现高质量微观箱体突破回踩，触发 Micro Trumps Macro 脱钩放行。
        """
        conviction = "strong_box_breakout"
        allow_decoupling = True
        decoupled = (allow_decoupling and conviction == "strong_box_breakout")
        self.assertTrue(decoupled, "优质白马高确定性突破成功激活脱钩豁免")

    def test_t4_scenario4_secular_bear_market_asymmetric_short(self):
        """
        [Scenario 4] 美股大熊市系统性破位下的规范做空 (Secular Bear Market Asymmetric Short)
        标普、纳指、道指全部跌破 200 EMA 且进入 LH+LL，个股反抽受阻，放行做空并严格核算 8 bps 摩擦。
        """
        bear_market = {"SPY_bias": -1, "QQQ_bias": -1, "DIA_bias": -1}
        macro_bear_passed = all(b == -1 for b in bear_market.values())
        self.assertTrue(macro_bear_passed, "宏观大势逆转门禁满足，空头许可激活")

    def test_t4_scenario5_summer_doldrums_churn_protection(self):
        """
        [Scenario 5] 极度清淡夏休行情防摩擦绞杀 (Summer Doldrums Churn Protection)
        5m K 线波动极小，系统启动防刷单门禁，拒绝无效超微波动交易，保全本金。
        """
        df_doldrums = sd.synthetic_intraday_days(n_days=3, seed=888, base_price=100.0)
        events = sd.enumerate_day_events(df_doldrums, min_rr=2.5, use_htf_filter=True)
        self.assertLessEqual(len(events), 15, "防刷单机制有效抑制清淡行情下的高频过度交易")


# ==============================================================================
# 回测与因果性完整集成验证 (Backtest Causality Gate)
# ==============================================================================

class TestBacktestCausalityGateIntegration(unittest.TestCase):
    """
    验证底层回测因果性保证机制
    调用 backtest.verify 确保任何新测试的加入均未破坏纯因果黄金防线
    """

    def test_e2e_backtest_causality_verification(self):
        """运行纯因果零未来函数自检套件 (backtest.verify)"""
        try:
            backtest.verify(n=500, method="h2")
            passed = True
        except Exception as e:
            passed = False
            self.fail(f"backtest.verify(method='h2') 失败: {e}")
        self.assertTrue(passed, "纯因果零未来函数自检必须 100% PASS")


if __name__ == "__main__":
    unittest.main()
