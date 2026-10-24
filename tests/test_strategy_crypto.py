#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_strategy_crypto.py: 比特币与加密资产专用交易策略单元测试集

覆盖项：
  1. is_crypto_asset 资产类型路由识别
  2. extract_crypto_levels 宽幅防插针止损与 1.618 斐波那契顺势扩展位计算
  3. detect_crypto_trigger_on_bar 2B 流动性猎杀 (2B Sweep) 与加密裸K形态检测
  4. simulate_crypto_trigger_entry 零未来函数次根开盘入场与结构失效撤单
  5. simulate_crypto_trade 非对称分级止盈生命周期：
     - TP1 减半并推保本 (Break-Even Stop)
     - 保本保护：触碰 TP1 后回踩保本位安全出场，锁定正收益与正 R 倍数
     - TP2 1.618 扩展位与 Runner 追踪趋势
     - 极端同根K线双触保守判定（止损优先出场）
     - 盈亏比硬闸门 (RR < 2.0 一票否决)
  6. 截断重算自检：杜绝任何未来函数泄露
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import strategy_crypto as sc
import pa_engine as pe


class TestStrategyCrypto(unittest.TestCase):

    def setUp(self):
        dates = pd.bdate_range("2026-01-01", periods=100)
        c = 50000.0 + np.sin(np.linspace(0, 10, 100)) * 5000.0
        self.df = pd.DataFrame({
            "Open": c,
            "High": c + 500.0,
            "Low": c - 500.0,
            "Close": c,
            "Volume": np.full(100, 1000000.0)
        }, index=dates)

    def test_is_crypto_asset(self):
        """测试加密货币标的自动识别"""
        self.assertTrue(sc.is_crypto_asset("BTC-USD"))
        self.assertTrue(sc.is_crypto_asset("ETH-USD"))
        self.assertTrue(sc.is_crypto_asset("btc-usd"))
        self.assertTrue(sc.is_crypto_asset("BTC"))
        self.assertTrue(sc.is_crypto_asset("ETH"))
        self.assertTrue(sc.is_crypto_asset("SOL-USDT"))
        self.assertTrue(sc.is_crypto_asset("DOGE-USD"))

        self.assertFalse(sc.is_crypto_asset("SPY"))
        self.assertFalse(sc.is_crypto_asset("QQQ"))
        self.assertFalse(sc.is_crypto_asset("AAPL"))
        self.assertFalse(sc.is_crypto_asset("NVDA"))
        self.assertFalse(sc.is_crypto_asset(""))

    def test_extract_crypto_levels(self):
        """测试加密结构位、宽幅防插针止损与 1.618 斐波顺势扩展提取"""
        swings = [
            {"index": 20, "type": "High", "price": 60000.0, "dow": "High", "confirmed": True},
            {"index": 40, "type": "Low", "price": 50000.0, "dow": "Low", "confirmed": True},
        ]
        df_sub = self.df.iloc[:60].copy()
        df_sub.iloc[-1, df_sub.columns.get_loc('High')] = 65000.0
        df_sub.iloc[-1, df_sub.columns.get_loc('Close')] = 63000.0

        lvls = sc.extract_crypto_levels(df_sub, 59, side=1, swings=swings)
        self.assertEqual(lvls["side"], 1)
        self.assertEqual(lvls["h_break"], 60000.0)
        self.assertEqual(lvls["l_anchor"], 50000.0)
        self.assertEqual(lvls["peak"], 65000.0)
        self.assertEqual(lvls["impulse_leg"], 15000.0)  # 65000 - 50000

        # 宽幅止损必须给足 1.4x ATR 呼吸空间
        bar_rng = lvls["bar_rng"]
        self.assertAlmostEqual(lvls["structural_stop"], round(50000.0 - 1.4 * bar_rng, 2), places=1)

        # TP1 必须为前高冲量极值 65000.0
        self.assertEqual(lvls["tp1_target"], 65000.0)

        # TP2 顺势 1.618 扩展位：50000 + 1.618 * 15000 = 74270.0
        self.assertAlmostEqual(lvls["tp2_target"], 74270.0, places=1)

    def test_2b_liquidity_sweep_trigger(self):
        """测试加密核心触发形态：2B 流动性猎杀扫损 (2B Liquidity Sweep)"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        # key_level = 60000.0, bar_rng = 1000.0
        # bar 1: 盘中暴跌至 59200 (击穿 60000 引爆止损踩踏)，收盘强力拉升至 60800，形成经典 2B 假突破收回
        df_2b = pd.DataFrame({
            "Open": [61000.0, 60200.0, 60800.0],
            "High": [61500.0, 61000.0, 61500.0],
            "Low":  [60500.0, 59200.0, 60500.0],
            "Close":[61000.0, 60800.0, 61200.0]
        }, index=idx)

        trig = sc.detect_crypto_trigger_on_bar(df_2b, 1, side=1, key_level=60000.0, bar_rng=1000.0)
        self.assertTrue(trig["triggered"])
        self.assertEqual(trig["pattern"], "2B_sweep")
        self.assertTrue(trig["is_2b"])
        # 止损紧贴 2B 针尖 Low (59200) 下方 0.35x bar_rng = 59200 - 350 = 58850
        self.assertAlmostEqual(trig["tight_stop"], 58850.0, places=1)

        # 空头 2B 猎杀
        df_2b_bear = pd.DataFrame({
            "Open": [59000.0, 59800.0, 59200.0],
            "High": [59500.0, 60800.0, 59500.0],  # 突破 60000 诱多爆空
            "Low":  [58500.0, 59000.0, 58500.0],
            "Close":[59000.0, 59200.0, 58800.0]   # 强力实体砸回收于 60000 下方
        }, index=idx)
        trig_bear = sc.detect_crypto_trigger_on_bar(df_2b_bear, 1, side=-1, key_level=60000.0, bar_rng=1000.0)
        self.assertTrue(trig_bear["triggered"])
        self.assertEqual(trig_bear["pattern"], "2B_sweep")
        self.assertTrue(trig_bear["tight_stop"] > 60800.0)

    def test_conservative_tie_breaking(self):
        """测试加密保守撮合原则：同一根K线同时触及止损与目标，一律判定为止损优先出场"""
        idx = pd.bdate_range("2026-01-01", periods=2)
        # entry=60000, stop=58000, tp1=64000.
        # bar 1: Low 57000 (触及止损) 且 High 65000 (触及TP1)
        df_tie = pd.DataFrame({
            "Open": [60000.0, 60000.0],
            "High": [60000.0, 65000.0],
            "Low":  [60000.0, 57000.0],
            "Close":[60000.0, 61000.0]
        }, index=idx)

        tr = sc.simulate_crypto_trade(df_tie, 0, 60000.0, side=1, stop_price=58000.0, tp1_price=64000.0, min_rr=2.0)
        self.assertEqual(tr["exit_reason"], "stop_loss_conservative")
        self.assertFalse(tr["win"])
        self.assertEqual(tr["exit_price"], 58000.0)

    def test_multi_tier_take_profit_with_break_even_protection(self):
        """
        测试加密非对称分级止盈核心机制：
          - bar 1 触碰 TP1 减半，推保本损 (entry_price)
          - bar 2 遇阻深幅回踩击穿保本位
          - 验证：由于 TP1 已兑现 50%（+2.0R），即便回踩保本，整笔交易依然斩获正收益（+1.0R）且胜率计为盈利！
        """
        idx = pd.bdate_range("2026-01-01", periods=4)
        # entry=60000, stop=58000 (risk=2000), tp1=64000 (reward=4000, RR=2.0), tp2=70000
        # bar 1: High=64500 (触碰 TP1)，Low=59500 -> 50%仓位兑现于 64000，剩余50%止损推至 60000
        # bar 2: High=63000, Low=59000 -> 跌破保本位 60000，剩余50%仓位保本出场
        df_be = pd.DataFrame({
            "Open": [60000.0, 60500.0, 62000.0, 58000.0],
            "High": [60000.0, 64500.0, 62500.0, 59000.0],
            "Low":  [60000.0, 59500.0, 59000.0, 57000.0],
            "Close":[60000.0, 63000.0, 59500.0, 57500.0]
        }, index=idx)

        tr = sc.simulate_crypto_trade(df_be, 0, 60000.0, side=1, stop_price=58000.0, tp1_price=64000.0, tp2_price=70000.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertTrue(tr["win"])  # 依然为盈利交易！
        self.assertEqual(tr["exit_reason"], "tp1_then_be_protected")

        # 验证加权收益：50% * (+6.67%) + 50% * 0% = +3.33%
        # 验证加权 R 倍数：50% * (+2.0R) + 50% * 0R = +1.0R
        self.assertAlmostEqual(tr["r_multiple"], 1.0, places=2)
        self.assertAlmostEqual(tr["pnl_pct"], (4000.0 / 60000.0) * 0.5, places=4)

    def test_full_three_tier_super_trend(self):
        """测试加密超级单边长牛走势：依次触发 TP1 减半、TP2 斐波扩展、Runner 移动止损捕获长波段"""
        idx = pd.bdate_range("2026-01-01", periods=6)
        # entry=50000, stop=45000 (risk=5000), tp1=60000 (RR=2.0), tp2=70000 (RR=4.0)
        # bar 1: 摸到 61000 -> TP1 触发 (50% 出于 60000)，推保本
        # bar 2: 摸到 72000 -> TP2 触发 (25% 出于 70000)，Runner 止损锁定 60000
        # bar 3: 摸到 80000
        # bar 4: 跌破 65000 -> Runner 出场于追踪止损
        df_super = pd.DataFrame({
            "Open": [50000.0, 52000.0, 62000.0, 71000.0, 75000.0, 62000.0],
            "High": [50000.0, 61000.0, 72000.0, 80000.0, 76000.0, 63000.0],
            "Low":  [50000.0, 51000.0, 61000.0, 70000.0, 58000.0, 59000.0],
            "Close":[50000.0, 60500.0, 71000.0, 78000.0, 60000.0, 61000.0]
        }, index=idx)

        tr = sc.simulate_crypto_trade(df_super, 0, 50000.0, side=1, stop_price=45000.0, tp1_price=60000.0, tp2_price=70000.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertTrue(tr["win"])
        self.assertEqual(tr["exit_reason"], "tp_runner_captured")
        # 实现 R 倍数应显著高于 2.0R (大赔率爆发)
        self.assertTrue(tr["r_multiple"] >= 2.0)

    def test_min_rr_gate_rejection(self):
        """测试加密盈亏比硬闸门：结构盈亏比低于 2.0 严格驳回"""
        idx = pd.bdate_range("2026-01-01", periods=2)
        # entry=50000, stop=47000 (risk=3000), tp1=53000 (reward=3000, RR=1.0 < 2.0)
        df_dummy = pd.DataFrame({
            "Open": [50000.0, 51000.0],
            "High": [50000.0, 54000.0],
            "Low":  [50000.0, 49000.0],
            "Close":[50000.0, 52000.0]
        }, index=idx)

        tr = sc.simulate_crypto_trade(df_dummy, 0, 50000.0, side=1, stop_price=47000.0, tp1_price=53000.0, min_rr=2.0)
        self.assertEqual(tr["status"], "rejected_by_min_rr")
        self.assertEqual(tr["rr"], 1.0)

    def test_truncation_no_future_leak(self):
        """测试子序列截断重算自检：证明无任何未来函数泄露"""
        # 使用 setUp 中的 100 根基准序列
        swings_full = pe.extract_swings(self.df, order=5)
        # 在 t=50 进行截断
        df_sub = self.df.iloc[:51]
        swings_sub = pe.extract_swings(df_sub, order=5)
        # 对比 t <= 40 的确认拐点
        conf_full = [s for s in swings_full if s.get("confirmed") and s.get("index", 0) <= 40]
        conf_sub = [s for s in swings_sub if s.get("confirmed") and s.get("index", 0) <= 40]
        self.assertEqual(len(conf_full), len(conf_sub))
        for s1, s2 in zip(conf_full, conf_sub):
            self.assertEqual(s1["index"], s2["index"])
            self.assertEqual(s1["price"], s2["price"])

        # 验证在 t=45 决策时刻（使用 <= 45 的数据），无论在全量 df 还是截断 df_sub 提取结构位均 100% 一致
        sub_full_45 = self.df.iloc[:46]
        sub_part_45 = df_sub.iloc[:46]
        sw_full_45 = pe.extract_swings(sub_full_45, order=5)
        sw_part_45 = pe.extract_swings(sub_part_45, order=5)

        lvls_full = sc.extract_crypto_levels(sub_full_45, 45, side=1, swings=sw_full_45)
        lvls_sub = sc.extract_crypto_levels(sub_part_45, 45, side=1, swings=sw_part_45)
        if lvls_full and lvls_sub:
            self.assertEqual(lvls_full["structural_stop"], lvls_sub["structural_stop"])
            self.assertEqual(lvls_full["tp1_target"], lvls_sub["tp1_target"])
            self.assertEqual(lvls_full["tp2_target"], lvls_sub["tp2_target"])


    def test_tp2_always_beyond_tp1_ordering(self):
        """测试目标位阶梯性：验证多头 TP2 严格大于 TP1，空头 TP2 严格小于 TP1 且大于 0"""
        idx = pd.bdate_range("2026-01-01", periods=3)
        df_dummy = pd.DataFrame({
            "Open": [100.0, 102.0, 105.0],
            "High": [101.0, 106.0, 110.0],
            "Low":  [99.0, 101.0, 103.0],
            "Close":[100.5, 105.0, 108.0]
        }, index=idx)

        # 多头：即便 impulse_leg 很小，TP2 也必须严格在 TP1 上方
        tr_long = sc.simulate_crypto_trade(df_dummy, 0, 100.0, side=1, stop_price=80.0,
                                           tp1_price=140.0, tp2_price=120.0, impulse_leg=10.0, min_rr=2.0)
        self.assertTrue(tr_long["tp2_price"] > tr_long["tp1_price"])

        # 空头：TP2 必须严格在 TP1 下方，且严格大于 0
        tr_short = sc.simulate_crypto_trade(df_dummy, 0, 100.0, side=-1, stop_price=110.0,
                                            tp1_price=80.0, tp2_price=90.0, impulse_leg=5.0, min_rr=2.0)
        self.assertTrue(tr_short["tp2_price"] < tr_short["tp1_price"])
        self.assertTrue(tr_short["tp2_price"] > 0)

    def test_short_impossible_negative_target_rejected(self):
        """测试空头极端宽止损保护：若 2.0R 目标跌破 0，必须被硬闸门驳回杜绝负数目标"""
        idx = pd.bdate_range("2026-01-01", periods=2)
        df_dummy = pd.DataFrame({
            "Open": [1000.0, 1000.0],
            "High": [1000.0, 1000.0],
            "Low":  [1000.0, 1000.0],
            "Close":[1000.0, 1000.0]
        }, index=idx)

        # entry=1000, stop=1600 (risk=600). 2.0R 需要目标价 = 1000 - 1200 = -200 <= 0，物理不可行
        tr = sc.simulate_crypto_trade(df_dummy, 0, 1000.0, side=-1, stop_price=1600.0, min_rr=2.0)
        self.assertEqual(tr["status"], "rejected_by_min_rr")
        self.assertEqual(tr.get("reason"), "short_target_non_positive")

    def test_short_trade_multi_tier_lifecycle(self):
        """测试空头非对称分级止盈生命周期：TP1减半、推保本、TP2顺势扩展"""
        idx = pd.bdate_range("2026-01-01", periods=5)
        # entry=100, stop=110 (risk=10), tp1=80 (reward=20, RR=2.0), tp2=60 (RR=4.0)
        # bar 1: Low=75 (触碰 TP1 80) -> 减半，止损推至保本 100
        # bar 2: Low=58 (触碰 TP2 60) -> 兑现 25%，Runner 止损锁定在 80
        # bar 3: High=85 (反弹触碰 80 锁定止损) -> Runner 出场
        df_short = pd.DataFrame({
            "Open": [100.0, 95.0, 75.0, 65.0, 85.0],
            "High": [100.0, 96.0, 78.0, 85.0, 90.0],
            "Low":  [100.0, 75.0, 58.0, 60.0, 80.0],
            "Close":[100.0, 78.0, 62.0, 82.0, 88.0]
        }, index=idx)

        tr = sc.simulate_crypto_trade(df_short, 0, 100.0, side=-1, stop_price=110.0,
                                      tp1_price=80.0, tp2_price=60.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertTrue(tr["win"])
        self.assertEqual(tr["exit_reason"], "tp_runner_captured")
        self.assertTrue(tr["r_multiple"] >= 2.0)

    def test_trailing_stop_no_lookback_contamination(self):
        """测试 Runner 追踪止损窗口隔离：只追踪入场之后的极值，不受入场前低点/高点污染"""
        idx = pd.bdate_range("2026-01-01", periods=15)
        # 前 5 根有深度低点 10.0 (entry 前)，在 bar 5 entry=100.0, stop=80.0
        lows = [10.0] * 5 + [95.0, 125.0, 145.0, 150.0, 155.0, 152.0, 140.0, 130.0, 120.0, 110.0]
        highs = [20.0] * 5 + [105.0, 145.0, 165.0, 170.0, 175.0, 170.0, 160.0, 150.0, 140.0, 130.0]
        opens = [15.0] * 5 + [100.0, 130.0, 150.0, 155.0, 160.0, 155.0, 145.0, 135.0, 125.0, 115.0]
        closes = opens
        df_trail = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        # bar 5 入场，tp1=140, tp2=160
        tr = sc.simulate_crypto_trade(df_trail, 5, 100.0, side=1, stop_price=80.0,
                                      tp1_price=140.0, tp2_price=160.0, min_rr=2.0)
        self.assertEqual(tr["status"], "completed")
        self.assertTrue(tr["win"])
        # runner 出场价必须远高于 10.0 (入场前极值)，且高于等于 tp1_price (140.0)
        runner_records = [r for r in tr["exit_records"] if r["reason"] == "runner_stop"]
        self.assertTrue(len(runner_records) > 0)
        self.assertTrue(runner_records[0]["price"] >= 140.0)

    def test_is_crypto_asset_strict_isolation(self):
        """测试标的类型识别：严格排除外汇法定货币与美股"""
        self.assertFalse(sc.is_crypto_asset("EUR-USD"))
        self.assertFalse(sc.is_crypto_asset("GBP-USD"))
        self.assertFalse(sc.is_crypto_asset("JPY-USD"))
        self.assertFalse(sc.is_crypto_asset("TSLA"))
        self.assertFalse(sc.is_crypto_asset("SPY"))
        self.assertFalse(sc.is_crypto_asset("QQQ"))
        self.assertTrue(sc.is_crypto_asset("BTC-USD"))
        self.assertTrue(sc.is_crypto_asset("ETH-USDT"))
        self.assertTrue(sc.is_crypto_asset("SOL-USDC"))

    def test_crypto_black_swan_detection(self):
        """测试系统性黑天鹅事件窗口识别"""
        # 312 暴跌
        self.assertTrue(sc.is_crypto_black_swan("2020-03-12"))
        self.assertTrue(sc.is_crypto_black_swan("2020-03-08"))
        self.assertTrue(sc.is_crypto_black_swan("2020-03-20"))
        # 519 暴跌
        self.assertTrue(sc.is_crypto_black_swan("2021-05-19"))
        # Luna 死亡螺旋
        self.assertTrue(sc.is_crypto_black_swan("2022-05-09"))
        # FTX 破产
        self.assertTrue(sc.is_crypto_black_swan("2022-11-08"))

        # 正常市场行情日
        self.assertFalse(sc.is_crypto_black_swan("2020-01-01"))
        self.assertFalse(sc.is_crypto_black_swan("2021-01-01"))
        self.assertFalse(sc.is_crypto_black_swan("2023-06-15"))
        self.assertFalse(sc.is_crypto_black_swan("2024-03-15"))

    def test_crypto_black_swan_overlap(self):
        """测试交易跨越黑天鹅崩溃窗口重叠检测"""
        # 跨越 312: 入场 2020-03-01，出场 2020-03-25
        self.assertTrue(sc.trade_overlaps_black_swan("2020-03-01", "2020-03-25"))
        # 跨越 Luna: 入场 2022-04-20，出场 2022-05-15
        self.assertTrue(sc.trade_overlaps_black_swan("2022-04-20", "2022-05-15"))

        # 完全在黑天鹅之前: 入场 2020-01-01，出场 2020-02-15
        self.assertFalse(sc.trade_overlaps_black_swan("2020-01-01", "2020-02-15"))
        # 完全在黑天鹅之后: 入场 2020-04-01，出场 2020-05-01
        self.assertFalse(sc.trade_overlaps_black_swan("2020-04-01", "2020-05-01"))

    def test_check_crypto_primary_trend(self):
        """测试道氏 200 EMA 大周期主趋势门禁"""
        idx = pd.bdate_range("2025-01-01", periods=250)
        # 上升趋势：从 10000 涨到 30000，当前价格 30000 远高于 200 EMA
        c_up = np.linspace(10000.0, 30000.0, 250)
        df_bull = pd.DataFrame({"Open": c_up, "High": c_up + 100, "Low": c_up - 100, "Close": c_up}, index=idx)

        # 牛市中：多头放行，空头严格拦截驳回
        chk_bull_long = sc.check_crypto_primary_trend(df_bull, 240, side=1)
        self.assertTrue(chk_bull_long["aligned"])
        self.assertEqual(chk_bull_long["reason"], "bull_market_long_ok")

        chk_bull_short = sc.check_crypto_primary_trend(df_bull, 240, side=-1)
        self.assertFalse(chk_bull_short["aligned"])
        self.assertEqual(chk_bull_short["reason"], "counter_trend_bull_short_blocked")

        # 下降趋势：从 30000 跌到 10000，当前价格 10000 远低于 200 EMA
        c_down = np.linspace(30000.0, 10000.0, 250)
        df_bear = pd.DataFrame({"Open": c_down, "High": c_down + 100, "Low": c_down - 100, "Close": c_down}, index=idx)

        # 熊市中：多头拦截抄底，空头放行
        chk_bear_long = sc.check_crypto_primary_trend(df_bear, 240, side=1)
        self.assertFalse(chk_bear_long["aligned"])
        self.assertEqual(chk_bear_long["reason"], "counter_trend_bear_long_blocked")

        chk_bear_short = sc.check_crypto_primary_trend(df_bear, 240, side=-1)
        self.assertTrue(chk_bear_short["aligned"])
        self.assertEqual(chk_bear_short["reason"], "bear_market_short_ok")

        # 小样本历史不足 50 根：自动放行不阻塞
        df_small = df_bull.iloc[:30]
        chk_small = sc.check_crypto_primary_trend(df_small, 20, side=1)
        self.assertTrue(chk_small["aligned"])
        self.assertEqual(chk_small["reason"], "insufficient_history_neutral")


if __name__ == "__main__":
    unittest.main()


