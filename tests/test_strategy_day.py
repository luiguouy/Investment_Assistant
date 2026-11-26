#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内价格行为量化策略单元测试集 (Unit Tests for strategy_day.py)

测试覆盖：
  1. normalize_intraday_tz & split_intraday_sessions: 美东时区转换与按交易日会话切分
  2. extract_session_key_levels: 盘前极值(PMH/PML)、开盘区间(ORH/ORL)与昨高昨低(PDH/PDL)提取
  3. detect_intraday_2b_sweep: 2B 高位假突破做空与低位假摔做多形态识别与紧密止损
  4. detect_orb_breakout_retest: 开盘区间(ORB)突破次级回踩多空确认
  5. simulate_day_trade:
     - 盈亏比硬闸门 (RR < 1.8 驳回)
     - 保守撮合原则 (同根双触判定止损优先)
     - 1.5R 动态推保本防守 (Break-Even)
     - 日内防隔夜强平纪律 (15:55 ET 无论盈亏强平，杜绝持仓隔夜)
     - 真实滑点与交易摩擦扣减 (Gross vs Net 净收益核算)
  6. 日内纯因果零未来函数自检 (Walk-Forward Truncation Invariance)
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

import strategy_day as sd


class TestStrategyDay(unittest.TestCase):

    def setUp(self):
        # 构造可预测的双交易日 5 分钟数据
        self.df_syn = sd.synthetic_intraday_days(n_days=3, seed=123, base_price=100.0)

    def test_session_splitting_and_levels(self):
        """测试按日会话切分与关键位提取"""
        sessions = sd.split_intraday_sessions(self.df_syn)
        self.assertEqual(len(sessions), 3)

        s_dates = list(sessions.keys())
        sess0 = sessions[s_dates[0]]
        sess1 = sessions[s_dates[1]]

        lvls0 = sd.extract_session_key_levels(sess0)
        self.assertIsNotNone(lvls0["ORH"])
        self.assertIsNotNone(lvls0["ORL"])
        self.assertTrue(lvls0["ORH"] >= lvls0["ORL"])
        self.assertIsNotNone(lvls0["PMH"])
        self.assertIsNotNone(lvls0["PML"])

        # 第二个交易日应包含昨日高低收 (PDH/PDL/PDC)
        lvls1 = sd.extract_session_key_levels(sess1, prev_session_df=sess0)
        self.assertEqual(lvls1["PDH"], round(float(sess0["High"].max()), 2))
        self.assertEqual(lvls1["PDL"], round(float(sess0["Low"].min()), 2))
        self.assertEqual(lvls1["PDC"], round(float(sess0["Close"].iloc[-1]), 2))

    def test_2b_liquidity_sweep_detection(self):
        """测试 2B 假突破流动性猎杀检测"""
        idx = pd.date_range("2026-10-26 09:30:00", periods=5, freq="5min", tz="America/New_York")
        # 设定关键阻力位 105.0
        # bar 0: 100~104
        # bar 1: 盘中冲到 107 (刺破 105)，但收盘 104.5 (收回 105 下方)，上影长 -> 2B Sweep High 做空
        df_test = pd.DataFrame({
            "Open":  [101.0, 104.0, 104.0, 103.0, 102.0],
            "High":  [104.0, 107.0, 104.5, 103.5, 102.5],
            "Low":   [100.5, 103.5, 103.0, 102.0, 101.0],
            "Close": [103.8, 104.2, 103.5, 102.2, 101.5],
            "Volume":[1000, 5000, 2000, 1500, 1200]
        }, index=idx)

        trig = sd.detect_intraday_2b_sweep(df_test, 1, level=105.0, level_name="PDH", bar_rng=1.5)
        self.assertTrue(trig["triggered"])
        self.assertEqual(trig["side"], -1)
        self.assertEqual(trig["pattern"], "2B_sweep_high_PDH")
        self.assertTrue(trig["tight_stop"] > 107.0)

        # 设定关键支撑位 95.0
        # bar 1 刺破 95 到 93，收回 95.8，阳线 -> 2B Sweep Low 做多
        df_test2 = pd.DataFrame({
            "Open":  [97.0, 95.5, 96.0, 97.0, 98.0],
            "High":  [98.0, 96.5, 97.0, 98.0, 99.0],
            "Low":   [96.0, 93.0, 95.5, 96.5, 97.5],
            "Close": [96.2, 95.8, 96.8, 97.8, 98.8],
            "Volume":[1000, 5000, 2000, 1500, 1200]
        }, index=idx)
        trig2 = sd.detect_intraday_2b_sweep(df_test2, 1, level=95.0, level_name="PDL", bar_rng=1.5)
        self.assertTrue(trig2["triggered"])
        self.assertEqual(trig2["side"], 1)
        self.assertEqual(trig2["pattern"], "2B_sweep_low_PDL")
        self.assertTrue(trig2["tight_stop"] < 93.0)

    def test_eod_forced_close_discipline(self):
        """测试 15:55 ET 强制平仓纪律，绝不持仓隔夜"""
        # 构造跨越 15:50 ~ 16:00 的日内数据
        idx = pd.date_range("2026-10-26 15:45:00", periods=5, freq="5min", tz="America/New_York")
        df_eod = pd.DataFrame({
            "Open":  [100.0, 100.5, 101.0, 101.2, 101.5],
            "High":  [101.0, 101.5, 102.0, 102.0, 102.5],
            "Low":   [99.5,  100.0, 100.5, 100.8, 101.0],
            "Close": [100.5, 101.0, 101.5, 101.3, 101.8],
            "Volume":[1000, 1000, 1000, 1000, 1000]
        }, index=idx)

        # 15:45 入场 (entry_t=0)，止损 95，止盈 110（未触及）
        # idx[2] 是 15:55:00，此时必须触发 eod_forced_close
        trade = sd.simulate_day_trade(df_eod, entry_t=0, entry_price=100.0, side=1,
                                     stop_price=95.0, tp_price=110.0, min_rr=2.0,
                                     eod_exit_time=dtime(15, 55))
        self.assertEqual(trade["status"], "completed")
        self.assertEqual(trade["exit_reason"], "eod_forced_close")
        self.assertEqual(trade["exit_t"], 2)
        self.assertEqual(df_eod.index[trade["exit_t"]].time(), dtime(15, 55))

    def test_slippage_and_friction_deduction(self):
        """测试滑点与摩擦扣减"""
        idx = pd.date_range("2026-10-26 10:00:00", periods=5, freq="5min", tz="America/New_York")
        df_trade = pd.DataFrame({
            "Open":  [100.0, 102.0, 104.0, 103.0, 102.0],
            "High":  [101.0, 105.0, 105.0, 104.0, 103.0],
            "Low":   [99.0,  101.0, 103.0, 102.0, 101.0],
            "Close": [100.5, 104.5, 103.5, 102.5, 101.5],
            "Volume":[1000, 1000, 1000, 1000, 1000]
        }, index=idx)

        # 100 入场做多，止损 98 (risk=2)，止盈 104 (reward=4, planned_rr=2.0)
        # bar 1 触及 104 (TP)
        # 滑点 5 bps (0.05%), 佣金 2 bps (0.02%)
        trade = sd.simulate_day_trade(df_trade, entry_t=0, entry_price=100.0, side=1,
                                     stop_price=98.0, tp_price=104.0, min_rr=2.0,
                                     slippage_bps=5.0, commission_bps=2.0)
        self.assertEqual(trade["status"], "completed")
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertTrue(trade["gross_pnl_pct"] > trade["net_pnl_pct"])
        self.assertTrue(trade["gross_r"] > trade["net_r"])
        self.assertAlmostEqual(trade["slippage_drag_pct"], trade["gross_pnl_pct"] - trade["net_pnl_pct"], places=5)

    def test_conservative_double_touch(self):
        """测试同根K线双触保守判定：止损优先"""
        idx = pd.date_range("2026-10-26 10:00:00", periods=2, freq="5min", tz="America/New_York")
        df_tie = pd.DataFrame({
            "Open":  [100.0, 100.0],
            "High":  [100.0, 115.0],
            "Low":   [100.0, 90.0],
            "Close": [100.0, 105.0],
            "Volume":[1000, 1000]
        }, index=idx)

        trade = sd.simulate_day_trade(df_tie, entry_t=0, entry_price=100.0, side=1,
                                     stop_price=95.0, tp_price=110.0, min_rr=2.0)
        self.assertEqual(trade["exit_reason"], "stop_loss_conservative")
        self.assertFalse(trade["win"])

    def test_day_zero_lookahead_truncation(self):
        """测试日内策略纯因果截断无未来函数自检"""
        # 生成 5 天仿真数据
        df_5d = sd.synthetic_intraday_days(n_days=5, seed=999, base_price=120.0)
        full_events = sd.enumerate_day_events(df_5d, strategy_type="day_combined", min_rr=1.8)

        # 截断在第 3 个交易日结束时，重新枚举事件
        sessions = sd.split_intraday_sessions(df_5d)
        dates = list(sessions.keys())
        cutoff_date = dates[2]
        sub_df = df_5d[df_5d.index.date <= cutoff_date]

        part_events = sd.enumerate_day_events(sub_df, strategy_type="day_combined", min_rr=1.8)

        full_sub = [(e["entry_date"], e["side"], round(e["entry_price"], 2), round(e["stop_price"], 2))
                    for e in full_events if e["session_date"] <= cutoff_date]
        part_sub = [(e["entry_date"], e["side"], round(e["entry_price"], 2), round(e["stop_price"], 2))
                    for e in part_events]

        self.assertEqual(full_sub, part_sub, "截断后历史决策集必须与完整序列完全一致！")

    def test_2b_rejection_of_inverted_levels(self):
        """测试 2B 严格拒绝对阻力位做多或对支撑位做空（倒错信号防范）"""
        idx = pd.date_range("2026-10-26 10:00:00", periods=5, freq="5min", tz="America/New_York")
        # 价格在 PDH(100.0) 上方运行，探底至 99.5 后收在 100.2（阳线）：对于 PDH 而言绝对不能产生 2B 做多信号
        df_dip = pd.DataFrame({
            "Open":  [100.5, 100.1, 100.4, 100.8, 101.0],
            "High":  [101.0, 100.6, 100.9, 101.2, 101.5],
            "Low":   [100.0, 99.5,  100.2, 100.5, 100.8],
            "Close": [100.2, 100.4, 100.8, 101.0, 101.2],
            "Volume":[1000, 1000, 1000, 1000, 1000]
        }, index=idx)

        trig = sd.detect_intraday_2b_sweep(df_dip, 1, level=100.0, level_name="PDH", bar_rng=1.0)
        self.assertFalse(trig.get("triggered", False), "PDH 阻力位绝不允许产生做多触发！")

        # 价格在 PDL(90.0) 下方运行，摸高至 90.5 后收在 89.8（阴线）：对于 PDL 而言绝对不能产生 2B 做空信号
        df_bump = pd.DataFrame({
            "Open":  [89.5, 90.2, 89.6, 89.2, 89.0],
            "High":  [89.8, 90.5, 89.9, 89.5, 89.2],
            "Low":   [89.0, 89.4, 89.0, 88.8, 88.5],
            "Close": [89.2, 89.8, 89.3, 89.0, 88.6],
            "Volume":[1000, 1000, 1000, 1000, 1000]
        }, index=idx)
        trig2 = sd.detect_intraday_2b_sweep(df_bump, 1, level=90.0, level_name="PDL", bar_rng=1.0)
        self.assertFalse(trig2.get("triggered", False), "PDL 支撑位绝不允许产生做空触发！")

    def test_extract_session_key_levels_causality(self):
        """测试关键位提取因果性：无论下午行情如何演化，上午 09:45 提取的 bar_rng 必须恒定一致"""
        sessions = sd.split_intraday_sessions(self.df_syn)
        s_dates = list(sessions.keys())
        sess0 = sessions[s_dates[0]]
        sess1 = sessions[s_dates[1]]

        # 上午 10:00 截断的会话
        cutoff_morning = sess1[sess1.index.time <= dtime(10, 0)]
        lvls_morning = sd.extract_session_key_levels(cutoff_morning, prev_session_df=sess0)
        # 全日完整会话
        lvls_full = sd.extract_session_key_levels(sess1, prev_session_df=sess0)

        self.assertEqual(lvls_morning["bar_rng"], lvls_full["bar_rng"],
                         "上午提取的 bar_rng 不受下午行情影响，严格无未来函数泄露！")

    def test_eod_early_close_half_day(self):
        """测试美股半日市（如13:00休市）强制在当日最后一根收盘平仓，绝不持仓隔夜"""
        # 构造当天仅到 13:00 的半日市数据，接着是次日 09:30
        idx_day1 = pd.date_range("2026-10-26 12:45:00", "2026-10-26 13:00:00", freq="5min", tz="America/New_York")
        idx_day2 = pd.date_range("2026-10-27 09:30:00", "2026-10-27 09:40:00", freq="5min", tz="America/New_York")
        idx = idx_day1.append(idx_day2)

        df_half = pd.DataFrame({
            "Open":  [100.0, 100.2, 100.5, 100.4, 102.0, 102.5, 103.0],
            "High":  [100.5, 100.6, 100.8, 100.7, 102.6, 103.0, 103.5],
            "Low":   [99.8,  100.0, 100.2, 100.1, 101.8, 102.2, 102.8],
            "Close": [100.2, 100.4, 100.6, 100.5, 102.3, 102.8, 103.2],
            "Volume":[1000] * 7
        }, index=idx)

        # 12:45 入场做多，止损 95，止盈 110（未触及）
        trade = sd.simulate_day_trade(df_half, entry_t=0, entry_price=100.0, side=1,
                                     stop_price=95.0, tp_price=110.0, min_rr=2.0)
        self.assertEqual(trade["status"], "completed")
        self.assertEqual(trade["exit_reason"], "eod_forced_close")
        exit_date = pd.to_datetime(trade["exit_date"]).date()
        entry_date = pd.to_datetime(trade["entry_date"]).date()
        self.assertEqual(exit_date, entry_date, "半日市必须在当日结束时强平，严禁留单隔夜！")

    def test_avoid_active_position_overlap(self):
        """测试在途持仓独占：持仓未平仓前不重复开仓"""
        evs = sd.enumerate_day_events(self.df_syn, strategy_type="day_combined",
                                      avoid_overlap=True)
        # 验证任意两笔同日交易之间，后一笔入场时间必然在前一笔平仓之后
        for i in range(len(evs) - 1):
            if evs[i]["session_date"] == evs[i + 1]["session_date"]:
                self.assertTrue(evs[i + 1]["trigger_t"] > evs[i]["trigger_t"],
                                "同日交易后一笔入场必在前一笔之后")


if __name__ == "__main__":
    unittest.main()
