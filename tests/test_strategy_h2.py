"""
H2 结构突破策略优化单元测试集
覆盖：
1. pe.dow_bias 与 pe.dow_bias_count 的边界条件与突破逻辑
2. pe.detect_mss 的真实位移穿越 (displacement) 与同向去重 (P1-7)
3. backtest.py 的未来函数自检与双模式 (h1/h2) 行为
4. 量能过滤 (volume_filter, 道氏第6条) 机制
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pa_engine as pe
import backtest as bt

class TestH2StrategyOptimization(unittest.TestCase):

    def setUp(self):
        # 构造基准数据
        dates = pd.bdate_range("2026-01-01", periods=100)
        # 生成一个明确的走势：先跌后涨
        prices = np.concatenate([
            np.linspace(100, 80, 40),   # 下跌波段
            np.linspace(80, 110, 60),   # 上涨反转波段
        ])
        spread = 0.5
        self.df = pd.DataFrame({
            "Open": prices,
            "High": prices + spread,
            "Low": prices - spread,
            "Close": prices,
            "Volume": np.full(100, 1000000.0)
        }, index=dates)

    def test_dow_bias_backward_compatibility(self):
        """测试 df=None 时优雅回退到旧版标签计数"""
        swings = [
            {"type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"type": "Low", "price": 90.0, "dow": "Low", "confirmed": True},
            {"type": "High", "price": 105.0, "dow": "HH", "confirmed": True},
            {"type": "Low", "price": 95.0, "dow": "HL", "confirmed": True},
        ]
        # 标签计数：HH+HL=2 -> +1
        self.assertEqual(pe.dow_bias(swings), 1)
        self.assertEqual(pe.dow_bias_count(swings), 1)

        # 空头标签：LH+LL=2 -> -1
        bear_swings = [
            {"type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"type": "Low", "price": 90.0, "dow": "Low", "confirmed": True},
            {"type": "High", "price": 95.0, "dow": "LH", "confirmed": True},
            {"type": "Low", "price": 85.0, "dow": "LL", "confirmed": True},
        ]
        self.assertEqual(pe.dow_bias(bear_swings), -1)
        self.assertEqual(pe.dow_bias_count(bear_swings), -1)

    def test_dow_bias_structural_break(self):
        """测试基于收盘价穿越最新确认波段高低点的判定"""
        swings = [
            {"index": 20, "type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"index": 40, "type": "Low", "price": 80.0, "dow": "Low", "confirmed": True},
        ]
        
        # 1. 收盘突破高点 100 -> +1
        df_up = self.df.iloc[:60].copy()
        df_up.iloc[-1, df_up.columns.get_loc('Close')] = 101.5
        bias = pe.dow_bias(swings, df=df_up)
        self.assertEqual(bias, 1)

        # 2. 收盘跌破低点 80 -> -1
        df_down = self.df.iloc[:60].copy()
        df_down.iloc[-1, df_down.columns.get_loc('Close')] = 78.5
        bias = pe.dow_bias(swings, df=df_down)
        self.assertEqual(bias, -1)

        # 3. 价格处于高低点之间 [80, 100]：趋势持续性
        # 如果此前刚刚突破过 100，回踩到 95，偏向依然为 +1
        df_pullback = self.df.iloc[:60].copy()
        df_pullback.iloc[-2, df_pullback.columns.get_loc('Close')] = 102.0  # 前一根突破
        df_pullback.iloc[-1, df_pullback.columns.get_loc('Close')] = 95.0   # 当前回踩
        bias = pe.dow_bias(swings, df=df_pullback)
        self.assertEqual(bias, 1)

    def test_dow_bias_buffer(self):
        """测试缓冲倍数过滤微小假突破"""
        swings = [
            {"index": 20, "type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"index": 40, "type": "Low", "price": 80.0, "dow": "Low", "confirmed": True},
        ]
        df_sub = self.df.iloc[:60].copy()
        # 仅超出 0.05 元，若 buffer_k=0.5 (振幅 1.0 时缓冲为 0.5)，则不应触发突破
        df_sub.iloc[-1, df_sub.columns.get_loc('Close')] = 100.05
        bias_nobuf = pe.dow_bias(swings, df=df_sub, buffer_k=0.0)
        self.assertEqual(bias_nobuf, 1)
        
        # 加上较大缓冲 (buffer=1.0) 后，100.05 未有效突破 100+1.0，且处于区间内无历史突破，预期为中性 0
        bias_buf = pe.dow_bias(swings, df=df_sub, buffer_k=1.0)
        self.assertEqual(bias_buf, 0)

    def test_detect_mss_displacement_and_dedupe(self):
        """测试 detect_mss 的位移穿越与同向去重 (P1-7)"""
        swings = [
            {"index": 10, "time": "2026-01-10", "type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"index": 20, "time": "2026-01-20", "type": "Low", "price": 90.0, "dow": "Low", "confirmed": True},
            {"index": 30, "time": "2026-01-30", "type": "High", "price": 105.0, "dow": "HH", "confirmed": True},
            {"index": 40, "time": "2026-02-10", "type": "Low", "price": 95.0, "dow": "HL", "confirmed": True},
        ]
        # 当前收盘价击穿前低点 95.0 -> 产生空翻 MSS
        df_sub = self.df.iloc[:60].copy()
        df_sub.iloc[-1, df_sub.columns.get_loc('Close')] = 93.0
        events = pe.detect_mss(swings, df=df_sub)
        self.assertTrue(len(events) >= 1)
        self.assertIn("空翻", events[-1]["type"])
        self.assertIn("95.00", events[-1]["detail"])

        # 同向去重测试：连续同向事件只保留最新
        dup_swings = [
            {"time": "t1", "dow": "LH", "price": 100},
            {"time": "t2", "dow": "LL", "price": 90},
            {"time": "t3", "dow": "LH", "price": 88},
            {"time": "t4", "dow": "LL", "price": 80},
        ]
        ev_dedup = pe.detect_mss(dup_swings)
        # 应去重，不出现连续两个空翻
        types = [e["type"] for e in ev_dedup]
        self.assertEqual(len(types), len(set(types)))

    def test_backtest_verify_both_modes(self):
        """测试 backtest.py 的 verify 自检函数在 h2 与 h1 模式下全绿"""
        ret_h2 = bt.verify(n=500, method="h2")
        self.assertEqual(ret_h2, 0)
        ret_h1 = bt.verify(n=500, method="h1")
        self.assertEqual(ret_h1, 0)

    def test_volume_filter(self):
        """测试道氏第6条量能验证过滤"""
        df_vol = self.df.copy()
        # 制造一个无量突破
        df_vol.iloc[-1, df_vol.columns.get_loc('Volume')] = 100.0  # 极低量
        w = bt.weekly_from_daily(df_vol)
        ev_all = bt.enumerate_events(df_vol, w, method="h2", volume_filter=False)
        ev_vol = bt.enumerate_events(df_vol, w, method="h2", volume_filter=True)
        # 量能过滤应该使事件数少于或等于无过滤事件数
        self.assertLessEqual(len(ev_vol), len(ev_all))

    def test_dow_bias_post_swing_confirmation_pullback_persistence(self):
        """测试破位新高确认成拐点后，价格在回调盘整期间保持多头偏向（消除虚假反转洗盘）"""
        # 构造空翻多反转行情：低点90，高点104，突破至106，随后在103盘整
        prices = [110, 108, 105, 103, 100, 98, 102, 104, 102, 100, 95, 93, 90, 92, 94, 96, 98, 103, 105, 106, 104, 103, 103, 103, 103, 103]
        n = len(prices)
        df_seq = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.5 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": [1e6] * n
        }, index=pd.bdate_range("2026-01-01", periods=n))

        # 在突破确立后 (t=18开始)，无论拐点是否确认、回调盘整期间均应保持偏向为 +1
        for t in range(18, n):
            sub = df_seq.iloc[:t + 1]
            sw = pe.extract_swings(sub, order=3)
            b = pe.dow_bias(sw, df=sub)
            self.assertEqual(b, 1, f"t={t} price={sub['Close'].iloc[-1]} 应当保持多头偏向 +1，实得 {b}")

    def test_detect_mss_exact_timing_and_bars_ago(self):
        """测试 detect_mss 在 df 模式下准确识别破位发生时的K线时间与 bars_ago"""
        # 构造一条先保持在 95 之上，在倒数第3根击穿 95.0 的序列
        dates = pd.bdate_range("2026-03-01", periods=20)
        # 前17根价格均在 96~105 之间，第18根(倒数第3根)击穿至93.0
        p_seq = [100.0, 98.0, 102.0, 104.0, 97.0, 98.0, 100.0, 105.0, 103.0, 101.0,
                 99.0, 98.0, 97.0, 99.0, 98.0, 97.0, 96.0, 93.0, 94.0, 94.5]
        df_sub = pd.DataFrame({
            "Open": p_seq, "High": [p + 0.5 for p in p_seq],
            "Low": [p - 0.5 for p in p_seq], "Close": p_seq,
            "Volume": [1e6] * 20
        }, index=dates)

        swings = [
            {"index": 0, "time": "2026-03-01", "type": "High", "price": 100.0, "dow": "High", "confirmed": True},
            {"index": 4, "time": "2026-03-05", "type": "Low", "price": 97.0, "dow": "Low", "confirmed": True},
            {"index": 7, "time": "2026-03-10", "type": "High", "price": 105.0, "dow": "HH", "confirmed": True},
            {"index": 12, "time": "2026-03-17", "type": "Low", "price": 95.0, "dow": "HL", "confirmed": True},
        ]

        events = pe.detect_mss(swings, df=df_sub)
        self.assertEqual(len(events), 1)
        # 触发时间应精准指向破位发生的那根 (-3 根)
        exp_time = str(df_sub.index[-3])[:16]
        self.assertEqual(events[0]["trigger_time"], exp_time)
        self.assertEqual(events[0]["bars_ago"], 2)


if __name__ == "__main__":
    unittest.main()
