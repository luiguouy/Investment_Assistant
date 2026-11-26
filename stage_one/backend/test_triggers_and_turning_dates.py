# -*- coding: utf-8 -*-
"""
验证 buy_triggers、sell_triggers、turning_dates 与空头方向逻辑
路径: stage_one/backend/test_triggers_and_turning_dates.py
"""

import unittest
import pandas as pd
import numpy as np
from strategy_adapter import run_strategy_analysis


class TestTriggersAndTurningDates(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        # 构造明显的空头下行通道与跌破 200 EMA 数据
        # 从 380 跌至 320
        downtrend = np.linspace(380, 320, 100)
        # 加上小幅震荡形成波浪次级高低点
        downtrend += np.sin(np.linspace(0, 20, 100)) * 5.0
        self.df_bear = pd.DataFrame(
            {
                "Open": downtrend + 1.0,
                "High": downtrend + 3.0,
                "Low": downtrend - 3.0,
                "Close": downtrend,
                "Volume": [1500] * 100,
            },
            index=dates,
        )

        # 构造明显的多头上升通道数据
        uptrend = np.linspace(300, 380, 100) + np.sin(np.linspace(0, 20, 100)) * 5.0
        self.df_bull = pd.DataFrame(
            {
                "Open": uptrend - 1.0,
                "High": uptrend + 3.0,
                "Low": uptrend - 3.0,
                "Close": uptrend,
                "Volume": [2500] * 100,
            },
            index=dates,
        )

    def test_bear_market_triggers_and_direction(self):
        """测试空头市场下 buy_triggers 与 sell_triggers 完整存在且 direction_mode 为 SHORT"""
        res = run_strategy_analysis("usTSLA", period="1d", mode="swing", df=self.df_bear)
        summary = res["summary"]
        plan = summary["trade_plan"]

        # 1. 验证 direction_mode 严格映射
        self.assertEqual(plan.get("direction_mode"), "SHORT")

        # 2. 即使处于观望/0R门禁，也必须有 buy_triggers
        buy_trig = summary.get("buy_triggers")
        self.assertIsNotNone(buy_trig, "空头趋势下必须暴露 buy_triggers")
        self.assertIn("aggressive", buy_trig)
        self.assertIn("conservative", buy_trig)
        self.assertIn("price_range", buy_trig["aggressive"])
        self.assertIn("price_range", buy_trig["conservative"])
        self.assertIn("condition", buy_trig["aggressive"])
        self.assertIn("condition", buy_trig["conservative"])

        # 3. 顺势卖出点 sell_triggers 必须完整
        sell_trig = summary.get("sell_triggers")
        self.assertIsNotNone(sell_trig, "空头趋势下必须暴露 sell_triggers")
        self.assertIn("rebound_sell", sell_trig)
        self.assertIn("take_profit", sell_trig)
        self.assertIn("breakdown_stop", sell_trig)
        self.assertIn("resistance_zone_1", sell_trig["rebound_sell"])
        self.assertIn("resistance_zone_2", sell_trig["rebound_sell"])

        # 4. 验证变盘日期推演 turning_dates
        turning_dates = summary.get("turning_dates")
        self.assertIsNotNone(turning_dates)
        self.assertIsInstance(turning_dates, list)
        if len(turning_dates) > 0:
            td = turning_dates[0]
            self.assertIn("date", td)
            self.assertIn("day_offset_text", td)
            self.assertIn("basis", td)
            self.assertIn("focus", td)

        # 5. 图表 swing 标注严禁在空头波段标注 HL
        markers = res["annotations"].get("markers", [])
        for m in markers:
            if m.get("id", "").startswith("swing_"):
                self.assertNotEqual(m.get("text"), "HL", "空头波段中严禁错误标注多头符号 HL")

    def test_bull_market_triggers(self):
        """测试多头市场下 triggers 正常输出且 direction_mode 正确映射"""
        # 测试不受美股大盘过滤器影响的加密资产
        res_crypto = run_strategy_analysis("BTC-USD", period="1d", mode="swing", df=self.df_bull)
        plan_c = res_crypto["summary"]["trade_plan"]
        self.assertEqual(plan_c.get("direction_mode"), "LONG")
        self.assertIsNotNone(res_crypto["summary"].get("buy_triggers"))
        self.assertIsNotNone(res_crypto["summary"].get("sell_triggers"))

        # 测试美股资产（当大盘空头阻断时映射为 WAIT，大盘顺风时为 LONG）
        res_us = run_strategy_analysis("usAAPL", period="1d", mode="swing", df=self.df_bull)
        plan_us = res_us["summary"]["trade_plan"]
        self.assertIn(plan_us.get("direction_mode"), ("LONG", "WAIT"))
        self.assertIsNotNone(res_us["summary"].get("buy_triggers"))
        self.assertIsNotNone(res_us["summary"].get("sell_triggers"))

    def test_waterfall_bear_anchor_invariants_and_dual_playbook(self):
        """验证单边暴跌下活跃锚点方向不倒挂、无美股文案串台、空头诊断措辞正确及双剧本并行"""
        # 构造前段震荡 + 末尾 20 根单边暴跌的极端空头数据（未形成新的 confirmed 波谷）
        dates = pd.date_range("2026-01-01", periods=140, freq="D")
        prices = [100.0 + 4.0 * np.sin(i / 4.0) for i in range(110)]
        # 末尾 30 天单边下跌至 ~60
        for j in range(30):
            prices.append(102.0 - j * 1.45)
        prices_arr = np.array(prices)
        df_crash = pd.DataFrame({
            "Open": prices_arr + 0.6,
            "High": prices_arr + 1.2,
            "Low": prices_arr - 1.2,
            "Close": prices_arr,
            "Volume": np.linspace(1000, 3000, 140),
        }, index=dates)

        # 1. 加密货币 (BTC-USD)
        res_c = run_strategy_analysis("BTC-USD", period="1d", mode="swing", df=df_crash, market="crypto")
        summary_c = res_c["summary"]
        cur_p = float(res_c["current_price"])
        bt_c = summary_c["buy_triggers"]
        st_c = summary_c["sell_triggers"]

        # 强方向性校验：空单止盈与多单清仓线必低于现价；空单止损必高于现价；多单稳健目标必高于入场价
        self.assertLess(st_c["take_profit"]["tp1"]["price"], cur_p, "空单下探止盈 TP1 必须低于现价")
        self.assertLess(st_c["take_profit"]["tp2"]["price"], st_c["take_profit"]["tp1"]["price"], "空单 TP2 必须低于 TP1")
        self.assertLess(st_c["breakdown_stop"]["long_clear_line"]["price"], cur_p, "多单无条件清仓线必须低于现价")
        self.assertGreater(st_c["breakdown_stop"]["short_stop_line"]["price"], cur_p, "空单突破止损线必须高于现价")
        self.assertGreater(
            bt_c["conservative"]["target"],
            bt_c["conservative"]["price_range"][1],
            "稳健做多目标位必须高于入场区间上沿",
        )
        self.assertNotIn("SPY", bt_c["conservative"]["condition"], "加密货币买点文案不得包含美股 SPY")

        # 空头深度诊断文案极性校验
        diag_c = summary_c.get("depth_diagnosis", "")
        self.assertNotIn("冲顶高抛止盈", diag_c, "空头模式下不得输出多头'冲顶高抛止盈'文案")
        self.assertIn("空头梯次目标", diag_c)

        # 双剧本并行：超跌共振必须激活 counter_trend_playbook (ACTIVE)
        ct_c = summary_c.get("counter_trend_playbook")
        self.assertIsNotNone(ct_c, "极度超卖单边暴跌下加密货币必须生成超跌反弹做多剧本")
        self.assertEqual(ct_c["mode"], "ACTIVE")
        self.assertLess(ct_c["stop_loss"], ct_c["entry_zone"][0])
        self.assertLessEqual(ct_c["entry_zone"][0], ct_c["entry_zone"][1])
        self.assertGreater(ct_c["tp1"], ct_c["entry_zone"][1])
        self.assertGreater(ct_c["tp2"], ct_c["tp1"])

        # 2. 美股 (usNVDA) 在同样暴跌下仅作为 OBSERVE_ONLY 高风险观察提示，且文案允许大盘共振
        res_us = run_strategy_analysis("usNVDA", period="1d", mode="swing", df=df_crash, market="us")
        ct_us = res_us["summary"].get("counter_trend_playbook")
        self.assertIsNotNone(ct_us)
        self.assertEqual(ct_us["mode"], "OBSERVE_ONLY")


if __name__ == "__main__":
    unittest.main()
