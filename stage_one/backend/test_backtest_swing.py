# -*- coding: utf-8 -*-
"""
单元测试：日线波段零未来函数回测引擎验证
路径: stage_one/backend/test_backtest_swing.py
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd

from backtest_swing import (
    ROUND_TRIP_FRICTION,
    _check_invariants,
    scan_adapter_signals,
    sim_adapter_counter_trend,
    sim_adapter_direct,
    sim_adapter_playbook,
    sim_strategy_core_mss,
    summarize_trades,
)


def _make_trend_df(n: int = 120, bull: bool = True) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    t = np.linspace(0, 4 * np.pi, n)
    if bull:
        closes = np.linspace(100.0, 160.0, n) + 6.0 * np.sin(t)
    else:
        closes = np.linspace(160.0, 100.0, n) - 6.0 * np.sin(t)
    return pd.DataFrame(
        {
            "Open": closes - (0.4 if bull else -0.4),
            "High": closes + 2.0,
            "Low": closes - 2.0,
            "Close": closes,
            "Volume": np.linspace(1000, 3000, n),
        },
        index=dates,
    )


class TestBacktestSwing(unittest.TestCase):
    def test_zero_lookahead_signal_scan(self):
        """修改 t 之后的未来K线，严禁影响 <= t 时刻的信号扫描结果。"""
        df1 = _make_trend_df(90, bull=True)
        df2 = df1.copy()
        # 剧烈篡改最后 10 根未来K线
        df2.iloc[-10:, df2.columns.get_loc("Close")] *= 3.0
        df2.iloc[-10:, df2.columns.get_loc("High")] *= 3.0
        df2.iloc[-10:, df2.columns.get_loc("Low")] *= 3.0

        plans1, errs1 = scan_adapter_signals("BTC-USD", df1, "crypto", start_idx=60)
        plans2, errs2 = scan_adapter_signals("BTC-USD", df2, "crypto", start_idx=60)
        self.assertEqual(errs1, [])
        self.assertEqual(errs2, [])

        # 到第 78 根（篡改发生前）所有决策必须 100% 逐字段一致
        for t in range(60, 79):
            self.assertEqual(plans1[t], plans2[t], f"Bar {t} leaked future data!")

    def test_check_invariants_catches_inverted_levels(self):
        bad_long = {
            "direction_mode": "LONG",
            "passed_gates": True,
            "entry": 100.0,
            "stop_loss": 105.0,
            "tp1": 120.0,
        }
        self.assertIsNotNone(_check_invariants(bad_long, 100.0, 10))

        bad_ct = {
            "direction_mode": "SHORT",
            "passed_gates": False,
            "counter_trend_playbook": {
                "entry_zone": [95.0, 98.0],
                "stop_loss": 99.0,
                "tp1": 105.0,
                "tp2": 110.0,
            },
        }
        self.assertIsNotNone(_check_invariants(bad_ct, 96.0, 10))

        good_short = {
            "direction_mode": "SHORT",
            "passed_gates": True,
            "entry": 100.0,
            "stop_loss": 110.0,
            "tp1": 80.0,
        }
        self.assertIsNone(_check_invariants(good_short, 100.0, 10))

    def test_summarize_trades_metrics_and_friction(self):
        df = _make_trend_df(40, bull=True)
        trades = [
            {
                "entry_t": 5,
                "exit_t": 10,
                "entry_date": df.index[5],
                "exit_date": df.index[10],
                "entry_price": 100.0,
                "exit_price": 110.0,
                "side": 1,
                "pnl_pct": 0.10,
                "r_multiple": 2.0,
                "hold_bars": 5,
                "mae": -0.02,
                "mfe": 0.12,
                "exit_reason": "take_profit",
                "signal_source": "direct",
            },
            {
                "entry_t": 12,
                "exit_t": 16,
                "entry_date": df.index[12],
                "exit_date": df.index[16],
                "entry_price": 110.0,
                "exit_price": 104.5,
                "side": -1,
                "pnl_pct": -0.05,
                "r_multiple": -1.0,
                "hold_bars": 4,
                "mae": -0.06,
                "mfe": 0.01,
                "exit_reason": "stop_loss",
                "signal_source": "direct",
            },
        ]
        summary = summarize_trades(trades, df, start_idx=0, friction=ROUND_TRIP_FRICTION, is_crypto=True)
        stats = summary["all"]
        self.assertEqual(stats["trades"], 2)
        self.assertEqual(stats["long_trades"], 1)
        self.assertEqual(stats["short_trades"], 1)
        self.assertEqual(stats["win_rate_pct"], 50.0)
        # Net: (1 + 0.0992) * (1 - 0.0508) - 1 = 0.04336 -> ~4.34%
        self.assertAlmostEqual(stats["total_net_return_pct"], 4.34, delta=0.05)
        self.assertAlmostEqual(stats["max_drawdown_pct"], 5.08, delta=0.05)

    def test_synthetic_end_to_end_tracks(self):
        df = _make_trend_df(100, bull=True)
        plans, errs = scan_adapter_signals("NQmain", df, "futures", start_idx=60)
        self.assertEqual(errs, [])
        res_dir = sim_adapter_direct(df, plans, 60, is_crypto=False)
        res_pb = sim_adapter_playbook(df, plans, 60, is_crypto=False, max_wait_bars=15)
        res_core = sim_strategy_core_mss(df, 60, is_crypto=False)
        self.assertIn("all", res_dir)
        self.assertIn("all", res_pb)
        self.assertIn("all", res_core)

    def test_counter_trend_causal_non_overlapping_and_open_end_bar(self):
        """验证 counter_trend GTC 挂单不重叠回溯，且末根 (n-1) 未平仓入场不计入已完成交易。"""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 8 + [95.0, 90.0],
                "High": [101.0] * 8 + [96.0, 91.0],
                "Low": [99.0] * 8 + [94.0, 89.5],
                "Close": [100.0] * 8 + [95.0, 90.0],
                "Volume": [1000.0] * 10,
            },
            index=dates,
        )
        # Bar 1 挂 96 限价单 (1..6 未成交 -> expired at 6 -> next_free_t = 7)
        # Bar 2 挂 99 限价单若无防重叠则会抢跑成交; Bar 8 挂 90 限价单在 Bar 9 (n-1) 成交
        plans = {
            1: {
                "direction_mode": "SHORT",
                "passed_gates": False,
                "counter_trend_playbook": {
                    "mode": "ACTIVE",
                    "entry_zone": [95.0, 96.0],
                    "stop_loss": 92.0,
                    "tp1": 102.0,
                    "tp2": 105.0,
                },
            },
            2: {
                "direction_mode": "SHORT",
                "passed_gates": False,
                "counter_trend_playbook": {
                    "mode": "ACTIVE",
                    "entry_zone": [98.0, 99.5],
                    "stop_loss": 92.0,
                    "tp1": 106.0,
                    "tp2": 110.0,
                },
            },
            8: {
                "direction_mode": "SHORT",
                "passed_gates": False,
                "counter_trend_playbook": {
                    "mode": "ACTIVE",
                    "entry_zone": [89.0, 90.5],
                    "stop_loss": 85.0,
                    "tp1": 96.0,
                    "tp2": 100.0,
                },
            },
        }
        out = sim_adapter_counter_trend(df, plans, start_idx=1, is_crypto=True, max_wait_bars=5)
        # Bar 2 不能因为 Bar 1 未来5根未成交而被回溯激活; Bar 8 在末根 n-1 成交应归入 open_at_end
        self.assertEqual(out["orders_placed"], 2)
        self.assertEqual(out["all"]["trades"], 0)
        self.assertEqual(len(out["open_at_end"]), 1)
        self.assertEqual(out["open_at_end"][0]["entry_date"], "2026-01-10")

    def test_crypto_daily_uses_utc_and_patch_does_not_duplicate(self):
        """验证 YahooProvider 加密日线按 UTC 解析日期，且 _patch_latest_bar_with_quote 不重复追加同日K线。"""
        from datetime import datetime, timezone
        from data_provider import YahooProvider
        from main import _patch_latest_bar_with_quote

        # 2026-11-20 00:00:00 UTC (对应美东 2026-11-19 19:00)
        ts_nov20_utc = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        ts_nov19_utc = ts_nov20_utc - 86400

        yp = YahooProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": [ts_nov19_utc, ts_nov20_utc],
                    "indicators": {
                        "quote": [{
                            "open": [62987.0, 61904.0],
                            "high": [64017.0, 62437.0],
                            "low": [61380.0, 60037.0],
                            "close": [61901.0, 60037.0],
                            "volume": [33000000000.0, 45000000000.0],
                        }]
                    },
                }]
            }
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("data_provider.get_proxy_client", return_value=mock_client):
            df = asyncio.run(yp._fetch_yahoo_direct("BTC-USD", "1d", False))

        self.assertEqual(list(df["date"]), ["2026-11-19", "2026-11-20"])

        # 模拟实时报价在美东 2026-11-20 10:00 (UTC 2026-11-20 15:00) 校准，不应新增重复行
        fake_quote = [{
            "symbol": "BTC-USD",
            "status": "ok",
            "price": 59950.0,
            "open": 61904.0,
            "high": 62437.0,
            "low": 59900.0,
            "update_time": "2026-11-20 10:00:00",
        }]
        with patch("main.quote_provider.get_quotes", new=AsyncMock(return_value=fake_quote)):
            patched = asyncio.run(_patch_latest_bar_with_quote("BTC-USD", "1d", df.copy()))

        self.assertEqual(len(patched), 2)
        self.assertEqual(list(patched["date"]), ["2026-11-19", "2026-11-20"])
        self.assertEqual(float(patched.iloc[-1]["close"]), 59950.0)

    def test_futures_1d_prefers_yahoo_with_volume_and_falls_back_to_sina(self):
        """验证期货 1d 优先 Yahoo 获取真实成交量，Yahoo 为空时回退新浪。"""
        from data_provider import DataProviderManager

        mgr = DataProviderManager()
        dates = pd.date_range("2026-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        y_df = pd.DataFrame({
            "date": dates,
            "open": [26000.0] * 10,
            "high": [26200.0] * 10,
            "low": [25900.0] * 10,
            "close": [26100.0] * 10,
            "volume": [500000.0] * 10,
        })
        s_df = y_df.copy()
        s_df["volume"] = 0.0

        mgr.yahoo.fetch_klines = AsyncMock(return_value=y_df)
        mgr.sina.fetch_klines = AsyncMock(return_value=s_df)
        res = asyncio.run(mgr.fetch_klines("NQmain", "1d", bypass_cache=True))
        self.assertEqual(mgr.last_source, "yahoo")
        self.assertEqual(float(res["volume"].iloc[-1]), 500000.0)
        mgr.sina.fetch_klines.assert_not_called()

        mgr.yahoo.fetch_klines = AsyncMock(return_value=pd.DataFrame())
        res_fb = asyncio.run(mgr.fetch_klines("NQmain", "1d", bypass_cache=True))
        self.assertEqual(mgr.last_source, "sina")
        self.assertEqual(len(res_fb), 10)


if __name__ == "__main__":
    unittest.main()
