"""Unit tests for okx_agent.watcher."""

import unittest

from okx_agent.watcher import evaluate_price_alerts, run_watcher


class TestOkxAgentWatcher(unittest.TestCase):
    def test_long_and_short_alert_zones(self):
        # Long: LIMIT_ENTRY, TP1_BE, SL_NEAR
        r_entry = evaluate_price_alerts(100150.0, "buy", entry=100000.0, tp1=103000.0, sl=99000.0, watch_entry=True)
        self.assertTrue(r_entry["triggered"])
        self.assertEqual(r_entry["event"], "LIMIT_ENTRY")

        r_tp1 = evaluate_price_alerts(102850.0, "buy", entry=100000.0, tp1=103000.0, sl=99000.0)
        self.assertTrue(r_tp1["triggered"])
        self.assertEqual(r_tp1["event"], "TP1_BE")

        r_sl = evaluate_price_alerts(99150.0, "buy", entry=100000.0, tp1=103000.0, sl=99000.0)
        self.assertTrue(r_sl["triggered"])
        self.assertEqual(r_sl["event"], "SL_NEAR")

        # Short: LIMIT_ENTRY, TP1_BE, SL_NEAR
        s_entry = evaluate_price_alerts(2996.0, "sell", entry=3000.0, tp1=2850.0, sl=3060.0, watch_entry=True)
        self.assertTrue(s_entry["triggered"])
        self.assertEqual(s_entry["event"], "LIMIT_ENTRY")

        s_tp1 = evaluate_price_alerts(2854.0, "sell", entry=3000.0, tp1=2850.0, sl=3060.0)
        self.assertTrue(s_tp1["triggered"])
        self.assertEqual(s_tp1["event"], "TP1_BE")

        s_sl = evaluate_price_alerts(3055.0, "sell", entry=3000.0, tp1=2850.0, sl=3060.0)
        self.assertTrue(s_sl["triggered"])
        self.assertEqual(s_sl["event"], "SL_NEAR")

    def test_hysteresis_prevents_ping_pong_until_rearmed(self):
        # Start disarmed inside TP1 zone (tp1=100, buffer=0.2% -> 99.8, clear=0.3% -> 99.7)
        prices = iter([99.85, 99.75, 99.60, 99.85])
        res = run_watcher(
            inst_id="SOL-USDT-SWAP",
            side="buy",
            tp1=100.0,
            sl=95.0,
            buffer_pct=0.002,
            hysteresis_pct=0.001,
            start_armed=False,
            interval_sec=0.0,
            price_fetcher=lambda _s: next(prices),
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res["event"], "TP1_BE")
        self.assertEqual(res["exit_code"], 0)
        # Must wait for 99.60 (< 99.70) to re-arm, then fire on 99.85 (poll #4)
        self.assertEqual(res["polls"], 4)

    def test_consecutive_error_circuit_breaker_and_recovery(self):
        def always_fail(_s):
            raise ConnectionError("SSL handshake timeout")

        res_trip = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            tp1=103000.0,
            sl=99000.0,
            max_consecutive_errors=5,
            interval_sec=0.0,
            price_fetcher=always_fail,
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res_trip["event"], "WATCHER_ERROR_CIRCUIT_BREAKER")
        self.assertEqual(res_trip["exit_code"], 3)
        self.assertEqual(res_trip["consecutive_errors"], 5)

        # Transient 2 errors followed by recovery and trigger should NOT trip circuit breaker
        seq = iter([ConnectionError("t1"), ConnectionError("t2"), 100500.0, 102900.0])

        def flaky_fetcher(_s):
            val = next(seq)
            if isinstance(val, Exception):
                raise val
            return val

        res_ok = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            tp1=103000.0,
            sl=99000.0,
            max_consecutive_errors=5,
            interval_sec=0.0,
            price_fetcher=flaky_fetcher,
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res_ok["event"], "TP1_BE")
        self.assertEqual(res_ok["exit_code"], 0)
        self.assertEqual(res_ok["polls"], 4)

    def test_max_polls_reached(self):
        res = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            tp1=103000.0,
            sl=99000.0,
            max_polls=3,
            interval_sec=0.0,
            price_fetcher=lambda _s: 100500.0,
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res["event"], "MAX_POLLS_REACHED")
        self.assertEqual(res["exit_code"], 4)
        self.assertEqual(res["polls"], 3)

    def test_p1_wick_detection_trail_milestone_and_time_stop(self):
        # 1m candle high wicked past TP1 even though last pulled back -> still triggers TP1_BE
        res_wick = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            tp1=103000.0,
            sl=99000.0,
            interval_sec=0.0,
            price_fetcher=lambda _s: {"last": 102100.0, "high": 103050.0, "low": 102000.0},
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res_wick["event"], "TP1_BE")
        self.assertEqual(res_wick["exit_code"], 0)

        # Trail milestone wakeup
        res_trail = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            sl=100100.0,
            trail_milestone_px=104000.0,
            interval_sec=0.0,
            price_fetcher=lambda _s: 104050.0,
            sleep_fn=lambda _t: None,
        )
        self.assertEqual(res_trail["event"], "TRAIL_MILESTONE")

        # Time stop review after max_hold_sec
        ticks = iter([0.0, 100.0, 4000.0])
        res_time = run_watcher(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            tp1=103000.0,
            sl=99000.0,
            max_hold_sec=3600.0,
            interval_sec=0.0,
            price_fetcher=lambda _s: 100500.0,
            sleep_fn=lambda _t: None,
            clock_fn=lambda: next(ticks),
        )
        self.assertEqual(res_time["event"], "TIME_STOP_REVIEW")
        self.assertEqual(res_time["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
