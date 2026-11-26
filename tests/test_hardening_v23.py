"""Regression tests for v2.3 hardening: Risk Ledger, >=2.5R net weighted RR, Fail-Closed gates, and MSS."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import pa_engine as pe
import risk_ledger as rl
import strategy_crypto as sc
import strategy_day as sd
import strategy_layer as sl


class TestHardeningV23(unittest.TestCase):
    def test_min_rr_unified_at_2_5(self):
        self.assertEqual(pe.MIN_RR, 2.5)
        self.assertEqual(sl.DEFAULT_MIN_RR, 2.5)
        self.assertEqual(sc.CRYPTO_MIN_RR, 2.5)

    def test_detect_mss_loop_catches_earlier_transition(self):
        sw = [
            {"time": "2026-01-01", "price": 100.0, "type": "Low", "dow": "HL"},
            {"time": "2026-01-02", "price": 110.0, "type": "High", "dow": "HH"},
            {"time": "2026-01-03", "price": 108.0, "type": "High", "dow": "LH"},
            {"time": "2026-01-04", "price": 95.0, "type": "Low", "dow": "LL"},
            {"time": "2026-01-05", "price": 98.0, "type": "High", "dow": "LH"},
        ]
        ev = pe.detect_mss(sw, df=None)
        self.assertEqual(len(ev), 1)
        self.assertIn("Bearish MSS", ev[0]["type"])

    def test_market_dow_gate_fail_closed_on_empty_dict(self):
        res = sl.verify_market_dow_gate(
            {"SPY": {"htf": {"phase": ""}, "mtf": {"phase": "", "swings": []}}},
            side=1,
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "rejected_by_missing_market_data")

    def test_build_trade_plan_v23_okx_swap_contract_sizing_and_net_rr(self):
        plan = pe.build_trade_plan(
            mode="swing",
            price=100.0,
            swings_mtf=[
                {"time": "t1", "price": 96.0, "type": "Low", "dow": "HL", "confirmed": True},
                {"time": "t2", "price": 105.0, "type": "High", "dow": "HH", "confirmed": True},
            ],
            swings_ltf=[
                {"time": "t1", "price": 96.0, "type": "Low", "dow": "HL", "confirmed": True},
                {"time": "t2", "price": 102.0, "type": "High", "dow": "HH", "confirmed": True},
            ],
            key_levels=[
                {"price": 115.0, "kind": "swing", "src": "test"},
                {"price": 125.0, "kind": "swing", "src": "far"},
            ],
            levels={},
            bar_rng=1.0,
            bias_htf=1,
            bias_mtf=1,
            use_trend_gate=False,
            venue="okx_swap",
            tp1_weight=0.6,
            ct_val=0.01,
            lot_sz=0.01,
            equity=10000.0,
            risk_pct=0.01,
        )
        self.assertEqual(plan["schema_version"], "2.3")
        self.assertEqual(plan["friction_bps"], 10.0)
        self.assertGreater(plan["gross_weighted_rr"], plan["weighted_rr"])
        self.assertGreater(plan["contracts"], 0.0)

    def test_intraday_htf_bias_timezone_and_no_sma_fallback(self):
        idx = pd.date_range("2026-11-20 09:30", periods=30, freq="5min")
        df5 = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5}, index=idx)
        bias = sd.extract_intraday_htf_bias(
            df5, cutoff_ts=pd.Timestamp("2026-11-20 11:30", tz="America/New_York")
        )
        self.assertEqual(bias, 0)

    def test_risk_ledger_caps_and_circuit_breakers(self):
        with tempfile.TemporaryDirectory() as td:
            lp = Path(td) / "risk_ledger.json"
            self.assertFalse(rl.check_pre_trade_risk("AAPL-USDT-SWAP", 0.02, lp)["passed"])

            rl.record_open("AAPL-USDT-SWAP", 0.01, 350.0, 355.0, ledger_path=lp)
            rl.record_open("NVDA-USDT-SWAP", 0.01, 150.0, 155.0, ledger_path=lp)
            self.assertEqual(
                rl.check_pre_trade_risk("QQQ", 0.01, lp)["reason"],
                "CORRELATION_BUCKET_CAP_EXCEEDED",
            )

            rl.record_open("BTC-USDT-SWAP", 0.01, 60000.0, 61000.0, ledger_path=lp)
            self.assertEqual(
                rl.check_pre_trade_risk("ETH-USDT-SWAP", 0.005, lp)["reason"],
                "TOTAL_OPEN_RISK_CAP_EXCEEDED",
            )

            rl.record_close("BTC-USDT-SWAP", -1.0, -0.01, ledger_path=lp)
            rl.record_close("AAPL-USDT-SWAP", -1.0, -0.01, ledger_path=lp)
            self.assertEqual(
                rl.check_pre_trade_risk("ETH-USDT-SWAP", 0.005, lp)["reason"],
                "DAILY_LOSS_CIRCUIT_BREAKER",
            )

            rl.record_close("NVDA-USDT-SWAP", -1.0, -0.005, ledger_path=lp)
            self.assertEqual(
                rl.check_pre_trade_risk("ETH-USDT-SWAP", 0.005, lp)["reason"],
                "CONSECUTIVE_LOSS_COOLDOWN",
            )


if __name__ == "__main__":
    unittest.main()
