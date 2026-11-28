"""Unit tests for okx_agent.reconcile and okx_agent.journal."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from okx_agent.journal import append_journal_entry, read_journal
from okx_agent.reconcile import (
    evaluate_realized_slippage,
    plan_breakeven_amend,
    plan_trailing_stop_amend,
    reconcile_state,
)
from risk_ledger import record_open


class TestOkxAgentReconcileAndJournal(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self._tmpdir.name) / "risk_ledger.json"
        self.journal_path = Path(self._tmpdir.name) / "okx_journal.jsonl"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_clean_reconciled_state_returns_ok(self):
        record_open("BTC-USDT-SWAP", 0.008, 100000.0, 99000.0, ledger_path=self.ledger_path)
        res = reconcile_state(
            positions=[{"instId": "BTC-USDT-SWAP", "pos": "8", "posSide": "net", "mgnMode": "isolated"}],
            open_orders=[],
            algo_orders=[
                {"instId": "BTC-USDT-SWAP", "algoId": "a1", "sz": "4", "slTriggerPx": "99000", "slTriggerPxType": "mark"},
                {"instId": "BTC-USDT-SWAP", "algoId": "a2", "sz": "4", "slTriggerPx": "99000", "slTriggerPxType": "mark"},
            ],
            ledger_path=self.ledger_path,
        )
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["actions"], [])
        self.assertEqual(res["warnings"], [])

    def test_asymmetric_partial_fill_and_non_mark_warning(self):
        record_open("BTC-USDT-SWAP", 0.008, 100000.0, 99000.0, ledger_path=self.ledger_path)
        res = reconcile_state(
            positions=[{"instId": "BTC-USDT-SWAP", "pos": "10", "posSide": "net", "mgnMode": "isolated"}],
            open_orders=[],
            algo_orders=[
                {"instId": "BTC-USDT-SWAP", "algoId": "a1", "sz": "5", "slTriggerPx": "99000", "slTriggerPxType": "last"},
            ],
            ledger_path=self.ledger_path,
        )
        self.assertEqual(res["status"], "ACTION_REQUIRED")
        self.assertEqual(len(res["warnings"]), 1)
        self.assertEqual(res["warnings"][0]["code"], "WARNING_SL_NOT_MARK_TRIGGERED")
        self.assertEqual(len(res["actions"]), 1)
        act = res["actions"][0]
        self.assertEqual(act["action"], "ACTION_PLACE_MISSING_SL")
        self.assertEqual(act["tool"], "swap_place_algo_order")
        self.assertEqual(act["params"]["sz"], "5")
        self.assertEqual(act["params"]["slTriggerPx"], "99000.0")
        self.assertEqual(act["params"]["slTriggerPxType"], "mark")
        self.assertTrue(act["params"]["reduceOnly"])

    def test_ledger_drift_detection(self):
        record_open("ETH-USDT-SWAP", 0.008, 3000.0, 2940.0, ledger_path=self.ledger_path)
        res = reconcile_state(
            positions=[{"instId": "BTC-USDT-SWAP", "pos": "4", "posSide": "net", "mgnMode": "isolated"}],
            open_orders=[],
            algo_orders=[{"instId": "BTC-USDT-SWAP", "algoId": "a1", "sz": "4", "slTriggerPx": "99000", "slTriggerPxType": "mark"}],
            ledger_path=self.ledger_path,
        )
        self.assertEqual(res["status"], "ACTION_REQUIRED")
        codes = {a["action"] for a in res["actions"]}
        self.assertIn("LEDGER_MISSING_LIVE_POSITION", codes)
        self.assertIn("LEDGER_STALE_OPEN_POSITION", codes)

    def test_plan_breakeven_amend_and_crossed_mark_fallback(self):
        safe_long = plan_breakeven_amend("BTC-USDT-SWAP", "buy", entry=100000.0, mark_px=101500.0, algo_id="a2")
        self.assertEqual(safe_long["action"], "AMEND_SL_TO_BREAKEVEN")
        self.assertEqual(safe_long["tool"], "swap_amend_algo_order")
        self.assertEqual(safe_long["params"]["newSlTriggerPx"], "100000.0")

        safe_short = plan_breakeven_amend("ETH-USDT-SWAP", "sell", entry=3000.0, mark_px=2920.0, algo_id="a3")
        self.assertEqual(safe_short["action"], "AMEND_SL_TO_BREAKEVEN")

        crossed = plan_breakeven_amend("BTC-USDT-SWAP", "buy", entry=100000.0, mark_px=99950.0, algo_id="a2")
        self.assertEqual(crossed["action"], "MARKET_CLOSE_REMAINING")
        self.assertEqual(crossed["tool"], "swap_close_position")

        # With fee buffer (10 bps), mark_px=100050 is below buffered BE 100100 -> market close
        buffered_crossed = plan_breakeven_amend(
            "BTC-USDT-SWAP", "buy", entry=100000.0, mark_px=100050.0, algo_id="a2", taker_fee_rate=0.0005
        )
        self.assertEqual(buffered_crossed["action"], "MARKET_CLOSE_REMAINING")
        self.assertEqual(buffered_crossed["breakeven_px"], 100100.0)

        buffered_safe = plan_breakeven_amend(
            "BTC-USDT-SWAP", "buy", entry=100000.0, mark_px=100250.0, algo_id="a2", taker_fee_rate=0.0005
        )
        self.assertEqual(buffered_safe["action"], "AMEND_SL_TO_BREAKEVEN")
        self.assertEqual(buffered_safe["params"]["newSlTriggerPx"], "100100.0")

    def test_p1_plan_trailing_stop_amend(self):
        # Tightening SL for long with >= 0.25R step succeeds
        ok = plan_trailing_stop_amend(
            "BTC-USDT-SWAP",
            "buy",
            current_sl=100100.0,
            candidate_sl=101500.0,
            mark_px=103000.0,
            algo_id="a2",
            entry=100000.0,
            initial_sl=99000.0,
        )
        self.assertEqual(ok["action"], "AMEND_TRAILING_SL")
        self.assertEqual(ok["params"]["newSlTriggerPx"], "101500.0")

        # Widening SL is rejected
        widen = plan_trailing_stop_amend(
            "BTC-USDT-SWAP", "buy", current_sl=100100.0, candidate_sl=99500.0, mark_px=103000.0, algo_id="a2"
        )
        self.assertEqual(widen["action"], "NO_TRAILING_AMEND")

        # Step too small (< 0.25R) is skipped
        tiny = plan_trailing_stop_amend(
            "BTC-USDT-SWAP",
            "buy",
            current_sl=100100.0,
            candidate_sl=100200.0,
            mark_px=103000.0,
            algo_id="a2",
            entry=100000.0,
            initial_sl=99000.0,
        )
        self.assertEqual(tiny["action"], "NO_TRAILING_AMEND")

    def test_evaluate_realized_slippage_and_liquidity_collapse(self):
        res_slip = evaluate_realized_slippage(entry=100.0, sl=98.0, exit_px=96.4, side="buy")
        self.assertAlmostEqual(res_slip["realized_r"], -1.8, places=4)
        self.assertTrue(res_slip["gap_slippage_breach"])
        self.assertTrue(res_slip["halt_instrument"])

        res_liq = evaluate_realized_slippage(
            entry=100.0, sl=98.0, exit_px=98.0, side="buy", top5_depth_usd=4000.0, notional_usd=5000.0
        )
        self.assertFalse(res_liq["gap_slippage_breach"])
        self.assertTrue(res_liq["liquidity_collapse"])
        self.assertTrue(res_liq["halt_instrument"])

    def test_journal_append_read_and_cli(self):
        append_journal_entry("SCAN_CONFLICT_HALT", "BTC-USDT-SWAP", {"reason": "htf vs mtf"}, journal_path=self.journal_path)
        append_journal_entry("PREFLIGHT_REJECTED", "ETH-USDT-SWAP", {"status": "REJECTED (< 2.5R)"}, journal_path=self.journal_path)
        all_rows = read_journal(journal_path=self.journal_path)
        self.assertEqual(len(all_rows), 2)
        btc_rows = read_journal(journal_path=self.journal_path, inst_id="btc-usdt-swap")
        self.assertEqual(len(btc_rows), 1)
        self.assertEqual(btc_rows[0]["event_type"], "SCAN_CONFLICT_HALT")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "okx_agent.journal",
                "--journal",
                str(self.journal_path),
                "tail",
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        tailed = json.loads(proc.stdout)
        self.assertEqual(len(tailed), 1)
        self.assertEqual(tailed[0]["inst_id"], "ETH-USDT-SWAP")


if __name__ == "__main__":
    unittest.main()
