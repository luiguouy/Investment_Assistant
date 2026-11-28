"""Unit tests for okx_agent.report and directional beta exposure cap."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from okx_agent.journal import append_journal_entry
from okx_agent.report import generate_performance_report
from risk_ledger import check_pre_trade_risk, record_close, record_open


class TestOkxAgentReportAndDirectionalCap(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self._tmpdir.name) / "risk_ledger.json"
        self.journal_path = Path(self._tmpdir.name) / "okx_journal.jsonl"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_performance_report_metrics_and_cli(self):
        record_open("BTC-USDT-SWAP", 0.01, 100000.0, 99000.0, side="buy", ledger_path=self.ledger_path)
        record_close("BTC-USDT-SWAP", 3.0, 0.03, ledger_path=self.ledger_path)

        record_open("ETH-USDT-SWAP", 0.01, 3000.0, 3060.0, side="sell", ledger_path=self.ledger_path)
        record_close("ETH-USDT-SWAP", -1.0, -0.01, ledger_path=self.ledger_path)

        record_open("BTC-USDT-SWAP", 0.01, 101000.0, 100000.0, side="buy", ledger_path=self.ledger_path)
        record_close("BTC-USDT-SWAP", 2.0, 0.02, ledger_path=self.ledger_path)

        append_journal_entry("SCAN_CONFLICT_HALT", "GOOGL-USDT-SWAP", {"reason": "htf vs mtf"}, journal_path=self.journal_path)
        append_journal_entry(
            "POSITION_CLOSED",
            "BTC-USDT-SWAP",
            {"planned_r": 2.5, "realized_r": 2.0},
            journal_path=self.journal_path,
        )

        rep = generate_performance_report(ledger_path=self.ledger_path, journal_path=self.journal_path)
        s = rep["summary"]
        self.assertEqual(s["total_trades"], 3)
        self.assertEqual(s["wins"], 2)
        self.assertEqual(s["losses"], 1)
        self.assertAlmostEqual(s["win_rate"], 0.6667, places=4)
        self.assertAlmostEqual(s["expectancy_r"], 1.3333, places=4)
        self.assertAlmostEqual(s["profit_factor"], 5.0, places=4)
        self.assertAlmostEqual(s["cumulative_r"], 4.0, places=4)
        self.assertAlmostEqual(s["max_drawdown_r"], -1.0, places=4)
        self.assertEqual(rep["by_symbol"]["BTC-USDT-SWAP"]["trades"], 2)
        self.assertEqual(rep["journal_audit"]["total_events"], 2)
        self.assertAlmostEqual(rep["journal_audit"]["avg_r_slippage_vs_plan"], -0.5, places=4)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "okx_agent.report",
                "--ledger",
                str(self.ledger_path),
                "--journal",
                str(self.journal_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cli_out = json.loads(proc.stdout)
        self.assertEqual(cli_out["summary"]["total_trades"], 3)

    def test_directional_bucket_and_net_beta_cap(self):
        # Two long crypto_majors block a 3rd long in same bucket, but allow an opposing short hedge
        record_open("BTC-USDT-SWAP", 0.008, 100000.0, 99000.0, side="buy", ledger_path=self.ledger_path)
        record_open("ETH-USDT-SWAP", 0.008, 3000.0, 2940.0, side="buy", ledger_path=self.ledger_path)

        same_dir = check_pre_trade_risk("SOL-USDT-SWAP", 0.008, side="buy", ledger_path=self.ledger_path)
        self.assertFalse(same_dir["passed"])
        self.assertEqual(same_dir["reason"], "CORRELATION_BUCKET_CAP_EXCEEDED")

        hedge_dir = check_pre_trade_risk("SOL-USDT-SWAP", 0.008, side="sell", ledger_path=self.ledger_path)
        self.assertTrue(hedge_dir["passed"], hedge_dir)

        # Cross-bucket net directional risk cap (0.025): 0.01 + 0.01 + 0.008 = 0.028 > 0.025
        ledger2 = Path(self._tmpdir.name) / "risk_ledger2.json"
        record_open("BTC-USDT-SWAP", 0.01, 100000.0, 99000.0, side="buy", ledger_path=ledger2)
        record_open("NVDA-USDT-SWAP", 0.01, 140.0, 136.0, side="buy", ledger_path=ledger2)
        net_cap = check_pre_trade_risk("AAPL-USDT-SWAP", 0.008, side="buy", ledger_path=ledger2)
        self.assertFalse(net_cap["passed"])
        self.assertEqual(net_cap["reason"], "NET_DIRECTIONAL_RISK_CAP_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
