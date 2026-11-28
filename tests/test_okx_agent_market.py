"""Unit tests for okx_agent.market and four-way consistency gate."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from okx_agent.market import (
    analyze_okx_market,
    check_four_way_consistency,
    parse_okx_candles,
)


def _make_okx_rows(
    n: int = 120,
    start_ms: int = 1700000000000,
    step_ms: int = 3600000,
    trend: str = "up",
    base: float = 100.0,
) -> list:
    """Build synthetic OKX candle rows [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm] newest-first."""
    rows = []
    prev_c = base
    for i in range(n):
        ts = start_ms + i * step_ms
        if trend == "up":
            drift = 0.35 * i + 2.5 * math.sin(i / 4.0)
        elif trend == "down":
            drift = -0.35 * i + 2.5 * math.sin(i / 4.0)
        else:
            drift = 0.05 * math.sin(i / 4.0)
        c = max(10.0, base + drift)
        o = prev_c if i > 0 else c - 0.1
        h = max(o, c) + 0.4
        l = min(o, c) - 0.4
        rows.append([
            str(ts),
            f"{o:.4f}",
            f"{h:.4f}",
            f"{l:.4f}",
            f"{c:.4f}",
            "1000",
            "10",
            "100000",
            "1",
        ])
        prev_c = c
    rows.reverse()  # OKX returns newest-first
    return rows


class TestOkxAgentMarket(unittest.TestCase):
    def test_parse_okx_candles_sorts_drops_unconfirmed_and_validates(self):
        rows = _make_okx_rows(n=10)
        # Prepend an unconfirmed newest bar (confirm == "0")
        unconfirmed = [
            str(int(rows[0][0]) + 3600000),
            "200.0",
            "205.0",
            "199.0",
            "204.0",
            "500",
            "5",
            "1000",
            "0",
        ]
        df = parse_okx_candles([unconfirmed] + rows, drop_unconfirmed=True)
        self.assertEqual(len(df), 10)
        self.assertTrue(df.index.is_monotonic_increasing)

        bad_row = [["1700000000000", "100", "99", "98", "100", "10", "1", "100", "1"]]
        with self.assertRaises(ValueError):
            parse_okx_candles(bad_row)

    def test_aligned_bullish_and_bearish_trends_allow(self):
        htf_up = parse_okx_candles(_make_okx_rows(n=120, step_ms=14400000, trend="up", base=100.0))
        mtf_up = parse_okx_candles(_make_okx_rows(n=120, step_ms=3600000, trend="up", base=100.0))
        res_up = analyze_okx_market("BTC-USDT-SWAP", htf_up, mtf_up)
        self.assertEqual(res_up["verdict"], "ALLOW", res_up["conflicts"])
        self.assertEqual(res_up["allowed_direction"], "LONG")
        self.assertEqual(res_up["conflicts"], [])

        htf_dn = parse_okx_candles(_make_okx_rows(n=120, step_ms=14400000, trend="down", base=200.0))
        mtf_dn = parse_okx_candles(_make_okx_rows(n=120, step_ms=3600000, trend="down", base=200.0))
        res_dn = analyze_okx_market("ETH-USDT-SWAP", htf_dn, mtf_dn)
        self.assertEqual(res_dn["verdict"], "ALLOW", res_dn["conflicts"])
        self.assertEqual(res_dn["allowed_direction"], "SHORT")
        self.assertEqual(res_dn["conflicts"], [])

    def test_four_way_consistency_conflict_halt(self):
        # 1. dow_bias=+1 vs mss_dir=-1
        c1 = check_four_way_consistency(
            {"dow_bias": 1},
            {"dow_bias": 1, "mss_dir": -1, "channel_dir": 1},
        )
        self.assertEqual(c1["verdict"], "CONFLICT_HALT")
        self.assertEqual(c1["allowed_direction"], "NONE")

        # 2. htf_bias=+1 vs mtf_bias=-1
        c2 = check_four_way_consistency(
            {"dow_bias": 1},
            {"dow_bias": -1, "mss_dir": -1, "channel_dir": -1},
        )
        self.assertEqual(c2["verdict"], "CONFLICT_HALT")

        # 3. dow_bias=+1 vs channel_dir=-1
        c3 = check_four_way_consistency(
            {"dow_bias": 1},
            {"dow_bias": 1, "mss_dir": 1, "channel_dir": -1},
        )
        self.assertEqual(c3["verdict"], "CONFLICT_HALT")

    def test_stock_token_external_pa_conflict(self):
        htf_up = parse_okx_candles(_make_okx_rows(n=120, step_ms=14400000, trend="up", base=200.0))
        mtf_up = parse_okx_candles(_make_okx_rows(n=120, step_ms=3600000, trend="up", base=200.0))
        res = analyze_okx_market("AAPL-USDT-SWAP", htf_up, mtf_up, external_pa_bias=-1)
        self.assertEqual(res["verdict"], "CONFLICT_HALT")
        self.assertEqual(res["allowed_direction"], "NONE")
        self.assertIn("EXTERNAL_PA_ENGINE_CONFLICT", {c["code"] for c in res["conflicts"]})

        # Missing US stock confirmation on stock_token must HALT (no silent pass)
        res_missing = analyze_okx_market("AAPL-USDT-SWAP", htf_up, mtf_up, external_pa_bias=None)
        self.assertEqual(res_missing["verdict"], "CONFLICT_HALT")
        self.assertIn("MISSING_US_STOCK_CONFIRMATION", {c["code"] for c in res_missing["conflicts"]})

        # Aligned US stock confirmation + normal basis allows LONG
        okx_last = float(mtf_up["Close"].iloc[-1])
        res_ok = analyze_okx_market(
            "AAPL-USDT-SWAP",
            htf_up,
            mtf_up,
            us_stock_ref={"available": True, "ticker": "AAPL", "bias": 1, "last_close": okx_last * 0.995},
        )
        self.assertEqual(res_ok["verdict"], "ALLOW", res_ok["conflicts"])
        self.assertEqual(res_ok["allowed_direction"], "LONG")

        # Severe off-hours basis dislocation (>3%) halts even if direction aligns
        res_basis = analyze_okx_market(
            "AAPL-USDT-SWAP",
            htf_up,
            mtf_up,
            us_stock_ref={"available": True, "ticker": "AAPL", "bias": 1, "last_close": okx_last * 0.94},
        )
        self.assertEqual(res_basis["verdict"], "CONFLICT_HALT")
        self.assertIn("US_STOCK_BASIS_DISLOCATION", {c["code"] for c in res_basis["conflicts"]})

    def test_neutral_wait_when_htf_and_mtf_neutral(self):
        res = check_four_way_consistency(
            {"dow_bias": 0},
            {"dow_bias": 0, "mss_dir": 0, "channel_dir": 0},
        )
        self.assertEqual(res["verdict"], "NEUTRAL_WAIT")
        self.assertEqual(res["allowed_direction"], "NONE")

    def test_cli_offline_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "market_input.json"
            htf_rows = _make_okx_rows(n=120, step_ms=14400000, trend="up", base=100.0)
            mtf_rows = _make_okx_rows(n=120, step_ms=3600000, trend="up", base=100.0)

            p.write_text(json.dumps({
                "inst_id": "BTC-USDT-SWAP",
                "htf_candles": htf_rows,
                "mtf_candles": mtf_rows,
            }), encoding="utf-8")
            proc_ok = subprocess.run(
                [sys.executable, "-m", "okx_agent.market", "--input", str(p)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_ok.returncode, 0, proc_ok.stderr)
            self.assertEqual(json.loads(proc_ok.stdout)["verdict"], "ALLOW")

            p.write_text(json.dumps({
                "inst_id": "AAPL-USDT-SWAP",
                "htf_candles": htf_rows,
                "mtf_candles": mtf_rows,
                "external_pa_bias": -1,
            }), encoding="utf-8")
            proc_halt = subprocess.run(
                [sys.executable, "-m", "okx_agent.market", "--input", str(p)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_halt.returncode, 2, proc_halt.stderr)
            self.assertEqual(json.loads(proc_halt.stdout)["verdict"], "CONFLICT_HALT")


if __name__ == "__main__":
    unittest.main()
