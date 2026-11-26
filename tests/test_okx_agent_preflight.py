"""Unit tests for okx_agent.preflight and okx_agent.universe."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from okx_agent.preflight import evaluate_preflight
from risk_ledger import record_close, record_open


def _make_book(mid: float, spread_bps: float = 2.0, sz_per_level: float = 100.0) -> tuple[list, list]:
    half_spread = mid * (spread_bps / 10000.0) / 2.0
    best_bid = mid - half_spread
    best_ask = mid + half_spread
    bids = [[str(round(best_bid - i * 0.5, 4)), str(sz_per_level)] for i in range(5)]
    asks = [[str(round(best_ask + i * 0.5, 4)), str(sz_per_level)] for i in range(5)]
    return bids, asks


class TestOkxAgentPreflight(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self._tmpdir.name) / "risk_ledger.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _base_btc_long(self) -> dict:
        bids, asks = _make_book(100000.0, spread_bps=1.0, sz_per_level=50.0)
        return {
            "inst_id": "BTC-USDT-SWAP",
            "side": "buy",
            "ord_type": "limit",
            "entry": 100000.0,
            "sl": 99000.0,       # 1R = 1000 (1.0%)
            "tp1": 102500.0,     # 2.5R
            "tp2": 104000.0,     # 4.0R -> gross weighted RR = 3.25R
            "w1": 0.5,
            "w2": 0.5,
            "risk_pct": 0.008,   # $80 risk on $10,000 equity
            "equity_usd": 10000.0,
            "leverage": 10.0,
            "acct_lv": 2,
            "ct_val": 0.01,
            "lot_sz": 0.01,
            "min_sz": 0.01,
            "tick_sz": 0.1,
            "bids": bids,
            "asks": asks,
            "funding_rate": 0.0001,
            "taker_fee_rate": "-0.0005",
            "ledger_path": str(self.ledger_path),
        }

    def test_happy_path_btc_long_and_eth_short_split_tp(self):
        res_long = evaluate_preflight(self._base_btc_long())
        self.assertTrue(res_long["passed"], res_long["rejections"])
        self.assertEqual(res_long["status"], "PASS")
        self.assertEqual(len(res_long["orders"]), 2)
        o1, o2 = res_long["orders"]
        self.assertEqual(o1["slTriggerPxType"], "mark")
        self.assertEqual(o1["tpTriggerPxType"], "last")
        self.assertTrue(o1["simulatedTrading"])
        self.assertEqual(o1["px"], "100000")
        self.assertEqual(o1["tpTriggerPx"], "102500")
        self.assertEqual(o2["tpTriggerPx"], "104000")
        self.assertAlmostEqual(float(o1["sz"]) + float(o2["sz"]), res_long["metrics"]["total_sz"], places=6)

        bids, asks = _make_book(3000.0, spread_bps=1.0, sz_per_level=200.0)
        eth_short = {
            "inst_id": "ETH-USDT-SWAP",
            "side": "sell",
            "ord_type": "limit",
            "entry": 3000.0,
            "sl": 3060.0,        # 1R = 60 (2.0%)
            "tp1": 2850.0,       # 2.5R
            "tp2": 2760.0,       # 4.0R
            "risk_pct": 0.01,
            "equity_usd": 10000.0,
            "leverage": 10.0,
            "acct_lv": 2,
            "ct_val": 0.1,
            "lot_sz": 0.01,
            "min_sz": 0.01,
            "tick_sz": 0.01,
            "bids": bids,
            "asks": asks,
            "funding_rate": 0.0001,
            "taker_fee_rate": 0.0005,
            "ledger_path": str(self.ledger_path),
        }
        res_short = evaluate_preflight(eth_short)
        self.assertTrue(res_short["passed"], res_short["rejections"])
        self.assertEqual(len(res_short["orders"]), 2)
        self.assertEqual(res_short["orders"][0]["side"], "sell")

    def test_market_order_omits_px(self):
        payload = self._base_btc_long()
        payload["ord_type"] = "market"
        res = evaluate_preflight(payload)
        self.assertTrue(res["passed"], res["rejections"])
        for order in res["orders"]:
            self.assertNotIn("px", order)

    def test_net_rr_veto_after_friction(self):
        payload = self._base_btc_long()
        payload["tp1"] = 102550.0
        payload["tp2"] = None
        payload["w1"] = 1.0
        payload["w2"] = 0.0
        # Gross RR = 2.55, but friction ~ 0.16R -> Net RR < 2.5
        res = evaluate_preflight(payload)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "REJECTED (< 2.5R)")
        codes = {r["code"] for r in res["rejections"]}
        self.assertIn("REJECTED_RR_BELOW_MIN", codes)

    def test_negative_taker_fee_normalized_as_cost(self):
        p_neg = self._base_btc_long()
        p_neg["taker_fee_rate"] = "-0.0005"
        p_pos = self._base_btc_long()
        p_pos["taker_fee_rate"] = "0.0005"
        r_neg = evaluate_preflight(p_neg)
        r_pos = evaluate_preflight(p_pos)
        self.assertEqual(r_neg["metrics"]["taker_fee_rate"], 0.0005)
        self.assertEqual(r_neg["metrics"]["net_weighted_rr"], r_pos["metrics"]["net_weighted_rr"])

    def test_funding_rate_sign_never_inflates_rr(self):
        payload = self._base_btc_long()
        payload["side"] = "sell"
        payload["sl"] = 101000.0
        payload["tp1"] = 97500.0
        payload["tp2"] = 96000.0
        payload["funding_rate"] = 0.005  # Shorts receive funding -> cost clamped to 0.0
        res = evaluate_preflight(payload)
        self.assertEqual(res["metrics"]["funding_cost_rate"], 0.0)

        payload_long = self._base_btc_long()
        payload_long["funding_rate"] = 0.005  # Longs pay funding -> > 0 cost
        res_long = evaluate_preflight(payload_long)
        self.assertGreater(res_long["metrics"]["funding_cost_rate"], 0.0)

    def test_account_level_one_rejected(self):
        payload = self._base_btc_long()
        payload["acct_lv"] = 1
        res = evaluate_preflight(payload)
        self.assertFalse(res["passed"])
        codes = {r["code"] for r in res["rejections"]}
        self.assertIn("REJECTED_ACCOUNT_LEVEL", codes)

    def test_spread_and_depth_rejected(self):
        payload = self._base_btc_long()
        bids, asks = _make_book(100000.0, spread_bps=8.0, sz_per_level=0.05)
        payload["bids"] = bids
        payload["asks"] = asks
        res = evaluate_preflight(payload)
        self.assertFalse(res["passed"])
        codes = {r["code"] for r in res["rejections"]}
        self.assertIn("REJECTED_SPREAD_EXCEEDED", codes)
        self.assertIn("REJECTED_DEPTH_INSUFFICIENT", codes)

    def test_liquidation_guard_crypto_vs_stock_token(self):
        # Stock token perp requires >= 4.0R liquidation distance
        bids, asks = _make_book(200.0, spread_bps=1.0, sz_per_level=5000.0)
        stock_payload = {
            "inst_id": "AAPL-USDT-SWAP",
            "side": "buy",
            "ord_type": "limit",
            "entry": 200.0,
            "sl": 194.0,       # 1R = 6.0 (3.0%)
            "tp1": 218.0,      # 3.0R
            "tp2": 224.0,      # 4.0R
            "risk_pct": 0.008,
            "equity_usd": 10000.0,
            "leverage": 10.0,  # ~9.5% away -> ~3.17R (< 4.0R for stock_token)
            "acct_lv": 2,
            "ct_val": 1.0,
            "lot_sz": 1.0,
            "min_sz": 1.0,
            "tick_sz": 0.01,
            "bids": bids,
            "asks": asks,
            "ledger_path": str(self.ledger_path),
        }
        res_stock = evaluate_preflight(stock_payload)
        self.assertFalse(res_stock["passed"])
        self.assertIn("REJECTED_LIQ_DISTANCE_TOO_CLOSE", {r["code"] for r in res_stock["rejections"]})

        # Same 3.17R passes for crypto perp (requires >= 2.0R)
        btc_payload = self._base_btc_long()
        btc_payload["sl"] = 97000.0  # 3.0% 1R, 10x lev -> ~3.17R liq distance
        btc_payload["tp1"] = 109000.0
        btc_payload["tp2"] = 112000.0
        res_btc = evaluate_preflight(btc_payload)
        self.assertTrue(res_btc["passed"], res_btc["rejections"])

        # Explicit liq_px inside SL rejected
        bad_liq = self._base_btc_long()
        bad_liq["liq_px"] = 99500.0
        res_bad_liq = evaluate_preflight(bad_liq)
        self.assertIn("REJECTED_LIQ_INSIDE_SL", {r["code"] for r in res_bad_liq["rejections"]})

    def test_sizing_and_split_min_sz_guard(self):
        # Below min_sz
        small = self._base_btc_long()
        small["equity_usd"] = 500.0
        small["risk_pct"] = 0.005  # $2.5 risk / ($1000 * 0.01) = 0.25 contracts -> if min_sz=1 -> rejected
        small["min_sz"] = 1.0
        res_small = evaluate_preflight(small)
        self.assertIn("REJECTED_SIZE_BELOW_MIN", {r["code"] for r in res_small["rejections"]})

        # Exactly min_sz fails split-TP, passes single-TP
        exact_min = self._base_btc_long()
        exact_min["lot_sz"] = 8.0
        exact_min["min_sz"] = 8.0
        res_split = evaluate_preflight(exact_min)
        self.assertFalse(res_split["passed"])
        self.assertIn("REJECTED_SPLIT_SIZE_BELOW_MIN", {r["code"] for r in res_split["rejections"]})

        exact_min["tp2"] = None
        exact_min["w1"] = 1.0
        exact_min["w2"] = 0.0
        exact_min["tp1"] = 103000.0
        res_single = evaluate_preflight(exact_min)
        self.assertTrue(res_single["passed"], res_single["rejections"])
        self.assertEqual(len(res_single["orders"]), 1)

    def test_risk_ledger_bucket_and_daily_loss_blocks(self):
        # Fill 2 crypto_majors slots
        record_open("BTC-USDT-SWAP", 0.008, 100000.0, 99000.0, ledger_path=self.ledger_path)
        record_open("ETH-USDT-SWAP", 0.008, 3000.0, 2940.0, ledger_path=self.ledger_path)

        bids, asks = _make_book(200.0, spread_bps=1.0, sz_per_level=500.0)
        sol_payload = {
            "inst_id": "SOL-USDT-SWAP",
            "side": "buy",
            "ord_type": "limit",
            "entry": 200.0,
            "sl": 198.0,
            "tp1": 206.0,
            "tp2": 208.0,
            "risk_pct": 0.008,
            "equity_usd": 10000.0,
            "leverage": 10.0,
            "acct_lv": 2,
            "ct_val": 1.0,
            "lot_sz": 0.01,
            "min_sz": 0.01,
            "tick_sz": 0.01,
            "bids": bids,
            "asks": asks,
            "ledger_path": str(self.ledger_path),
        }
        res_bucket = evaluate_preflight(sol_payload)
        self.assertFalse(res_bucket["passed"])
        self.assertIn("LEDGER_CORRELATION_BUCKET_CAP_EXCEEDED", {r["code"] for r in res_bucket["rejections"]})

        # Daily loss circuit breaker (-2.5R)
        ledger2 = Path(self._tmpdir.name) / "risk_ledger2.json"
        record_open("BTC-USDT-SWAP", 0.008, 100000.0, 99000.0, ledger_path=ledger2)
        record_close("BTC-USDT-SWAP", -2.5, -0.02, ledger_path=ledger2)
        p2 = self._base_btc_long()
        p2["ledger_path"] = str(ledger2)
        res_daily = evaluate_preflight(p2)
        self.assertFalse(res_daily["passed"])
        self.assertIn("LEDGER_DAILY_LOSS_CIRCUIT_BREAKER", {r["code"] for r in res_daily["rejections"]})

    def test_invalid_geometry_and_unlisted_instrument(self):
        bad_geo = self._base_btc_long()
        bad_geo["sl"] = 100500.0  # SL above entry for buy
        res_geo = evaluate_preflight(bad_geo)
        self.assertIn("REJECTED_INVALID_PRICE_GEOMETRY", {r["code"] for r in res_geo["rejections"]})

        unlisted = self._base_btc_long()
        unlisted["inst_id"] = "DOGE-USDT-SWAP"
        res_unlisted = evaluate_preflight(unlisted)
        self.assertIn("REJECTED_INSTRUMENT_NOT_IN_UNIVERSE", {r["code"] for r in res_unlisted["rejections"]})

    def test_cli_exit_codes(self):
        ok_payload = self._base_btc_long()
        proc_ok = subprocess.run(
            [sys.executable, "-m", "okx_agent.preflight", "--json", json.dumps(ok_payload)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_ok.returncode, 0, proc_ok.stderr)
        out_ok = json.loads(proc_ok.stdout)
        self.assertTrue(out_ok["passed"])

        bad_payload = self._base_btc_long()
        bad_payload["acct_lv"] = 1
        proc_bad = subprocess.run(
            [sys.executable, "-m", "okx_agent.preflight", "--json", json.dumps(bad_payload)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_bad.returncode, 1)
        out_bad = json.loads(proc_bad.stdout)
        self.assertFalse(out_bad["passed"])


if __name__ == "__main__":
    unittest.main()
