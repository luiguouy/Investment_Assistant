"""Unit tests for okx_agent.levels capital-aware SL/TP/trailing calculator."""

import unittest

from okx_agent.levels import (
    compute_chandelier_trailing_sl,
    compute_swing_levels,
    recommend_capital_profile,
)


class TestOkxAgentLevels(unittest.TestCase):
    def test_capital_profile_micro_vs_standard(self):
        micro = recommend_capital_profile(141.5, perp_type="stock_token")
        self.assertEqual(micro["tier"], "micro")
        self.assertEqual(micro["risk_pct"], 0.02)
        self.assertEqual(micro["w1"], 0.5)
        self.assertEqual(micro["leverage"], 4.0)

        std = recommend_capital_profile(5000.0, perp_type="crypto")
        self.assertEqual(std["tier"], "standard")
        self.assertEqual(std["risk_pct"], 0.01)
        self.assertEqual(std["w1"], 0.5)
        self.assertEqual(std["leverage"], 8.0)

    def test_live_micro_account_googl_split_and_auto_degrade(self):
        # 141.5 USDT on GOOGL-USDT-SWAP (minSz=0.01) -> can split 60/40
        res = compute_swing_levels(
            inst_id="GOOGL-USDT-SWAP",
            side="buy",
            entry=280.0,
            swing_anchor=276.0,
            atr=2.0,
            equity_usd=141.5,
            ct_val=1.0,
            lot_sz=0.01,
            min_sz=0.01,
            tick_sz=0.01,
            is_2b_sweep=True,
        )
        self.assertTrue(res["executable"], res["warnings"])
        self.assertEqual(res["capital_tier"], "micro")
        self.assertEqual(res["split_mode"], "split_two_tranche")
        self.assertEqual(res["sl"], 274.0)  # 276 - 1.0*2.0
        self.assertGreaterEqual(res["net_weighted_rr"], 2.5)
        self.assertGreater(res["breakeven_px"], 280.0)

        # If min_sz=0.4 and total_sz=0.47 (< 2*min_sz) -> auto degrades to single_with_trailing
        deg = compute_swing_levels(
            inst_id="GOOGL-USDT-SWAP",
            side="buy",
            entry=280.0,
            swing_anchor=276.0,
            atr=2.0,
            equity_usd=141.5,
            ct_val=1.0,
            lot_sz=0.01,
            min_sz=0.4,
            tick_sz=0.01,
            is_2b_sweep=True,
        )
        self.assertTrue(deg["executable"], deg["warnings"])
        self.assertEqual(deg["split_mode"], "single_with_trailing")
        self.assertIsNone(deg["tp2"])
        self.assertGreaterEqual(deg["net_weighted_rr"], 2.5)

    def test_stop_width_corridor_blocks_too_narrow_or_too_wide(self):
        narrow = compute_swing_levels(
            inst_id="BTC-USDT-SWAP",
            side="buy",
            entry=100000.0,
            swing_anchor=99900.0,
            atr=100.0,  # ~0.22% < 0.6%
            equity_usd=5000.0,
            ct_val=0.01,
            lot_sz=0.01,
            min_sz=0.01,
            tick_sz=0.1,
        )
        self.assertFalse(narrow["executable"])
        self.assertTrue(any("STOP_WIDTH_OUT_OF_CORRIDOR" in w for w in narrow["warnings"]))

    def test_chandelier_trailing_sl_monotonic(self):
        sl_long = compute_chandelier_trailing_sl(
            side="buy",
            current_sl=280.42,
            atr=2.0,
            peak_or_trough_px=296.0,   # 296 - 4.0 = 292.0
            confirmed_swing_px=290.0,  # 290 - 1.0 = 289.0
            tick_sz=0.01,
        )
        self.assertEqual(sl_long, 292.0)

        # Must never widen if peak drops
        sl_no_widen = compute_chandelier_trailing_sl(
            side="buy",
            current_sl=292.0,
            atr=2.0,
            peak_or_trough_px=293.0,   # 293 - 4.0 = 289.0 < 292.0
            tick_sz=0.01,
        )
        self.assertEqual(sl_no_widen, 292.0)


if __name__ == "__main__":
    unittest.main()
