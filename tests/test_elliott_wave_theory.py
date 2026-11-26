import unittest
import numpy as np
import pandas as pd

import pa_engine as pe


def _make_df(closes, start="2025-01-01"):
    dates = pd.date_range(start=start, periods=len(closes), freq="B")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": c,
        "High": c * 1.01,
        "Low": c * 0.99,
        "Close": c,
        "Volume": np.full(len(c), 1_000_000.0),
    }, index=dates)


class TestElliottWaveTheory(unittest.TestCase):
    def test_corrective_subtypes_zigzag_and_flats(self):
        # 1. Zigzag: 100 -> 80 (A=20), B=90 (0.50), C=70 (breaks A=80)
        sw_zz = [
            {"type": "High", "price": 100.0, "confirmed": True},
            {"type": "Low", "price": 80.0, "confirmed": True},
            {"type": "High", "price": 90.0, "confirmed": True},
            {"type": "Low", "price": 70.0, "confirmed": True},
        ]
        self.assertEqual(pe.classify_corrective_subtypes(sw_zz)["subtype"], "zigzag")

        # 2. Regular Flat: 100 -> 80, B=99 (0.95), C=80.5 (~1.0)
        sw_rf = [
            {"type": "High", "price": 100.0, "confirmed": True},
            {"type": "Low", "price": 80.0, "confirmed": True},
            {"type": "High", "price": 99.0, "confirmed": True},
            {"type": "Low", "price": 80.5, "confirmed": True},
        ]
        self.assertEqual(pe.classify_corrective_subtypes(sw_rf)["subtype"], "regular_flat")

        # 3. Expanded Flat: 100 -> 80, B=104 (breaks 100), C=75 (breaks 80)
        sw_ef = [
            {"type": "High", "price": 100.0, "confirmed": True},
            {"type": "Low", "price": 80.0, "confirmed": True},
            {"type": "High", "price": 104.0, "confirmed": True},
            {"type": "Low", "price": 75.0, "confirmed": True},
        ]
        self.assertEqual(pe.classify_corrective_subtypes(sw_ef)["subtype"], "expanded_flat")

        # 4. Running Flat: 100 -> 80, B=104 (breaks 100), C=85 (holds above 80)
        sw_run = [
            {"type": "High", "price": 100.0, "confirmed": True},
            {"type": "Low", "price": 80.0, "confirmed": True},
            {"type": "High", "price": 104.0, "confirmed": True},
            {"type": "Low", "price": 85.0, "confirmed": True},
        ]
        self.assertEqual(pe.classify_corrective_subtypes(sw_run)["subtype"], "running_flat")

    def test_corrective_triangle_and_sparse_edge(self):
        # Contracting Triangle: Highs 120 > 115 > 110; Lows 90 < 95 < 100
        sw_tri = [
            {"type": "High", "price": 120.0, "confirmed": True},
            {"type": "Low", "price": 90.0, "confirmed": True},
            {"type": "High", "price": 115.0, "confirmed": True},
            {"type": "Low", "price": 95.0, "confirmed": True},
            {"type": "High", "price": 110.0, "confirmed": True},
            {"type": "Low", "price": 100.0, "confirmed": True},
        ]
        res = pe.classify_corrective_subtypes(sw_tri)
        self.assertEqual(res["subtype"], "contracting_triangle")
        self.assertEqual(pe.classify_corrective_subtypes([])["subtype"], "none")

    def test_fibonacci_and_lucas_time_windows_causal(self):
        df = _make_df(np.linspace(100, 130, 30))
        # Pivot at index 9 -> at len=30 (curr_idx=29), bars_since_pivot = 29 - 9 + 1 = 21 (Fib 21)
        sw_fib = [{"index": 9, "type": "Low", "price": 105.0, "confirmed": True}]
        tw_fib = pe.calc_time_windows(sw_fib, df)
        self.assertTrue(tw_fib["active"])
        self.assertEqual(tw_fib["bars_since_pivot"], 21)
        self.assertEqual(tw_fib["fib_hit"], 21)
        self.assertTrue(tw_fib["is_window_hit"])

        # Truncate df at index 19 (11 bars since pivot 9 -> Lucas 11)
        tw_lucas = pe.calc_time_windows(sw_fib, df.iloc[:20])
        self.assertEqual(tw_lucas["bars_since_pivot"], 11)
        self.assertEqual(tw_lucas["lucas_hit"], 11)
        self.assertTrue(tw_lucas["is_window_hit"])

        # Empty edge
        self.assertFalse(pe.calc_time_windows([], df)["active"])

    def test_elliott_12_and_24_channels(self):
        # Stage 1: 1-2 temporary channel (0 at idx 0 p=100, 1 at idx 5 p=110, 2 at idx 10 p=105)
        # Slope = (105 - 100) / 10 = 0.5; rail at idx 15 = 110 + 0.5 * (15 - 5) = 115.0
        closes_w3 = np.linspace(100, 120, 16)
        closes_w3[-1] = 118.0  # > 115.0 -> breakout confirmed
        df_w3 = _make_df(closes_w3)
        sw_w3 = [
            {"index": 0, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "type": "High", "price": 110.0, "confirmed": True},
            {"index": 10, "type": "Low", "price": 105.0, "confirmed": True},
        ]
        ch_12 = pe.calc_elliott_channel(sw_w3, df_w3)
        self.assertEqual(ch_12["channel_stage"], "1-2_temporary")
        self.assertTrue(ch_12["breakout_confirmed"])

        # Stage 2: 2-4 primary channel (2 at idx 10 p=105, 3 at idx 15 p=130, 4 at idx 20 p=120)
        # Slope = (120 - 105) / 10 = 1.5; baseline at idx 25 = 105 + 1.5 * 15 = 127.5
        closes_w5 = np.linspace(100, 135, 26)
        closes_w5[-1] = 125.0  # < 127.5 -> 2-4 baseline broken!
        df_w5 = _make_df(closes_w5)
        sw_w5 = sw_w3 + [
            {"index": 15, "type": "High", "price": 130.0, "confirmed": True},
            {"index": 20, "type": "Low", "price": 120.0, "confirmed": True},
        ]
        ch_24 = pe.calc_elliott_channel(sw_w5, df_w5)
        self.assertEqual(ch_24["channel_stage"], "2-4_primary")
        self.assertTrue(ch_24["trend_broken"])

    def test_wave4_shallow_overlap_extension_and_alternate_counts(self):
        # Shallow overlap: P0=100, P1=120 (L1=20), P2=110, P3=150, P4=118 (overlap=2/20=10% <= 15%), P5=190 (L5=72 >= 1.618*20)
        sw_shallow = [
            {"index": 0, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 15, "type": "Low", "price": 110.0, "confirmed": True},   # w2_bars = 10
            {"index": 25, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 28, "type": "Low", "price": 118.0, "confirmed": True},   # w4_bars = 3
            {"index": 40, "type": "High", "price": 190.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(sw_shallow)
        self.assertFalse(wp["rules"]["rule3"]["passed"])
        self.assertTrue(wp["rules"]["rule3"]["shallow_overlap_orthodox_ok"])
        self.assertTrue(wp["rules"]["alternation"]["passed"])
        self.assertEqual(wp["extension_info"]["extended_wave"], "wave_5")
        self.assertTrue(wp["extension_info"]["irregular_top_risk"])

        # Valid developing_wave_5 -> evaluate_wave_position includes composite fib & wave_counts
        df = _make_df(np.linspace(100, 160, 35))
        sw_dev5 = [
            {"index": 0, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 5, "type": "High", "price": 120.0, "confirmed": True},
            {"index": 10, "type": "Low", "price": 110.0, "confirmed": True},
            {"index": 20, "type": "High", "price": 150.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 135.0, "confirmed": True},
        ]
        ev = pe.evaluate_wave_position(sw_dev5, df=df, current_price=155.0)
        self.assertIn("0.618_w0_to_w3", ev["fib_projections"])
        self.assertIn("1.382_w0_to_w3", ev["fib_projections"])
        self.assertEqual(ev["elliott_channel"]["channel_stage"], "2-4_primary")
        self.assertTrue(ev["time_windows"]["active"])
        self.assertGreaterEqual(len(ev["wave_counts"]["alternate_counts"]), 1)

    def test_bearish_impulse_symmetry_shallow_overlap_alternation_and_extension(self):
        # 1. 空头浅刺重叠 + K线耗时交替 + 第5浪延伸
        # P0=200, P1=180 (L1=20), P2=190 (w2_retrace=0.50, w2_bars=10), P3=150 (L3=40),
        # P4=182 (侵入P1=180, overlap=2/20=10% <= 15%, p4 < p0, w4_retrace=0.80, w4_bars=3), P5=140 (L5=42 >= 1.618*20)
        sw_bear_shallow = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 190.0, "confirmed": True},   # w2_bars = 10
            {"index": 25, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 28, "type": "High", "price": 182.0, "confirmed": True},   # w4_bars = 3
            {"index": 40, "type": "Low", "price": 140.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(sw_bear_shallow)
        self.assertEqual(wp["direction"], -1)
        self.assertFalse(wp["rules"]["rule3"]["passed"])
        self.assertTrue(wp["rules"]["rule3"]["shallow_overlap_orthodox_ok"])
        self.assertAlmostEqual(wp["rules"]["rule3"]["overlap_depth_pct"], 0.1)
        self.assertTrue(wp["rules"]["alternation"]["passed"])
        self.assertEqual(wp["rules"]["alternation"]["w2_bars"], 10)
        self.assertEqual(wp["rules"]["alternation"]["w4_bars"], 3)
        self.assertEqual(wp["extension_info"]["extended_wave"], "wave_5")
        self.assertTrue(wp["extension_info"]["irregular_top_risk"])

        # 2. 空头深度重叠 (> 15%) -> shallow_overlap_orthodox_ok 为 False
        sw_bear_deep = list(sw_bear_shallow)
        sw_bear_deep[4] = {"index": 28, "type": "High", "price": 185.0, "confirmed": True}  # 5/20 = 25%
        wp_deep = pe.classify_wave_pattern(sw_bear_deep)
        self.assertFalse(wp_deep["rules"]["rule3"]["shallow_overlap_orthodox_ok"])
        self.assertAlmostEqual(wp_deep["rules"]["rule3"]["overlap_depth_pct"], 0.25)

        # 3. 仅靠 K线耗时交替 (>=1.5x) 触发 alternation，且第3浪延伸
        # P0=200, P1=180 (L1=20), P2=190 (w2_retrace=0.50, w2_bars=12),
        # P3=150 (L3=40 >= 1.618*20), P4=170 (w4_retrace=0.50, w4_bars=4), P5=150 (L5=20)
        sw_bear_w3_ext = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 17, "type": "High", "price": 190.0, "confirmed": True},  # w2_bars = 12
            {"index": 25, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 29, "type": "High", "price": 170.0, "confirmed": True},  # w4_bars = 4 (ratio=3.0 >= 1.5)
            {"index": 35, "type": "Low", "price": 150.0, "confirmed": True},
        ]
        wp_w3 = pe.classify_wave_pattern(sw_bear_w3_ext)
        self.assertTrue(wp_w3["is_valid_impulse"])
        self.assertTrue(wp_w3["rules"]["alternation"]["passed"])
        self.assertEqual(wp_w3["extension_info"]["extended_wave"], "wave_3")
        self.assertFalse(wp_w3["extension_info"]["irregular_top_risk"])

    def test_bearish_elliott_12_and_24_channels(self):
        # Stage 1: 空头 1-2 临时通道 (0@idx0 p=200, 1@idx5 p=180, 2@idx10 p=190)
        # Slope = (190 - 200)/10 = -1.0; parallel rail @ idx15 = 180 + (-1.0)*(15-5) = 170.0
        closes_w3 = np.linspace(200, 165, 16)
        closes_w3[-1] = 165.0  # < 170.0 -> downward breakout confirmed
        df_w3 = _make_df(closes_w3)
        sw_w3 = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 10, "type": "High", "price": 190.0, "confirmed": True},
        ]
        ch_12 = pe.calc_elliott_channel(sw_w3, df_w3)
        self.assertEqual(ch_12["channel_stage"], "1-2_temporary")
        self.assertEqual(ch_12["direction"], -1)
        self.assertTrue(ch_12["breakout_confirmed"])
        self.assertIn("主跌加速确认", ch_12["desc"])

        # 未跌破下轨时 breakout_confirmed 为 False
        closes_w3_hold = closes_w3.copy()
        closes_w3_hold[-1] = 175.0
        ch_12_hold = pe.calc_elliott_channel(sw_w3, _make_df(closes_w3_hold))
        self.assertFalse(ch_12_hold["breakout_confirmed"])

        # Stage 2: 空头 2-4 主通道 (2@idx10 p=190, 3@idx15 p=150, 4@idx20 p=165)
        # Slope = (165 - 190)/10 = -2.5; baseline @ idx25 = 190 + (-2.5)*15 = 152.5
        closes_w5 = np.linspace(200, 145, 26)
        closes_w5[-1] = 155.0  # > 152.5 -> 2-4 baseline broken upward!
        df_w5 = _make_df(closes_w5)
        sw_w5 = sw_w3 + [
            {"index": 15, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 20, "type": "High", "price": 165.0, "confirmed": True},
        ]
        ch_24 = pe.calc_elliott_channel(sw_w5, df_w5)
        self.assertEqual(ch_24["channel_stage"], "2-4_primary")
        self.assertEqual(ch_24["direction"], -1)
        self.assertTrue(ch_24["trend_broken"])

    def test_no_index_and_none_boundaries(self):
        # 无 index/time 字段且 df=None 时，classify_wave_pattern 不崩溃且耗时为 None
        sw_no_idx = [
            {"type": "Low", "price": 100.0, "confirmed": True},
            {"type": "High", "price": 120.0, "confirmed": True},
            {"type": "Low", "price": 110.0, "confirmed": True},
            {"type": "High", "price": 150.0, "confirmed": True},
            {"type": "Low", "price": 135.0, "confirmed": True},
            {"type": "High", "price": 170.0, "confirmed": True},
        ]
        wp = pe.classify_wave_pattern(sw_no_idx, df=None)
        self.assertTrue(wp["is_valid_impulse"])
        self.assertIsNone(wp["rules"]["alternation"]["w2_bars"])
        self.assertIsNone(wp["rules"]["alternation"]["w4_bars"])

        df = _make_df(np.linspace(100, 150, 20))
        self.assertFalse(pe.calc_time_windows(sw_no_idx, df)["active"])
        self.assertFalse(pe.calc_time_windows(sw_no_idx, None)["active"])
        self.assertEqual(pe.calc_elliott_channel(sw_no_idx, df)["channel_stage"], "none")
        self.assertEqual(pe.calc_elliott_channel(sw_no_idx, None)["channel_stage"], "none")
        self.assertEqual(pe.classify_corrective_subtypes(None)["subtype"], "none")

        # swings=None 及含非字典元素的列表不崩溃
        self.assertEqual(pe.classify_wave_pattern(None)["pattern"], "unclassified")
        self.assertEqual(pe.classify_wave_pattern([None, "bad"])["pattern"], "unclassified")
        self.assertEqual(pe.evaluate_wave_position(None, df=df)["status"], "invalid")
        self.assertEqual(pe.calc_elliott_channel(None, df)["channel_stage"], "none")
        self.assertEqual(pe.generate_wave_counts(None, df=df)["primary_count"]["label"], "unclassified")
        self.assertFalse(pe.calc_time_windows(None, df)["active"])
        self.assertEqual(pe.classify_corrective_subtypes([None, {}])["subtype"], "none")

        # 负索引边界不得被当作有效历史拐点
        sw_neg_idx = [
            {"index": -1, "type": "Low", "price": 100.0, "confirmed": True},
            {"index": 3, "type": "High", "price": 110.0, "confirmed": True},
            {"index": 6, "type": "Low", "price": 105.0, "confirmed": True},
        ]
        self.assertEqual(pe.calc_elliott_channel(sw_neg_idx, df)["channel_stage"], "none")
        self.assertFalse(pe.calc_time_windows([{"index": -1, "type": "Low", "price": 100.0, "confirmed": True}], df)["active"])

        ev_empty = pe.evaluate_wave_position([], df=None, current_price=None)
        self.assertEqual(ev_empty["status"], "invalid")
        self.assertEqual(ev_empty["wave_pattern"], "unclassified")
        self.assertEqual(ev_empty["time_windows"], {})
        self.assertEqual(ev_empty["elliott_channel"], {})
        self.assertEqual(ev_empty["signal"], "neutral")
        self.assertEqual(ev_empty["action"], "neutral")

    def test_bearish_evaluate_wave_position_and_wave_counts_causality(self):
        df = _make_df(np.linspace(200, 140, 35))
        sw_bear_dev5 = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 10, "type": "High", "price": 190.0, "confirmed": True},
            {"index": 20, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 25, "type": "High", "price": 165.0, "confirmed": True},
        ]
        df_trunc = df.iloc[:29].copy()
        ev1 = pe.evaluate_wave_position(sw_bear_dev5, df=df_trunc, current_price=float(df_trunc["Close"].iloc[-1]))
        wc1 = pe.generate_wave_counts(sw_bear_dev5, df=df_trunc)

        self.assertEqual(ev1["status"], "valid")
        self.assertEqual(ev1["wave_pattern"], "developing_wave_5")
        self.assertEqual(ev1["signal"], "neutral")
        self.assertEqual(ev1["action"], "neutral")
        self.assertEqual(ev1["wave4_invalidation_level"], 180.0)
        self.assertIn("0.618_w0_to_w3", ev1["fib_projections"])
        self.assertIn("1.382_w0_to_w3", ev1["fib_projections"])
        self.assertLess(ev1["fib_projections"]["1.618"], 165.0)
        self.assertEqual(ev1["elliott_channel"]["channel_stage"], "2-4_primary")
        self.assertEqual(ev1["elliott_channel"]["direction"], -1)
        self.assertTrue(ev1["time_windows"]["active"])
        self.assertEqual(wc1["primary_count"]["direction"], -1)
        self.assertGreaterEqual(len(wc1["alternate_counts"]), 1)
        self.assertIn("浪4高点", wc1["alternate_counts"][0]["summary"])

        # 因果性验证：篡改 t=29 之后的未来数据，截断到 t=28 的输出必须 100% 一致
        df_poison = df.copy()
        df_poison.iloc[29:, df_poison.columns.get_loc("Close")] = 9999.0
        df_poison_trunc = df_poison.iloc[:29].copy()
        ev2 = pe.evaluate_wave_position(sw_bear_dev5, df=df_poison_trunc, current_price=float(df_poison_trunc["Close"].iloc[-1]))
        wc2 = pe.generate_wave_counts(sw_bear_dev5, df=df_poison_trunc)
        self.assertEqual(ev1, ev2)
        self.assertEqual(wc1, wc2)
        self.assertEqual(ev1["corrective_subtype"]["subtype"], "none")

    def test_bear_symmetry_channels_and_causality_guard(self):
        # Bear shallow overlap: P0=200, P1=180 (L1=20), P2=190, P3=150, P4=182 (overlap=2/20=10% <= 15%), P5=140 (L5=42 >= 1.618*20)
        sw_bear_shallow = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 190.0, "confirmed": True},   # w2_bars = 10
            {"index": 25, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 28, "type": "High", "price": 182.0, "confirmed": True},   # w4_bars = 3
            {"index": 40, "type": "Low", "price": 140.0, "confirmed": True},
        ]
        wp_bear = pe.classify_wave_pattern(sw_bear_shallow)
        self.assertEqual(wp_bear["direction"], -1)
        self.assertFalse(wp_bear["rules"]["rule3"]["passed"])
        self.assertTrue(wp_bear["rules"]["rule3"]["shallow_overlap_orthodox_ok"])
        self.assertEqual(wp_bear["rules"]["rule3"]["overlap_depth_pct"], 0.1)
        self.assertTrue(wp_bear["rules"]["alternation"]["passed"])
        self.assertEqual(wp_bear["extension_info"]["extended_wave"], "wave_5")
        self.assertTrue(wp_bear["extension_info"]["irregular_top_risk"])

        # Bear 1-2 temporary channel breakout
        closes_b3 = np.linspace(200, 165, 16)
        closes_b3[-1] = 170.0  # rail at idx 15 = 180 + (-1.0)*10 = 170? Wait: slope=(190-200)/10=-1.0, rail(15)=180-10=170.0
        closes_b3[-1] = 168.0  # < 170.0 -> breakout confirmed
        df_b3 = _make_df(closes_b3)
        sw_b3 = sw_bear_shallow[:3]
        ch_b12 = pe.calc_elliott_channel(sw_b3, df_b3)
        self.assertEqual(ch_b12["channel_stage"], "1-2_temporary")
        self.assertEqual(ch_b12["direction"], -1)
        self.assertTrue(ch_b12["breakout_confirmed"])

        # Bear 2-4 primary channel & causality guard
        # P2=(15, 190), P4=(28, 175) -> slope = -15/13
        closes_b5 = np.linspace(200, 145, 32)
        closes_b5[-1] = 185.0  # > baseline -> trend_broken
        df_b5 = _make_df(closes_b5)
        sw_b5 = [
            {"index": 0, "type": "High", "price": 200.0, "confirmed": True},
            {"index": 5, "type": "Low", "price": 180.0, "confirmed": True},
            {"index": 15, "type": "High", "price": 190.0, "confirmed": True},
            {"index": 25, "type": "Low", "price": 150.0, "confirmed": True},
            {"index": 28, "type": "High", "price": 175.0, "confirmed": True},
        ]
        ch_b24 = pe.calc_elliott_channel(sw_b5, df_b5)
        self.assertEqual(ch_b24["channel_stage"], "2-4_primary")
        self.assertEqual(ch_b24["direction"], -1)
        self.assertTrue(ch_b24["trend_broken"])
        self.assertIn("向上突破基线", ch_b24["desc"])

        # Causality guard: if df is truncated before i4 (28), channel must not project backwards
        ch_future_guard = pe.calc_elliott_channel(sw_b5, df_b5.iloc[:20])
        self.assertEqual(ch_future_guard["channel_stage"], "none")

        # Bear corrective subtypes (Low -> High -> Low -> High)
        sw_bear_zz = [
            {"type": "Low", "price": 100.0, "confirmed": True},
            {"type": "High", "price": 120.0, "confirmed": True},
            {"type": "Low", "price": 110.0, "confirmed": True},
            {"type": "High", "price": 130.0, "confirmed": True},
        ]
        self.assertEqual(pe.classify_corrective_subtypes(sw_bear_zz)["subtype"], "zigzag")
        self.assertEqual(pe.classify_corrective_subtypes(sw_bear_zz)["direction"], -1)


if __name__ == "__main__":
    unittest.main()
