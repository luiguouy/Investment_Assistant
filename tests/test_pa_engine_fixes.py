#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pa_engine.py v2.1 修复项回归测试（离线运行，不依赖行情源）

用法：
    python test_pa_engine_fixes.py

覆盖 4 项 P0 修复：
    FIX-1 拐点确认状态（未来函数 / 重绘）
    FIX-2 盘前与开盘区间窗口（会话截断 / 跨日混入）
    FIX-3 斐波那契共振区聚类（单链漂移）
    FIX-4 取数失败处理（重试 / 熔断 / 退出码）
"""

import os
import sys
import io
import contextlib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pa_engine as pe

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def walk(n, start=100.0, index=None, seed=42):
    rng = np.random.default_rng(seed)
    c = start + np.cumsum(rng.normal(0.1, 1.0, n))
    return pd.DataFrame(
        {"Open": c, "High": c + rng.uniform(0.2, 1.2, n),
         "Low": c - rng.uniform(0.2, 1.2, n), "Close": c,
         "Volume": rng.integers(5e5, 5e6, n)},
        index=index if index is not None else pd.bdate_range("2025-01-01", periods=n),
    )


# ---------------------------------------------------------------- FIX-1
def test_swing_confirmation():
    print("\n[FIX-1] 拐点确认状态")
    rng = np.random.default_rng(7)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {"Open": close, "High": close + rng.uniform(.2, 1., n),
         "Low": close - rng.uniform(.2, 1., n), "Close": close,
         "Volume": rng.integers(1e5, 1e6, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="D"))

    for k in (100, 105, 110, 115, 120):
        conf, pend = pe._split_swings(df.iloc[:k], 5)
        last = conf[-1]["index"] if conf else -1
        check(f"bars={k}: 末位已确认拐点 index={last} <= {k - 1 - 5}", last <= k - 1 - 5,
              f"待确认 {len(pend)} 个")

    conf, _ = pe._split_swings(df, 5)
    check("拐点严格高低交替", all(conf[i]["type"] != conf[i + 1]["type"] for i in range(len(conf) - 1)))
    check("已确认拐点均带 dow 标签", all("dow" in s for s in conf))


# ---------------------------------------------------------------- FIX-2
def test_intraday_window():
    print("\n[FIX-2] 盘前 / 开盘区间窗口")
    idx = []
    for d in ("2026-10-20", "2026-10-21"):
        idx += list(pd.date_range(f"{d} 04:00", f"{d} 19:55", freq="5min"))
    df = pd.DataFrame({"Open": 1., "High": 1., "Low": 1., "Close": 1., "Volume": 100},
                      index=pd.DatetimeIndex(idx))
    prev = df.index.date == pd.Timestamp("2026-10-20").date()
    cur = df.index.date == pd.Timestamp("2026-10-21").date()
    df.loc[prev, "High"], df.loc[prev, "Low"] = 200.0, 50.0
    df.loc[cur, "High"], df.loc[cur, "Low"] = 101.0, 99.0

    lv = pe.calc_fibonacci_and_levels([], df_intraday=df)
    check(f"PMH={lv.get('PMH')} 只取当日 101.0（不被昨日 200 污染）", lv.get("PMH") == 101.0)
    check(f"会话锁定={lv.get('session')}", lv.get("session") == "2026-10-21")
    check(f"ORH={lv.get('ORH')} 取当日 101.0", lv.get("ORH") == 101.0)

    # 盘前数据缺失 -> 必须输出 N/A，不给近似值
    no_pm = pd.DataFrame({"Open": 1., "High": 1.5, "Low": 0.5, "Close": 1., "Volume": 100},
                         index=pd.date_range("2026-10-21 09:30", "2026-10-21 15:55", freq="5min"))
    lv2 = pe.calc_fibonacci_and_levels([], df_intraday=no_pm)
    check("无盘前数据时不输出 PMH", "PMH" not in lv2, str(lv2.get("intraday_missing")))
    check("此时 ORH 仍可正常计算", lv2.get("ORH") == 1.5)

    # 昨日K线定位
    daily = pd.DataFrame({"Open": [1, 2, 3], "High": [10, 20, 30], "Low": [1, 2, 3],
                          "Close": [5, 15, 25]},
                         index=pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"]))
    check("日线含当日 -> PDH 取前一日",
          pe.calc_fibonacci_and_levels([], df_daily=daily, df_intraday=df).get("PDH") == 20.0)
    check("日线不含当日 -> PDH 仍取前一日",
          pe.calc_fibonacci_and_levels([], df_daily=daily.iloc[:2], df_intraday=df).get("PDH") == 20.0)


# ---------------------------------------------------------------- FIX-3
def test_fib_cluster():
    print("\n[FIX-3] 斐波那契共振区聚类")
    swings, p = [], 100.0
    for i in range(8):
        swings.append({"index": i, "time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                       "type": "High" if i % 2 == 0 else "Low", "price": p,
                       "dow": "HH" if i % 2 == 0 else "LL"})
        p *= 1.009

    res = pe.calc_multi_leg_fib(swings, tolerance_pct=0.01)
    span = 100 * (swings[-1]["price"] / swings[0]["price"] - 1)
    print(f"  输入跨度 {span:.1f}%，输出 {len(res['clusters'])} 个簇")
    check("不再退化为单一超大簇", all(
        (lambda lo, hi: (hi - lo) / lo <= 0.0101)(
            *[float(x.strip('$')) for x in c["zone"].split(" - ")])
        for c in res["clusters"]))
    check("强共振要求来自 >=2 条腿", all(
        c["strength"] != "🔥 强共振" or c.get("legs", 1) >= 2 for c in res["clusters"]))


# ---------------------------------------------------------------- FIX-4
def test_fetch_behaviour():
    print("\n[FIX-4] 取数失败处理")
    original_backoff = pe.FETCH_BACKOFF_BASE
    pe.FETCH_BACKOFF_BASE = 0.01

    def mkdf(n=30):
        return walk(n, index=pd.bdate_range("2026-09-01", periods=n))

    class OnceFlaky:
        calls = 0

        def __init__(self, sym):
            pass

        def history(self, **kw):
            OnceFlaky.calls += 1
            if OnceFlaky.calls == 1:
                raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
            return mkdf()

    class Dead:
        calls = 0

        def __init__(self, sym):
            pass

        def history(self, **kw):
            Dead.calls += 1
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    try:
        pe.yf.Ticker = OnceFlaky
        pe.reset_fetch_state()
        df = pe.fetch_klines("AAA", "1d", "1mo")
        check("单次限流后重试成功", len(df) == 30, f"请求 {OnceFlaky.calls} 次")

        before = OnceFlaky.calls
        pe.fetch_klines("AAA", "1d", "1mo")
        check("运行内缓存生效", OnceFlaky.calls == before)

        pe.yf.Ticker = Dead
        pe.reset_fetch_state()
        Dead.calls = 0
        pe.fetch_klines("B1", "1d", "1mo")
        first = Dead.calls
        pe.fetch_klines("B2", "1d", "1mo")
        pe.fetch_klines("B3", "1d", "1mo")
        check("连续限流后熔断，不再发起请求", Dead.calls == first, f"总请求 {Dead.calls} 次")
        check("熔断原因写入失败记录", any("熔断" in x for x in pe._FETCH_FAILURES))

        # 退出码
        pe.reset_fetch_state()
        pe.fetch_klines = lambda *a, **k: pd.DataFrame()
        sys.argv = ["pa_engine.py", "--ticker", "AAA", "--mode", "day"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code_fail = pe.main()
        check("全部取数失败 -> 退出码 1", code_fail == 1, f"实得 {code_fail}")

        good = lambda t, i, p, prepost=False, retries=3: walk(300, seed=abs(hash(t + i)) % 9999)
        pe.fetch_klines = good
        sys.argv = ["pa_engine.py", "--ticker", "AAA", "--mode", "swing"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code_ok = pe.main()
        check("正常数据 -> 退出码 0", code_ok == 0, f"实得 {code_ok}")

        # 批量单点异常
        def crash_one(t, i, p, prepost=False, retries=3):
            if t == "BBB":
                raise ValueError("boom")
            return good(t, i, p, prepost, retries)

        pe.fetch_klines = crash_one
        sys.argv = ["pa_engine.py", "--watchlist", "AAA,BBB,CCC", "--mode", "swing"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = pe.main()
        check("单点异常不中断整批", "AAA" in out.getvalue() and "CCC" in out.getvalue())
        check("异常标的计入失败并返回 1", code == 1 and "BBB" in err.getvalue())
    finally:
        pe.FETCH_BACKOFF_BASE = original_backoff


# ---------------------------------------------------------------- P1-1
def _swings(seq):
    out = []
    for i, (t, p) in enumerate(seq):
        out.append({"index": i, "time": f"2026-10-{i + 1:02d}", "type": t, "price": p,
                    "dow": ("HH" if t == "High" else "HL"), "confirmed": True})
    return out


def test_trade_plan():
    print("\n[P1-1] 交易计划与风控闸门")
    mtf = _swings([("Low", 95), ("High", 105), ("Low", 98), ("High", 110),
                   ("Low", 103), ("High", 115)])
    ltf = _swings([("Low", 106), ("High", 112), ("Low", 108), ("High", 114), ("Low", 109)])
    kl = [{"price": 109.0, "kind": "swing", "src": "LTF-HL"},
          {"price": 105.0, "kind": "fib", "src": "Fib0.618"},
          {"price": 112.0, "kind": "swing", "src": "MTF-HH"},
          {"price": 118.0, "kind": "day_extreme", "src": "PDH"},
          {"price": 124.0, "kind": "gap", "src": "Gap"}, ]
    levels = {"fib_extensions": {"1.272": 121.0, "1.618": 131.0},
              "fib_retracements": {"0.618": 105.0}}

    p = pe.build_trade_plan("swing", 110.0, mtf, ltf, kl, levels, 1.0, 1, 1,
                            equity=100000, risk_pct=0.01)
    check("多头方向识别", p["side"] == "long")
    check(f"止损锚定最近已确认低点 109 - 0.8 = {p['stop']}", p["stop"] == 108.2)
    check(f"入场取下方支撑 ${p['entry']}", p["entry"] == 109.0)
    check(f"TP1 取上方结构位 ${p['tp1']}", p["tp1"] == 112.0)
    check(f"RR1 计算正确（实得 {p['rr1']}）", p["rr1"] == 3.75)
    check(f"股数 = 1000 / 0.8 = {p['shares']}", p["shares"] == 1250)
    check("RR 达标 -> 可执行", p["status"].startswith("✅"))

    # RR 不足必须被拦截
    kl_bad = [{"price": 109.0, "kind": "swing", "src": "x"},
              {"price": 109.9, "kind": "fib", "src": "Fib"}]
    p2 = pe.build_trade_plan("swing", 110.0, mtf, ltf, kl_bad, {"fib_extensions": {}},
                             1.0, 1, 1)
    check(f"RR {p2['rr1']} < 2 被判不成立", str(p2["status"]).startswith("❌"), p2["status"])

    # HTF/MTF 冲突
    p3 = pe.build_trade_plan("swing", 110.0, mtf, ltf, kl, levels, 1.0, 1, -1)
    check("HTF/MTF 冲突 -> 不出计划", "冲突" in p3["status"])

    # 无拐点
    p4 = pe.build_trade_plan("swing", 110.0, [], [], kl, levels, 1.0, 1, 1)
    check("无已确认拐点 -> 拒绝出计划", "无已确认拐点" in p4["status"])

    # 空头镜像
    mtf_s = _swings([("High", 115), ("Low", 105), ("High", 112), ("Low", 102),
                     ("High", 109), ("Low", 99)])
    ltf_s = _swings([("High", 104), ("Low", 100), ("High", 102), ("Low", 99),
                     ("High", 100.8)])
    kl_s = [{"price": 100.5, "kind": "swing", "src": "LTF-LH"},
            {"price": 98.0, "kind": "swing", "src": "x"},
            {"price": 96.0, "kind": "gap", "src": "Gap"}]
    p5 = pe.build_trade_plan("swing", 100.0, mtf_s, ltf_s, kl_s,
                             {"fib_extensions": {"1.618": 94.0}}, 1.0, -1, -1)
    check("空头方向识别", p5["side"] == "short")
    check(f"空头止损在最近高点 100.8 之上（{p5['stop']}）", p5["stop"] == 101.6)
    check(f"空头 TP1 位于下方（{p5['tp1']}）", p5["tp1"] == 98.0)
    check(f"空头 TP2 取 1.618 扩展（{p5['tp2']}）", p5["tp2"] == 94.0)
    check(f"空头 RR 计算正确（{p5['rr1']}）", p5["rr1"] == 2.27)

    # 现价已在结构之外 -> 拒绝
    p6 = pe.build_trade_plan("swing", 108.0, mtf, ltf, kl, levels, 1.0, 1, 1)
    check("现价跌破最近低点 -> 结构失效", "结构失效" in p6["status"])


# ---------------------------------------------------------------- P1-2 / P1-3
def test_key_levels_and_location():
    print("\n[P1-2 / P1-3] 关键位分级归并与形态位置过滤")
    lv = [{"price": 109.0, "kind": "swing", "src": "MTF-HL"},
          {"price": 109.05, "kind": "fib", "src": "Fib0.618"},
          {"price": 109.02, "kind": "cluster", "src": "Cluster"}]
    merged = pe.build_key_levels({"fib_retracements": {}}, [], [], [], [], 1.0)
    check("空输入不报错", merged == [])

    kl = pe.build_key_levels({"fib_retracements": {"0.618": 109.05}},
                             [{"price": 109.0, "dow": "HL"}], [], [], [], 1.0)
    check(f"三处相近价位归并为 1 个（实得 {len(kl)}）", len(kl) == 1)
    check("保留 tier 最高的来源", kl[0]["kind"] == "swing")
    check("记录同源汇聚", len(kl[0].get("confluence", [])) >= 1)

    # 2B 只接受 tier>=3：构造一根「刺破 100 后收回」的 K 线
    df2b = pd.DataFrame(
        {"Open": [99.5, 99.4, 99.5, 99.6, 99.6],
         "High": [99.7, 99.6, 99.8, 99.7, 100.6],
         "Low": [99.3, 99.2, 99.4, 99.3, 99.5],
         "Close": [99.4, 99.5, 99.6, 99.5, 99.7],
         "Volume": 1000},
        index=pd.date_range("2026-10-21 09:30", periods=5, freq="5min"))
    fib_only = [{"price": 100.0, "kind": "fib", "src": "Fib0.5"}]
    swing_lv = [{"price": 100.0, "kind": "swing", "src": "Swing"}]
    s_fib = pe.scan_price_action(df2b, key_levels=fib_only, bar_rng=1.0)
    s_sw = pe.scan_price_action(df2b, key_levels=swing_lv, bar_rng=1.0)
    check("2B 不接受 fib 级水平位", not any("2B" in s["pattern"] for s in s_fib))
    check("2B 接受 swing 级水平位", any("2B" in s["pattern"] for s in s_sw),
          str([s["pattern"] for s in s_sw]))

    # 位置过滤：无任何关键位时不输出形态
    c = np.array([100, 100.2, 100.1, 100.3, 100.2, 100.0])
    o = np.array([99.9, 100.0, 100.2, 100.1, 100.3, 100.3])
    h = np.array([100.3, 100.4, 100.4, 100.5, 100.45, 100.5])
    l = np.array([99.8, 99.9, 100.0, 100.0, 100.1, 99.7])
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000},
                      index=pd.date_range("2026-10-21 09:30", periods=6, freq="5min"))
    s_far = pe.scan_price_action(df, key_levels=[{"price": 130.0, "kind": "swing", "src": "x"}],
                                 bar_rng=1.0)
    check("远离关键位的形态被过滤", len(s_far) == 0, f"剩余 {len(s_far)} 条")
    s_open = pe.scan_price_action(df, key_levels=[{"price": 130.0, "kind": "swing", "src": "x"}],
                                  bar_rng=1.0, require_location=False)
    check("关闭位置过滤后形态恢复", len(s_open) > 0, f"共 {len(s_open)} 条")


# ---------------------------------------------------------------- P1-4 / P3-2
def test_unclosed_bar_and_json():
    print("\n[P1-4 / P3-2] 未收盘K线识别 与 JSON 输出")
    idx = pd.date_range(pd.Timestamp.now(tz="America/New_York").floor("5min") - pd.Timedelta("10min"),
                        periods=10, freq="5min", tz="America/New_York")
    df = walk(10, index=idx)
    check("末根仍在运行时被识别", pe._is_last_bar_open(df) is True)

    fixed_now = pd.Timestamp("2026-10-21 12:00", tz="America/New_York")
    idx2 = pd.date_range("2026-10-21 09:30", periods=20, freq="5min", tz="America/New_York")
    df2 = walk(20, index=idx2)
    check("末根已收盘时不误判", pe._is_last_bar_open(df2, now=fixed_now) is False)

    daily = walk(30, index=pd.bdate_range("2026-01-01", periods=30))
    check("日线不做未收盘判断", pe._is_last_bar_open(daily) is False)

    # JSON 落盘
    def fake(ticker, interval, period, prepost=False, retries=3):
        if interval == "1wk":
            return walk(104, index=pd.date_range("2024-10-25", periods=104, freq="W-FRI"))
        if interval == "1d" and period == "1y":
            return walk(250, index=pd.bdate_range("2025-10-22", periods=250))
        if interval == "1d":
            return walk(22, index=pd.bdate_range("2026-09-23", periods=22))
        if interval == "1h":
            idx3 = []
            for d in pd.bdate_range("2026-09-23", periods=21):
                idx3 += list(pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:30", freq="1h"))
            return walk(len(idx3), index=pd.DatetimeIndex(idx3, tz="America/New_York"))

    import tempfile, os, json
    out_path = os.path.join(tempfile.gettempdir(), "pa_engine_test.json")
    if os.path.exists(out_path):
        os.remove(out_path)

    pe.fetch_klines = fake
    pe.reset_fetch_state()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pe.analyze_single_ticker("NVDA", "swing", json_out=out_path, equity=50000)

    ok = os.path.exists(out_path)
    check("JSON 文件已生成", ok)
    if ok:
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        check("JSON 含 plan.status", "status" in data.get("plan", {}))
        check("JSON 含 key_levels", isinstance(data.get("key_levels"), list))
        check("JSON 可被下游直接消费（含 price/phase/bias）",
              all(k in data for k in ("price", "phase", "bias", "plan")))
        os.remove(out_path)


# ---------------------------------------------------------------- 附带
def test_volume_missing_column():
    print("\n[附带] 缺 Volume 列")
    df = pd.DataFrame({"Open": np.arange(30.) + 100, "High": np.arange(30.) + 101,
                       "Low": np.arange(30.) + 99, "Close": np.arange(30.) + 100.5},
                      index=pd.date_range("2026-01-01", periods=30, freq="D"))
    try:
        r = pe.analyze_volume(df)
        check("不崩溃且标记不可用", r.get("available") is False)
    except Exception as e:
        check(f"不崩溃（实际抛出 {type(e).__name__}）", False)


if __name__ == "__main__":
    test_swing_confirmation()
    test_intraday_window()
    test_fib_cluster()
    test_fetch_behaviour()
    test_trade_plan()
    test_key_levels_and_location()
    test_unclosed_bar_and_json()
    test_volume_missing_column()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"❌ 失败 {len(FAILS)} 项：")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("✅ 全部通过")
