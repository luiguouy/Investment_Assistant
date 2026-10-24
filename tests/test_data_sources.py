#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_sources.py 的离线自检

为什么要离线测：本环境代理拒绝对 www.okx.com 建 CONNECT 隧道（返回 502），
拿不到真实响应。但「解析对不对」和「网络通不通」是两件事，必须分开验证——
否则等网络恢复时，可能拿到的是一张字段错位的表却毫无察觉。

用法：python test_data_sources.py
"""

import os
import json
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import data_sources as ds

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- 夹具
def build_okx_pages(n=250, start="2024-01-01"):
    """
    造 n 根日线，按 OKX 真实格式返回：
      [ts_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
      时间**倒序**，单页最多 100 根。
    刻意让 high/low 与 open/close 的关系成为唯一自洽解，
    一旦解析时 h/l 顺序搞反，结构校验会立刻报错。
    """
    idx = pd.date_range(start, periods=n, freq="D")
    rows = []
    for i, d in enumerate(idx):
        o = 100 + i * 0.5
        c = o + (1.0 if i % 2 == 0 else -1.0)
        hi = max(o, c) + 0.8
        lo = min(o, c) - 0.8
        ts = int(d.timestamp() * 1000)
        confirm = "0" if i == n - 1 else "1"      # 最后一根设为「未收盘」
        rows.append([str(ts), f"{o:.2f}", f"{hi:.2f}", f"{lo:.2f}", f"{c:.2f}",
                     "1234", "1234", "1234", confirm])
    rows.reverse()                                 # OKX 返回倒序
    pages = []
    for i in range(0, len(rows), 100):
        pages.append(rows[i:i + 100])
    return pages


def fake_http_factory(pages):
    """按 after 参数返回对应页；记录调用次数，便于验证翻页逻辑"""
    calls = {"n": 0, "afters": []}

    def fake_get(url, timeout=20):
        calls["n"] += 1
        after = ""
        if "after=" in url:
            after = url.split("after=")[1].split("&")[0]
        calls["afters"].append(after)

        if not after:
            page = pages[0]
        else:
            # 找到「最老时间戳 = after」的下一页
            page = None
            for p in pages:
                if p[-1][0] == after:
                    nxt = pages.index(p) + 1
                    page = pages[nxt] if nxt < len(pages) else []
                    break
            if page is None:
                page = []

        body = {"code": "0", "msg": "", "data": page}
        return json.dumps(body), {"transport": "fake", "status": 200}

    return fake_get, calls


# ---------------------------------------------------------------- 测试
def test_okx_parser():
    print("\n[OKX] 解析与翻页")
    original = ds.http_get
    try:
        pages = build_okx_pages(250)
        ds.http_get, calls = fake_http_factory(pages)

        src = ds.OKXSource()
        df = src.fetch("BTC", 250)

        check("取到数据", not df.empty, f"{len(df)} 根")
        check("未收盘的那根被剔除（250 取到 249）", len(df) == 249, f"实得 {len(df)}")
        check("发生翻页（>=3 次请求）", calls["n"] >= 3, f"请求 {calls['n']} 次")
        check("日期升序且唯一",
              df.index.is_monotonic_increasing and not df.index.duplicated().any())

        # 字段顺序：high 必须 >= max(open,close)，low <= min(open,close)
        ok_hl = bool((df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
                     and (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all())
        check("OHLC 字段顺序正确（H/L 包住 O/C）", ok_hl)

        # 抽查一根：第 0 根 o=100.0, c=101.0(偶数根涨), h=101.8, l=99.2
        r = df.iloc[0]
        check("首根数值精确匹配夹具",
              abs(r["Open"] - 100.0) < 1e-6 and abs(r["Close"] - 101.0) < 1e-6
              and abs(r["High"] - 101.8) < 1e-6 and abs(r["Low"] - 99.2) < 1e-6,
              f"O={r['Open']} C={r['Close']} H={r['High']} L={r['Low']}")

        # 时间戳是毫秒：2024-01-01
        check("毫秒时间戳解析正确", str(df.index[0].date()) == "2024-01-01",
              str(df.index[0].date()))

        # 末根应是第 249 根（原 250 根的倒数第二）
        check("末根日期正确（未收盘那根之前）", str(df.index[-1].date()) == "2024-09-05",
              str(df.index[-1].date()))
    finally:
        ds.http_get = original


def test_routing():
    print("\n[路由] BTC/ETH 只走 OKX，其余只走股票源")
    okx = ds.OKXSource()
    em = ds.EastmoneySource()
    tx = ds.TencentSource()

    check("OKX 接 BTC", okx.accepts("BTC") is True)
    check("OKX 接 ETH", okx.accepts("ETH") is True)
    check("OKX 拒 SOL", okx.accepts("SOL") is False)
    check("OKX 拒 AAPL", okx.accepts("AAPL") is False)
    check("东财拒 BTC", em.accepts("BTC") is False)
    check("腾讯拒 ETH", tx.accepts("ETH") is False)
    check("腾讯接 AAPL", tx.accepts("AAPL") is True)
    check("OKX 在源列表首位", ds.SOURCES[0] is ds.OKXSource)


def test_okx_unsupported():
    print("\n[OKX] 不支持的标的应直接失败而非乱拉")
    src = ds.OKXSource()
    df = src.fetch("SOL", 100)
    check("SOL 返回空", df.empty)
    check("原因写明只支持 BTC/ETH", "仅支持" in src.last_reason, src.last_reason)


def test_hl_guard():
    print("\n[校验] 字段错位必须报错而不是静默通过")
    bad = [["2024-01-01", "100", "101", "99", "100.5", "1"]]   # 把 h/l 传成 l/h
    # 构造：High=99 < max(O,C)=100.5  -> 应触发校验
    try:
        ds._to_frame([["2024-01-01", "100", "100.5", "99", "101.5", "1"]], source="bad")
        check("错位数据触发异常", False, "未报错")
    except ValueError as e:
        check("错位数据触发异常", True, str(e)[:40])


if __name__ == "__main__":
    test_okx_parser()
    test_routing()
    test_okx_unsupported()
    test_hl_guard()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"❌ 失败 {len(FAILS)} 项：")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("✅ 全部通过")
