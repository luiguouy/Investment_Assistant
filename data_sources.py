#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多来源行情适配器：缓存 → 东财 → 腾讯 → yfinance → stooq，按顺序降级

设计要点（都来自实测失败，不是假设）：
  1. 双传输：本环境出网必须走 HTTP_PROXY，且代理对 Python TLS 与 curl 的放行不一致
     （curl 通、urllib 断的现象确实发生过）。所以 HTTP 先走 urllib，连接层失败再退 curl。
  2. 逐源熔断：某个源连续失败 2 次即在本轮不再尝试，避免把 30 个标的都耗在死源上。
  3. 来源留痕：每个标的记录「数据来自哪个源、多少根、复权口径」，写进 data/_meta.json。
     回测结论必须能追溯到数据来源，否则无法复现。
  4. 结构校验：解析后检查 High >= max(Open,Close) 且 Low <= min(Open,Close)。
     字段顺序搞错会静默污染全部结构判定，宁可报错。

用法：
    python data_sources.py --probe                     # 探测各源可用性
    python data_sources.py --fetch AAPL,MSFT --bars 1600
    python data_sources.py --fetch-file tickers.txt
    python data_sources.py --status                    # 查看缓存清单
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
META_PATH = os.path.join(DATA_DIR, "_meta.json")
SECID_PATH = os.path.join(DATA_DIR, "_secid.json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
OHLC = ["Open", "High", "Low", "Close", "Volume"]
THROTTLE = 0.35
MAX_GAP_DAYS = 20      # 相邻K线最大允许间隔；超过即判定数据缺失整段


def check_continuity(df: pd.DataFrame) -> tuple:
    """
    数据连续性校验。真实案例：新浪的 QQQ 序列在 2005–2010 之间有 2307 天的空洞，
    若直接用，回测会把六年压缩成一根 K 线的跳变——结论完全失真且毫无提示。
    """
    if df is None or len(df) < 3:
        return False, "样本过短"
    gaps = df.index.to_series().diff().dt.days.dropna()
    if gaps.empty:
        return False, "无法计算间隔"
    worst = int(gaps.max())
    if worst > MAX_GAP_DAYS:
        idx = gaps.idxmax()
        return False, f"数据不连续：最大间隔 {worst} 天（断点 {idx.date()} 附近，疑似整段缺失）"
    return True, ""


# ==============================================================================
# 0. 传输层：urllib 优先，连接层失败退 curl
# ==============================================================================

# 确保国内金融数据接口直连，规避本地代理的 SSL 握手阻断与空包
_DOMESTIC_HOSTS = "eastmoney.com,push2his.eastmoney.com,sina.com.cn,gtimg.cn,ifzq.gtimg.cn"
_existing_no_proxy = os.environ.get("no_proxy", "")
if not any(h in _existing_no_proxy for h in ["eastmoney.com", "gtimg.cn"]):
    os.environ["no_proxy"] = f"{_existing_no_proxy},{_DOMESTIC_HOSTS}".strip(",")


def http_get(url: str, timeout: int = 20):
    """返回 (text 或 None, meta)。meta 含 transport / status / error。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = getattr(r, "status", 200)
            return r.read().decode("utf-8", "ignore"), {"transport": "urllib", "status": status}
    except Exception as e1:
        err1 = f"{type(e1).__name__}"

    try:
        cmd = ["curl", "-s", "-m", str(timeout), "--noproxy", _DOMESTIC_HOSTS, "-A", UA, "-w", "\n__ST__%{http_code}", url]
        out = subprocess.run(
            cmd,
            capture_output=True, timeout=timeout + 15)
        txt = out.stdout.decode("utf-8", "ignore")
        status = 0
        if "__ST__" in txt:
            body, _, st = txt.rpartition("__ST__")
            txt, status = body, int((st or "0").strip() or 0)
        if txt.strip():
            return txt, {"transport": "curl", "status": status, "urllib_error": err1}
        return None, {"transport": "curl", "status": status, "error": "empty body"}
    except Exception as e2:
        return None, {"error": f"urllib:{err1} / curl:{type(e2).__name__}"}


# ==============================================================================
# 1. 解析与校验
# ==============================================================================

def _to_frame(rows: list, date_col: int = 0, o: int = 1, c: int = 2, h: int = 3,
              l: int = 4, v: int = 5, source: str = "", epoch: str = None) -> pd.DataFrame:
    """按 (date, open, close, high, low, volume) 的位置解析，并做结构校验。
    epoch：若日期列是时间戳（如 OKX 的毫秒），传 'ms' / 's'。"""
    recs = []
    for r in rows:
        try:
            raw = r[date_col]
            ts = (pd.to_datetime(int(raw), unit=epoch, utc=True).tz_localize(None)
                  if epoch else pd.Timestamp(str(raw)[:10]))
            recs.append({
                "Date": ts.normalize() if epoch else ts,
                "Open": float(r[o]), "Close": float(r[c]),
                "High": float(r[h]), "Low": float(r[l]),
                "Volume": float(r[v] or 0),
            })
        except (ValueError, TypeError, IndexError):
            continue
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs).drop_duplicates(subset="Date").set_index("Date").sort_index()
    df = df[OHLC].astype(float)

    # 结构校验：H/L 必须包住 O/C。错误率过高说明字段顺序猜错，宁可报错。
    body_hi = df[["Open", "Close"]].max(axis=1)
    body_lo = df[["Open", "Close"]].min(axis=1)
    bad = ((df["High"] < body_hi) | (df["Low"] > body_lo)).sum()
    if len(df) and bad / len(df) > 0.05:
        raise ValueError(f"{source} 字段顺序可疑：{bad}/{len(df)} 根 K 线 High/Low 不含 Open/Close")

    return df[(df[OHLC] > 0).all(axis=1)]


# ==============================================================================
# 2. 各数据源
# ==============================================================================

class Source:
    name = "base"
    down_after = 2          # 连续失败多少只标的后熔断

    def __init__(self):
        self.fails = 0
        self.ok = 0
        self.last_reason = ""

    @property
    def down(self) -> bool:
        return self.fails >= self.down_after

    def accepts(self, ticker: str) -> bool:
        """是否负责该标的。用于把 BTC/ETH 路由到 OKX、其余走股票源。"""
        return True

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        raise NotImplementedError

    def _fail(self, reason: str, transport: bool = False) -> pd.DataFrame:
        """transport=True 才算「源级故障」（影响熔断）；标的缺失/代码不存在不算。"""
        if transport:
            self.fails += 1
        else:
            self.misses = getattr(self, "misses", 0) + 1
        self.last_reason = reason
        return pd.DataFrame()

    def _ok(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fails = 0
        self.ok += 1
        return df


class OKXSource(Source):
    r"""
    OKX 现货日线（仅 BTC / ETH，按需求限定）。

    为什么不用 OAuth：`D:\AI_Workspace\OKX_Agent\okx_oauth_bridge.js` 那套 PKCE 流程是给
    **交易 MCP**（/api/v1/mcp/trading-oauth）用的；K 线属于公开行情，鉴权不需要 token。
    但那个脚本有两点值得复用：
      1. 它证明 www.okx.com 在正常网络下可达；
      2. 它手动实现了 HTTP CONNECT 代理隧道 —— 说明该站点必须走 CONNECT，
         所以本适配器依赖 curl 的自动 CONNECT（urllib 在部分环境下会被代理断开）。

    接口：GET /api/v5/market/history-candles?instId=BTC-USDT&bar=1D&limit=100&after={ts}
      - 单次最多 100 根，按 after（返回早于该时间戳的数据）向前翻页
      - 返回行：[ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]，**时间倒序**
      - confirm='0' 表示该根尚未收盘（当日进行中），必须剔除，否则最后一根会重绘
    """
    name = "okx"
    BASE = "https://www.okx.com"
    INST = {"BTC": "BTC-USDT", "ETH": "ETH-USDT"}   # 只允许这两个

    def accepts(self, ticker: str) -> bool:
        return ticker.upper() in self.INST

    def _page(self, inst: str, after: str = ""):
        # 用 1Dutc：日线按 UTC 零点切分，避免 UTC+8 切分导致的日期错位
        url = (f"{self.BASE}/api/v5/market/history-candles"
               f"?instId={inst}&bar=1Dutc&limit=100")
        if after:
            url += f"&after={after}"
        txt, meta = http_get(url)
        if txt is None:
            return None, f"传输失败: {meta.get('error', meta.get('status'))}"
        try:
            d = json.loads(txt)
        except json.JSONDecodeError:
            return None, f"非 JSON（HTTP {meta.get('status')}），多为代理拒绝 CONNECT"
        if str(d.get("code")) != "0":
            return None, f"接口拒绝: code={d.get('code')} msg={d.get('msg')}"
        rows = d.get("data") or []
        if not rows:
            return None, "无数据（可能已到该交易对上市之前）"
        return rows, ""

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        inst = self.INST.get(ticker.upper())
        if not inst:
            return self._fail(f"仅支持 {list(self.INST)}，收到 {ticker}")

        rows, after, err = [], "", ""
        while len(rows) < bars:
            page, err = self._page(inst, after)
            if page is None:
                break
            rows += page
            oldest = page[-1][0]          # 该页最老的一根（返回是倒序）
            if oldest == after:
                break
            after = oldest
            time.sleep(THROTTLE)

        if not rows:
            return self._fail(err or "无数据")

        # 只保留已收盘的 K 线，并转成升序
        rows = [r for r in rows if len(r) >= 9 and str(r[8]) == "1"]
        rows.reverse()
        try:
            # OKX 列序：ts, open, high, low, close, vol（注意 h 在 l 前面）
            df = _to_frame(rows, date_col=0, o=1, h=2, l=3, c=4, v=5,
                           source=f"{self.name}/{inst}", epoch="ms")
        except ValueError as e:
            return self._fail(str(e))
        if df.empty:
            return self._fail("解析后为空")
        return self._ok(df.tail(bars))


class EastmoneySource(Source):
    """东方财富：前复权日线。secid 前缀 105=纳斯达克 / 106=纽交所 / 107=美交所"""
    def accepts(self, ticker: str) -> bool:
        return ticker.upper() not in OKXSource.INST and not ticker.upper().endswith("-USD")

    name = "eastmoney"

    def __init__(self):
        super().__init__()
        self.secids = {}
        if os.path.exists(SECID_PATH):
            try:
                with open(SECID_PATH, encoding="utf-8") as f:
                    self.secids = json.load(f)
            except Exception:
                self.secids = {}

    def _url(self, secid: str, bars: int) -> str:
        return ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
                f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
                "&fields2=f51,f52,f53,f54,f55,f56,f57"
                f"&klt=101&fqt=1&beg=19900101&end=20500101&lmt={max(bars, 100)}")

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        cands = [self.secids[ticker]] if ticker in self.secids else \
                [f"105.{ticker}", f"106.{ticker}", f"107.{ticker}"]

        last = "无匹配 secid"
        transport_hit = False
        for secid in cands:
            txt, meta = http_get(self._url(secid, bars))
            if txt is None:
                last = f"传输失败: {meta.get('error', meta.get('status'))}"
                transport_hit = True
                continue
            if not txt.lstrip().startswith("{"):
                last = f"非 JSON 响应（HTTP {meta.get('status')}）"
                continue
            try:
                d = json.loads(txt)
            except json.JSONDecodeError:
                last = "JSON 解析失败"
                continue

            data = d.get("data")
            if not isinstance(data, dict) or not data.get("klines"):
                last = f"无 K 线（secid={secid}）"
                continue

            rows = [ln.split(",") for ln in data["klines"]]
            try:
                df = _to_frame(rows, source=f"{self.name}/{secid}")
            except ValueError as e:
                last = str(e)
                continue
            if df.empty:
                last = "解析后为空"
                continue

            if self.secids.get(ticker) != secid:
                self.secids[ticker] = secid
                _save_json(SECID_PATH, self.secids)
            return self._ok(df)

        return self._fail(last, transport=transport_hit)


class TencentSource(Source):
    """
    腾讯：前复权日线，单次最多 640 根，按 end 日期向前翻页。
    美股代码需带交易所后缀：.OQ 纳斯达克 / .N 纽交所 / .A 美交所。
    """
    def accepts(self, ticker: str) -> bool:
        return ticker.upper() not in OKXSource.INST and not ticker.upper().endswith("-USD")

    name = "tencent"
    SUFFIXES = [".OQ", ".N", ".A"]

    def __init__(self):
        super().__init__()
        self.suffix = {}
        meta = _load_json(META_PATH)
        for k, v in meta.items():
            if v.get("source") == "tencent" and v.get("symbol"):
                self.suffix[k] = v["symbol"]

    def _page(self, symbol: str, end: str, count: int = 640):
        # beg 必须显式给出：留空会导致接口按 "bad params" 拒绝，翻页拿不到历史
        url = ("https://web.ifzq.gtimg.cn/appstock/app/usfqkline/get"
               f"?param={symbol},day,1990-01-01,{end},{count},qfq")
        txt, meta = http_get(url)
        if txt is None:
            return None, f"传输失败: {meta.get('error', meta.get('status'))}"
        try:
            d = json.loads(txt)
        except json.JSONDecodeError:
            return None, f"非 JSON（HTTP {meta.get('status')}）"
        if d.get("code") != 0:
            return None, f"接口拒绝: {d.get('msg')}"
        node = (d.get("data") or {}).get(symbol) or {}
        for key in ("qfqday", "day"):
            arr = node.get(key)
            if isinstance(arr, list) and len(arr) >= 2:
                return arr, ""
        return None, "无 K 线"

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        tries = [self.suffix[ticker]] if ticker in self.suffix else \
                [f"us{ticker}{s}" for s in self.SUFFIXES]

        last = "无匹配代码"
        for symbol in tries:
            chunks, end = [], pd.Timestamp.today().strftime("%Y-%m-%d")
            err = ""
            while len(chunks) < bars:
                arr, err = self._page(symbol, end)
                if arr is None:
                    break
                chunks = arr + chunks
                first = pd.Timestamp(str(arr[0][0])[:10])
                nxt = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if nxt >= end:
                    break
                end = nxt
                time.sleep(THROTTLE)
                if first.year < 2000:
                    break

            if not chunks:
                last = err or "无数据"
                continue
            try:
                df = _to_frame(chunks, source=f"{self.name}/{symbol}")
            except ValueError as e:
                last = str(e)
                continue
            if df.empty:
                last = "解析后为空"
                continue

            if self.suffix.get(ticker) != symbol:
                self.suffix[ticker] = symbol
            return self._ok(df)

        # 只有传输层/非 JSON 才算源级故障；「该标的无历史」属标的级缺失，不应熔断
        return self._fail(last, transport=("传输失败" in last or "非 JSON" in last))


class SinaSource(Source):
    r"""
    新浪财经美股日 K：历史最长（SPY 可回溯到 2001），但**未复权**。

    接口：GET stock.finance.sina.com.cn/usstock/api/json_v2.php/US_MinKService.getDailyK
          ?symbol=SPY&___qn=3
    返回：[{d,o,h,l,c,v,a}, ...]，升序，无需翻页。

    为什么需要它：腾讯对 SPY / DIA / IWM 只返回 1 行占位（不提供该标的历史）；
    而道氏理论原本研究的正是指数。没有这个源，「指数 vs 个股」这个检验就做不了。
    代价是未复权——分红日会有约 1~2% 的价格断层，对结构判定是噪声源，
    但不影响「趋势延续性」这类中等尺度的检验。此限制需在读结论时记住。
    """
    name = "sina"

    def accepts(self, ticker: str) -> bool:
        return ticker.upper() not in OKXSource.INST and not ticker.upper().endswith("-USD")

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        url = ("https://stock.finance.sina.com.cn/usstock/api/json_v2.php/"
               f"US_MinKService.getDailyK?symbol={ticker.upper()}&___qn=3")
        txt, meta = http_get(url, timeout=30)
        if txt is None:
            return self._fail(f"传输失败: {meta.get('error', meta.get('status'))}", transport=True)
        body = txt.lstrip()
        if not body.startswith("["):
            return self._fail(f"非数组响应（HTTP {meta.get('status')}）")
        try:
            arr = json.loads(txt)
        except json.JSONDecodeError:
            return self._fail("JSON 解析失败", transport=True)
        if not arr:
            return self._fail("返回空数组（代码可能不存在）")

        rows = []
        for r in arr:
            if isinstance(r, dict) and "d" in r:
                rows.append([r["d"], r.get("o"), r.get("h"), r.get("l"),
                             r.get("c"), r.get("v") or 0])
        try:
            df = _to_frame(rows, date_col=0, o=1, h=2, l=3, c=4, v=5,
                           source=f"{self.name}/{ticker.upper()}")
        except ValueError as e:
            return self._fail(str(e))
        if df.empty:
            return self._fail("解析后为空")
        return self._ok(df.tail(bars))


class YfinanceSource(Source):
    """yfinance：仅作兜底。本环境实测整域 403，库会误报为 YFRateLimitError。"""
    def accepts(self, ticker: str) -> bool:
        return ticker.upper() not in OKXSource.INST

    name = "yfinance"
    down_after = 1

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        try:
            import pa_engine as pe
            period = "5y" if bars <= 1260 else "10y"
            df = pe.fetch_klines(ticker, "1d", period, retries=1)
            if df.empty:
                return self._fail("返回空（Yahoo 多为 403 整域拦截，非限流）", transport=True)
            return self._ok(df[OHLC].dropna())
        except Exception as e:
            return self._fail(f"{type(e).__name__}: {e}")


class StooqSource(Source):
    """stooq：纯 CSV。本环境返回 200 + JS 工作量证明页，检测到后明确报「反爬挑战」。"""
    def accepts(self, ticker: str) -> bool:
        return ticker.upper() not in OKXSource.INST

    name = "stooq"
    down_after = 1

    def fetch(self, ticker: str, bars: int) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        txt, meta = http_get(url)
        if txt is None:
            return self._fail(f"传输失败: {meta.get('error', meta.get('status'))}", transport=True)
        if txt.lstrip().startswith("<"):
            return self._fail("返回 HTML（JS 工作量证明挑战页），需浏览器环境", transport=True)
        lines = txt.strip().splitlines()
        if len(lines) < 3 or not lines[0].lower().startswith("date"):
            return self._fail("CSV 格式异常")
        rows = [ln.split(",") for ln in lines[1:]]
        try:
            df = _to_frame(rows, source=self.name)
        except ValueError as e:
            return self._fail(str(e))
        return self._ok(df.tail(bars)) if not df.empty else self._fail("解析后为空")


# ==============================================================================
# 3. 调度：缓存优先，逐源降级
# ==============================================================================

SOURCES = [OKXSource, EastmoneySource, TencentSource, SinaSource,
           YfinanceSource, StooqSource]


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cache_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.upper()}.csv")


def read_cache(ticker: str) -> pd.DataFrame:
    p = cache_path(ticker)
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True).dropna(subset=OHLC[:4])
    except Exception:
        return pd.DataFrame()
    ok, why = check_continuity(df)
    if not ok:
        print(f"[警告] 缓存 {ticker} 不可用 -> {why}", file=sys.stderr)
        return pd.DataFrame()
    return df


def write_cache(ticker: str, df: pd.DataFrame, source: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(cache_path(ticker))
    meta = _load_json(META_PATH)
    meta[ticker.upper()] = {"source": source.split("/")[0],
                            "symbol": source.split("/")[1] if "/" in source else ticker.upper(),
                            "bars": int(len(df)),
                            "start": str(df.index[0].date()),
                            "end": str(df.index[-1].date()),
                            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_json(META_PATH, meta)


def fetch_history(ticker: str, bars: int = 1600, allow_network: bool = True,
                  sources: list = None, force_update: bool = False,
                  max_age_days: int = 5) -> tuple:
    """返回 (df, note)。note 说明数据来自哪里、为什么失败。"""
    ticker = ticker.upper()
    cached = read_cache(ticker)
    if not force_update and not cached.empty and len(cached) >= min(300, bars * 0.5):
        if not allow_network:
            return cached, f"cache({len(cached)} 根)"
        last_ts = pd.Timestamp(cached.index[-1])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_localize(None)
        if (pd.Timestamp.now() - last_ts).days <= max_age_days:
            return cached, f"cache({len(cached)} 根)"

    if not allow_network:
        return (cached, f"cache({len(cached)} 根)") if not cached.empty else (pd.DataFrame(), "无缓存且未允许联网")

    for src in (sources if sources is not None else SOURCES):
        inst = src() if isinstance(src, type) else src
        if inst.down or not inst.accepts(ticker):
            continue
        try:
            got = inst.fetch(ticker, bars)
        except Exception as e:
            got = inst._fail(f"未捕获异常 {type(e).__name__}: {e}")
        if not got.empty:
            if not cached.empty:
                merged = pd.concat([cached, got])
                final_df = merged[~merged.index.duplicated(keep="last")].sort_index()
            else:
                final_df = got
            ok, why = check_continuity(final_df)
            if not ok:
                # 不写入缓存，继续换下一个源
                print(f"[警告] {inst.name} 的 {ticker} 合并后校验被拒 -> {why}", file=sys.stderr)
                inst.last_reason = why
                time.sleep(THROTTLE)
                continue
            write_cache(ticker, final_df, f"{inst.name}/{inst.suffix.get(ticker, ticker)}"
                       if hasattr(inst, "suffix") else f"{inst.name}/{ticker}")
            action_desc = "更新至" if not cached.empty else "拉取"
            return final_df, f"{inst.name}({action_desc} {len(final_df)} 根，止于 {final_df.index[-1].date()})"
        time.sleep(THROTTLE)

    if not cached.empty:
        return cached, f"全部网络源失败，保留旧缓存({len(cached)} 根)"
    return pd.DataFrame(), "全部数据源不可用"


def fetch_many(tickers: list, bars: int = 1600, force_update: bool = False) -> dict:
    srcs = [s() for s in SOURCES]
    report = {}
    for i, tk in enumerate(tickers, 1):
        df, note = fetch_history(tk, bars=bars, allow_network=True, sources=srcs, force_update=force_update)
        report[tk] = {"ok": not df.empty, "bars": len(df), "note": note}
        flag = "OK " if not df.empty else "FAIL"
        print(f"  [{i:>2}/{len(tickers)}] {flag} {tk:<8} {note}", file=sys.stderr)
    print("\n数据源使用情况：", file=sys.stderr)
    for s in srcs:
        print(f"  {s.name:<10} 成功 {s.ok:>3} 次，连续失败 {s.fails} 次"
              f"{'（已熔断）' if s.down else ''}"
              f"{'  最近原因: ' + s.last_reason if s.last_reason else ''}", file=sys.stderr)
    return report


# ==============================================================================
# 4. CLI
# ==============================================================================

def probe(tickers=("AAPL", "BTC")) -> int:
    print(f"{'源':<12}{'标的':<8}{'HTTP':>6}{'传输':>10}{'根数':>7}  结果")
    print("-" * 78)
    alive = 0
    for cls in SOURCES:
        inst = cls()
        for tk in tickers:
            if not inst.accepts(tk):
                print(f"{inst.name:<12}{tk:<8}{'—':>6}{'':>10}{0:>7}  不负责该标的")
                continue
            try:
                df = inst.fetch(tk, 400)
            except Exception as e:
                df = pd.DataFrame()
                inst.last_reason = f"{type(e).__name__}: {e}"
            meta = _load_json(META_PATH)
            status = "—"
            note = inst.last_reason or "ok"
            print(f"{inst.name:<12}{tk:<8}{status:>6}{'':>10}{len(df):>7}  {note}")
            time.sleep(THROTTLE)
        if inst.ok:
            alive += 1
    print("-" * 78)
    print(f"可用源: {alive}/{len(SOURCES)}")
    return 0 if alive else 1


def status() -> int:
    meta = _load_json(META_PATH)
    if not meta:
        print(f"缓存为空（{DATA_DIR}）")
        return 1
    print(f"{'标的':<8}{'来源':<12}{'根数':>6}  {'起':<12}{'止':<12}拉取时间")
    print("-" * 72)
    for tk, m in sorted(meta.items()):
        print(f"{tk:<8}{m.get('source',''):<12}{m.get('bars',0):>6}  "
              f"{m.get('start',''):<12}{m.get('end',''):<12}{m.get('fetched_at','')}")
    print(f"\n共 {len(meta)} 个标的")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="多来源行情适配器")
    ap.add_argument("--probe", action="store_true", help="探测各源可用性")
    ap.add_argument("--fetch", type=str, default="", help="逗号分隔标的")
    ap.add_argument("--fetch-file", type=str, default="", help="每行一个标的")
    ap.add_argument("--bars", type=int, default=1600, help="目标根数（默认 1600 ≈ 6.4 年）")
    ap.add_argument("--status", action="store_true", help="查看缓存清单")
    ap.add_argument("--update", "--force", dest="force_update", action="store_true",
                    help="强制从网络拉取最新数据并增量合并到本地缓存")
    args = ap.parse_args()

    if args.probe:
        return probe()
    if args.status:
        return status()

    tickers = [t.strip().upper() for t in args.fetch.split(",") if t.strip()]
    if args.fetch_file:
        ff_path = args.fetch_file
        if not os.path.exists(ff_path) and os.path.exists(os.path.join("config", ff_path)):
            ff_path = os.path.join("config", ff_path)
        with open(ff_path, encoding="utf-8") as f:
            tickers += [ln.split("#")[0].strip().upper() for ln in f
                        if ln.split("#")[0].strip()]
    if not tickers:
        print("[错误] 请指定 --probe / --fetch / --fetch-file / --status", file=sys.stderr)
        return 1

    rep = fetch_many(tickers, bars=args.bars, force_update=args.force_update)
    ok = sum(1 for v in rep.values() if v["ok"])
    print(f"\n成功 {ok}/{len(tickers)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
