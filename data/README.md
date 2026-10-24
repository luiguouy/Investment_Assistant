# data/ 目录：日线数据缓存

`backtest.py` 优先读这里的 CSV，读不到才联网。

## 文件名

```
data/AAPL.csv
data/MSFT.csv
```

## 格式

| 项 | 要求 |
|:---|:---|
| 索引列 | 第一列，列名为空或 `Date`，格式 `YYYY-MM-DD` |
| 必需列 | `Open,High,Low,Close,Volume` |
| 复权 | **建议前复权**（拆分/分红已调整）。未复权的数据会把拆股当成跳空，污染结构与缺口判定 |
| 排序 | 日期升序 |
| 行数 | 越长越好，建议 ≥ 1200 根（约 5 年） |

最小示例：

```csv
Date,Open,High,Low,Close,Volume
2021-01-04,133.52,133.61,126.76,129.41,143301900
2021-01-05,128.89,131.74,128.43,131.26,97664900
```

## 怎么拿数据

四条路，按成本排序：

1. **自动（股票）**：`python data_sources.py --fetch-file tickers.txt` —— 走 缓存→东财→腾讯→yfinance→stooq 降级链。腾讯当前可用，30 只约 2.5 分钟。
2. **自动（BTC/ETH）**：`python data_sources.py --fetch-file tickers_crypto.txt` —— 走 OKX 公开行情（`/api/v5/market/history-candles`，`1Dutc`）。**需要 okx.com 可达**；本沙箱的代理拒绝该域名，见下。
3. **券商/行情软件导出**：多数支持导出 CSV，注意勾选"前复权"。
4. **Yahoo Finance 网页**：`finance.yahoo.com` → Historical Data → Download。浏览器能打开，Python 直连会被 403（环境代理所致，详见 `../backtest_protocol.md` 第七节）。

### 关于 OKX

本环境实测：`CONNECT www.okx.com:443` 被本地代理拒绝（**502**），明文 HTTP 同样 502；作为对照，`web.ifzq.gtimg.cn` 正常返回 200。币安 / Coinbase / Bybit 也全部不可达。

也就是说 **OKX 的适配器代码已就绪、解析层已用离线夹具验证通过**（见 `../test_data_sources.py`），缺的只是网络放行。换到 OKX 可达的网络后直接执行第 2 条即可，无需改代码。

## 为什么不让程序自己绕过

stooq 的验证页是工作量证明（SHA-256 找前导零），Yahoo 的 403 是访问控制，代理的 502 是出网策略。破解或伪造都会绕过访问控制，且随时失效——建立在这样的通道上的回测结论不可复用。


## 校验

放好数据后，先看一眼有没有问题：

```bash
python -c "
import pandas as pd, glob
for p in sorted(glob.glob('data/*.csv')):
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    print(f'{p:28s} {len(d):5d} 根  {d.index[0].date()} -> {d.index[-1].date()}  '
          f'空值={int(d.isna().sum().sum())}  最低价={d.Low.min():.2f}')
"
```

`最低价 ≤ 0` 或空值 > 0 的，先修数据再跑回测。
