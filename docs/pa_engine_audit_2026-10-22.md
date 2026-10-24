# pa_engine.py 代码审计与优化建议

- **审计对象**：`D:/Antigravity_Workspaces/Investment_Assistant/pa_engine.py`（851 行，v2.0）
- **配套规范**：`.agents/skills/us-stock-price-action-analysis/SKILL.md`、`.antigravity/skills/us_stock_elliott_dow_pa.md`
- **审计日期**：2026-10-22
- **审计方式**：静态通读 + 7 组可复现对照实验（本地注入合成数据，不依赖行情源）+ 1 次真实运行
- **运行环境**：Python 3.12.5 / numpy 2.2.6 / pandas 2.3.3 / scipy 1.16.2 / yfinance 1.2.0

> **阅读说明（v2.2 更新）**：本文第一至五节描述的是 **v2.0 基线代码**的问题与建议，保留原文以便对照。
> 其中 P0 四项已在 **v2.1** 修复，P1-1 / P1-2 / P1-3 / P1-4 / P3-2 已在 **v2.2** 修复，详见第七节与第七节之二。
> 因此第一节「判断 1」（风控完全不在代码层）已被 v2.2 部分推翻：止损、盈亏比、仓位现在由代码计算并强制拦截。

---

## 一、结论先行

### 判断 1：该文件不是"交易策略"，是"结构数据采集 + 终端渲染"脚本

全文没有任何交易决策要素：无信号生成、无入场/止损/止盈计算、无仓位规模、无盈亏比、无持仓状态、无回测。`analyze_single_ticker()` 的返回值是 `None`，唯一输出是 `print`。

这意味着 **"风险控制"在本项目中目前不存在于代码层**，而是由 LLM 读取终端文本后按 SKILL.md 模板手写。而 SKILL.md 第 108 行写着硬规则：

> 预估盈亏比 `(TP - Entry) / (Entry - SL)`（若低于 1:2 必须提示放弃）

该规则既不可执行、不可审计、不可复现——它依赖 LLM 每次对四个数字做心算。一旦算错或漏算，没有任何机制拦截。**这是本次审计最重要的发现，也是最高优先级的缺口。**

### 判断 2：存在 3 个会直接产出错误结论的正确性缺陷

不是风格问题，是会算错数的问题。均已用实验复现（见第二节）：

| 缺陷 | 后果 |
|:---|:---|
| 末位拐点未确认即被当作当前结构 | 96% 的运行中，报告里的"最后拐点"都是**尚未成立的拐点**；回测必然使用未来信息，实盘每次刷新结论会漂移 |
| 盘前/开盘区间按"根数窗口"截取 | 5m 数据下 `PMH` 只覆盖 07:30–09:30，**04:00–07:30 的盘前高低被丢弃**；日内 2B 猎杀位直接错位 |
| 斐波那契共振区用"单链聚类" | 8 个跨度 6.5% 的价位被合并成 1 个 $103.07–$106.11 的"🔥强共振"，宽度是设定容差的 3 倍 |

### 判断 3：参数体系是 12 个魔法数，且没有注入通道

`order`（3/5/7、6/5/4）、缺口阈值 0.008、共振容差 0.01、量比 1.5/1.2/0.7/0.5、背离 0.8、`lookback=20`、`recent=5`、`tail(150)` 全部硬编码。既无法 A/B 对照，也无法按标的波动率缩放。**在这些参数被标定之前，"策略表现如何"这个问题无法回答**——任何关于胜率/期望值的说法都缺乏依据。

---

## 二、验证方法与实验结果

以下实验全部用合成数据，可复现，不依赖行情源。

| 编号 | 实验内容 | 结果 |
|:---:|:---|:---|
| E1 | 同一序列逐步加长（100→120 根），观察 `extract_swings(order=5)` 的末位拐点 | bars=100 时末位拐点落在第 100 根（**0 根确认**）；bars=105 时同一根被新拐点取代 → 重绘 |
| E2 | 200 次随机游走，统计末位拐点距 K 线末端的根数 | **96% 的样本距末端 <5 根（未确认），中位距离 1 根** |
| E3 | 40 个相邻 0.9% 递增价位，容差 1%，测共振区聚类 | 输出 1 个簇，跨度 $103.07–$106.11（3%），`count=18`，标注"🔥强共振" → 单链漂移 |
| E4 | 2 个交易日 × 192 根 5m（prepost 全天），测 `PMH` | `tail(150)` 起点为次日 07:30；当日 66 根盘前仅 24 根参与计算 → 漏 04:00–07:30 |
| E5 | 10 个拐点（1 次真实结构破位）喂 `detect_mss` | 输出 2 条事件，全部标注"多头结构被破坏" → 重复触发 |
| E6 | 同一 HH/HL 序列，间隔 7 天 vs 间隔 365 天 | `_determine_dow_phase` 输出完全相同 → 无时间衰减 |
| E7 | 真实运行 `python pa_engine.py --ticker AAPL --mode day` | 9 次请求全部 `YFRateLimitError`，仅 1 行"严重错误"，**退出码 0** |
| E8 | 无 `Volume` 列的 30 根 DataFrame 喂 `analyze_volume` | `KeyError: 'Volume'` 崩溃 |
| E9 | 6 根 5m K 线，末根为长下影，喂 `scan_price_action` | 末根（进行中 K 线）触发 2 条信号：Pin Bar + 2B |

E7 尤其值得注意：**当前这台机器上 yfinance 已处于限流状态，脚本每次都以"成功"退出**。任何把它挂进定时任务或自动化流水线的方案，都会在数据完全缺失的情况下持续产出空报告而不报警。

---

## 三、问题清单（按优先级）

### 优先级总览

| ID | 级别 | 问题 | 影响面 | 修复成本 |
|:---|:---:|:---|:---|:---:|
| P0-1 | 阻断 | 末位拐点未确认（未来函数 + 重绘） | 所有结构结论、回测 | 低 |
| P0-2 | 阻断 | 盘前/开盘区间窗口截断、无会话过滤 | 日内模式全部关键位 | 低 |
| P0-3 | 阻断 | 共振区单链聚类 | Fib 共振、2B 信号 | 低 |
| P0-4 | 阻断 | 取数失败静默降级、退出码恒 0 | 自动化可靠性 | 中 |
| P1-1 | 高 | 无止损/仓位/盈亏比 → 风控不在代码层 | 策略可执行性 | 中 |
| P1-2 | 高 | key_levels 无分级去重 → 2B 信号通胀 | 信号信噪比 | 中 |
| P1-3 | 高 | PA 形态无位置过滤（脱离关键位） | 信号信噪比 | 低 |
| P1-4 | 高 | 扫描窗口含未收盘 K 线、同 bar 多信号叠加 | 实盘可用性 | 低 |
| P1-5 | 高 | 成交量用 20 根均量做日内基准（U 型季节性） | 量能结论 | 中 |
| P1-6 | 高 | 量价背离判据过宽、顶底可同时成立 | 背离结论 | 低 |
| P1-7 | 中 | MSS 重复触发、无新鲜度过滤 | "近期无结构转换"不可信 | 低 |
| P1-8 | 中 | 道氏阶段判定无时间衰减、无幅度阈值 | 大势定性 | 低 |
| P1-9 | 中 | 艾略特"三大铁律"完全未实现 | 文档宣称≠代码能力 | 高 |
| P2-1 | 中 | `order` 魔法数，不随数据量/波动自适应 | 拐点密度（决定一切下游） | 中 |
| P2-2 | 中 | 全部阈值硬编码，无配置层 | 无法调参/验证 | 中 |
| P2-3 | 中 | Fib 比例用绝对容差 0.05 → 虚假精确匹配 | 波浪比例结论 | 低 |
| P2-4 | 中 | Fib 基准腿不筛选（未完成/噪声/逆向腿） | 回撤位质量 | 低 |
| P2-5 | 中 | `auto_adjust=False` → 拆股被误判为缺口 | 缺口/拐点 | 低 |
| P3-1 | 低 | 请求数 10 次/标的，无缓存无并发 | 吞吐、限流 | 中 |
| P3-2 | 低 | print-only，无结构化输出 | 可测试性、可集成性 | 中 |
| P3-3 | 低 | watchlist 分支仍做全量计算、`pa[0]` 取最早信号 | 性能、可用性 | 低 |
| P3-4 | 低 | `from datetime import time` 遮蔽标准库 | 维护性 | 极低 |
| P3-5 | 低 | `except Exception: pass` 吞掉时区/索引错误 | 静默失效 | 极低 |
| P3-6 | 低 | 缺 `Volume` 列直接 KeyError | 鲁棒性 | 低 |
| P3-7 | 低 | 两份 skill 文档内容完全相同 | 维护漂移 | 极低 |

---

### P0-1 末位拐点未确认：未来函数与重绘

**现象**：`extract_swings()` 返回的最后一个拐点，经常就是最后一根或倒数第二根 K 线。

**证据**（E1 / E2）：200 次随机样本中，96% 的末位拐点距 K 线末端不足 5 根，中位距离 1 根。

**根因**：`argrelextrema(values, np.greater_equal, order=5)` 的语义是"该点在左右各 5 根范围内最大"。第 i 根能否成为拐点，取决于 i+1…i+5 根的数据——即需要**未来 5 根**才能确认。代码把 `swings[-1]` 直接当作"当前结构"，交给：

- `_determine_dow_phase()` → 定性"当前"趋势
- `calc_fibonacci_and_levels(recent_mtf)` → 用 `swings[-2] → swings[-1]` 当基准腿
- `key_hlines` → 当作 2B 猎杀位
- `calc_multi_leg_fib()` → 计算腿长比例

**影响**：
1. 若用于回测，等于在 t 时刻使用了 t+5 的信息，胜率会被系统性高估；
2. 实盘中每来一根新 K 线，末位拐点可能消失或被替换，同一标的连续两次分析可得出相反的"道氏阶段"；
3. 以未完成的腿计算斐波那契扩展位，其数值本身没有意义（腿还没走完）。

**修法**：拐点增加 `confirmed` 标记，结构判定只用已确认拐点；未确认的单独以 `pending` 呈现，明确标注"待确认"。

```python
def extract_swings(df: pd.DataFrame, order: int = 5, confirmed_only: bool = True) -> list:
    ...
    cutoff = len(df) - order - 1          # 右侧不足 order 根 → 无法确认
    for pt in clean_swings:
        pt["confirmed"] = pt["index"] <= cutoff

    if confirmed_only:
        while clean_swings and not clean_swings[-1]["confirmed"]:
            clean_swings.pop()            # 只裁尾部，历史点不受影响
    return clean_swings
```

同时建议：`analyze_single_ticker()` 额外输出一行"末位未确认拐点：$xxx（待 N 根确认）"，让报告读者知道结构还在变动。

---

### P0-2 盘前/开盘区间：窗口截断 + 无会话过滤

**现象**：`PMH/PML/ORH/ORL` 的数值与实际不符。

**证据**（E4）：两日 5m 数据（prepost，192 根/日）共 384 根，`tail(150)` 起点落在**次日 07:30**。当日实际有 66 根盘前 K 线（04:00–09:30），但只有 24 根进入 `PMH` 计算。

**根因**（`calc_fibonacci_and_levels` 第 289–306 行）有双重缺陷：

```python
df_recent = df_intraday.tail(150).copy()   # ① 按"根数"截取，不是按"会话"
times = df_recent.index.time
pre_market_mask = [(t >= time(4, 0)) and (t < time(9, 30)) for t in times]  # ② 只按时刻过滤，不按日期
```

- ① 150 根 5m ≠ 150 根跨 1 天。prepost 全天数据下 150 根只有 0.78 个交易日，盘前窗口被拦腰截断；若数据密度更低（如仅常规时段 78 根/日，或半日市），窗口又会跨到前一交易日。
- ② 掩码不含日期条件，一旦窗口跨日，前一日的盘前高/开区间高会混进来。

**影响**：`PMH/PML` 系统性偏低或偏高 → 2B 猎杀位错位 → 日内交易计划的入场/止损全部建立在错误价位上。这比"信号少一点"严重得多，因为它**错得很像对的**。

**修法**：先锁定最后一个交易日，再取时刻窗口；数据不足时显式输出 `N/A` 而非给出错值。

```python
if df_intraday is not None and not df_intraday.empty:
    dfi = df_intraday
    if dfi.index.tz is not None:
        dfi = dfi.tz_convert("America/New_York")   # 显式对齐交易所时区
    session = dfi.index[-1].date()
    dfi = dfi[dfi.index.date == session]            # 关键：先锁定会话
    t = dfi.index.time
    pm = dfi[[(x >= time(4, 0)) and (x < time(9, 30)) for x in t]]
    if len(pm) >= 24:                              # 覆盖不足则不输出
        levels["PMH"], levels["PML"] = round(pm["High"].max(), 2), round(pm["Low"].min(), 2)
```

---

### P0-3 斐波那契共振区：单链聚类产生假共振

**现象**：`calc_multi_leg_fib()` 输出的 `zone` 宽度经常是 `tolerance_pct` 的数倍，却标"🔥强共振"。

**证据**（E3）：8 个拐点（100.00 → 106.47，跨度 6.5%），容差 1%，引擎输出单一簇 `$103.07–$106.11`（宽 3%），`count=18`。

**根因**（第 380 行）：

```python
if group and abs(p - group[-1]) / max(abs(group[-1]), 0.01) <= tolerance_pct:
```

只与组内**上一个元素**比较，构成单链（single-linkage）：100.0 → 100.9 → 101.8 → …每步都在 1% 内，但首尾可相差任意远。这是聚类算法的经典退化情形。

次要问题：`count` 统计的是 Fib 数值个数，同一价位被多条腿重复生成时会重复计数，进一步夸大"汇聚强度"。

**影响**：假强共振 → 其 `center` 又被塞进 `key_hlines`（第 790 行）→ 反过来喂给 2B 检测 → 假信号放大。

**修法**：与组首（或组中心）比较，并以簇总宽度封顶；`count` 改为按来源腿去重。

```python
group = [all_fib[0]]
for p in all_fib[1:]:
    lo, hi = min(group[0], p), max(group[0], p)   # 与组首比较
    if (hi - lo) / max(abs(lo), 0.01) <= tolerance_pct:
        group.append(p)
    else:
        flush(group); group = [p]   # flush(): 组内去重后输出簇，同步原第 384 行逻辑
flush(group)
```

---

### P0-4 取数失败静默降级、退出码恒 0

**现象**：所有 K 线请求失败时，脚本仍以退出码 0 结束。

**证据**（E7，真实运行）：

```
[警告] 拉取 SPY (1h, 1mo) 失败: Too Many Requests. Rate limited. ...
...（共 9 条）
[严重错误] 未能获取 AAPL 的核心K线数据，分析中止。
=== EXIT CODE = 0 ===
```

**根因**：
1. `fetch_klines()` 用 `except Exception` 把限流、网络、符号不存在全部压成"返回空 DataFrame"，无重试、无退避、无缓存；
2. `analyze_single_ticker()` 只对 `df_mtf` 判空，HTF/LTF 为空时静默降级（`htf_phase` 变"数据不足"、`avg_volume=0`），报告照常输出；
3. `main()` 无失败计数，无 `sys.exit(非零)`；
4. 诊断信息一半在 stdout（`[严重错误]`）、一半在 stderr（`[警告]`），日志无法统一采集。

**影响**：自动化链路无法区分"分析完成"与"完全失败"。配合 E7 的现实（限流是常态，不是异常），这会让流水线持续产出空报告。

**修法**：

```python
# 1. 结构化失败 + 指数退避 + 本地缓存
def fetch_klines(ticker, interval, period, prepost=False, retries=3) -> pd.DataFrame:
    for i in range(retries):
        try:
            df = yf.Ticker(ticker).history(interval=interval, period=period,
                                           prepost=prepost, auto_adjust=True)
            if not df.empty:
                return _clean(df)
        except YFRateLimitError:
            time.sleep(2 ** i)          # 2s / 4s / 8s
    raise DataUnavailable(f"{ticker} {interval}/{period}")

# 2. 分级 fail-fast
#    MTF 缺失 → raise（终止该标的）
#    LTF 缺失 → 标注 "LTF 数据缺失，裸K信号不可用"，继续
#    HTF 缺失 → 标注 "HTF 缺失，大势验证跳过"，继续

# 3. main 汇总
failed = [t for t in tickers if not ok(t)]
sys.exit(1 if failed else 0)
```

---

### P1-1 风控不在代码层：无止损、无仓位、无盈亏比

**现象**：SKILL.md 第 98–108 行要求输出"狙击手交易计划"（硬止损、TP1、TP2、盈亏比，RR<1:2 必须放弃），但引擎不产出其中任何一项的客观锚点。

**根因**：`analyze_single_ticker()` 止步于"描述结构"，不进入"形成决策"。缺失三个层次：

| 缺失 | 应有输出 | 当前状态 |
|:---|:---|:---|
| 证伪位（Invalidation） | 最近已确认 swing 外侧 ± 缓冲 | 无 |
| 盈亏比（R:R） | 由 entry / stop / TP 计算 | 无（LLM 手算） |
| 仓位（Size） | `risk_pct × equity / stop_distance` | 无 |

**修法**：新增 `build_trade_plan()`，把风控从"文档承诺"变成"代码闸门"。

```python
def build_trade_plan(levels, swings, last_price, atr, side, risk_pct=0.006, equity=None):
    conf = [s for s in swings if s.get("confirmed")]
    if not conf:
        return {"status": "❌ 结构未确认，不成立"}

    if side == "long":
        anchor = min(s["price"] for s in conf[-3:] if s["type"] == "Low")
        stop = round(anchor - 0.25 * atr, 2)
        tp1 = levels["fib_extensions"].get("1.272")
        tp2 = levels["fib_extensions"].get("1.618")
    else:
        anchor = max(s["price"] for s in conf[-3:] if s["type"] == "High")
        stop = round(anchor + 0.25 * atr, 2)
        tp1, tp2 = levels["fib_extensions"].get("1.272"), levels["fib_extensions"].get("1.618")

    risk = abs(last_price - stop)
    if not tp1 or not risk:
        return {"status": "❌ 目标位或止损距离缺失，不成立"}
    plan = {"entry": last_price, "stop": stop, "tp1": tp1, "tp2": tp2,
            "rr1": round(abs(tp1 - last_price) / risk, 2),
            "rr2": round(abs(tp2 - last_price) / risk, 2) if tp2 else None}
    if equity:
        plan["shares"] = int(equity * risk_pct / risk)
    plan["status"] = "✅ 可执行" if plan["rr1"] >= 2 else "❌ RR<1:2，放弃"
    return plan
```

**关于 ATR 与"禁用滞后指标"的冲突**：ATR 是平均量，严格按 SKILL.md 第 15 行的字面禁令也属禁用词。建议区分**方向性指标**（MACD/RSI/MA，确实禁用）与**尺度归一化量**（平均振幅，不产生方向信号）。若坚持字面合规，可用 `np.median(high-low)[-14:]`（中位振幅）替代，功能等价且不叫 ATR。

---

### P1-2 key_levels 无分级、无去重 → 2B 信号通胀

**现象**：`scan_price_action()` 的 2B 分支会对**每一个** key_level 逐一判定。

**证据**：单次扫描喂入约 20 个水平位，其中 8 个重复（Fib 位与 swing 价格天然重叠）。每个位 × 4 根 K 线 = 80 次独立判定。

**根因**：第 776–790 行把五类异质水平平铺进一个 list：

```python
key_hlines.extend(list(levels.get("fib_retracements", {}).values()))  # 5 个
key_hlines += [PDH, PDL, ORH, ORL]                                    # 4 个
key_hlines += [gap levels]                                            # 3 个
key_hlines += [recent_mtf[-4:] prices] + [recent_ltf[-4:] prices]      # 8 个
key_hlines += [cluster centers]                                       # n 个
```

而 SKILL.md 第 52 行对 2B 的定义是"刺破**关键高低点（Swing High/Low 或 PDH/PDL）**"——Fib 回撤位与共振中枢不在其列。**代码的实现范围超出了自己的规范**。

**影响**：几乎每次扫描都会命中至少一条"流动性猎杀"，信号失去区分度；报告中的"2B 猎杀"变成噪声而非警报。这恰好违背了 SKILL.md"拒绝模棱两可"的立意。

**修法**：水平位带 tier 与来源，按重要性排序，归并后再参与判定。

```python
TIER = {"swing": 4, "day_extreme": 3, "gap": 3, "fib": 2, "cluster": 1}
MERGE_ATR = 0.10        # 相距 < 0.1×中位振幅的位视为同一价

def build_key_levels(raw: list[dict], bar_range: float) -> list[dict]:
    raw.sort(key=lambda x: (-TIER[x["kind"]], x["price"]))
    out = []
    for lv in raw:
        if all(abs(lv["price"] - o["price"]) > MERGE_ATR * bar_range for o in out):
            out.append(lv)
    return out

# 2B 仅接受 tier >= 3 的水平位
sweep_levels = [lv["price"] for lv in key_levels if TIER[lv["kind"]] >= 3]
```

---

### P1-3 PA 形态脱离位置：形态在没有关键位的地方也报警

**现象**：Pin Bar / 吞没 / 孕线的判定完全不看 `key_levels`；`key_levels` 只被 2B 分支使用。

**根因**：第 488–520 行的三个形态分支是纯几何判定，无位置上下文。而价格行为的核心前提恰恰是"位置优先于形态"——一根锤子出现在真空区与出现在前低支撑上，信息量差一个数量级。

**影响**：报告里"看涨锤子线（多头拒绝下跌）"会频繁出现但无交易含义，读者被迫自行筛选。

**修法**：加 proximity gate + 趋势上下文。

```python
def _near_level(price, levels, tol):
    return any(abs(price - lv) <= tol for lv in levels)

tol = 0.25 * bar_range
if lower_shadow >= 2 * body and lower_shadow / rng >= 0.60:
    if _near_level(c_low, levels, tol):          # 必须发生在关键位附近
        emit("看涨锤子线 @ 关键位")
```

---

### P1-4 扫描窗口含未收盘 K 线，同一根 K 线多信号叠加

**证据**（E9）：6 根 5m 数据，末根 09:55 触发 2 条信号（Pin Bar + 2B Sweep Low）。

**根因**：`recent_df = df.tail(5)`，循环 `range(1, len(recent_df))` 的最后一个元素正是最后一根 K 线。实盘中这一根还在运行——`High/Low/Close` 都会继续变化。

**影响**：
1. 形态重绘：09:55 报"锤子线"，09:58 可能变成大阴线；
2. 同 bar 多信号：同一根 K 线既可报 Pin Bar 又可报 2B，报告读者无法判断该看哪个。

**修法**：

```python
# 1. 剔除未收盘 bar（按 bar 时长与当前时刻比较，或简单用 df.iloc[:-1]）
if df.index[-1] + bar_delta > pd.Timestamp.now(tz=df.index.tz):
    df = df.iloc[:-1]

# 2. 每 bar 只保留最高优先级形态
PRIORITY = ["2B", "Engulfing", "Pin Bar", "Inside Bar"]
```

---

### P1-5 成交量基准失当：日内 U 型季节性 + 与自身规范矛盾

**现象**：`analyze_volume()` 用以 20 根均量为基准的比值分档（>1.5x 放量 / <0.5x 极缩量）。

**根因**：日内成交量有强 U 型季节性——开盘 15 分钟与尾盘 30 分钟天然是全天均量的数倍。用"20 根均值"做分母，结果**开盘第一根必然"🔴放量"、午盘必然"🔵缩量"**，信号是时钟的函数，不是资金的函数。

**附加矛盾**：模块 docstring 第 5 行"严禁使用任何滞后性技术指标"，但 20 根均量就是一个移动平均；`argrelextrema(order=N)` 同样需要 N 根确认。表述与实现不一致，容易被对抗性 review 抓住。

**修法**：按时刻分桶计算相对量（RVOL），或至少与"同时间桶的历史均值"比较。

```python
def rvol(df, buckets=13):        # 30 分钟一桶
    df = df.assign(bucket=df.index.hour * 2 + df.index.minute // 30)
    base = df.groupby("bucket")["Volume"].transform(lambda s: s.shift(1).rolling(10).mean())
    return df["Volume"] / base
```

若认同该改法，建议同步把 docstring 措辞改为"不使用派生振荡指标（MACD/RSI/KDJ 等）"，避免自我否定。

---

### P1-6 量价背离判据过宽，顶底可同时成立

**根因**（第 585–600 行）：

```python
if last_5['High'].max() > prev_5['High'].max():        # 仅是"近5根最高 > 前5根最高"
    if last_5['Volume'].mean() < prev_5['Volume'].mean() * 0.8:
        → 顶背离
if last_5['Low'].min() < prev_5['Low'].min():          # 独立 if，不互斥
    if last_5['Volume'].mean() < prev_5['Volume'].mean() * 0.8:
        → 底背离
```

"近 5 根最高 > 前 5 根最高"在震荡中高频成立，不构成"创新高"；两个 `if` 独立 → 同一时刻可同时输出顶背离与底背离，自相矛盾。

**修法**：以已确认 swing 的极值为锚，顶底互斥，并要求价格与量能同向递减。

```python
hi_now, hi_prev = swings_conf[-1], swings_conf[-3]   # 同类拐点比较
if hi_now["price"] > hi_prev["price"] and vol_at(hi_now) < vol_at(hi_prev):
    → 顶背离（价格新高、量能阶梯下降）
elif lo_now["price"] < lo_prev["price"] and vol_at(lo_now) < vol_at(lo_prev):
    → 底背离
```

---

### P1-7 MSS 重复触发、无新鲜度过滤

**证据**（E5）：10 个拐点（真实结构破位 1 次）→ 输出 2 条事件，文案均为"多头结构被破坏"。

**根因**：第 173 行条件 `prev.dow == "LH" and curr.dow == "LL"` 只比较**相邻两个**拐点标签。一旦进入空头结构，每一对 LH→LL 都会重复满足；`prior_bullish` 只看前 5 个拐点，是任意阈值。

更根本的问题：真实的 MSS 需要**收盘价穿越**前一个结构位（displacement），而非仅标签组合。

**影响**：
- 事件列表被重复项污染，`[-2:]` 取到的两条可能是同一次破位；
- 第 703 行 `if not has_mss:` 输出"✅ 近期无结构转换，当前趋势延续中"——但 `has_mss` 只要历史上有事件就为真，**没有时效判定**，这句话可信度为零。

**修法**：

```python
# 1. 要求收盘穿越 + 缓冲
broken = prev_low_price
if c_close < broken - 0.1 * bar_range and prev_high_label == "LH":
    emit_mss()

# 2. 同向事件只保留最新一次，并加新鲜度
events = dedupe_by_direction(events)
fresh = [e for e in events if e["bars_ago"] <= 10]
```

---

### P1-8 道氏阶段：无时间衰减、无幅度阈值

**证据**（E6）：`[100,105,110,115]` 的 HH/HL 序列，间隔 7 天与间隔 365 天，输出完全相同："📈 多头趋势 (HH+HL 主导)"。

**根因**：`_determine_dow_phase()` 只统计最后 4 个拐点的标签计数（`bull>=3` → 多头）。两点缺失：
1. 无时效：周线 2 年数据里，"最后 4 个拐点"可能横跨 18 个月，报告却当作当下结论；
2. 无幅度阈值：`pt["price"] >= last_high` 用 `>=`，价格相等即算 HH；1 美分的抬升也算 HH。

**修法**：

```python
def _determine_dow_phase(swings, df, lookback_bars=40, min_move=0.005):
    cutoff_idx = len(df) - lookback_bars                 # 只看近期窗口内的拐点
    recent = [s for s in swings if s["index"] >= cutoff_idx][-4:]
    # 幅度阈值：变动需超过 min_move 才计入 HH/HL
```

并把输出从二值标签升级为带强度的表述（如"多头（强，4/4 结构位抬升）"），避免读者误读为确定性结论。

---

### P1-9 艾略特"三大铁律"未实现

**现象**：SKILL.md 第 27–33 行规定"一票否决制"三条铁律，第 87 行要求报告"显式声明符合"。代码里没有任何浪序枚举与规则校验——`calc_multi_leg_fib()` 只算腿长比例，不做浪形编号。

**影响**：所谓"一票否决"实际是 LLM 目测后的自我声明，**没有任何客观约束**。这是文档能力与代码能力的差距，也是"对抗性 review"最容易击穿的部位。

**修法**（可选，成本较高）：
1. 最小可行版：实现规则校验器 `validate_impulse(legs) -> (ok, violated_rule)`，输入候选 5 浪序列，检查浪2不破起点、浪3非最短、浪4不破浪1顶；
2. 若不实现，则应把文档措辞从"一票否决制"改为"人工核对清单"，避免过度承诺。

---

### P2-1 至 P2-6 参数设置

| ID | 现状 | 问题 | 建议 |
|:---|:---|:---|:---|
| P2-1 | `order`：swing 3/5/7，day 6/5/4 | 决定拐点密度，等于决定全部下游结论；不随数据量/波动自适应 | 标定方式：要求"每腿 ≥ k 根且腿幅 ≥ 1.5×中位振幅"，反推 order；暴露为 CLI 参数 |
| P2-2 | 0.008 / 0.01 / 1.5 / 1.2 / 0.7 / 0.5 / 0.8 / 20 / 5 / 150 全硬编码 | 无配置层，无法 A/B，无法按波动率缩放 | 统一 `@dataclass Params`，阈值以振幅百分比表达（如 `gap = max(0.005, 0.3 × bar_range%)`） |
| P2-3 | `_match_fib_ratio` 绝对容差 0.05 | 1.050→"近似 1.000"、2.570→"≈2.618 ✅"（相对误差 1.8% 标为精确命中） | 改相对容差（≤2%），三档输出：命中 / 近似 / 无匹配 |
| P2-4 | Fib 基准腿取 `swings[-2] → swings[-1]` | 不校验腿是否完成、是否与 HTF 同向、幅度是否够大 | 取最近一段已确认且幅度 ≥1.5×振幅的推动腿；仅当价格处于该腿区间时才输出回撤位 |
| P2-5 | `auto_adjust=False` | 拆股/除权产生原始价格断层 → 2:1 拆股被 `detect_gaps` 报成"强引力缺口"，并扭曲 swing | 改 `auto_adjust=True`；另注意第 54 行 fallback 用的是默认 `auto_adjust=True`，**主备路径复权口径不一致**，应统一 |
| P2-6 | 统一 2 位小数 | 对 <$1 标的层级粒度过粗（$0.85 与 $0.86 视为一位） | 按量级自适应：≥100 → 2 位，≥10 → 3 位，<10 → 4 位 |

---

### P3 执行效率与工程质量

**P3-1 请求数过多、无缓存、无并发**

单标的完整分析最多 **10 次**网络请求：`check_dow_indices` 6 次（SPY/QQQ/DIA × HTF+MTF）+ 标的自身 3 次 + 第 707 行额外的 1d 拉取 1 次。watchlist 模式为 `6 + 3N`。

问题点：
- `check_dow_indices()` 在标的有效性校验**之前**执行（第 846 行），无效代码也会先烧 6 次请求；
- 若标的恰为 SPY/QQQ/DIA，同一数据被拉两次；
- 每次 `fetch_klines` 都新建 `yf.Ticker`，无 Session 复用；
- 无本地缓存 → 同一标的重复分析重复下载；
- 无并发 → watchlist 10 个标的串行 36 次请求。

E7 已证明：9 次请求即可触发限流。

**建议**：参数校验前置 → 指数数据 TTL 缓存（盘中 60s / 盘后 30min）→ 同 interval 请求合并去重 → watchlist 用 `ThreadPoolExecutor(max_workers=4)` + 全局限速。

**P3-2 print-only 架构**

所有函数通过 `print` 输出，`analyze_single_ticker()` 返回 `None`。后果：无法单元测试（需捕获 stdout）、无法回测、无法被下游程序消费、无法 `--json` 导出。

**建议**：拆三层 —— `compute()` 返回 dict/dataclass（纯函数，可测）→ `render_text()` / `render_json()` 消费同一份数据。这是把本项目从"脚本"升级为"引擎"的前提，也是接入自动化流水线的必要条件。

**P3-3 冗余计算**：watchlist 分支在 `if is_watchlist` 之前已计算 `swings_htf` / `swings_ltf` / `ltf_phase`，但精简模式只用 `htf_phase` 与 `df_ltf`。另外 `pa_summary = pa[0]["pattern"]` 取的是 5 根窗口里**最早**的一条信号，应取最新或最高优先级。

**P3-4** `from datetime import time`（第 22 行）遮蔽标准库 `time` 模块。当前无冲突，但后续若需 `time.sleep()` 做退避（P0-4 修法必需）就会踩坑。建议 `import datetime as dt` 后写 `dt.time(4, 0)`。

**P3-5** 第 307 行 `except Exception: pass` 会静默吞掉时区缺失、索引非 DatetimeIndex 等错误，导致日内关键位整体缺失且无任何提示。应收窄到 `(AttributeError, TypeError, ValueError)` 并至少 `print(..., file=sys.stderr)`。

**P3-6** 缺 `Volume` 列时 `analyze_volume()` 直接 `KeyError` 崩溃（E8 已复现）。`fetch_klines` 已用 `cols` 过滤缺失列，但下游未做防御。建议 `if "Volume" not in df: return {"unavailable": True}`。

**P3-7** `.agents/skills/us-stock-price-action-analysis/SKILL.md` 与 `.antigravity/skills/us_stock_elliott_dow_pa.md` 内容完全相同（`diff` 为空）。两份副本会随时间漂移，建议保留一份 + 软链接，或明确"以 .agents 为准"。

---

## 四、边界情况清单

| 场景 | 当前行为 | 风险 | 建议 |
|:---|:---|:---:|:---|
| 数据不足（`len < order*2+3`） | `swings=[]`，报告仍打印"数据处理完成" | 高 | 显式输出"⚠️ 数据不足，结论不成立"并退出非零 |
| `diff == 0`（两拐点同价 / 单边一字板） | Fib 各档全等于 end_price，进而被聚成"共振" | 中 | `if diff < 0.5*bar_range: 跳过 Fib 计算` |
| 同一根 K 线同时是 High 与 Low 候选 | 生成零长度腿 | 中 | 按 `High-Low` 振幅阈值过滤，或同 time 去重 |
| 时区非交易所本地（UTC 等） | `time(4,0)` 掩码错位，被 `except: pass` 吞掉 | 高 | 显式 `tz_convert("America/New_York")` + 断言 |
| `df_daily` 不含当日 | `iloc[-2]` = 前日的前日 → PDH 错一位 | 中 | 用日期比对确定"昨日"，而非固定 `-2` |
| 半日市（13:00 收盘） | ORH 正常，无影响 | 低 | 无需处理 |
| 长假 / 停牌 | 假期跳空被当作结构缺口；无交易日历 | 中 | 引入交易日历校验，或标注"间隔 N 日，可能为假期缺口" |
| 成交量为 0 | 量比 0 → 全部"⚪极缩量" | 低 | `avg_vol == 0` 时输出"成交量不可用" |
| 无效代码 | 先烧 6 次请求再报错；退出码 0 | 中 | 参数校验前置 + 非零退出 |
| 部分标的失败（watchlist） | 单点异常可中断整个扫描（无 per-ticker try） | 中 | 每个标的独立 `try/except`，失败记入汇总 |
| 重绘 | 两次运行结论可能不同 | 高 | 见 P0-1 |

---

## 五、优化路线图

### 阶段一：修复正确性（1–2 天，改动集中在 3 个函数）

1. `extract_swings()` 加 `confirmed` 与 `confirmed_only`（P0-1）
2. `calc_fibonacci_and_levels()` 会话锁定 + 时区显式化（P0-2）
3. `calc_multi_leg_fib()` 聚类改与组首比较（P0-3）
4. `fetch_klines()` 加退避重试 + `main()` 非零退出（P0-4）
5. `analyze_volume()` 缺列防御（P3-6）

**验收标准**：E1–E9 全部回归通过；重复运行同一标的，结构输出一致（重绘消失）。

### 阶段二：建立策略层（3–5 天）

6. 拆 `compute()` / `render()`，`compute()` 返回 dataclass（P3-2）
7. 新增 `build_trade_plan()`：证伪位、TP1/TP2、RR、仓位（P1-1）
8. key_levels 分级去重 + 2B 限 tier（P1-2）
9. PA 加位置过滤 + 未收盘 bar 剔除 + 每 bar 单信号（P1-3、P1-4）
10. 量能改 RVOL（P1-5）、背离改 swing 锚定（P1-6）

**验收标准**：单标的可产出结构化 JSON，含 `plan.rr` 与 `plan.status`；RR<2 的方案被代码拦截。

### 阶段三：参数标定与验证（需要历史数据，1–2 周）

> ⚠️ **前置条件**：动手前先读 `strategy_theory.md`。那份文档说明了三层理论的分工、当前实现与理论的偏离、以及由理论推导出的七条回测硬约束。**回测只能验证假设，不能产生假设。**

11. 抽出 `Params` 配置层，以振幅百分比表达阈值（P2-1、P2-2）
12. 用 2–3 年 5m/15m 历史数据做参数敏感性扫描（order × 缺口阈值 × 共振容差）
13. 建立最小回测框架：入场 = 2B 收盘确认，出场 = 止损 / TP1 / TP2，统计胜率、期望值、最大回撤
14. 只有在第 13 步产出正期望且样本量足够（≥100 笔）之后，才谈"策略有效性"

**关键提醒**：第 13 步之前，**任何关于该策略盈利能力的判断都缺乏依据**。当前代码不产生可回测的信号序列，这正是判断 1 所指的缺口。

---

## 六、回归验证清单

可执行测试见 `test_pa_engine_fixes.py`（离线，无需行情源）。当前状态：

- [x] 同一序列逐根加长，末位拐点不再出现在距末端 `<order` 根的位置，或明确标注 `pending`
- [x] 5m/2d 数据下 `PMH` 覆盖完整 04:00–09:30 窗口，或显式输出 `N/A`
- [x] 输入跨度 6.5% 的 8 个价位，共振区输出 ≥2 个簇，且每簇宽度 ≤ 容差
- [x] 所有请求失败时退出码非 0，stderr 有可机读的错误摘要
- [ ] 10 个拐点（1 次破位）输入 `detect_mss`，输出 1 条事件（P1-7 未修）
- [ ] 同一 HH/HL 序列间隔 7 天 vs 365 天，阶段判定不同（P1-8 未修）
- [x] 无 `Volume` 列不崩溃
- [x] 末根 K 线（未收盘）不产生 PA 信号
- [x] 单标的分析产出可序列化 dict，含 `plan.rr` / `plan.status`

---

## 七、修复记录（v2.1，2026-10-22）

P0 四项已落地，`pa_engine.py` 版本升至 v2.1（851 → 1121 行；随后 v2.2 又扩到 1556 行）。回归测试随包提供：`test_pa_engine_fixes.py`。

| ID | 改动 | 关键实现 | 验证结果 |
|:---|:---|:---|:---|
| P0-1 | 拐点加确认状态 | `extract_swings(..., confirmed_only=True)` 返回 `confirmed` / `bars_to_confirm`；新增 `_split_swings()` 分层；报告新增「⏳ 待确认」行，斐波那契基准腿改用已确认拐点 | 末位已确认拐点恒满足 `index ≤ n-1-order`；96% 的未确认点不再进入结构判定 |
| P0-2 | 日内窗口重写 | 新增 `_to_exchange_tz` / `_infer_bar_minutes` / `_prev_session_bar`；先锁定最后交易日再取时刻窗口；按 bar 周期校验覆盖率（阈值 60%），不足输出 `N/A`；附带修正 PDH 的 `iloc[-2]` 硬编码 | 两日数据下 `PMH` 只取当日 101.0，昨日 200 不再污染；无盘前数据时输出 N/A 与非零说明 |
| P0-3 | 共振区聚类重写 | 与组首比较 + 簇总宽封顶 + 记录来源腿；`strength` 要求 `count≥3 且 legs≥2` | 同一输入从 1 个 3% 宽假簇变为 3 个 ≤1% 宽的簇；单腿汇聚不再标「强共振」 |
| P0-4 | 取数失败处理 | 指数退避重试（2s/4s）+ 限流熔断（连续 2 次后跳过剩余请求）+ 运行内缓存 + 失败汇总 + 非零退出码；单标的异常不中断批量；`$0.00` 改为 `N/A` | 真实限流环境下 9 次请求从「静默成功」变为约 2 秒内失败、退出码 1、诊断完整 |

**运行行为变化（需注意）**：

1. `fetch_klines` 复权口径由 `auto_adjust=False` 改为 `True`，拆股不再被误判为缺口；历史缺口与拐点数值会与旧版不同，属预期修正。
2. 行情源不可用时命令以退出码 1 结束。若有定时任务/流水线依赖「总是成功」，需改为判断退出码。
3. 报告新增三类行：「⏳ 待确认」「N/A（数据不足）」「⚠️ 未计算: 原因」。下游若解析文本，需容忍这些新增行。

**仍未处理（保留原优先级）**：P1-5 日内量能基准、P1-6 背离判据、P1-7 MSS 新鲜度、P1-8 阶段时间衰减、P1-9 艾略特铁律校验、P2 参数配置层。（P1-1~P1-4 与 P3-2 已在 v2.2 完成，见第七节之二。）

---

## 七之二、v2.2 修复记录（策略层，2026-10-22）

本次把「风控」从 `SKILL.md` 的文字承诺搬进代码。`pa_engine.py` v2.1 → v2.2，新增 CLI 参数 `--json` / `--equity` / `--risk-pct`。

| ID | 改动 | 关键实现 | 验证 |
|:---|:---|:---|:---|
| P1-1 | 新增 `build_trade_plan()` | 结构偏向取 HTF+MTF 共识，冲突即不出计划；止损 = 最近已确认波段低/高点 ± `0.8×中位振幅`；入场取最近支撑/阻力，TP1 取上方 tier≥2 结构位，TP2 取 1.618 扩展；RR 由代码计算，`RR1 < 2` 直接判「按规则放弃」；`--equity` + `--risk-pct` 产出股数 | 多头 RR 1:3.75 → 可执行，股数 1250；RR 1:1.13 → 放弃；HTF/MTF 冲突、无确认拐点、现价已破结构 三种情形均正确拒绝 |
| P1-2 | 新增 `build_key_levels()` | 五类水平带 tier（swing 4 / 日内极值 3 / 缺口 3 / fib 2 / cluster 1），按 `0.15×中位振幅` 归并，保留最高 tier 并记录同源汇聚；`scan_price_action` 的 2B 只接受 tier≥3 | 三处相近价位归并为 1 个；同一价位 fib 级不触发 2B、swing 级触发 |
| P1-3 | 形态位置闸门 | Pin Bar / 吞没 / 孕线 必须落在关键位 `±0.25×中位振幅` 内才输出；`require_location=False` 可关闭 | 远离关键位的 3 条形态被过滤，关闭闸门后恢复 |
| P1-4 | 剔除未收盘 K 线 | `_is_last_bar_open()` 按「当前时刻 < 末根起始 + 周期」判断，仅对日内周期生效（日线以上返回 False，避免误判） | 进行中的 5m 末根被识别并剔除；已收盘时不误判；日线不受影响 |
| P3-2 | 结构化输出 | `--json <path>` 导出 price / phase / bias / swings / levels / key_levels / gaps / volume / pa_signals / plan，`default=str` 处理时间戳 | 落盘后可被 `json.load` 读取，含 `plan.status` 与 `key_levels` |

**设计取舍（值得记录）**：

1. **止损缓冲兼作最小止损距离**（`STOP_BUFFER_K = 0.8`）。若不设下限，止损紧贴结构时会退化成噪声级距离（如每股风险 $0.25），RR 被虚增到 1:13 —— 闸门形同虚设。这是把「盈亏比」写成代码后才会暴露的问题。
2. **中位振幅代替 ATR**。AGENTS 规范禁用滞后指标，而止损缓冲、归并宽度又必须有一个尺度。用最近 14 根 `High-Low` 的中位数，只做尺度归一化、不产生方向信号，比 ATR 抗极端值。
3. **风险预算默认 1%**，`--risk-pct` 限制在 (0, 0.1]；`--equity` 缺省时跳过股数计算并明示，不猜账户规模。
4. **止损锚点取「最近一个」而非「最近几个里最低的」**。后者会把止损推到很早以前的深低点，RR 被无谓压低；前者与 `SKILL.md` 的「波浪证伪点」语义一致。

**仍未解决的真问题**：计划是否盈利，取决于 P1-5/P1-6/P1-7/P1-8 与参数标定。本次只是让「算错」和「没算」这两类问题消失，不代表策略有效。


---

## 八、已排查但确认不是问题的项

保持审计的对称性，以下几项在初读时可疑，实测后排除：

| 疑点 | 实测结论 |
|:---|:---|
| Windows 中文环境下打印 emoji 会 `UnicodeEncodeError` | **未复现**。本机重定向到文件时 `sys.stdout.encoding = utf-8`，打印 `📈` 正常，退出码 0。环境相关，非代码缺陷 |
| `detect_gaps()` 循环内 `df.iloc[i+1:]` 造成 O(n²) 性能问题 | **实测影响很小**。250 / 500 / 1000 / 2000 根分别为 16 / 26 / 56 / 118 ms，当前数据规模下不构成瓶颈，无需优先处理 |
| `_match_fib_ratio` 在比例接近 0 时除零 | 不涉及除法，无此风险 |
| `argrelextrema` 平台期产生重复拐点 | 同向归并逻辑（第 101–114 行）已正确处理 |

---

## 附：一句话总结

代码的**结构提取**能力是扎实的（拐点归并、道氏打标、缺口追踪的写法都正确），问题集中在三处：**把未确认的结构当成已确认**（P0-1）、**窗口/聚类的边界处理**（P0-2/P0-3）、以及**整条策略与风控链条的缺位**（P1-1）。前两者是修 3 个函数就能解决的 bug，后者是架构层面的缺口——它也解释了为什么"这个策略好不好"这个问题目前无法回答。

**v2.2 之后的现状**：前三项已全部落地，风控（止损 / 盈亏比 / 仓位）已是代码层的硬闸门而非文档承诺。但"策略是否盈利"依然无法回答——缺的是参数标定与回测，不是修 bug。下一步应是阶段三，而不是继续加功能。

