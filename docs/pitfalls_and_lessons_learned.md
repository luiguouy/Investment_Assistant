# 价格行为学量化系统 · 踩坑与避错实战经验总结库 (Pitfalls & Lessons Learned)

> **版本**：v1.0 | **创建日期**：2026-10-26  
> **核心原则**：*回测里犯下的每一个隐式假设错误，实盘都会用真金白银加倍清算。*  
> 本库汇集了本量化交易系统（道氏理论 + 艾略特波浪 + 裸K价格行为 + 箱体与通道分析）在全周期研发、代码重构、数学验证和实证回测中排查出的全部关键 Bug、设计盲区与理论认知偏差，作为后续系统演进与实盘部署的一级防错规范。

---

## 快速检索索引表 (Quick Diagnostic Index)

| 编号 | 错误类别 | 典型故障现象 / 痛点 | 根本原因 (Root Cause) | 终极防错机制 / 测试守护 | 关联源码文件 |
|:---:|:---|:---|:---|:---|:---|
| **D-01** | 数据传输 | 国内金融接口返回空包或握手超时 | 本地 HTTP 代理对国内直连域名的 CONNECT 隧道阻断 | 注入 `_DOMESTIC_HOSTS` 并设置 `no_proxy` 白名单 | [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) |
| **D-02** | 数据缓存 | 取数命令无法拉取最新交易日数据 | 缓存读取无脑短路，且全量覆盖会洗脱长历史 | 引入增量合并 `pd.concat` 去重与 `--update` 强制参数 | [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) |
| **D-03** | 数据质量 | 某标的回测指标突变、穿越数年 | 数据源存在数年静默断层（如 2000+ 天缺失） | 建立 `check_continuity` 严格断层熔断降级 | [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) |
| **D-04** | 数据校验 | 价格行为形态完全失真却零报错 | 字段列序颠倒（如把 Close 误当 High） | 引入 OHLC 结构硬哨兵校验：$H \ge \max(O,C), L \le \min(O,C)$ | [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) |
| **M-01** | 未来函数 | 回测胜率虚高，截断重算后结果不一致 | 决策日 $t$ 提前使用了当日盘中或收盘数据 | 确立严格时序分离：$dec\_t = entry\_t - 1$；自检 `T1` 守护 | [`backtest.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/backtest.py) |
| **M-02** | 未来函数 | 周线顺势门禁泄漏未来数天行情 | 日线聚合周线时混入了当前尚未走完的“半截周” | 强制过滤 `label <= 当前决策日`；自检 `T3` 守护 | [`backtest.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/backtest.py) |
| **M-03** | 撮合偏差 | 同根 K 线暴涨暴跌导致盈亏比虚假大增 | 撮合引擎盲目假设“先触碰止盈 TP” | 建立极度保守撮合铁律：同根双触严格**止损优先**；自检 `T7` | [`strategy_layer.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_layer.py) |
| **M-04** | 数学奇点 | 通道门禁在测试时产生常数抵消 Bug | 纯数学严直线上前 29 根无残差导致标准误与偏差抵消 | 测试用例必须注入市场内生微观波动项 $\sin(x)$ | [`tests/test_box_and_channel.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/tests/test_box_and_channel.py) |
| **S-01** | 策略逻辑 | 日内 2B 假突破出现高位追多、低位追空 | 2B 检测未校验水平线性质，多空方向倒错（占 46.7%） | 限制 PDH 仅做空，PDL 仅做多；自检 `T16` 守护 | [`strategy_day.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_day.py) |
| **S-02** | 策略逻辑 | 震荡股（如 INTC）胜率仅 21.4% 被反复割肉 | 在横盘整理区间机械套用单边突破追单 | 研发箱体五大状态机，箱体内严禁追单，改为 2B 猎杀 | [`pa_engine.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/pa_engine.py) |
| **S-03** | 策略逻辑 | 长牛资产做空胜率仅 12.5%，大盘门禁对空头盲目放行 | 忽视美股年化 8%~10% 长牛漂移与借券摩擦，机械对称做空 | 建立非对称做空硬红线：跌破 200 EMA / 1h 空头 + 2B 诱多；自检 `T10` | [`strategy_layer.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_layer.py) |
| **S-04** | 策略逻辑 | 裸 K 形态（Pin Bar/吞没）发出大量假信号 | 形态脱离了关键水平位置，沦为半空中的杂波 | 强制 `require_location=True`，无位置形态一票否决 | [`pa_engine.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/pa_engine.py) |
| **S-05** | 资产特性 | 加密货币策略频繁被美股指数空头误伤 | 机械套用跨资产门禁，忽视了加密原生周期 | 资产解耦：加密资产改用原生 200 EMA 与 4 年周期顺势 | [`strategy_crypto.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_crypto.py) |
| **S-06** | 策略逻辑 | 主观数浪预测幻觉与波浪 5 浪衰竭一票否决 | 强行主观数浪诱发幻觉，5 浪衰竭在单边市误杀主升浪 | 波浪降级为客观事后证伪止损（浪4不破浪1顶）与斐波比例尺，100% 决策锚定道氏/箱体/通道/2B | [`pa_engine.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/pa_engine.py) |
| **S-07** | 策略逻辑 | SPY/QQQ 高重叠同义反复误杀非科技独立 Alpha | 科技七巨头权重重叠度 0.9+，大盘一刀切冻结全行业 | 引入 QQQ vs DIA 风格背离、SPY vs IWM 市场广度及行业 ETF，确立“微观战胜宏观”脱钩规则 | [`strategy_layer.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_layer.py) |
| **S-08** | 理论认知 | “纯裸K禁提指标”极端洁癖与 200 EMA 认知互搏 | 宣称严禁指标但底层强依赖 200 EMA 防大回撤，提示词与代码撕裂 | 正名 200 EMA 为“道氏主要趋势连续平滑代理”，严禁买卖交叉但授权宏观顺势准入门禁 | [`.agents/skills/.../SKILL.md`](file:///d:/Antigravity_Workspaces/Investment_Assistant/.agents/skills/us-stock-price-action-analysis/SKILL.md) |
| **I-01** | 日内微观 | 日内 5m 策略毛利很高但净收益全军覆没 | 5m 单笔浮盈极小，8 bps 真实摩擦吞噬了 40% 利润，参数冲突 (1.8 vs 2.0 vs 2.5) | 强制全链路最低盈亏比 $\ge 2.5R$，强制 1h HTF 趋势共振门禁，禁低波动刮头皮 | [`strategy_day.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_day.py) |
| **I-02** | 日内风控 | 半日市或数据缺漏导致日内单变成隔夜单 | 仅靠 `time >= 15:55` 触发，提前休市时条件无法满足 | 跨日比较与收盘强平兜底，隔夜单归零；自检 `T15` | [`strategy_day.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_day.py) |

---

## 模块一：数据源、网络通信与本地缓存的踩坑教训

### 1. 本地代理对国内金融域名的 SSL 握手阻断 (Empty Body)
- **踩坑场景**：系统调用东方财富 (`push2his.eastmoney.com`)、新浪 (`stock.finance.sina.com.cn`)、腾讯 (`web.ifzq.gtimg.cn`) 抓取 A 股或美股前复权日线时，接口频繁返回 `status=0, empty body` 或 `SSL: UNEXPECTED_EOF_WHILE_READING`。
- **排查真相**：本地操作系统环境变量中配置了 `https_proxy = http://127.0.0.1:7897`。国内金融站点禁止或不支持境外代理节点的 HTTP CONNECT 隧道代理，导致代理直接切断连接。
- **沉淀规范**：
  在 [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) 初始化时主动注入域名直连白名单：
  ```python
  _DOMESTIC_HOSTS = "eastmoney.com,push2his.eastmoney.com,sina.com.cn,gtimg.cn,ifzq.gtimg.cn"
  _existing_no_proxy = os.environ.get("no_proxy", "")
  if not any(h in _existing_no_proxy for h in ["eastmoney.com", "gtimg.cn"]):
      os.environ["no_proxy"] = f"{_existing_no_proxy},{_DOMESTIC_HOSTS}".strip(",")
  ```
  在 curl 子进程调用中显式加入 `--noproxy _DOMESTIC_HOSTS`，可用源数量立刻由 3/6 恢复至 5/6。

### 2. 缓存静态短路阻断更新 vs 无脑覆盖冲洗历史
- **踩坑场景**：执行 `python data_sources.py --fetch-file watchlist.txt` 时，终端直接打印 `cache(1920 根)` 并退出，无法更新最新的收盘价；但若强制清空缓存重新拉取，部分短序列接口（如腾讯单次 640 根）会冲掉标的已有的 4,000+ 根历史长缓存（如 SPY 从 2008 年至今的历史直接被阉割）。
- **沉淀规范**：
  - 增加 `--update` / `--force` 显式控制开关；
  - 无论何时拉取新数据，一律执行**基于主键索引的增量合并**：
    ```python
    if not cached.empty:
        merged = pd.concat([cached, got])
        final_df = merged[~merged.index.duplicated(keep="last")].sort_index()
    ```
    既保障了末端 K 线更新至最新收盘日，又完全保护了早期 20 年的底层历史账目。

### 3. 长历史数据静默断层污染形态判断
- **踩坑场景**：早期对标的进行回测时，发现某些品种在 2008 年或 2015 年附近突发巨幅收益，检查才发现新浪的 QQQ 数据在 2005-2010 之间缺失了整整 2,307 天，东方财富的 NVDA 在 2015-11 前缺失了 2,863 天。
- **后果**：回测系统把几年的停滞直接当成一根跳空 K 线，导致全部摆动拐点（Swing）和趋势判断完全崩溃。
- **沉淀规范**：
  建立物理断层校验熔断器 `check_continuity(df)`：
  ```python
  gaps = df.index.to_series().diff().dt.days.dropna()
  if int(gaps.max()) > MAX_GAP_DAYS (20天):
      return False, f"数据不连续：最大间隔 {gaps.max()} 天（疑似整段缺失）"
  ```
  校验失败时不予入库，并自动降级切换至备用数据源。

### 4. 字段列序颠倒导致无报错静默污染
- **踩坑场景**：接入 OKX 交易所和国内行情源时，各家 API 返回的数组顺序差异巨大（OKX 是 `ts, o, h, l, c`，部分国内源是 `d, o, c, h, l`）。如果不慎将 High 与 Close 颠倒，代码运行完全不报语法错，但回测的最高价全被平抑，Pin Bar 与假突破彻底失效。
- **沉淀规范**：
  在 `_to_frame()` 中设立不可违背的几何防错哨兵：
  ```python
  bad = ((df["High"] < df[["Open", "Close"]].max(axis=1)) | 
         (df["Low"] > df[["Open", "Close"]].min(axis=1))).sum()
  if len(df) and bad / len(df) > 0.05:
      raise ValueError("字段顺序可疑：High/Low 未包含 Open/Close 实体")
  ```

---

## 模块二：时间序列、未来函数与数学回测的致命漏洞

### 1. 决策日与入场日的时序混淆 ($dec\_t$ vs $entry\_t$)
- **踩坑场景**：回测中如果直接在循环 $t$ 时刻根据 $df.iloc[t]$ 判定突破，并假设在第 $t$ 根成交，或者在当日收盘前去读当日的高低点作为门禁，就会引入严重的未来函数（Lookahead Bias）。
- **沉淀规范**：
  严格物理时序分离公式：
  $$\text{决策点 (Decision)}: t \quad (\text{以第 } t \text{ 根收盘价确立信号})$$
  $$\text{入场点 (Entry)}: t+1 \quad (\text{以第 } t+1 \text{ 根开盘价 } Open_{t+1} \text{ 撮合成交})$$
  并在 [`backtest.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/backtest.py) 中建立 **`[T1]` 截断重算自检**：截断历史在 $t=299, 449, 599$ 时的信号集，必须与全样本运行时前 $t$ 根发出的信号 **100% 精确一致**。

### 2. 周线聚合中的“未走完周”未来函数渗透
- **踩坑场景**：在日线回测时，为了获取宏观周线（HTF）趋势，通常使用 `df.resample('W-FRI')`。如果在周二做日线决策，而引用的周线包含了截至周五的整周极值，那么周三到周五的信息就被偷窥了。
- **沉淀规范**：
  日线取周线时，必须强制加入时间戳左截断：
  ```python
  # 只允许使用标签日期 <= 当前决策日期的已收盘周
  valid_weekly = df_weekly[df_weekly.index <= current_daily_date]
  ```
  设立 `[T3]` 周线不越界自检，违规越界次数必须恒为 0。

### 3. 同根 K 线双触的乐观撮合陷阱 (Double-Touch Bias)
- **踩坑场景**：持仓过程中，某根日线或日内 K 线波动剧烈，其最高价达到了止盈位（$High \ge TP$），而最低价也打穿了止损位（$Low \le SL$）。在未精确切片 tick 数据的前提下，许多回测系统默认“先打到止盈”。
- **实盘清算**：实盘中往往是先插针止损出局，随后行情再向目标位奔去。乐观撮合会导致胜率被虚假拔高 15%~25%。
- **沉淀规范**：
  在 [`strategy_layer.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_layer.py) 与 [`strategy_day.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_day.py) 中确立**极端保守撮合铁律**：
  ```python
  if is_stop_hit and is_tp_hit:
      # 同一根 K 线内双触，无条件判定为止损优先出场！
      exit_reason = "stop_loss_conservative"
  ```
  设立 `[T7]` 自检保证双触必须被无情判定为亏损出局。

### 4. 纯数学直线导致通道残差抵消奇点
- **踩坑场景**：在编写 `tests/test_box_and_channel.py` 自检时，为了测试通道下轨支撑拦截，构造了纯直线价格序列 `close = 100.0 + 0.5 * x`，最后一根下跌 `close[-1] - 1.2`，断言应落入通道下轨支撑区（$0.0 \le pos \le 0.25$）并拦截空头。然而断言报错，显示通道位置恒为 $-0.7361$（击穿下轨）。
- **数学本质**：当历史 29 根数据完全严格共线时，前 29 根回归残差全为 0。此时全样本残差标准误 $s_e$ 与最后一根的跌幅 $|\Delta|$ 严格成正比。在通道位置公式中：
  $$pos = 0.5 + \frac{C_t - mid}{4 s_e} = 0.5 + \frac{\Delta}{4 \cdot k |\Delta|} \equiv \text{常数 } -0.7361$$
  分子偏差与分母标准误完全相消！
- **沉淀规范**：真实金融序列永远具备微观布朗运动与分形噪声，单测合成用例必须注入客观微观波动项 `1.0 * np.sin(0.4 * x)`，通道带宽方能真实体现。

---

## 模块三：价格行为学、波浪与道氏理论落地的策略认知偏差

### 1. 2B 流动性猎杀的方向倒错 Bug（致命逻辑陷阱）
- **踩坑场景**：系统在日内 2B 扫描中，对关键水平线盲目放行双向交易，导致在昨日最高价（PDH）上形成了“跌破做多”、在昨日最低价（PDL）上形成了“突破做空”，共产生了 57 笔倒错交易（占 2B 样本的 46.7%）。
- **理论本质**：
  - 阻力高位（PDH / PMH / ORH）是**买方流动性池 (Buy-Side Liquidity, BSL)**：价格冲过高点是机构引诱散户追多、触发空头止损买单，随后反手打压收阴，**只能做空 (Sweep High, side=-1)**；
  - 支撑低位（PDL / PML / ORL）是**卖方流动性池 (Sell-Side Liquidity, SSL)**：价格跌破低点是机构诱空并猎杀多头止损卖单，随后强力收阳收回，**只能做多 (Sweep Low, side=1)**。
- **沉淀规范**：
  在代码中固化方向约束，任何阻力位放行做多或支撑位放行做空的行为，均被视为代码级非法操作，由 `[T16]` 自检实行一票否决。

### 2. 震荡市中盲目套用单边突破追单 (Chasing False Breakouts)
- **踩坑场景**：在对 Intel (INTC) 回测时，原装突破策略胜率仅有惨淡的 21.4%（PF 1.10），大量交易在冲高突破后迅速反转被止损。
- **本质归因**：资产处于大级别矩形横盘箱体内部时，突破大多是资金试盘或假突破，突破追单在统计学上是负期望行为。
- **沉淀规范**：
  - 引入**震荡箱体识别器 (`Consolidation Box`) 与五大状态机**；
  - 箱体内（`IN_BOX`）启动强制门禁，**全面封杀突破追单**，仅允许在箱体外沿做 2B 假突破反手猎杀；
  - 升级后，INTC 胜率跃升至 33.3%，PF 飙升至 1.91（+73.6%），期望收益提升 957%！

### 3. 长牛资产摸顶做空的非对称毁灭性与无条件放行漏洞
- **踩坑场景**：在 Google (GOOGL) 上，原策略做多胜率高达 66.7%（PF 4.39，单笔期望 +7.62%），但做空胜率仅 12.5%（PF 0.11，单笔期望 -12.78%）。
- **排查代码真相**：审查 `strategy_layer.py:verify_market_dow_gate` 发现，代码对多头交易设立了重重 SPY/QQQ 道氏门禁，但对空头计划（`side == -1`）竟然**完全无条件直接放行（`"passed": True, "reason": "空头计划放行"`）**！
- **理论本质与实盘风险**：
  - 美股由于名义 GDP 增长、通胀与上市公司股份回购，内生具备年化 8%~10% 的长牛向上漂移特性；
  - 做空存在天然的不对称风险：潜在收益有限（最大 100%）、潜在亏损无限，并伴随高昂的融券借贷利息（Borrow Fees）、券商强制召回（Recall）以及灾难性的轧空挤压（Short Squeeze）。机械镜像无门禁做空是系统潜在的巨大实盘爆仓隐患。
- **沉淀规范（非对称做空三红线）**：
  彻底废除空头无条件放行漏洞，做空必须同时满足三大严苛门禁：
  1. **宏观大势破位确认**：标的收盘价必须有效跌破日线 200 EMA 或周线出现明确道氏空头结构（LH+LL），严禁在大级别牛市中做空；
  2. **关键流动性猎杀证据**：仅放行在重要阻力位（PDH, 箱顶, 显著前高）出现明显的 2B 假突破诱多收阴（Sweep High），严禁低位追空杀跌；
  3. **融券与持仓摩擦约束**：做空交易施加严格的持仓时限惩罚与紧凑的保本推损纪律，严禁逆势扛单。

### 4. 脱离位置的蜡烛形态只是随机噪声
- **踩坑场景**：早期如果在全图任意位置出现锤子线（Pin Bar）或吞没形态就提示入场，实测胜率低于 35%。
- **价格行为学第一性原理**：
  *“形态只是决战的证据，位置才是发动决战的战场。”*
- **沉淀规范**：
  代码中默认锁定 `require_location=True`。任何蜡烛形态必须精准落在**顶底转换位、斐波那契黄金回撤位或箱体通道边界**的 $\pm 0.5 \times ATR$ 容忍度内，半空中的“完美形态”一律判定为伪信号并直接丢弃。

### 5. 跨资产机械套用门禁（加密货币与美股指数的错配）
- **踩坑场景**：早期策略将美股标普 500（SPY）和纳指（QQQ）的大盘门禁套用在比特币（BTC-USD）上，导致 BTC 在 2020-2021 年多头单边主升浪中，多次因美股短期震荡被大盘门禁误杀拦截。
- **沉淀规范**：
  加密资产实行资产解耦，改用加密货币原生的 **200 EMA 大周期顺势门禁** 与 4 年减半牛熊周期独立运行。

### 6. 主观数浪预测幻觉与波浪 5 浪衰竭一票否决陷阱
- **踩坑场景**：上层 Skill 曾设立“艾略特波浪三大铁律一票否决制”，强制要求标明当前处于大浪级的第几浪，并给出主选与备选数浪方案；底层 `pa_engine.py` 的 `evaluate_wave_position` 试图预测第 3 浪突破与 C 浪底，并在 `build_trade_plan` 中将 `wave5_exhaustion` 挂接为一票否决门禁。
- **实证真相与认识论危害**：
  - 波浪理论具有天然的分形多义性，同一段价格序列存在多种合法数浪方案。强行让 LLM 或算法在事前断言“第几浪”，必然催生主观臆测与幻觉；
  - 在强势单边长牛趋势中，所谓“第 5 浪”往往出现强烈的浪型延伸（Wave Extension），机械设立 `wave5_exhaustion` 一票否决门禁，导致系统在标的主升浪最丰厚的利润段被频繁自我截肢（如在 NVDA, TSLA 的主升走势中因“5浪衰竭”错失暴利）。
- **沉淀规范（波浪职能全面降级）**：
  1. 彻底废除波浪预测性入场信号与 5 浪衰竭一票否决门禁，买卖决策 100% 锚定道氏摆动点、箱体状态机、动态通道与 2B 裸K反转；
  2. 波浪铁律回归其本质——**客观事后证伪止损位**：以铁律三（浪4底不破浪1顶 $P_4 > P_1$）作为推动浪失效的客观止损点（`wave4_invalidation_stop`），一旦收盘刺破即刻确认结构破坏并坚决止损；
  3. 斐波那契仅作为多腿空间比例尺与共振集群（Fib Clusters）度量衡。

### 7. SPY/QQQ 高重叠同义反复误杀非科技独立 Alpha
- **踩坑场景**：`strategy_layer.py:verify_market_dow_gate` 曾仅比对 SPY 与 QQQ，只要任一指数出现空头结构（LH+LL）或破位，个股多头交易即被 `rejected_by_market_dow` 一票否决。
- **排查真相**：
  - 标普 500 (`SPY`) 与 纳斯达克 100 (`QQQ`) 的前七大科技巨头市值占比极高，二者日常相关性达 0.9+。以二者做双重验证，本质是“大科技权重的同义反复”；
  - 当市场出现科技回调而传统工业/金融/公用事业走强的风格轮动时（QQQ 弱但 DIA/XLF 强），非科技板块优质标的的突破被系统无情误杀。
- **沉淀规范（多维独立印证与微观脱钩规则）**：
  1. **构建独立宏观矩阵**：对比 `QQQ` 与 `DIA`（科技 vs 价值风格背离）、`SPY` 与 `IWM`（市场广度与小盘投机偏好），以及细分行业板块 ETF（如 SMH, XLF, XLE）；
  2. **确立“微观战胜宏观 (Micro Trumps Macro)”脱钩放行规则**：当个股在关键结构位完成**放量真突破（成交量 $\ge 1.5\times$ 均量）**且所属细分板块走强时，允许与大盘次级回调脱钩，提高盈亏比门禁至 $\ge 3.0R$ 并放行交易，严禁粗暴一刀切冻结。

### 8. “纯裸K禁提指标”极端洁癖与 200 EMA 顺势认知互搏
- **踩坑场景**：上层 Skill 曾声明“严禁使用、计算或提及任何指标（绝对禁止出现 MA、EMA 等词汇）”。但在 2026-10-24 一次拔除 200 EMA 的尝试中，策略表现严重恶化。
- **消融实验证据**：
  - 查阅 `docs/crypto_stat_validation.md` 与 `docs/index_stat_validation.md`：关闭 200 EMA 门禁后，策略盈亏比 PF 从 1.59 暴跌至 0.84，期望收益由正转负（-1.64%）；200 EMA 是全系统唯一具有统计显著性的宏观防大回撤核心组件。
  - 原教旨主义裸K洁癖导致“嘴上禁指标、底层靠均线”，使提示词、理论文档与底层实现发生严重撕裂。
- **沉淀规范（为 200 EMA 正名）**：
  - 在全系统理论与 Skill 中正式确立 200 EMA 为**“道氏理论大级别主要趋势（Primary Trend）的连续平滑数学代理 (Continuous Smooth Proxy for Dow Primary Trend)”**；
  - 严禁将均线用作滞后买卖金叉死叉信号，绝对禁止引入 MACD、RSI、KDJ、BOLL 等振荡器指标；
  - 200 EMA 仅作为**大周期宏观趋势准入闸门**：价格在其上方放行多头，在其下方禁止盲目抄底做多，打通上层提示词与底层逻辑。

---

## 模块四：日内高频、微观结构与真实交易摩擦的实战陷阱

### 1. 固定交易摩擦对微观波动的毁灭性侵蚀与全链路 $\ge 2.5R$ 统一
- **踩坑场景**：日内 5 分钟级别策略在不计交易摩擦时，毛盈亏比（Gross PF）能达到 1.30 ~ 4.22；但一旦施加真实机构环境下的 **8 bps 固定摩擦（单边 3 bps 滑点 + 1 bps 规费，往返 8 bps）**，净盈亏比暴跌至 0.56 附近，小单微利被瞬间吞噬。
- **参数冲突与代码撕裂**：审查发现，`strategy_day.py` 第 37 行设定 `DEFAULT_DAY_MIN_RR = 1.8`；`backtest.py` 第 1529 行强制将日内 `min_rr` 回退为 1.8；而 `SKILL.md` 声称 2.0R。在 8 bps 摩擦侵蚀下，1.8R/2.0R 会使系统扣除摩擦后的净期望彻底转负。
- **数学本质**：5 分钟级别由于单笔持仓时间极短，平均浮盈（MFE）中位数仅为 +0.20% ~ +0.35%。8 bps（0.08%）的摩擦直接硬生生切走了 **40% 的潜在总利润**！
- **沉淀规范**：
  1. **严禁在日内进行高频“刮头皮（Scalping）”**：严禁低波动标的刷单，开仓必须确保风险距离 $\ge 2 \times \text{friction}$；
  2. **全链路统一最小盈亏比 $\ge 2.5R$**：彻底终结参数冲突，全链路将日内最小盈亏比硬门禁锁定为 **$\ge 2.5R$**，低于 2.5R 一律一票否决（`rejected_by_min_rr`）；
  3. **强制日内 1h HTF 趋势共振门禁 (`use_htf_filter=True`)**：日内 5m 价格行为必须完全顺应 1 小时宏观周期趋势（1h 多头仅做多，1h 空头仅做空），彻底废除脱离大周期的盲目开单。

### 2. 半日市与数据缺失时的隔夜穿透漏洞
- **踩坑场景**：美股在感恩节翌日或圣诞前夕通常提前于美东时间 13:00 休市。早期日内策略依靠 `bar_time >= 15:55` 触发收盘平仓，导致在半日市时该判断永远无法触发，日内仓位被动持有隔夜，承受次日开盘跳空暴跌打击。
- **沉淀规范**：
  在 [`strategy_day.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_day.py) 中构筑双重安全锁：
  ```python
  # 锁 1：跨日比较，严禁任何单子越过当天午夜
  if bar_date != entry_date:
      force_exit(price=open_price, reason="force_eod_cross_day")
  # 锁 2：提前休市与常规休市（>= 15:55 或该会话最后一根有效 K 线）
  if bar_time >= dtime(15, 55) or is_last_bar_of_session:
      force_exit(price=close_price, reason="force_eod_market_close")
  ```
  配合 `[T15]` 自检，确保跨日隔夜留单数绝对为 0。

---

## 模块五：工程架构与防回归防线 (The Verification Shield)

为了防止上述错误在后续的代码迭代中出现任何形式的死灰复燃，本系统建立了两道不可逾越的自动化防线：

### 防线一：73 项全量单元测试套件 (`tests/`)
涵盖数据层解析、拐点提取、波浪铁律、裸K形态、箱体状态机、通道动态轨道以及日内撮合等各个模块：
```powershell
python -m unittest discover tests
# 运行结果：73 项测试 100% OK，零 Regression
```

### 防线二：T1 ~ T16 纯因果零未来函数截断自检 (`backtest.py --verify`)
任何策略逻辑修改后，必须运行全量零未来函数自检：
```powershell
python backtest.py --verify
```
1. **`[T1]` 决策集未来函数自检**：子序列截断重算，决策集 100% 一致。
2. **`[T2]` 入场时点因果自检**：第 $t$ 根决策，第 $t+1$ 根开盘入场。
3. **`[T3]` 周线不越界自检**：只用完整收盘周，越界次数恒为 0。
4. **`[T4]` MAE/MFE 口径自检**：$MAE \le 0 \le MFE$。
5. **`[T5-Limit]` 位置层挂单自检**：截断重算成交集完全一致。
6. **`[T6-Trigger]` 触发层形态自检**：截断重算触发入场点完全一致。
7. **`[T7-Execution]` 保守撮合铁律自检**：同根 K 线双触严格判定止损优先。
8. **`[T8-RR-Gate]` 盈亏比硬闸门自检**：$RR < 2.0$ 严格一票否决。
9. **`[T9-Strategy]` 全闭环未来函数自检**：截断重算交易集完全一致。
10. **`[T10-Dow-Wave]` 道氏波浪闭环自检**：截断重算入场出场完全一致。
11. **`[T11-Crypto]` 加密分级止盈自检**：截断重算阶梯止盈完全一致。
12. **`[T12-Box-Channel]` 箱体通道闭环自检**：截断重算箱体判定与轨道门禁完全一致。
13. **`[T13-Swing-Tiered]` 波段分级止盈自检**：TP1 减半推保本与 Runner 跟踪截断重算完全一致。
14. **`[T14-Day-Strategy]` 日内事件枚举自检**：微观撮合截断重算完全一致。
15. **`[T15-Day-EOD-Discipline]` 日内隔夜归零自检**：跨日留单数严格为 0。
16. **`[T16-Day-Inversion-Guard]` 2B 关键位防倒错自检**：倒错反向信号严格为 0。

---

> [!TIP]
> **开发警言**：在量化交易系统的演进中，**永远不要相信未经“截断重算”检验的代码，永远不要相信未经“交易摩擦清洗”的策略**。本库中的每一条规则，都是实战与测试中用大量心血换来的宝贵资产，后续任何新增模块均须严格遵守本库规范。
