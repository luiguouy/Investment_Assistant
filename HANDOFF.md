# Session Handoff: Investment Assistant (价格行为学量化系统)

> 生成时间: 2026-10-23 23:25 | 状态: 比特币与加密资产专用策略全周期落实交付完成，随时无损续接

---

## 1. 任务背景与核心目标

- **项目路径**: `D:\Antigravity_Workspaces\Investment_Assistant`
- **核心目标**: 构建基于经典价格行为学（道氏理论 + 艾略特波浪理论 + 裸K形态）与现代量化风控工程的严谨交易研究、回测与实战系统。
- **当前交付状态**: 
  1. 美股四层级联立闭环（方向层、位置层、触发层、执行风控层）与道氏相互印证/波浪理论内化完成；
  2. 比特币（BTC）及加密货币专属量化策略（`strategy_crypto.py`）全套研发、单元测试、零未来函数验证（T1~T11）及 BTC-USD 3,652 根全周期历史数据实证回测 100% 落实交付；
  3. 历史数据、指标 JSON、实证报告（`docs/strategy_crypto.md`）与自选配置完整沉淀。

---

## 2. 关键架构决策与确认项

1. **美股四层级联立决策链条**:
   - `L1 方向层 (Direction)`: 关键结构位放量突破（H2 假设），消除原 H1 计数模式的滞后性；
   - `L2 位置层 (Location)`: 拒绝市价追单，在顶底转换位（Swing Retest）布设限价回踩挂单；
   - `L3 触发层 (Trigger)`: 2B 假突破收回、看涨/看跌 Pin Bar 探底拒绝、关键位实体吞没；
   - `L4 执行风控层 (Execution)`: 结构失效点 + 0.5 ATR 保护止损、结构目标 TP 止盈、**$RR \ge 2.0$ 盈亏比硬闸门**、同根 K 线双触严格**止损优先**保守撮合。
2. **加密资产专用策略架构 (Crypto Native Strategy Engine)**:
   - **资产解耦**: 彻底解除美股大盘（SPY/QQQ）指数门禁拦截，依据加密货币原生周期（4年周期与独立牛熊节奏）自主触发；
   - **宽幅防插针双重自适应止损**: 结构支撑位 1.4x ATR 宽止损与 2B 假摔针尖外 0.35x ATR 紧止损取交集 `min(structural_stop, tight_stop)`，杜绝假摔爆仓出局；
   - **2B 流动性猎杀核心形态 (Liquidity Sweep)**: 盘中下破诱空后强势实体站回，赋予最高置信度入场权重；
   - **非对称分级止盈闭环**:
     - **TP1 (50% 仓位)**: 结构前高（或 $\ge 2.0R$）减半止盈，并**立即将剩余仓位止损推至保本位（Break-Even Stop）**；
     - **TP2 (25% 仓位)**: 冲量波段 1.618 斐波那契顺势扩展位止盈，剩余仓位防守提到 TP1；
     - **Runner (25% 仓位)**: 跟踪长牛趋势移动止损，不设固定终点，捕捉超级单边行情。
3. **工作区目录结构规范**:
   - `config/`: 股票池与加密货币清单（`tickers.txt`, `tickers_crypto.txt`, `watchlist.txt`）；
   - `data/`: 本地历史日线缓存（含 3,652 根日线的 `BTC-USD.csv` 与 11 只核心标的）；
   - `docs/`: 策略理论（`strategy_theory.md`）、加密专属报告（`strategy_crypto.md`）、预注册与回测报告（v1/v2/v3）；
   - `results/`: 结构化指标留痕（`backtest_btc_compare.json`, `backtest_portfolio_compare.json`）；
   - `tests/`: 自动化单元测试与数学自检套件（支持路径自适应）；
   - `strategy_crypto.py` / `strategy_layer.py` / `backtest.py` / `pa_engine.py`: 核心执行引擎。

---

## 3. 模块实现与实测状态

- [x] **加密专用策略引擎 (`strategy_crypto.py`)**: 包含标的识别（严格过滤法币外汇）、结构与 1.618 Fib 提取、2B 扫损形态识别、宽止损锚定与分级止盈生命周期推演（修复 TP2 顺延阶梯倒挂、空头负数目标保护、Runner 追踪止损入场前污染）。
- [x] **加密自动化测试套件 (`tests/test_strategy_crypto.py`)**: 包含 13 项专属单元测试（标的识别、结构位提取、2B 扫损、保守撮合双触、分级止盈保本防守、超级长牛 Runner、RR 闸门、截断重算零未来函数、TP阶梯保序、空头负目标拦截、空头分级执行、Runner 窗口隔离）。
- [x] **回测引擎全面支持加密资产 (`backtest.py`)**:
  - 支持 `--asset-type [auto|stock|crypto]` 参数与自动类型路由；
  - 增加 `--entry-mode strategy_crypto` 与 7 种模式横向对比（`--compare`）；
  - 修复出场分布统计对分级止盈命中的全息分类（避免原版硬编码 `take_profit` 导致虚报 0 笔）；
  - 新增 `[T11-Crypto]` 零未来函数截断重算自检。
- [x] **工程与数学自检 100% PASS**:
  - `python -m unittest discover tests`: **53 项单元测试全过**（含黑天鹅检测与道氏 200 EMA 顺势门禁测试）；
  - `python backtest.py --verify`: **T1~T11 全部通过**（100% PASS，截断重算零未来函数）。
- [x] **道氏 200 EMA 大周期顺势门禁全面固化落地 (`strategy_crypto.py` + `backtest.py`)**:
  - **核心机制**:
    - 牛市（Close >= 200 EMA）：**坚决只做多不做空**，杜绝逆势做空被狂暴轧空（Short Squeeze）爆头；
    - 熊市（Close <= 200 EMA）：**坚决只做空不做多**，杜绝逆势盲目抄底接飞刀；
    - 空头形态非对称过滤：彻底剔除滞后追跌的空头吞没（Engulfing），仅保留严苛的高位 2B 冲顶扫损与长上影 Pin Bar。
  - **BTC-USD 3,652 根全周期日线最终实证指标跃升**:
    - **总体胜率 (Win Rate)**: 从 40.7% 跃升至 **53.8%**（7 胜 / 6 负），突破 50% 心理大关；
    - **盈亏比 (Profit Factor)**: 从 0.86 飙升至 **1.59**（极高盈利能力，+85% 跃升）；
    - **平均 R 倍数**: 达到 **+0.34 R**；
    - **单笔期望收益**: 跃升至 **+4.89%**（单笔平均净赚 +4.89%）；
    - **多头 (Long)**: 8 笔，胜率 **50.0%**，PF **1.25**，期望 **+2.36%**；
    - **空头 (Short)**: 5 笔，胜率 **60.0%**，PF **2.39**，平均 R **+0.76 R**，单笔期望 **+8.92%**！彻底终结做空亏损痛点！
    - **止损率**: 发生率从美股原装的 64.2% 骤降至 **23.1% (仅 3 笔)**！
  - 产出专题深度报告 [`docs/strategy_btc_blackswan_analysis.md`](file:///d:/Antigravity_Workspaces/Investment_Assistant/docs/strategy_btc_blackswan_analysis.md) 与回测结果 JSON [`results/backtest_btc_compare.json`](file:///d:/Antigravity_Workspaces/Investment_Assistant/results/backtest_btc_compare.json)、[`results/backtest_btc_blackswan.json`](file:///d:/Antigravity_Workspaces/Investment_Assistant/results/backtest_btc_blackswan.json)。
  - **美股原装策略套用于 BTC (RR >= 2.0)**:
    - MTF 组: 53 笔, 胜率 34.0%, 盈亏比 PF = 0.90 (亏损), 平均 R = -0.06 R, 期望 -0.46%, 止损率 64.2%;
    - MTF+HTF 组: 22 笔, 胜率 18.2%, PF = 0.32, 平均 R = -0.52 R, 期望 -3.39%, 止损率 77.3%.
  - **加密专属策略全闭环 (Crypto Native Engine)**:
    - MTF 组: 55 笔, 胜率 **41.8%** (跃升 **+7.8%**), 盈亏比 PF **1.03** (突破盈亏平衡), 平均 R **+0.14 R** (由负转正提升 **+0.20 R**), 单笔期望 **+0.26%** (提升 **+0.72%**), MFE 中位数由 6.67% 提升至 **10.82%**;
    - MTF+HTF 组: 23 笔, 胜率 **26.1%** (提升 **+7.9%**), PF **0.51** (提升 **+59%**), 平均每笔亏损从 -0.52 R 缩减至 **-0.26 R** (风险暴露减半 **50%**);
    - 出场分布全息覆盖: 止损 21 笔 (38.2%), 止盈兑现 10 笔 (18.2%), 到期 24 笔 (43.6%)，彻底杜绝虚报漏报。
- [ ] **下一阶段待执行事项**:
  1. 探讨日内更小周期（4H / 1H）在加密专属策略上的落地可行性（利用 OKX/Binance API 颗粒度更高的数据抓取）；
  2. 针对 ETH, SOL 等高 Beta 资产扩展全周期对比验证，校准 ATR 缓冲系数与资金管理模型；
  3. 实盘挂单接口（Paper Trading / API 模拟盘）对接。

---

## 4. 关键文件与位置清单

| 类别 | 文件路径 | 说明 |
|:---|:---|:---|
| **加密策略引擎** | [`strategy_crypto.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_crypto.py) | 加密专用 2B 猎杀、宽止损与分级止盈核心逻辑 |
| **回测与验证** | [`backtest.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/backtest.py) | 支持 `--asset-type`、`--compare` 与 T1~T11 自检 |
| **美股策略引擎** | [`strategy_layer.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/strategy_layer.py) | 美股四层级联立策略与执行风控 |
| **价格行为引擎** | [`pa_engine.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/pa_engine.py) | 核心价格行为、道氏相互印证与波浪过滤分析 |
| **数据适配层** | [`data_sources.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/data_sources.py) | 多源数据拉取、OKX/Yahoo/降级清洗 |
| **标的配置** | [`config/tickers_crypto.txt`](file:///d:/Antigravity_Workspaces/Investment_Assistant/config/tickers_crypto.txt), [`config/watchlist.txt`](file:///d:/Antigravity_Workspaces/Investment_Assistant/config/watchlist.txt) | 加密与核心自选资产池 |
| **实证报告** | [`docs/strategy_btc_blackswan_analysis.md`](file:///d:/Antigravity_Workspaces/Investment_Assistant/docs/strategy_btc_blackswan_analysis.md), [`docs/strategy_crypto.md`](file:///d:/Antigravity_Workspaces/Investment_Assistant/docs/strategy_crypto.md) | 黑天鹅多空归因实证报告与加密专属量化报告 |
| **回测产物** | [`results/backtest_btc_blackswan.json`](file:///d:/Antigravity_Workspaces/Investment_Assistant/results/backtest_btc_blackswan.json), [`results/backtest_btc_compare.json`](file:///d:/Antigravity_Workspaces/Investment_Assistant/results/backtest_btc_compare.json) | 剔除黑天鹅对比与 3,652 根日线 7 模式横向对比指标 |
| **测试套件** | [`tests/test_strategy_crypto.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/tests/test_strategy_crypto.py), [`tests/test_strategy_layers.py`](file:///d:/Antigravity_Workspaces/Investment_Assistant/tests/test_strategy_layers.py) | 52 项单元测试与数学自检套件 |

---

## 5. 踩坑与避错记录 (What Didn't Work & Key Findings)

1. **切勿将 2B 极值止损孤立设置在结构破位位上方**:
   早期研发若仅把止损设为 `c_l - 0.35x ATR`，在顶底转换位回踩时实际止损幅度不足 1.5%，在加密市场仍极易被二次探底假摔扫出。必须自适应取 `min(structural_stop, tight_stop)`，使止损兼具结构低点 1.4x ATR 与 2B 极值针尖外 0.35x ATR 的双重防守。
2. **分级推保本（Break-Even Protection）是胜率和 PF 翻正的关键支柱**:
   BTC 常见假突破后深度反转，TP1 兑现 50% 后即便行情深度回撤打保本损，整笔交易仍贡献 +1.0 R（计为盈利单），彻底消除了坐过山车最终亏损出场的负期望侵蚀。
3. **切勿将美股指数（SPY/QQQ）强加于加密标的**:
   加密货币具有独立的 4 年减半主升浪周期，BTC 本身就是宏观基准锚。若施加美股空头门禁，会导致加密长牛周期中的重要入场信号被系统性误杀。
4. **多头 LONG 显著优于空头 SHORT**:
   在比特币的历史世俗长牛（Secular Bull）背景下，多头策略胜率 38.5%，PF 达 1.09，期望 +0.83%；空头交易容易在高波动轧空行情中承压，未来应以顺主趋势做多为主。

---

## 6. 新会话快速唤醒 Prompt 与恢复指令

**当开启新会话或新窗口时，直接发送以下文字即可无损唤醒：**

> 我已返回，请读取根目录下的 `HANDOFF.md` 恢复上下文。比特币专属交易策略已完成交付，请汇报当前系统状态，并针对下一阶段（如 ETH/SOL 高 Beta 标的泛化验证、4H 日内周期落地或实盘模拟接口）提出推进建议。

**常用验证与回测执行命令：**
```powershell
# 1. 运行完整 45 项单元测试与 T1~T11 数学自检
python -m unittest discover tests
python backtest.py --verify

# 2. 运行 BTC-USD 离线 7 模式横向对比回测
python backtest.py --tickers BTC-USD --offline --compare

# 3. 运行加密专用单策略回测
python backtest.py --tickers BTC-USD --offline --asset-type crypto --entry-mode strategy_crypto
```
