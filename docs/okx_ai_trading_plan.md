# `okx-ai-trading` 分支架构与改造规划

> 分支基线：`main @ db830e9`  
> 目标：在保留既有价格行为学分析内核与风控账本的基础上，构建一套**专为 AI Agent（LLM + `okx-trading` MCP）在 OKX 执行实盘/模拟盘交易**的确定性工程底座与标准化 SOP。

---

## 1. 分支定位与核心设计原则

1. **专注 OKX AI Agent 实盘/模拟交易闭环**
   - 覆盖标的：OKX `*-USDT-SWAP` 主流加密货币永续合约（如 `BTC-USDT-SWAP`、`ETH-USDT-SWAP`、`SOL-USDT-SWAP`）及可选的美股代币化永续合约（如 `AAPL-USDT-SWAP` 等，按更严格流动性与强平安全垫约束执行）。
   - 交互方式：AI Agent 负责结构研判、上下文综合决策、向用户确认并编排 `okx-trading` MCP 工具；**所有数值计算、门禁校验与状态对账全部下沉到确定性 Python 模块**。
2. **核心工程原则：Deterministic Python Computes, AI Agent Decides & Orchestrates MCP**
   - 杜绝由 LLM 手算 `1R`、净 `Weighted_RR`、合约张数（`ctVal` / `lotSz` / `minSz`）、强平安全距离、订单簿深度/点差或四向一致性校验。
   - 任何下单前必须由 Python 预检模块输出可审计的 JSON 决策票据（`PASS` / `REJECTED` / `CONFLICT_HALT`），并与 `risk_ledger.py` 持久化联动。

---

## 2. 现有资产复用与解耦边界

| 模块 / 资产 | 处置策略 | 说明 |
| :--- | :--- | :--- |
| `pa_engine.py` | **核心复用** | 复用 Swing、Dow Phase、MSS、Box、Channel、Wave 客观失效与 `bias_htf/mtf` 引擎，作为信号与四向一致性校验底层。 |
| `strategy_crypto.py` | **核心复用** | 复用加密原生 2B 扫损触发、防插针宽止损（`1.4x` / `0.35x` ATR）与分级止盈参数体系。 |
| `risk_ledger.py` | **核心复用与扩展** | 复用单笔风险上限（`0.5%~1.0%`）、组合总敞口上限（`<= 3.0%`）、相关性桶上限（`<= 2`）、日内 `-2.0% / -2.5R` 熔断及 3 连亏 24h 冷却机制。 |
| `AGENTS.md` | **规则固化** | 将 §1 预检硬门禁、§2 零上下文监控、§3 引擎交叉验算、§4 异常处置预案固化为 `okx_agent/` 可执行代码与自动化测试。 |
| `strategy_layer.py` / `strategy_day.py` / `backtest.py` | **保持原样、职责解耦** | 美股现货/日内回测引擎保持稳定，不与 OKX 实盘执行层耦合；仅在分析 OKX 美股代币永续时作为美股正股时段/HTF 交叉参考。 |
| `.agents/skills/us-stock-price-action-analysis/` | **保持原样** | 美股纯技术分析 Skill 与新建 OKX AI Agent 交易 Skill 相互独立。 |

---

## 3. 目标目录与模块规划

```text
Investment_Assistant/
├── config/
│   └── okx_universe.json          # OKX 交易标的池、相关性桶映射、perp 类型 (crypto/stock_token) 与事件静默窗口配置
├── okx_agent/
│   ├── __init__.py
│   ├── market.py                  # OKX 公共 K 线适配器 (15m/1H/4H/1D) + pa_engine/strategy_crypto 联动 + 四向一致性强制校验
│   ├── preflight.py               # 实盘/模拟下单前确定性硬门禁 (acctLv/深度/点差/净Weighted_RR>=2.5/张数/强平距离/risk_ledger) + 双单分批止盈 JSON 票据生成
│   ├── watcher.py                 # 标准化零上下文 15s 后台价格哨兵 (AGENTS.md §2)，带滞回缓冲区与 >=5 次 SSL 超时熔断
│   ├── reconcile.py               # OKX 实盘状态 (持仓/挂单/条件单) 与 data/risk_ledger.json 对账器，覆盖 AGENTS.md §4 异常预案
│   └── journal.py                 # 追加式结构化交易与拒绝原因审计日志 (data/okx_journal.jsonl)
├── .agents/skills/
│   └── okx-ai-trading/
│       └── SKILL.md               # AI Agent 专用 OKX 交易标准作业程序 (SOP)
├── docs/
│   └── okx_ai_trading_plan.md     # 本分支架构与改造规划文档
└── tests/
    ├── test_okx_agent_preflight.py
    ├── test_okx_agent_market.py
    ├── test_okx_agent_watcher.py
    └── test_okx_agent_reconcile.py
```

### 各核心模块职责说明

1. **`okx_agent/market.py`**
   - 拉取并校验 OKX `15m` / `1H` / `4H` / `1D` 执行期与高周期 K 线，输出与 `pa_engine.py` 兼容的 OHLCV 结构。
   - 执行强制**四向一致性校验（Four-Way Consistency Check）**：交叉比对 `[二] Dow Phase`、`[二b] MSS`、`[三c] Channel Slope` 与 `[八] bias_htf / bias_mtf`。
   - 若任意两项存在方向冲突，直接输出 `status: "CONFLICT_HALT"` 及冲突明细，阻断下单；美股代币化永续若与 `pa_engine.py` 结论冲突同样触发 `CONFLICT_HALT`。

2. **`okx_agent/preflight.py`**
   - 严格实现 `AGENTS.md` §1 全部硬门禁：
     - **账户与合约检查**：`acctLv >= 2`；读取 `ctVal`、`lotSz`、`minSz`。
     - **盘口与资金费率检查**：top-5 深度 `>= 3x` 订单名义价值、买卖价差 `<= 5 bps`；扣减实际 taker 手续费（或默认 `10 bps` 往返）+ 滑点 + 资金费率成本。
     - **风控账本与仓位计算**：调用 `risk_ledger.py` 校验组合敞口、相关性桶、日亏损与冷却期；按 `Risk_USD / (1R * ctVal)` 计算合约张数并按 `lotSz` 向下/规范取整且校验 `>= minSz`。
     - **强平安全垫校验**：加密永续 `|Liq_Px - Entry| >= 2.0 * 1R`；美股代币永续 `|Liq_Px - Entry| >= 4.0 * 1R`。
     - **净盈亏比硬一票否决**：`Weighted_RR < 2.5` 一律输出 `REJECTED (< 2.5R)`，禁止靠拉宽 SL 或压缩 TP 凑单。
   - 校验通过后，自动生成两笔**原子分批止盈挂单参数**（例如 `50%` 仓位绑定 `TP1 + mark-SL`、`50%` 仓位绑定 `TP2 + mark-SL`），供 Agent 直接传给 `okx-trading` MCP `swap_place_order`，保证 100% 仓位自成交毫秒起均有 `mark` 价触发的原子止损保护。

3. **`okx_agent/watcher.py`**
   - 标准化本地后台轻量轮询脚本（每 15s 轮询公共行情，不依赖高频 LLM cron），仅在价格触及**限价挂单区**、**TP1 触发保本移损窗口**或**临近 SL 警报区**时退出并向 Agent 发送结构化 JSON 唤醒事件（`LIMIT_ENTRY`、`TP1_BE`、`SL_NEAR`）。
   - 内置**滞回（hysteresis）缓冲**防止边界反复震荡触发唤醒乒乓，以及连续 `>= 5` 次 SSL/网络超时自动退出告警机制。

4. **`okx_agent/reconcile.py`**
   - 接收 `swap_get_positions`、`swap_get_orders(status="open")`、`swap_get_algo_orders(status="pending")` 与本地 `data/risk_ledger.json` 进行状态核对，落地 `AGENTS.md` §4 四类异常预案：
     1. **分批挂单单边部分成交**：检测无原子 SL 覆盖的裸露仓位，计算补充 `swap_place_algo_order` 参数并重算实际 `Weighted_RR`。
     2. **TP1 成交后保本移损前置校验**：在调用原子 `swap_amend_algo_order` 前校验 `markPx` 与 `Entry` 的相对位置；若已越过 `Entry` 亏损侧，提示立即市价平仓或贴近 `markPx` 缩紧止损（严禁先撤后挂）。
     3. **哨兵崩溃 / 重启恢复**：一键输出交易所实盘状态与本地账本差异清单。
     4. **流动性骤降 / 滑点超 `1.5 * 1R`**：检测实际平仓 `-R` 超标或盘口深度跌破 `1x` 持仓名义价值，自动记录完整 `-R` 至 `risk_ledger.py` 并标记暂停交易。

5. **`okx_agent/journal.py`**
   - 以 JSON Lines 格式追加写入 `data/okx_journal.jsonl`，统一留存：
     - 每次扫描与四向一致性判定结果（含 `CONFLICT_HALT`）；
     - 每次 `preflight` 拒绝记录（含 `REJECTED (< 2.5R)`、深度不足、账本熔断原因）；
     - 实际下单、部分成交调整、TP1 保本移损与最终平仓复盘指标（计划 `1R` vs 实际实现 `R`）。

6. **`.agents/skills/okx-ai-trading/SKILL.md`**
   - 为 AI Agent 提供精简、无歧义的 OKX 交易 SOP：
     `市场扫描 (market.py) → 四向一致性门禁 → 预检票据生成 (preflight.py) → 用户确认 / MCP 原子下单 → 零上下文哨兵 (watcher.py) → 状态对账与异常处理 (reconcile.py) → 审计归档 (journal.py)`。

---

## 4. 分阶段实施路线图（M1–M4）

| 阶段 | 核心交付物 | 验收标准 |
| :--- | :--- | :--- |
| **M1：安全底座（Safety Floor）** | `config/okx_universe.json`<br>`okx_agent/preflight.py`<br>`tests/test_okx_agent_preflight.py` | 100% 覆盖 `AGENTS.md` §1 计算与门禁逻辑；`< 2.5R`、深度 `< 3x`、强平垫不足、账本超限全部确定性拦截并输出原子双单 JSON。 |
| **M2：OKX 行情与四向信号引擎** | `okx_agent/market.py`<br>`tests/test_okx_agent_market.py` | 支持 OKX 多周期 K 线拉取与离线 mock 测试；四向一致性冲突 100% 触发 `CONFLICT_HALT`。 |
| **M3：零上下文哨兵与对账器** | `okx_agent/watcher.py`<br>`okx_agent/reconcile.py`<br>`okx_agent/journal.py`<br>`tests/test_okx_agent_watcher.py`<br>`tests/test_okx_agent_reconcile.py` | 覆盖滞回唤醒、连续 5 次网络超时熔断、单边部分成交补 SL、BE 移损前置校验及 `okx_journal.jsonl` 写入验证。 |
| **M4：Agent Skill 联动与模拟盘 E2E** | `.agents/skills/okx-ai-trading/SKILL.md`<br>OKX Demo 全流程演练 | 使用 `okx-trading` MCP 在 OKX 模拟盘完成一次完整闭环（分析 → 预检 → 下单 → 哨兵 → 对账 → 归档）。 |
