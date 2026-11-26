# Investment Assistant (StockVision) · 全栈跨周期智能交易平台与 OKX AI Agent 系统

集成 **StockVision 全市场跨周期 Web/桌面技术分析平台**、**OKX AI Agent 确定性实盘/模拟盘交易工具链** 与 **价格行为学（Price Action / 道氏理论 / 艾略特波浪 / 2B 扫损）Walk-Forward 量化回测系统** 的统一工作区。

---

## 目录结构

```text
Investment_Assistant/
├── okx_agent/               # OKX AI Agent 确定性交易工具链 (M1-M4)
│   ├── universe.py          # OKX 永续合约标的池与分层强平规则加载器
│   ├── market.py            # OKX 多周期 K 线适配与四向一致性强制门禁 (CONFLICT_HALT)
│   ├── preflight.py         # 下单前硬门禁 (acctLv/深度/点差/净RR>=2.5/张数/强平/账本) 与原子双单生成
│   ├── watcher.py           # 零上下文 15s 后台价格哨兵 (带滞回缓冲与 5 次超时熔断)
│   ├── reconcile.py         # 交易所持仓/挂单/条件单与 risk_ledger 对账及异常处置预案
│   └── journal.py           # 结构化交易与拒绝原因审计日志 (data/okx_journal.jsonl)
├── stage_one/               # StockVision 全栈 Web & 桌面分析平台
│   ├── backend/             # FastAPI + WebSockets 后端服务、SQLite 缓存、MCP 接口与策略核心
│   │   ├── main.py          # 核心 RESTful & WebSocket API
│   │   ├── strategy_adapter.py # 多周期价格行为、通道与波浪算法适配
│   │   ├── strategy_core/   # 后端内嵌策略核心子包
│   │   ├── data_provider.py # 历史 K 线获取与聚合
│   │   ├── quote_provider.py# 实时行情分层轮询与缓存调度
│   │   └── desktop_main.py  # 桌面内嵌 Webview 运行入口
│   └── frontend/            # Vue 3 + Vite + Tailwind CSS + Lightweight Charts v5 前端
├── .agents/skills/          # AI Agent 专用技能库
│   ├── okx-ai-trading/      # OKX AI Agent 7 步标准交易作业程序 (SOP)
│   └── us-stock-price-action-analysis/ # 美股纯技术流价格行为分析技能
├── config/                  # 标的清单与配置
│   ├── okx_universe.json    # OKX 永续合约标的池与风控默认参数
│   ├── tickers.txt          # 美股核心股票池 (30 只)
│   └── tickers_crypto.txt   # 加密货币标的池
├── data/                    # 历史行情缓存、元数据与风控/审计账本
├── docs/                    # 策略理论、防错经验库与架构设计文档
├── scripts/                 # 前后端启动脚本与波浪/箱体可视化脚本
├── tests/                   # 340+ 项自动化单元测试、因果自检与对抗强化套件
├── pa_engine.py             # 价格行为学核心分析引擎 (Swing/Dow/MSS/Box/Channel/Wave)
├── strategy_layer.py        # 美股策略层 (200 EMA 顺势代理/四指数协同/微观脱钩)
├── strategy_day.py          # 日内 5m 高分辨率微观策略引擎 (8 bps 摩擦/2.5R/1h HTF 门禁)
├── strategy_crypto.py       # 加密资产专用策略引擎 (2B 猎杀/防插针宽止损/分级止盈)
├── risk_ledger.py           # 持久化组合风控账本 (单笔上限/相关性桶/日亏熔断/连亏冷却)
├── backtest.py              # Walk-Forward 回测引擎与 T1~T16 零未来函数因果自检
├── data_sources.py          # 多源行情拉取、降级与连续性校验适配器
├── start_dev.bat            # Web 开发环境一键启动脚本 (FastAPI :8000 + Vite :5173)
├── start_desktop.bat        # 独立桌面应用一键启动脚本
└── stockvision.cmd          # StockVision CLI 快速入口
```

---

## 快速上手与常用命令

### 1. 启动 StockVision 全栈 Web / 桌面平台
```bash
# 一键启动前后端开发环境 (FastAPI @ http://127.0.0.1:8000 + Vite @ http://localhost:5173)
start_dev.bat

# 启动独立桌面端窗口
start_desktop.bat
```

### 2. OKX AI Agent 确定性交易工具链 (`okx_agent`)
```bash
# 1) 多周期行情分析与四向一致性校验 (冲突退出码 2)
python -m okx_agent.market --inst-id BTC-USDT-SWAP --htf 4H --mtf 1H --ltf 15m

# 2) 下单前硬门禁预检与原子分批止盈单生成 (拒绝退出码 1)
python -m okx_agent.preflight --input ticket.json

# 3) 零上下文 15s 后台价格哨兵
python -m okx_agent.watcher --inst-id BTC-USDT-SWAP --side buy --entry 100000 --tp1 102500 --sl 99000

# 4) 实盘/模拟盘状态与本地风控账本对账
python -m okx_agent.reconcile --input live_snapshot.json

# 5) 查看组合风控账本与交易审计日志
python risk_ledger.py status
python -m okx_agent.journal tail --limit 20
```

### 3. 运行单元测试与零未来函数因果自检
```bash
# 运行根目录完整单元与对抗测试套件 (含 okx_agent 全部测试)
python -m unittest discover tests

# 运行后端服务与加密 V2.1 特性测试
cd stage_one/backend && python -m unittest test_crypto_features.py test_dow_elliott_refactor.py test_backtest_swing.py

# 执行 T1~T16 零未来函数纯因果数学自检
python backtest.py --verify
```
