# Investment Assistant Workspace Rules

## 1. Pre-Order Risk-Reward & Sizing Hard Gate (实盘/模拟下单前硬门禁)

Before proposing or placing ANY live or simulated order (via `okx-trading` MCP or trade plan):

1. **Account & Venue Pre-Flight**:
   - For `SWAP` orders, verify `acctLv >= 2` via `account_get_config` (if `acctLv == "1"`, stop and ask the user to switch to Single-currency Margin mode).
   - Check `market_get_instruments` (`ctVal`, `lotSz`, `minSz`), `market_get_orderbook` (require top-5 depth `>= 3x` order notional and spread `<= 5 bps`), `market_get_funding_rate`, and `account_get_trade_fee(instType="SWAP")` (omit `instId` to avoid `50016`).
   - For OKX `*-USDT-SWAP` (including stock-token swaps like `AAPL-USDT-SWAP`), use OKX `market_get_candles` (`1H`/`15m`) as execution-timeframe truth for S/R, 2B sweeps, and volume; treat Yahoo `pa_engine.py` / local CSV as US-session / HTF context only.
2. **Position Sizing, Portfolio Cap & Liquidation Guard**:
   - Cap `1R` risk at `0.5%–1.0%` of account equity (`Risk_USD = Equity * Risk_Pct`, `Size = Risk_USD / (1R * ctVal)` rounded to `lotSz`, `>= minSz`).
   - **Portfolio & Correlation Cap**: Sum of open `1R` risks across all concurrent positions must be `<= 2.5%–3.0%` of equity; max `1–2` positions per correlated beta/sector bucket (e.g., US mega-cap tech `AAPL/NVDA/MSFT/QQQ` or crypto majors `BTC/ETH/SOL`).
   - **Persistent Risk Ledger Enforcement**: Before proposing or placing ANY live/demo order, run `python risk_ledger.py check --symbol <SYMBOL> --risk-pct <PCT>` (and call `open` / `update-risk` / `close` on fills/BE/exits) so daily loss (`-2.0%` / `-2.5R`), max drawdown (`-5.0R`), 3-loss/rolling-loss 24h cooldown, and correlation bucket caps persist across sessions.
   - **Loss Circuit Breakers**: Daily loss limit `-2.0%` (or `-2.5R`) halts new entries for the day; `3` consecutive `-1R` losses triggers a 24h cooldown and halves `1R` to `0.25%–0.5%`.
   - **Liquidation & Stock-Token Perp Guard**: Require `|Liq_Px - Entry| >= 2.0 * 1R` on standard crypto perps and `>= 4.0 * 1R` on OKX stock-token perps (`AAPL-USDT-SWAP`, etc.); avoid holding full size across weekends or earnings gaps.
3. **Compute Exact Risk (`1R`) & Weighted R:R**:
   - `1R = |Entry - SL|`.
   - Single TP: `RR = |TP - Entry| / 1R`; Split TPs: `Weighted_RR = (w1 * |TP1 - Entry| + w2 * |TP2 - Entry|) / 1R`.
   - Subtract **venue-aware friction**: `8 bps` for US cash equities; `10 bps` taker round-trip (or actual `account_get_trade_fee` rate) + spread/slippage + funding for OKX `SWAP`.
4. **Hard Veto (`< 2.5R`)**:
   - If net `Weighted_RR < 2.5` (or net `RR < 2.5`), **DO NOT propose, confirm, or execute the order**.
   - State exact `1R` and `Weighted_RR`, mark **REJECTED (< 2.5R)**, and either wait for a better pullback/retest entry or skip the setup. Never widen SL or narrow TP1/TP2 just to force a fill.
5. **Atomic Split-TP Execution (`TP1 + TP2`)**:
   - `swap_place_order` / `spot_place_order` attach only 1 TP + 1 SL per order. To execute split TPs without an unprotected post-fill window, split the entry into **two separate orders** (e.g. `50%` with `TP1 + SL`, `50%` with `TP2 + SL`) so 100% of size has an atomic SL from millisecond zero.
   - Default `slTriggerPxType: "mark"` (anti-wick) and `tpTriggerPxType: "last"` on perps. Note that attached `attachAlgoOrds` appear on `swap_get_orders(status="open")`, not `swap_get_algo_orders(status="pending")`, until the parent order fills.

## 2. Zero-Context Position Monitoring (零上下文持仓监控)

When monitoring an open position or pending entry over time:
- **Do NOT use recurring LLM cron (`schedule(CronExpression=...)`)** to poll prices every minute—this bloats context.
- Instead, launch a lightweight local Python background watcher (`run_command` with `WaitMsBeforeAsync=1000`, `IsDaemon=false`) that polls public ticker data every 15s and exits only when price enters a critical alert buffer near **pending limit entry**, **TP1** (to move remaining SL to breakeven), or **SL**.
- On alert wakeup, verify actual position/orders/algo orders via `okx-trading` MCP. When moving remaining SL to Breakeven after TP1 fills, use **atomic `swap_amend_algo_order`** (never cancel-then-replace) and verify `markPx` vs `Entry` before amending; if price oscillates on the alert border without triggering, widen/adjust the alert buffer slightly (with hysteresis) before restarting the watcher to avoid ping-pong wakeups.

## 3. Engine Output Cross-Verification (引擎结果交叉验算)

When analyzing markets or tickers with `pa_engine.py` or `analyze_symbol.py`:
- **Never draw final trend conclusions from `--watchlist` summary lines alone.** Always run full single-ticker analysis (`--ticker <SYMBOL>`) for primary indices (`SPY`, `QQQ`) or any trade candidate.
- **Four-Way Consistency Check**: Cross-verify `[二] Dow Phase`, `[二b] MSS`, `[三c] Channel Slope`, and `[八] bias_htf / bias_mtf`.
- **Conflict Halt**: If any two sections contradict (e.g., Phase says bullish while MSS shows a recent breakdown or `bias_htf/mtf == -1`), **halt directional calls/orders**, report the contradiction explicitly, and inspect the underlying pivots/code before proceeding.
- **Engine Arbitration & Session Boundary**: Use `pa_engine.py` as truth for US cash equities and `stockvision-trading-system` + OKX live `1H`/`15m` candles for crypto (`BTC`/`ETH`/`SOL`). For OKX stock-token perps (`AAPL-USDT-SWAP`, etc.), if `pa_engine.py` and OKX live structure disagree, **HALT / NO TRADE**. Narrow-stop `--mode day` setups must flatten before `15:55 ET` and must never be held overnight on 24/7 stock-token perps.

## 4. Failure-Mode Playbooks (异常与失败场景处置预案)

1. **Asymmetric Partial Fill (分批挂单单边部分成交)**:
   - On any partial-fill wakeup, inspect `swap_get_positions` and `swap_get_orders(status="open")`. Reconcile exact filled size so 100% of open position size has an active `mark`-triggered SL (`swap_place_algo_order` if uncovered) and recalculate actual `Weighted_RR` before continuing.
2. **TP1 Filled & `markPx` Already Crossed `Entry` (保本改单被拒)**:
   - Never cancel the existing SL before amending. If `swap_amend_algo_order` to `Entry` is rejected because `markPx` already crossed `Entry` on the losing side, immediately market-close the remaining tranche (or tighten SL strictly behind current `markPx` if structure holds).
3. **Watcher Crash / `>= 5` SSL Timeouts / Server Restart**:
   - On restart or watcher failure, immediately call `swap_get_positions` + `swap_get_orders(status="open")` + `swap_get_algo_orders(status="pending")` to reconcile live state before relaunching the watcher.
4. **In-Position Liquidity Collapse or Gap Slippage (`> 1.5 * 1R`)**:
   - If realized SL slippage exceeds `1.5 * 1R` (or off-hours top-5 book depth collapses below `1x` position notional), record the full realized `-R` into `risk_ledger.py` immediately and halt off-hours trading on that thin-book instrument.

## 5. OKX AI Agent Deterministic CLI Toolchain (`okx-ai-trading`)

Follow [.agents/skills/okx-ai-trading/SKILL.md](.agents/skills/okx-ai-trading/SKILL.md) and delegate all numeric/gate computations to `okx_agent`:
- `python -m okx_agent.market` — OKX multi-timeframe candle fetch + `pa_engine` Four-Way Consistency Check (`CONFLICT_HALT` exits `2`).
- `python -m okx_agent.preflight` — Pre-order hard gate (`acctLv >= 2`, depth `>= 3x`, spread `<= 5 bps`, net `Weighted_RR >= 2.5`, liquidation guard, `risk_ledger.py`) + atomic split-TP `swap_place_order` payload generator (rejections exit `1`).
- `python -m okx_agent.watcher` — Zero-context 15s background price watcher with hysteresis and 5-error circuit breaker (exit `3`).
- `python -m okx_agent.reconcile` — Live position/order/algo vs `risk_ledger.json` reconciler and failure-mode playbook planner.
- `python -m okx_agent.journal` — Append-only trade and rejection audit log (`data/okx_journal.jsonl`).
