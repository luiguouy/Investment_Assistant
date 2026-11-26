---
name: okx-ai-trading
description: Standard operating procedure for AI Agent autonomous/semi-autonomous perpetual swap trading on OKX using deterministic okx_agent Python gates and okx-trading MCP tools.
---

# OKX AI Agent Trading SOP (`okx-ai-trading`)

> Core principle: **Deterministic Python computes; AI Agent decides, confirms with user, and orchestrates `okx-trading` MCP.**
> Never hand-compute `1R`, `Weighted_RR`, contract sizing, liquidation distance, or four-way consistency in prose.

---

## Step 1: Market Scan & Four-Way Consistency Gate

1. Check `config/okx_universe.json` (`BTC-USDT-SWAP`, `ETH-USDT-SWAP`, `SOL-USDT-SWAP`, or `AAPL/NVDA/TSLA-USDT-SWAP`).
2. Run the deterministic multi-timeframe analyzer:
   ```bash
   python -m okx_agent.market --inst-id <INST_ID> --htf 4H --mtf 1H --ltf 15m
   ```
   - For stock-token perps (`AAPL-USDT-SWAP`, etc.), first check `pa_engine.py --ticker <BASE>` and pass `--external-pa-bias <-1|0|1>`.
3. **Gate**:
   - Exit code `2` (`verdict == "CONFLICT_HALT"`) or `verdict == "NEUTRAL_WAIT"` → **HALT / NO TRADE**.
   - Log the skip via `python -m okx_agent.journal append --event-type SCAN_CONFLICT_HALT --inst-id <INST_ID> --payload '<json>'` and stop.
   - Only proceed when `verdict == "ALLOW"` (`LONG` or `SHORT`).

---

## Step 2: Live Venue Snapshot (`okx-trading` MCP)

Collect live account and market specs before sizing:
1. `account_get_config(simulatedTrading=<bool>)` — require `acctLv >= 2`.
2. `account_get_balance(simulatedTrading=<bool>)` — read available USDT `eq` (`equity_usd`).
3. `market_get_instruments(instType="SWAP", instId="<INST_ID>")` — read `ctVal`, `lotSz`, `minSz`, `tickSz`.
4. `market_get_orderbook(instId="<INST_ID>", sz=5)` — read `bids` & `asks`.
5. `market_get_funding_rate(instId="<INST_ID>")` — read `fundingRate`.
6. `account_get_trade_fee(instType="SWAP", simulatedTrading=<bool>)` — omit `instId` to avoid `50016`; pass `taker` fee rate directly (normalizer handles negative OKX values).

---

## Step 3: Deterministic Pre-Flight Hard Gate

1. Build a JSON ticket combining Step 1 levels (`entry`, `sl`, `tp1`, optional `tp2`, `risk_pct`, `leverage`) and Step 2 venue snapshot.
2. Run:
   ```bash
   python -m okx_agent.preflight --input <ticket.json>
   ```
3. **Hard Veto**:
   - Exit code `1` (`passed == false`) → **DO NOT propose or place any order**.
   - Report exact `status` (e.g., `REJECTED (< 2.5R)`), `metrics`, and `rejections` to the user, log to `okx_agent.journal`, and stop.

---

## Step 4: User Confirmation & Atomic Split-TP Execution

1. Present the `PASS` ticket summary (`inst_id`, `side`, `entry`, `sl`, `tp1`/`tp2`, `total_sz`, `net_weighted_rr`, `liq_dist_r`, `simulatedTrading`) and obtain user confirmation (default `simulatedTrading: true` unless live is explicitly requested).
2. Execute each order in `result["orders"]` verbatim via `okx-trading` MCP `swap_place_order` (two atomic orders for split-TP, 100% covered by `mark`-triggered SL from millisecond zero).
3. On fill/acceptance, record in both ledgers:
   ```bash
   python risk_ledger.py open --symbol <INST_ID> --risk-pct <actual_risk_pct> --entry <entry> --sl <sl> --venue okx_swap
   python -m okx_agent.journal append --event-type ORDER_PLACED --inst-id <INST_ID> --payload '<json>'
   ```

---

## Step 5: Zero-Context Background Monitoring

Launch the lightweight 15s background watcher (`run_command` with `WaitMsBeforeAsync=1000`, `IsDaemon=false`):
```bash
python -m okx_agent.watcher --inst-id <INST_ID> --side <buy|sell> --entry <entry> --tp1 <tp1> --sl <sl> [--watch-entry]
```
- **Do NOT** use `schedule(CronExpression=...)` to poll prices.
- End your turn and wait for the watcher wakeup notification.

---

## Step 6: Wakeup Reconciliation & Breakeven Move

On any watcher wakeup (`LIMIT_ENTRY`, `TP1_BE`, `SL_NEAR`, or `WATCHER_ERROR_CIRCUIT_BREAKER` exit code `3`):
1. Fetch live state via `okx-trading` MCP:
   - `swap_get_positions(simulatedTrading=<bool>)`
   - `swap_get_orders(status="open", simulatedTrading=<bool>)`
   - `swap_get_algo_orders(status="pending", simulatedTrading=<bool>)`
2. Run deterministic reconciliation:
   ```bash
   python -m okx_agent.reconcile --input <live_snapshot.json>
   ```
   - If `ACTION_PLACE_MISSING_SL` is returned (asymmetric partial fill), immediately call `swap_place_algo_order` with the generated `params`.
3. On `TP1_BE` wakeup (after TP1 tranche fills), validate `markPx` vs `Entry` using `okx_agent.reconcile.plan_breakeven_amend(...)`:
   - If `AMEND_SL_TO_BREAKEVEN`: call atomic `swap_amend_algo_order` (never cancel-then-replace).
   - If `MARKET_CLOSE_REMAINING`: call `swap_close_position` immediately.
4. If restarting the watcher while price hovers near a boundary, pass `--disarmed` so hysteresis (`buffer_pct + hysteresis_pct`) prevents ping-pong wakeups.

---

## Step 7: Position Close, Slippage Audit & Journal

When the position closes:
1. Check `okx_agent.reconcile.evaluate_realized_slippage(entry, sl, exit_px, side, top5_depth_usd, notional_usd)`:
   - If `halt_instrument == true` (`realized_r < -1.5` or liquidity collapse), halt further trading on `<INST_ID>`.
2. Record realized PnL in `risk_ledger.py` and `okx_agent.journal`:
   ```bash
   python risk_ledger.py close --symbol <INST_ID> --pnl-r <realized_r> --pnl-pct <realized_pct>
   python -m okx_agent.journal append --event-type POSITION_CLOSED --inst-id <INST_ID> --payload '<json>'
   ```
