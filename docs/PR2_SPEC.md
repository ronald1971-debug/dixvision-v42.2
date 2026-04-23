# PR #2 Specification — Per-Form Dashboards, SL/TP, Modes, Memecoin Trio, Forex, Stocks, Fully Customizable Cockpit

**Status:** spec-only. No code lands from this PR until operator (you) approves.
**Scope:** dashboard re-architecture + new trading-form tabs + SL/TP engine + mode pipeline + memecoin copy/normal/sniper + forex + stocks + NFT + full user customization in all autonomy modes.
**Not in scope for this PR:** live wiring of new exchange keys; that stays gated behind operator approval + WARMUP clock + sandbox promotion.

---

## 0. Guiding principles

1. **Safety floors are never editable.** 4 % max drawdown, kill-switch, dead-man, WARMUP clock (30 d), sandbox gate, fast-path frozen functions, two-person gate for kill-switch override + fast-path amend. UI renders each as a read-only padlock widget with a tooltip citing the manifest clause that locks it.
2. **Everything else is user-adjustable in every autonomy mode** (USER_CONTROLLED, SEMI_AUTO, FULL_AUTO). Every change emits `OPERATOR/SETTINGS_CHANGED` with before/after values + mode-at-time + ISO timestamp, and is reversible from the Ledger tab.
3. **Same SL/TP engine everywhere.** What you tune in backtest behaves identically when promoted to paper → shadow → canary → live.
4. **Mode promote chain is gated.** Every promotion (Backtest → Paper → Shadow → Canary → Live) runs through the existing sandbox patch pipeline + operator click per manifest §15.
5. **Online-verified feature set.** Widgets and defaults reflect what the best 2026 desks actually ship (sources cited inline).

---

## 1. Widget-based cockpit architecture

Rewrite the current one-scroll SPA into a widget grid, similar to Grafana / TradingView / Bookmap.

### 1.1 Runtime

- **Grid:** React + `react-grid-layout` for drag / resize / dock / tab. Each widget has stable id, type, settings JSON.
- **Layout store:** per-profile JSON saved to `data/layouts/<profile>.json`. Sync to cloud when the cockpit is connected to the hosted 24/7 instance.
- **Hot reload of settings:** each widget subscribes to `OPERATOR/SETTINGS_CHANGED` and re-renders on next animation frame. No app restart for any non-floor change.
- **Widget types (v1):** Chart, Depth, Tape, OrderForm, Positions, PnL, Funding, LiqHeatmap, RouteGraph, PoolHealth, SlippageSim, GasEstimator, WalletCap, SL-TP-Builder, Strategies, AutonomyMode, OperatorApprovals, CustomStrategies, WeeklyScout, RiskCache, Safety, Traders, Chat, LedgerTail, Economicsaps, OptionsChain, OptionsRisk, Sweep, FloorChart, RarityLens, CopyLeaderboard, SnipeQueue, HolderDist, RugScore, AlertsHub, ThemeDial, PadlockFloors (manifest read-only).
- **Layout profiles:** `Conservative`, `Standard`, `Aggressive`, `Custom`, plus global templates `Desk-morning`, `Overnight`, `Weekend-memecoin-watch`.

### 1.2 Personalization

- Theme: dark / light / high-contrast.
- Font size, density, number locale, timezone, language (14+ today; extend as operator adds).
- Keyboard shortcuts: fully remappable, exportable JSON.
- Sound alerts: per-alert WAV, configurable per widget.

### 1.3 Audit

Every layout save, widget add/remove, settings dial, profile switch writes `OPERATOR/SETTINGS_CHANGED` + `OPERATOR/LAYOUT_CHANGED` to the ledger. Roll-back button in the Ledger tab restores any prior snapshot.

---

## 2. Chart library choice — TradingView Lightweight Charts v5

- **Pick:** `lightweight-charts@^5.1.0` (Apache 2.0, 35 KB gzipped, TypeScript-native), wrapped in `lightweight-charts-react-components@1.4` for the React layer.
- **Why:** Apache 2.0 license = no attribution liability, no keyed API, no CDN lock-in. v5 (released 2025-12-16) ships the unified `addSeries` API, tree-shakable series imports, first-class plugin typings. Bundle is ~2 % of the heavier Advanced Charts drop-in (700 KB) and works inside the Windows portable `.exe`.
- **Plugin slots we use:** candlestick, line, baseline (funding/SL/TP markers), histogram (volume), area (PnL), custom-series (ladder fills), watermarks for mode banner (`PAPER` / `SHADOW` / `CANARY` in bright diagonal text so the operator never confuses modes).
- **Fallback for heavy desk views:** allow operator to opt into the free-but-proprietary `charting_library` drop-in (Pine indicators, drawing tools) on a per-widget basis; default stays Lightweight.
- **Refs:** https://github.com/tradingview/lightweight-charts (v5.1.0 release), https://tradingview.github.io/lightweight-charts/docs/migrations/from-v4-to-v5, https://www.npmjs.com/package/lightweight-charts-react-components.

---

## 3. Per-form dashboards

Every trading form below gets its own tab with a suggested default layout (operator can rebuild freely). Every tab has the **Autonomy Mode**, **Kill-Switch**, **Dead-Man**, **Wallet-Policy clock** pinned along the top bar (padlock floors).

### 3.1 Spot

Default widgets:
- **Chart** (candlestick + volume, multi-timeframe, indicators: EMA, VWAP, RSI, MACD).
- **Depth Ladder** (aggregated L2 across enabled venues).
- **Time-and-Sales** tape.
- **OrderForm** (market / limit / stop-limit / OCO / bracket) with integrated SL/TP presets.
- **Positions + PnL** (realized + unrealized, per venue).
- **SL/TP Builder** (see §4).
Refs: TradingView Stock Heatmap ([link](https://www.tradingview.com/heatmap/stock/)), Coinbase Advanced Trade, Binance Pro, Kraken Pro.

### 3.2 Perps / Leverage (2026 native)

Default widgets:
- **Chart** with funding-rate histogram overlay.
- **Funding Table** per venue, next-funding countdown, cumulative-funding PnL estimator.
- **Liquidation Heatmap** (open-interest bands + projected liq price).
- **Margin Widget** (cross / isolated toggle, leverage slider, maintenance-margin ratio).
- **OrderForm** with reduce-only flag + bracket orders.
- **Per-venue oracle spread** (oracle price vs exec price, divergence alarm).
- **HIP-3 builder selector** (Hyperliquid) — pick trade.xyz stock-perp builder, Felix FLX, etc.
2026 enhancements:
- **Hyperliquid HIP-3 markets** (stock perps on-chain, $25 B+ volume since Oct 2025 mainnet; builder codes + 500k HYPE stake registered in `immutable_core` only for display). Ref: https://hyperliquidguide.com/ecosystem/hip-3-builder-codes, https://www.coingecko.com/learn/hyperliquid-hip3-hip4-tokenized-stocks-and-prediction-markets.
- **Hyperliquid HIP-4 prediction markets** tab (read-only first).
- **dYdX unlimited** + Drift V2 funding-arb widget.
- Bybit Copy-Trading leaderboard cross-linked into §3.4.1 copy-trader.

### 3.3 DEX / DeFi

Default widgets:
- **Chart** (on-chain TWAP + CEX reference overlay).
- **Route Graph** (Jupiter Juno / Iris / JupiterZ RFQ / DFlow / OKX splits visualized as a sankey).
- **Pool Health** (liquidity, 24h volume, utilization, LP concentration).
- **Slippage Sim** (quoted vs expected with price-impact curve).
- **Gas Estimator** (priority fee suggester: Helius p50/p75/p90, Ethereum base-fee + priority-tip).
- **Swap OrderForm** with max-slippage guard, MEV-protected RPC toggle, limit-order loop (DEX has no native stops — we synthesize).
2026 enhancements:
- **Jupiter Juno** self-learning aggregator replaces Metis as the default routing engine on Solana; Ultra Swap API for the best-execution path. Refs: https://station.jup.ag/docs/routing, https://dev.jup.ag/routing, https://www.dextools.io/tutorials/top-5-dex-aggregators-2026.
- **Intent-based execution** (1inch Fusion+, Odos v3, CowSwap-style solver auction) — route widget surfaces the winning solver + fallback.
- **MEV-Blocker / MEV-Share / Jito bundle** toggle per swap.

### 3.4 Memecoin (trio: copy + normal + sniper)

Shared widgets at the top of the tab:
- **Pair Card** (price, MC, FDV, 5m / 1h / 6h / 24h change, buys vs sells, unique wallets).
- **Holder Concentration** (top-10 %, top-50 %, dev-wallet %, LP lock status, mint/freeze authority status).
- **Rug Score** + inline safety badges (Photon-style: mint revoked, freeze revoked, LP burned, >10 % holder alerts). Ref: Photon Sol, BullX NEO, GMGN.ai 2026 terminals.
- **Live tape** of buys / sells with PnL per wallet (Axiom-style speed).

#### 3.4.1 Copy-trading (`mind/memecoin/copy_trader.py`)

- Watch up to 500 leader wallets (GMGN 2026 cap) across Solana + Ethereum mempools + Jito shred-stream.
- Per-leader settings: min $ size to mirror, max slippage, daily cap, auto-exit on leader exit, PnL cutoff (stop mirroring a leader after N losing trades).
- Target mirror latency sub-0.4 s (GMGN benchmark).
- Mirrors through our order engine — so **autonomy mode + wallet policy + kill-switch + dead-man + SL/TP engine all apply** exactly like a first-party trade.
- UI widgets: `CopyLeaderboard`, `LeaderPnL`, `MirrorQueue`, `CopyRiskDial`.
Refs: GMGN.ai docs, Bybit Copy-Trading, Banana Pro.

#### 3.4.2 Normal (`mind/memecoin/signal_trader.py`)

- Signal = DexScreener + Birdeye + GeckoTerminal momentum + Helius on-chain volume + X/TG sentiment (through existing providers) + holder growth + LP lock + rug score.
- Entry: operator-set composite threshold. Exit: SL/TP ladders (see §4.5).
- Honeypot + blacklist check: simulate the sell tx (Tenderly + Helius simulate) before placing the buy. If the sell simulation reverts, refuse the buy.
- Dev-dump watchdog: if dev wallet moves > N % of holdings → instant exit.
- UI widgets: `SignalTracker`, `CompositeScore`, `HolderGrowth`, `HoneypotCheck`.

#### 3.4.3 Sniper (`mind/memecoin/sniper.py`)

- Watches Pump.fun launches + Raydium pool creation + Uniswap V2 `PairCreated` / V3 `PoolCreated` via WebSocket.
- Pre-signs the buy + SL/TP exits in a single **Jito bundle** (Solana) or **Flashbots bundle** (EVM) so the first-block entry is atomic with stop orders already in the mempool.
- Mandatory filters: LP size ≥ threshold, liquidity locked, mint & freeze authorities revoked (Solana) or ownership renounced (EVM), honeypot-safe, dev wallet % of supply under cap, social presence (X/TG non-empty).
- UI widgets: `SnipeQueue`, `LaunchFirehose`, `FilterDial`, `BundleStatus`.
Refs: Jito Bundles docs, Flashbots Protect, Helius Priority Fees.

### 3.5 Forex

Default widgets:
- **Chart** (candlestick, pair tree: majors / minors / exotics).
- **Session Clock** (Sydney / Tokyo / London / New York, current / next / previous overlay).
- **ECN Depth** ladder (L2, actual ECN venue book where available).
- **Economic Calendar** (ForexFactory + TradingEconomics feed) with auto-pause hooks for high-impact events (FOMC / NFP / ECB / CPI).
- **Pip Calculator** + lot-size helper (standard / mini / micro).
- **Swap/Rollover** display (cost of holding overnight per pair).
- **Correlation Matrix** + risk-on/off regime dashboard.
Broker adapters (contracts scaffolded, live wiring per operator):
- OANDA v20 API, IG REST API, Interactive Brokers TWS / IBKR API, FXCM, Dukascopy, MT4 / MT5 bridge (read-only first; signing after operator approval).
2026 refs: MT5 DOM + economic-calendar updates; cTrader ECN depth; TradingView FX heatmap.

### 3.6 Stocks

Default widgets:
- **Chart** (candlestick + volume + VWAP + ORB).
- **Level-2** + time-and-sales.
- **Options Chain** (calls/puts, IV skew, open-interest, Greeks; ToS-style). Read-only first; order entry gated.
- **Fundamentals Pane** (P/E, P/B, FCF, debt, insider trades from EDGAR provider, institutional holdings, short interest, float).
- **Earnings + Dividend Calendar**.
- **Pre/Post-Market tape** (Nasdaq TotalView where available).
- **Tax-Lot Tracker** (FIFO / LIFO / specific-lot) + wash-sale flag (US).
- **Sector Heatmap** (IBKR Desktop v2.1 style — asset-class ETFs across equities, fixed income, commodities, REITs, international). Ref: https://supa.is/article/ibkr-desktop-heatmap-cross-asset-stocks-commodities-bonds-how-to-use-2026.
Broker adapters (contracts scaffolded):
- Alpaca (REST + streaming), Interactive Brokers, Tradier, Schwab/ThinkOrSwim bridge. Note: ThinkOrSwim is now a Schwab property after the TD Ameritrade absorption; we integrate via Schwab OAuth + the ToS bridge. Refs: https://www.schwab.com/learn/story/using-market-heat-map-on-thinkorswim-desktop, https://www.schwab.com/learn/story/setting-trailing-stops-on-thinkorswim-desktop (March 2026).

### 3.7 NFT

Default widgets:
- **Floor Chart** (timeseries per collection, aggregated across Blur / OpenSea Pro / Magic Eden / Tensor).
- **Trait-Floor Grid** (rarity-aware floors per trait).
- **Sweep Cart** with trait filters (Blur + Magic Eden 2026 style).
- **Collection-Bid Ladder** (Blur-style: bid at floor, floor-1 %, floor-2 %, etc.).
- **Blend Loans** widget (read-only; show open loans, expiries, liquidation risk). Ref: https://cryptoadventure.com/blur-review-2026-pro-nft-marketplace-features-fees-and-blend-lending/.
- **Auto-listing** rules (Magic Eden Pro-style).
- **Rarity Lens** (floors stratified by rarity band).

---

## 4. SL/TP engine (identical across all modes + all forms)

Each trading-form tab gets an **SL/TP Builder** widget. Same engine, same primitives, different defaults per form.

### 4.1 Primitives

- **Hard SL**: % or absolute price.
- **Trailing SL**: % from high-water mark (long) / low-water mark (short), with ratchet.
- **Timed SL**: auto-exit after N minutes if price hasn't moved ±X %.
- **TP ladders**: up to 5 legs with per-leg size % and trigger price/%.
- **Breakeven after first TP**: toggle.
- **OCO** (one-cancels-other) and **Bracket** orders at the primitive level.

### 4.2 Presets per form

| Form | Conservative | Standard | Aggressive |
|---|---|---|---|
| Spot | SL 3 %, TP 5/10/20 % (33/33/34) | SL 5 %, TP 20/50/100 % (25/25/50) | SL 8 %, TP 50/100/200 % + trailing runner |
| Perps | SL 1 % price + margin 50 % buffer, TP 3/6/12 % | SL 3 %, TP 10/25/50 % | SL 5 %, TP 25/50/100 % |
| DEX | SL 5 %, TP 20/50/100 % (limit-loop) | SL 10 %, TP 50/100/200 % | SL 15 %, TP 2x/5x/10x |
| Memecoin-Copy | Mirrors leader + independent SL 30 % + TP 2x/5x | Mirrors leader + SL 40 % + TP 3x/7x | Mirrors leader + SL 50 % + TP 5x/10x + runner |
| Memecoin-Normal | SL 30 %, TP 2x/5x ladder | SL 40 %, TP 2x/5x/10x | SL 50 %, TP 5x/10x/50x + 30 % trailing runner |
| Memecoin-Sniper | SL 40 % in-bundle, TP 2x/5x | SL 50 % in-bundle, TP 3x/10x | SL 60 % in-bundle, TP 5x/25x/100x |
| Forex | SL 10 pips, TP 20/40/80 pips | SL 20 pips, TP 40/80/160 | SL 30 pips, TP 60/120/240 |
| Stocks | SL 2 %, TP 4/8/16 % bracket | SL 4 %, TP 10/20/40 % | SL 6 %, TP 20/40/80 % |
| NFT | SL -15 % floor, TP +25 %/50 % | SL -20 %, TP +50 %/100 % | SL -25 %, TP +100 %/300 % |

Operator can override at global, per-form, per-strategy, or per-trade level. Overrides are ledger-audited.

### 4.3 Simulation

"Simulate" button runs the current SL/TP ruleset against the last N ticks and shows where it would have fired. Integrates with §5 Backtest for longer horizons.

### 4.4 DEX / Memecoin specific logic

- **Limit-order loop** on DEX: our engine watches price locally and submits a swap with `max_slippage` guard when SL/TP triggers; stops are not native to the pool.
- **Rug-trip SL**: if LP pulled, mint/freeze authority comes back live, blacklist added, or honeypot sell simulation starts failing → instant emergency exit (or flag "unable to exit" if honeypot confirmed).
- **Dev-dump SL**: if dev wallet moves > N % → exit.
- **Bundle SL/TP**: sniper pre-signs SL + TP into the same Jito/Flashbots bundle as entry → exits are already in the mempool.

### 4.5 Perps-specific logic

- Margin-%-aware SL (auto-close at `liq_price - safety_buffer`).
- Funding-flip SL: auto-flatten if funding flips adverse past operator threshold (e.g. > +0.05 % / 8h when long).
- Reduce-only guard on TP legs to avoid unintentionally opening the opposite side.

---

## 5. Mode pipeline — Backtest / Paper / Shadow / Canary / Live / Replay

Every strategy traverses the chain **Backtest → Paper → Shadow → Canary → Live**; each transition is a sandbox-pipeline promotion + operator click per manifest §15.

### 5.1 Backtest Lab

Dedicated tab with: strategy picker, date range, venue picker, param grid (grid or Bayesian), walk-forward out-of-sample validation, slippage+latency simulation flags.

Report view:
- Equity curve + drawdown chart
- Sharpe / Sortino / Calmar / R², win-rate, avg R:R, hold-time histogram
- SL/TP hit frequency + slippage histogram
- Trade log CSV export + per-trade ledger event
- Param-sweep heatmap

Data sources:
- Ledger tape (event store) — preferred, deterministic.
- Exchange REST candles.
- On-chain TX stream (Helius, Bitquery, Dune).

### 5.2 Paper Desk (forward-test)

Live signals, **simulated fills** at the next tick / VWAP / mid / operator-chosen fill model. Virtual $10 k balance (operator-adjustable). Same UI as Live — same SL/TP engine, same positions widget, same ledger events but tagged `PAPER_*`. Operator can run any number of Paper desks in parallel.

### 5.3 Shadow Desk

Real live market data + real signal generation + real order placement logic, **but orders are never sent to venues**. Writes `SHADOW_*` ledger events. Cockpit compares Shadow-fills vs Paper-fills vs what Live would have been, so the operator sees the gap between the simulator and reality before risking capital.

### 5.4 Canary Widget

Tiny live budget (operator-set, default $5, capped at WARMUP $100) for N minutes (default 30), with auto-rollback on:
- First drawdown breach (> 2 %).
- SL/TP engine anomaly.
- Shadow divergence > operator threshold.
- Any kill-switch / dead-man trigger.

All Canary outcomes write `CANARY_*` events.

### 5.5 Live Desk

Full autonomy-mode + wallet-policy + risk gates. Per-form tab is the live surface.

### 5.6 Replay Console

Re-runs a past live session (chosen by time range or by a specific ledger-event id) against a patched strategy to prove the patch would not have blown up. Must run cleanly before any `Shadow → Canary → Live` promotion after a patch.

### 5.7 Promote chain UI

Mode panel widget on every per-form tab shows: `[Backtest] → [Paper] → [Shadow] → [Canary] → [Live]` with a promote button between each stage. Each click:
1. Runs the sandbox patch pipeline (authority_lint + unit tests + dep scan + shadow test + canary run).
2. Requires operator click (two-person gate for fast-path-touching promotions).
3. Writes `GOVERNANCE/PROMOTION` to the ledger.
4. Advances the strategy's stage, bound to the ledger.

---

## 6. Autonomy × customizability matrix

| Setting | USER_CONTROLLED | SEMI_AUTO | FULL_AUTO |
|---|---|---|---|
| Caps ($ daily / per-trade / per-wallet / per-strategy / per-venue) | ✓ adjustable | ✓ adjustable | ✓ adjustable (change applies next tick) |
| Risk (size %, leverage, concurrent positions, portfolio heat) | ✓ | ✓ | ✓ |
| SL/TP (hard / trailing / ladders / breakeven / OCO / bracket) | ✓ | ✓ | ✓ |
| Strategies on/off, weightings, profiles | ✓ | ✓ | ✓ |
| Data sources on/off, cadence, priority | ✓ | ✓ | ✓ |
| Execution venue preference, slippage, MEV guard, algo (TWAP/VWAP/POV/iceberg) | ✓ | ✓ | ✓ |
| Timeouts (dead-man grace within manifest floor, latency budget, heartbeat cadence) | ✓ | ✓ | ✓ |
| UI (layout, theme, language, density, timezone, hotkeys) | ✓ | ✓ | ✓ |
| Alerts (what fires push, sound, kill-switch) | ✓ | ✓ | ✓ |
| **Live override in FULL_AUTO** (close-now / hold / trail to +X % on a single open position) | n/a | ✓ | ✓ |
| Custom strategies (paste Python → sandbox → approve → live) | ✓ | ✓ | ✓ |
| **Manifest floors** (4 % DD, kill-switch, dead-man, WARMUP clock, fast-path frozen, sandbox gate) | 🔒 | 🔒 | 🔒 |
| **Kill-switch override / fast-path amend** | 🔒 two-person | 🔒 two-person | 🔒 two-person |

Behavior after an adjustment:
- USER_CONTROLLED: change takes effect immediately; next intent still waits for operator click.
- SEMI_AUTO: change takes effect on next tick; auto-trading inside the new envelope.
- FULL_AUTO: change takes effect on next tick; auto-trading inside the new envelope without asking. One-click fall-back from FULL_AUTO to SEMI_AUTO or USER_CONTROLLED while trades are running.

---

## 7. Manifest-pinned floors (read-only padlock widget)

Top-bar widget on every tab with padlock icons and tooltips citing the manifest clause:

- 🔒 **Max drawdown 4.00 %** — §22 axiom (`immutable_core.foundation`, enforced in `governance/constraint_compiler.py`).
- 🔒 **Kill-switch** — §1 + §3.
- 🔒 **Dead-man switch** — §3.
- 🔒 **Wallet policy clock** — §8 (WARMUP 30 d → SUPERVISED 30 d / $100/day → OPERATOR-SET).
- 🔒 **Sandbox gate on code** — §15.
- 🔒 **Fast-path frozen functions** — `fast_execute_trade`, `fast_risk_cache`; amend requires two-person hardware-key.
- 🔒 **Manifest file is read-only** — addenda only through the sandbox pipeline.

Operator can **see** all of these and click for audit history, but **cannot** disable any.

---

## 8. 2026 enhancements rolled in

| Enhancement | Applied to | Source |
|---|---|---|
| Hyperliquid **HIP-3** builder-deployed perp DEXs, stock perps on-chain (trade.xyz, Felix FLX) | §3.2 Perps, §3.6 Stocks perp sub-mode | https://hyperliquidguide.com/ecosystem/hip-3-builder-codes; https://www.coingecko.com/learn/hyperliquid-hip3-hip4-tokenized-stocks-and-prediction-markets |
| Hyperliquid **HIP-4** prediction markets | §3.2 (read-only first) | same |
| **Jupiter Juno** self-learning aggregator (combines Iris + JupiterZ RFQ + DFlow + OKX) | §3.3 DEX route graph | https://station.jup.ag/docs/routing; https://dev.jup.ag/routing |
| **GMGN sub-0.4 s copy** up to 500 wallets, HL perps 50x integration | §3.4.1 Copy | https://www.dextools.io/tutorials/top-5-dex-aggregators-2026 + GMGN.ai 2026 release notes |
| **Photon Smart-MEV** Fast/Secure dual mode + inline mint/LP/holder badges | §3.4 safety badges, SL trip rules | Photon Sol 2026 terminal docs |
| **Axiom** raw speed + widget bar | §3.4 pair cards, live tape | Axiom 2026 terminal |
| **BullX NEO** + **Banana Pro** multi-chain widget bars | §3.4 cross-chain | BullX NEO, Banana Pro 2026 |
| **IBKR Desktop v2.1 Heatmap** (asset-class ETFs: equities + fixed-income + commodities + REIT + international, March 2026) | §3.6 Stocks sector heatmap | https://supa.is/article/ibkr-desktop-heatmap-cross-asset-stocks-commodities-bonds-how-to-use-2026 |
| **ToS on Schwab** migration (OAuth + bridge) | §3.6 broker adapters | https://www.schwab.com/learn/story/using-market-heat-map-on-thinkorswim-desktop |
| **Blur Blend loans** widget (read-only first) | §3.7 NFT | https://cryptoadventure.com/blur-review-2026-pro-nft-marketplace-features-fees-and-blend-lending/ |
| **OpenSea + Magic Eden + Tensor** sweep-by-trait + auto-listing | §3.7 NFT | https://www.coingabbar.com/en/crypto-blogs-details/blur-vs-opensea-vs-magic-eden-best-nft-marketplace-2026 |
| **Lightweight Charts v5** unified `addSeries` API, tree-shakable, Apache 2.0 | §2 chart library | https://github.com/tradingview/lightweight-charts; https://tradingview.github.io/lightweight-charts/docs/migrations/from-v4-to-v5 |
| **MEV-Blocker / MEV-Share / Jito bundles** toggle | §3.3 DEX, §3.4.3 Sniper | Jito, Flashbots, MEV-Blocker docs |
| **Intent-based routing** (1inch Fusion+, Odos v3, CowSwap solver auction) | §3.3 DEX | DEXTools 2026 aggregator ranking |
| **Economic-event auto-pause** (FOMC / NFP / ECB / CPI) | §3.5 Forex | ForexFactory, TradingEconomics API |

---

## 9. Migration + implementation plan (when this PR is approved)

1. **Scaffolding PR** — `cockpit/static/dashboard-v2/` with React + `react-grid-layout` + `lightweight-charts-react-components`, shell layout with today's 10 panels wrapped as widgets. No behavior change; feature-flag `DASHBOARD_V2=false` default.
2. **Per-form tab PR** — add Spot / Perps / DEX / Memecoin / Forex / Stocks / NFT tabs behind the flag, each with default widget set from §3.
3. **SL/TP engine PR** — `execution/sl_tp/` with the primitives from §4 + simulate endpoint + ledger events.
4. **Mode pipeline PR** — `modes/` module with Backtest / Paper / Shadow / Canary / Replay desk entry points + promote chain UI.
5. **Memecoin trio PR** — `mind/memecoin/{copy_trader,signal_trader,sniper}.py` + widgets.
6. **Customization + profiles PR** — layout store, profiles, hotkeys, padlock-floors widget.
7. **Flip the flag** — after all above merge + CI green + operator approval, `DASHBOARD_V2=true` becomes default.

Every PR in the chain:
- Ships behind the sandbox pipeline (authority_lint + tests + dep scan + shadow test + canary).
- Requires operator click to merge.
- Preserves manifest floors (no change to `immutable_core`, `governance/constraint_compiler`, `enforcement/*` safety rails).
- Adds regression tests for every new widget's wiring + every new endpoint.

---

## 10. Open questions for operator

Please confirm before I start coding:

1. **Chart library final call.** Default Lightweight Charts v5, allow opt-in to free-proprietary Advanced Charts per-widget? (Recommended: yes.)
2. **Forex broker priority.** OANDA + IG + IBKR + MT5 bridge in that order — agree?
3. **Stock broker priority.** Alpaca + IBKR + Schwab/ToS in that order — agree?
4. **Memecoin leaderboard seeding.** Start with a curated GMGN-style list + let operator edit, or empty + operator-only?
5. **Canary default size.** $5 for 30 min — too low / too high?
6. **FULL_AUTO live override visibility.** Should open-position cards show a permanent `close-now` button, or hide it behind a right-click menu?

Waiting on your OK + answers to the above before any code PR lands.

---

## 11. Canonical screen layouts (post-research expansion)

The widget grid from §1 ships with six canonical layouts the operator
can switch between instantly (`Ctrl+1..6`). Each is a validated
arrangement matching how professional desks actually use screen real
estate; all widgets remain drag/resize-overridable.

### 11.1 Single-monitor 4-pane grid (default desktop, 40/25/20/15)

```
+-----------------------------------+---------------------+
|                                   |                     |
|        Chart (40%)                |   Order book /      |
|        multi-TF,                  |   depth / tape      |
|        drawing tools,             |     (25%)           |
|        indicator stack            |                     |
|                                   +---------------------+
|                                   |                     |
|                                   |  OrderForm (20%)    |
|                                   |  + SL-TP builder    |
|                                   |                     |
+-----------------------------------+---------------------+
|  Positions / PnL / Alerts (15% full width)              |
+---------------------------------------------------------+
```

- **40 % Chart:** primary asset, configurable indicator stack, drawing
  tools persisted per symbol.
- **25 % Depth/Tape:** order book, time-and-sales, liquidation heatmap
  (derivatives), route graph (DEX).
- **20 % OrderForm + SL/TP builder:** all order types from §4, same
  SL/TP primitives across backtest / paper / shadow / canary / live.
- **15 % Positions / PnL / Alerts:** consolidated bottom bar with live
  P&L, open positions, latest alerts, dead-man lamp, WARMUP-clock lamp.

### 11.2 Dual-monitor split (standard desk)

- **Monitor 1 — Execution:** Chart (60 %) + Depth/Tape (25 %) + OrderForm (15 %).
- **Monitor 2 — Context + Risk:** Positions · PnL · Funding · Options chain / gamma
  (if mode=options) · Memecoin smart-money feed (if mode=memecoin) · LedgerTail · Alerts · PadlockFloors.

### 11.3 Triple-monitor split (power desk)

- **Monitor 1 — Execution** (as 11.2).
- **Monitor 2 — Context:** Positions · PnL · Funding · Route graph · Pool health · Economics ap.
- **Monitor 3 — Risk + Research:** RiskCache · Safety · PadlockFloors · LedgerTail · Strategies · WeeklyScout · CustomStrategies · AlertsHub.

### 11.4 Mobile portrait (PWA)

Stacked single-column, collapsible sections, top-to-bottom priority:

```
[PadlockFloors lamp bar]  (dead-man / WARMUP / kill-switch / drawdown)
[AutonomyMode selector]
[Chart — mini, configurable TF]
[OrderForm — compact]
[Positions — swipe-left to close]
[Alerts — infinite scroll]
[LedgerTail — last 20 events]
```

### 11.5 Tablet landscape

Two-column split; left = Chart + Depth, right = Positions + OrderForm +
Alerts. Full widget grid available via `Layout → Custom`.

### 11.6 Ultra-wide (32:9 / 49")

Four vertical columns: Chart · Depth+Tape · OrderForm+SL/TP · Positions+PnL+Alerts.
Gives the 4-pane grid without vertical splits.

All layouts render the **PadlockFloors lamp bar** persistently — dead-man,
WARMUP clock, kill-switch, drawdown floor, fast-path-frozen badge — so
the operator cannot lose sight of any safety floor regardless of
viewport.

---

## 12. Memecoin-mode cockpit tab (expansion of §3 memecoin trio)

The Memecoin tab ships with a dedicated widget set purpose-built for
the Solana / meme market, consuming the specs codified in
`docs/MEMECOIN_TRADING_SPEC.md` and
`docs/DEX_AND_BOT_ADAPTER_ROADMAP.md`.

### 12.1 Widgets

| Widget | What it shows |
|--------|---------------|
| **SafetyGate60s** | Live 60-second pre-trade pipeline per candidate token: mint/freeze/update authority · bundle detect · dev-wallet history · LP status · honeypot-sim. Row-red for any fail. |
| **SmartMoneyFeed** | Live net-buys from the operator-curated smart-money wallet list (GMGN / Birdeye / Solscan) — token · wallet winrate · net-buy 5m SOL · smart-money holder count. |
| **BundleAlerts** | Real-time alarm stream from the bundle detector — "3 fresh-funded wallets bought X in block 1, candidate auto-rejected." |
| **RugWarnings** | Post-entry alarm stream — slow-rug / mint-abuse / tax-trap / LP-ownership-moved / honeypot-late-enable. Click row → force-exit confirm. |
| **MemecoinPositions** | Per-position: entry, current, unrealized % / SOL, staged TP progress (2× / 3× / 5×), stop-loss, trailing status, smart-money-follower exit signal. |
| **BurnerWalletCap** | Memecoin burner bankroll usage vs. hard cap (governance limit), per-bot tier-2 burner balances, top-up queue. |
| **ModeSelectorMeme** | Off / Cautious / Aggressive with operator-TOTP transition. |
| **ExitEngine** | Staged TP targets, stop-loss, trailing config, smart-money-follower exit toggle. |
| **LPLockMonitor** | LP-ownership change watcher across all open memecoin positions. |
| **PumpFunGraduates** | Pump.fun / PumpSwap graduation stream + post-graduation liquidity curve. |
| **LaunchSniperQueue** | When Memecoin-Aggressive is active: pending Phase 1 + Phase 2 snipes with cancel button. |

### 12.2 Default Memecoin tab layout

```
+-------------------------+--------------------+----------------+
| SmartMoneyFeed          | SafetyGate60s      | ModeSelectorMeme
|                         |                    +----------------+
|                         |                    | BurnerWalletCap
|                         |                    +----------------+
|                         |                    | PadlockFloors
+-------------------------+--------------------+----------------+
| MemecoinPositions                           | ExitEngine      |
+---------------------------------------------+-----------------+
| BundleAlerts | RugWarnings | LPLockMonitor  | LaunchSniperQueue
+---------------------------------------------+-----------------+
| PumpFunGraduates (full width)                                 |
+---------------------------------------------------------------+
```

### 12.3 Safety interlocks (cockpit-side)

- The OrderForm on the Memecoin tab **will not submit** a buy until
  SafetyGate60s returns green; the submit button is disabled with a
  tooltip listing which checks are still pending or red.
- Position size input is hard-clamped to the governance cap for the
  active Memecoin mode — the UI cannot submit a size above the cap.
- Any red row on RugWarnings flashes the padlock lamp bar; operator
  cannot navigate away from the Memecoin tab without acknowledging.

---

## 13. AI-platform feature-parity checklist

Built as a public checklist so every operator request can be traced to
a researched platform. Each row is a widget or a strategy template in
the cockpit — not a new autonomous agent. Governance authority is
unchanged.

| # | Platform | Claimed capability | DIX widget / template |
|---|----------|--------------------|-----------------------|
| 1 | Trade Ideas (Holly) | Daily high-probability trade setups | `AISignalsFeed` widget consuming Holly API (read-only) |
| 2 | TrendSpider | Automated TA, chart pattern recognition | `PatternScanner` widget + chart overlay |
| 3 | QuantConnect | Institutional algorithmic backtesting | `BacktestRunner` widget + `CustomStrategies` template registry |
| 4 | Tickeron | AI pattern recognition (equity + crypto) | Additional feed into `AISignalsFeed` |
| 5 | Alpaca | API-first developer broker | Broker adapter (Phase 2); paper + live desks |
| 6 | MenthorQ | Dealer positioning, gamma, volatility | `OptionsRisk` + `GammaHeatmap` widgets |
| 7 | PredictIndicators.ai | Forward-looking indicator forecasts | `IndicatorForecast` widget (MACD / RSI / BB / Stoch with confidence gradient) |
| 8 | TradeAlgo | Options analytics (Greeks, margin, AI risk) | `OptionsRisk` widget + `MarginTracker` |
| 9 | Sharesight | Multi-broker portfolio performance | `PortfolioAttribution` widget (Sharpe, max DD, alpha/beta) |
| 10 | Portfolio Visualizer | Performance attribution analytics | Same widget as #9, additional tabs |
| 11 | Kubera | Unified view (brokerage + crypto + DeFi) | `UnifiedPortfolio` widget aggregating all DIX-connected venues |
| 12 | Wealthica | Unified portfolio aggregation | Merged into `UnifiedPortfolio` |

**Every feed is read-only.** Signals are surfaced as **advisory input**
to the operator and (optionally, gated) to Indira's feature weights.
Nothing on this list places a trade, approves a trade, modifies a
rule, or bypasses the governance kernel. Same N1/N7 constraints as the
neuromorphic triad.

---

## 14. Broker / venue adapter integration plan (cockpit-side)

The cockpit surfaces every adapter listed in
`docs/DEX_AND_BOT_ADAPTER_ROADMAP.md` through a uniform
**BrokerStatus** widget + a uniform OrderForm routing dropdown.

### 14.1 BrokerStatus widget

- Per-adapter row: connection status · auth-token age (with rotate
  button) · rate-limit headroom · p95 latency · dead-man timestamp ·
  last error · burner-wallet balance (DEX/bot rows only).
- Color: green / yellow (degraded) / red (dead-man tripped or
  circuit-open) — matches Grafana palette for consistency.

### 14.2 Routing dropdown on OrderForm

- Shows only adapters valid for the active tab (e.g. Memecoin tab only
  lists Jupiter / Raydium / PumpSwap / Four.meme / GMGN / BullX /
  Trojan / Axiom / BonkBot / Banana Gun / Photon / Maestro).
- Greys out any adapter whose dead-man is stale or whose circuit is
  open.
- Greys out any adapter whose burner is below the per-trade floor.

### 14.3 Phase schedule for cockpit adapter rows

| Phase | Adapters visible in cockpit |
|-------|---|
| PR #2 scaffolding | Placeholder rows for every adapter, `status=PENDING_PHASE_2` |
| Phase 2 batch 1 | Jupiter · Raydium · Pump.fun/PumpSwap · Binance · Coinbase · IBKR |
| Phase 2 batch 2 | Uniswap v4 · Four.meme · Hyperliquid · dYdX · Alpaca |
| Phase 5 | Sniper-bot gateways (GMGN / BullX / Trojan / Axiom / BonkBot / Banana Gun / Photon / Maestro) |
| Phase 6 | Forex broker batch 1 (Exness · IC Markets · Pepperstone · FP Markets · Axi) |
| Phase 6 | Forex broker batch 2 (IG · FxPro · ThinkMarkets · Tickmill · OANDA · XM · RoboForex · Capital.com) |
| Phase 6 | Forex broker batch 3 (BlackBull · Hantec · FXTM · AvaTrade · HFM · HYCM · IQ Option · Libertex · Deriv · Quotex) |
| Phase 6 | Stock brokers (ProRealTime+IBKR · TradingView+broker · NinjaTrader) |

Each adapter row exposes its authority_lint class (C2 / C3 / W1 / W2)
in a hover tooltip so the operator can verify the authority boundary
at a glance.

---

## 15. Browser / session hardening

The web cockpit is a control surface for a live trading system. It
ships with mandatory client-side hardening beyond the existing
`cockpit/auth.py`.

| Control | Requirement |
|--------|-------------|
| **HTTPS** | TLS-only. HSTS with 1-year `max-age` + `includeSubDomains` + `preload`. HTTP requests redirected 308. |
| **Cert pinning** | Cockpit PWA embeds a public-key pin for the hosted 24/7 instance; mismatch = refuse to connect + operator alert. |
| **2FA / TOTP** | Every login requires TOTP. TOTP also re-prompts for: mode change, canary→live promote, kill-switch override, wallet top-up, seed-list edit, authority_lint rule change. |
| **Device binding** | Cockpit generates a device key pair at pairing time; session tokens are bound to that device key (WebAuthn). Sessions cannot be replayed from a different device. |
| **Session timeout** | Idle timeout: 15 min. Absolute max session: 12 h. Re-auth forces TOTP. |
| **Login-history monitor** | Widget `LoginHistory` shows every successful + failed login attempt with IP / device-ID / geo / timestamp. Anomaly (new geo, new device, impossible-travel) → lockout + operator alert. |
| **Dedicated browser profile** | Cockpit recommends a dedicated Chromium profile with no extensions, no sync, no saved credentials outside the device key. One-click script to provision. |
| **Download / paste controls** | Cockpit never downloads files to disk except explicitly requested exports. Paste of private keys into any form is intercepted + blocked + alerted. |
| **Content-Security-Policy** | Strict CSP: `default-src 'self'`; no inline scripts; explicit allowlist for chart / telemetry origins. |
| **Rate limit / brute-force lockout** | 5 failed logins → 15-min lockout; 15 failures → permanent lockout + operator unlock required. |
| **Operator-override double-click** | Any "live" operator-override button (force exit all, kill-switch, mode change to FULL_AUTO) requires double-click + TOTP within 30 s window. |

Every hardening control maps to a manifest safety clause and is
rendered as a read-only row on the `PadlockFloors` widget.

---

## 16. Observability — Prometheus + Grafana dashboards (Phase 7)

The cockpit is the operator's window; Grafana is the forensic layer.
Phase 7 ships a fixed set of Grafana dashboards + their Prometheus
metric definitions, no ad-hoc dashboards.

### 16.1 Metric families (exported by every service)

- `dix_ledger_events_total{type}` — counter, per event type.
- `dix_hazard_events_total{severity}` — counter.
- `dix_deadman_last_tick_seconds{sensor}` — gauge (age of last
  proof-of-life per neuromorphic / DYON / memecoin sensor).
- `dix_adapter_latency_seconds{adapter,stage}` — histogram.
- `dix_adapter_error_total{adapter,kind}` — counter.
- `dix_adapter_circuit_state{adapter}` — gauge (0 closed / 1 half / 2 open).
- `dix_position_pnl_bps{asset,mode}` — gauge.
- `dix_drawdown_bps{account}` — gauge.
- `dix_memecoin_safety_gate_total{check,result}` — counter.
- `dix_memecoin_rug_alerts_total{pattern}` — counter.
- `dix_operator_approvals_total{action,result}` — counter.
- `dix_autolearn_fetches_total{domain,status}` — counter.
- `dix_autolearn_filter_total{verdict}` — counter.
- `dix_autolearn_approvals_total{result}` — counter.
- `dix_sandbox_pipeline_stage{stage,status}` — gauge.

### 16.2 Dashboards shipped in Phase 7

| # | Dashboard | Panels |
|---|-----------|--------|
| 1 | **Safety floors** | Dead-man lamp matrix · WARMUP clock · drawdown vs. 4 % floor · kill-switch state · hazard-event rate by severity · frozen-fast-path attestation. |
| 2 | **Adapter health** | Per-adapter latency p50/p95/p99 · error rate · circuit state · token-age / rate-limit headroom. |
| 3 | **Trading** | Positions · P&L by account / mode / asset · drawdown · Sharpe rolling 30d · win/loss by mode. |
| 4 | **Memecoin** | Safety-gate pass/fail by check · rug-alerts by pattern · burner-wallet balances · per-bot tier-2 burner balances · sniper queue depth · mode-transition log. |
| 5 | **Autolearn** | Crawler fetch rate by domain · filter verdict split · operator approve/reject rate · pending-buffer depth · per-source trust drift. |
| 6 | **Sandbox pipeline** | Per-stage status (lint / tests / axiom-check / shadow / canary) · promote-chain velocity · operator approval latency. |
| 7 | **Operator** | Login attempts (success/fail/geo) · TOTP-prompt rate · override-button rate · settings-change rate. |
| 8 | **Governance** | Policy-rule hits · constraint-violation attempts · mode-transition log · two-person-gate events. |
| 9 | **System / node** | CPU · RSS · GC · disk I/O · ledger write latency · hazard-bus backlog · event-loop lag. |
| 10 | **Options / gamma** | Gamma exposure · dealer positioning · implied vol surface · MenthorQ-parity panels. |

### 16.3 Alert routing (Phase 7)

- Grafana alerts feed a single `SYSTEM_HAZARD` channel back into the
  DIX hazard bus (severity-mapped).
- Critical alerts mirror to the cockpit `AlertsHub` widget and to the
  operator's pager.
- Nothing in Grafana can execute, approve, or modify state — it is a
  sink for metrics, a source of operator-visible alerts, and nothing
  else.

---

## 17. What this expansion does NOT do

- No new runtime code in this PR; it remains spec-only.
- No commitment to a specific chart-library vendor for the AI-parity
  widgets (IndicatorForecast / PatternScanner / OptionsRisk) — that
  decision stays in §10 open questions.
- No changes to the existing authority_lint rules C1/C2; W1/W2/C3 are
  specified in the companion docs (MEMECOIN_TRADING_SPEC, INDIRA_WEB_AUTOLEARN_SPEC)
  and implemented in Phase 5/6.
- No change to the sandbox-gated promote chain; every widget listed
  above rides the same chain once it has runtime code.

