# DIX VISION v42.2 — DEX, Bot & Venue Adapter Roadmap

**Status:** reference / priority list (no adapters built yet)
**Implementation phase:** Phase 2 (core adapters) · Phase 5 (safety-wrapped memecoin execution)
**Governing document:** see `MEMECOIN_TRADING_SPEC.md` for memecoin-specific safety rules.

This document lists every trading venue, bot, safety API, analytics
provider, and CEX gateway that DIX will eventually integrate with,
grouped by role. Each row is a target only — **no adapter exists yet**.

Every adapter must obey:

- `authority_lint` C2 (no forbidden imports into `governance.*`,
  `mind.fast_execute`, `execution.engine` from adapter code).
- `authority_lint` W1 (memecoin adapters never touch main wallet).
- N2 (event-only outputs — adapters emit events, never decide).
- N4 (every outbound trade + every inbound fill = ledger row).
- N5 (each adapter exposes `check_self()` — dead-man proof-of-life).
- Rate-limit + circuit-breaker wrappers (Phase 3 DYON).
- Sandbox-test suite before promotion to live.

No adapter may import governance, policy, or fast-path modules directly.

---

## 1. Execution Adapters — Solana / Memecoin-native

| # | Venue | Kind | Chains | Auth | Notes |
|---|-------|------|--------|------|-------|
| 1 | **Jupiter** | DEX aggregator | Solana | API key optional | Routes across 20+ DEXs; best-price engine; quote-sim used for honeypot check (§1.1 of memecoin spec) |
| 2 | **Raydium** | CLMM AMM | Solana | RPC + wallet | Graduation destination from Pump.fun; deepest Solana meme liquidity |
| 3 | **Pump.fun / PumpSwap** | launchpad + AMM | Solana | wallet | Source of 11M+ tokens; bonding curve discovery; pre-graduation sniping |
| 4 | **Four.meme** | launchpad + AMM | BNB | wallet | Binance-ecosystem equivalent of Pump.fun |
| 5 | **Uniswap v4 + Unichain** | AMM + hooks | Ethereum / Unichain / Base | RPC + wallet | Hooks enable on-chain stop-loss / take-profit |
| 6 | **Hyperliquid** | perp DEX | Hyperliquid L1 | API key + signing | High-leverage perp DEX; memecoin perps |
| 7 | **dYdX v4** | perp DEX | dYdX Cosmos | API key + signing | Backup perp DEX |

Priority order for Phase 2 build-out: **Jupiter → Raydium → Pump.fun / PumpSwap → Uniswap v4 → Four.meme → Hyperliquid → dYdX**.

---

## 2. Execution Adapters — Sniper / Trading Bots

These are **external tools** that DIX can optionally route orders
through. Each integration is opt-in per mode and per operator toggle.
None of them are allowed to be autonomous; DIX is the caller, they are
the callee. DIX must be able to run memecoin trading fully without them.

| # | Bot | Interface | Best for | Key risk |
|---|-----|-----------|----------|----------|
| 1 | **GMGN** | Web + Telegram bot | Smart-money wallet tracking + sniping | Telegram bot imports private key — use burner only |
| 2 | **BullX** | Web + API | Ultra-fast sniping, bundle detection | Web key exposure |
| 3 | **Trojan** | Telegram bot + Web | Mobile / on-the-go sniping, MEV protection | Telegram bot imports private key — use burner only |
| 4 | **Axiom** | Web dashboard | Power-user sniping | Complexity — requires careful config |
| 5 | **BonkBot** | Telegram bot | Pump.fun-specific sniping | Private key exposure; use burner |
| 6 | **Banana Gun** | Telegram bot | Multi-chain sniping | Historical hack (2024, $3M loss) — burner-only, low balance |
| 7 | **Photon** | Web | Solana sniping, price alerts | Fee structure per-trade |
| 8 | **Maestro** | Telegram bot | EVM-focused sniping, copy-trading | Multi-chain exposure |

**Integration rule:** any bot that requires a private key uses a
dedicated burner wallet, never the main burner, never the treasury.
This creates a tier-2 burner scheme: Main Treasury → Main Memecoin
Burner → Per-Bot Burner. `authority_lint` rule W2 enforces this.

---

## 3. Discovery / Analytics APIs

Observe-only. These feed signals into Indira and the safety gate. They
never place orders, never modify state.

| # | Service | What it provides |
|---|---------|------------------|
| 1 | **GMGN smart-money tracker** | Wallet ranking, win rate, live new-buys feed |
| 2 | **DexTools** | Real-time DEX liquidity, rug alerts, contract audits |
| 3 | **DexScreener** | Cross-chain pair data, volume, liquidity, live charts |
| 4 | **Birdeye.so** | Solana token analytics, holder distribution, wallet P&L |
| 5 | **Solscan** | On-chain transaction explorer (Solana) |
| 6 | **Etherscan / BaseScan / BscScan** | On-chain transaction explorers (EVM) |
| 7 | **CoinMarketCap Meme Tracker** | Market-cap / volume ranking by chain |
| 8 | **LunarCrush** | Social-sentiment analytics (X, Discord, Reddit) |
| 9 | **Mudrex Learn meme tracker** | Risk-scored curated meme list |
| 10 | **Kubera / Wealthica** | Unified portfolio aggregation (multi-broker + crypto) |
| 11 | **Sharesight / Portfolio Visualizer** | Performance attribution, Sharpe, max drawdown, alpha/beta |
| 12 | **TradeAlgo** | Options Greeks, margin tracking, AI risk alerts |
| 13 | **MenthorQ QUIN** | Volatility / gamma / dealer-positioning analytics |
| 14 | **Trade Ideas (Holly AI)** | Equity trade-setup signals (for AI-parity benchmark) |
| 15 | **TrendSpider** | Automated TA, chart pattern recognition |
| 16 | **Tickeron** | AI pattern recognition for equities + crypto |
| 17 | **PredictIndicators.ai** | Forward-looking indicator forecasts (MACD, RSI, BB, Stoch) with confidence gradient |

Integration pattern: each is a read-only HTTP client with aggressive
rate-limit / backoff / circuit-breaker wrappers; responses are
normalized into `signal` / `metric` / `social_score` events that feed
the knowledge store and the dashboard. Authority_lint rule C3 forbids
any of these clients from importing `execution.*` or `governance.*`.

---

## 4. Safety APIs

Hard-dependency for the 60-second pre-trade safety stack.

| # | Service | Role |
|---|---------|------|
| 1 | **Rugcheck.xyz** | Mint/freeze/update authority + LP lock + holder checks (Solana) |
| 2 | **Token Sniffer** | Scam / rug detection (multi-chain) |
| 3 | **Honeypot.is / GoPlus** | EVM honeypot simulation |
| 4 | **De.Fi / CertiK Skynet** | Smart-contract audit lookup |
| 5 | **DEX quote simulation** (Jupiter simulate / Raydium simulate / Uniswap callStatic) | Internal honeypot check via round-trip quote |

If any safety API is **unreachable**, the safety gate **fail-closes**
— all new memecoin entries are blocked until the service is back
(dead-man applies here too).

---

## 5. CEX Gateways

For major-cap assets, hedging, on/off-ramp, and tokenized-TradFi
exposure.

| # | Venue | Role |
|---|-------|------|
| 1 | **Binance** | Primary spot + perp liquidity (majors) |
| 2 | **Coinbase Advanced** | US-compliant spot + Base on-ramp |
| 3 | **Kraken** | Security-first CEX; Ink L2; institutional margin |
| 4 | **BingX** (ChainSpot + futures) | CeDeFi on-chain pivot via USDT; up to 50× leverage on majors; AI copy-trading ecosystem |
| 5 | **OKX** | Multi-chain wallet + DEX + CEX |
| 6 | **Bybit** | Perps + memecoin pre-launch listings |
| 7 | **KuCoin** | Long-tail altcoin liquidity |

---

## 6. Broker Gateways — TradFi

### 6.1 Forex (adapter priority by cost + regulation)

1. Exness
2. IC Markets
3. Pepperstone
4. FP Markets
5. Axi
6. IG
7. FxPro
8. ThinkMarkets
9. Tickmill
10. OANDA
11. RoboForex
12. XM
13. Capital.com
14. BlackBull Markets
15. Hantec Markets
16. FXTM
17. AvaTrade
18. HFM
19. HYCM
20. IQ Option
21. Libertex
22. Deriv
23. Quotex

### 6.2 Stock brokers

1. **Interactive Brokers (IBKR TWS)** — multi-asset, API-rich, primary
2. **ProRealTime + IBKR** — charting front-end driving IBKR execution
3. **TradingView + connected broker** — chart-driven order routing
4. **NinjaTrader** — futures + tick-level backtesting reference
5. **Alpaca** — API-first US equities (paper + live)

Each broker adapter is a thin, typed client wrapping the vendor SDK
with the same authority-lint + dead-man + ledger constraints as DEX
adapters.

---

## 7. Phased Build-Out

| Phase | What lands |
|-------|-----------|
| 2 | Adapter base class + Jupiter + Raydium + Pump.fun/PumpSwap + Binance + Coinbase + IBKR base |
| 2 | Analytics clients: GMGN / DexScreener / Birdeye / Rugcheck (read-only) |
| 4 | Memecoin risk advisor (neuromorphic) + Memecoin Mode config |
| 5 | Bot gateway wrappers: GMGN / BullX / Trojan / Axiom / BonkBot (operator-opt-in per bot) |
| 5 | Safety gate (60 s stack) + honeypot simulator + bundle detector + dev-wallet history |
| 5 | MEV guard (Jito bundle integration + private mempool on EVM) |
| 6 | `authority_lint` W1 + W2 rules; CEX gateway adapters; forex broker adapters batch 1 (Exness, IC, Pepperstone, FP, Axi) |
| 7 | Observability: Prometheus exporters per adapter; Grafana Trading board |
| PR #2 chain | Cockpit tabs: Memecoin, AI-signals, Portfolio-analytics, Broker-status |

Each adapter is its own small, test-covered, authority-lint-enforced
PR. Do not batch multiple adapters into a single PR.
