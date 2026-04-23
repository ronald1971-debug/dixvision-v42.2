# DIX VISION v42.2 — Memecoin / Solana Trading Spec

**Status:** reference specification (no runtime code yet)
**Owners:** Indira (perception) · Governance (authority) · Execution (adapters)
**Implementation phase:** Phase 5 (execution + MEV guard) on top of Phase 2 (adapters)

This document is the single source of truth for how DIX will trade memecoins
and low-cap on-chain assets (Solana Pump.fun / Raydium / Jupiter ecosystem,
BNB Chain Four.meme, Ethereum/Base via Uniswap v4, etc.).

Unlike major-pair trading, memecoin trading has a radically different risk
surface: the majority of new launches are scams, rugs, honeypots, or
bundled insider dumps. The correct default stance is **pass 95 % of
opportunities**; tooling exists to identify the 5 % worth touching and to
cap the damage when we are wrong.

Nothing in this document gives any component decision authority.
All execution paths remain governed by the kernel (axioms N1 / C2 / E-series).
Memecoin trading runs under a dedicated **Memecoin Mode** with its own
hard caps, its own burner-wallet policy, and its own kill-switch.

---

## 0. Non-negotiables

1. **Burner wallet enforcement.** Memecoin paths never touch the main
   treasury wallet. A dedicated, topped-up burner is used. `authority_lint`
   rule `W1` must block any memecoin adapter from importing or resolving
   the main wallet path.
2. **Position cap is hard, not advisory.** Memecoin Mode has its own
   per-trade and per-day notional caps, enforced by the governance kernel
   before execution is even routed.
3. **No autonomous "YOLO" mode.** Every adapter, bot integration, and
   signal source is observe-and-propose only. Final decisions go through
   governance exactly like every other trading path.
4. **Every trade leaves a ledger row.** N4 applies: entry, exit, rug
   detection, safety-check pass/fail — all ledgered.
5. **Dead-man applies.** Memecoin sensor plugins (volume, liquidity,
   social-sentiment, rug-detect) each expose `check_self()`. If a sensor
   goes silent, Memecoin Mode fail-closes.

---

## 1. The 60-Second Pre-Trade Safety Stack

Before any memecoin buy order is even proposed to governance, the
candidate token must pass the following pipeline. Total wall-clock
budget: ~60 s. Any fail short-circuits the pipeline and rejects the
candidate.

| # | Check | Source(s) | Time | Pass condition |
|---|-------|-----------|------|----------------|
| 1 | **Mint authority** | Rugcheck API + on-chain | 10 s | Revoked |
| 2 | **Freeze authority** | Rugcheck API + on-chain | 5 s | Revoked |
| 3 | **Update authority** | Rugcheck API + on-chain | 5 s | Revoked |
| 4 | **Bundle detection** | GMGN / BullX API | 15 s | No ≥3-wallet buys in same block as creation; no fresh-funded whale in block 1 |
| 5 | **Dev wallet history** | GMGN / Solscan / Birdeye | 15 s | Deployer has ≥1 prior token that lived >24 h OR is a known legitimate builder |
| 6 | **Liquidity status** | DexScreener / DexTools / on-chain | 10 s | LP burned or locked; pool ≥ configured floor (default $10k for Memecoin-Cautious) |

Pass = every row green. Anything else = **auto-reject**, ledger a
`MEMECOIN_SAFETY_REJECT` event with the failed check(s) and move on.

### 1.1 Honeypot simulation

Before the first real buy on any new token, the adapter issues a
**simulated sell** via the DEX's quote endpoint (Jupiter quote / Raydium
simulate). If the simulated sell reverts or returns < 80 % of the buy
value at the same size, the token is flagged as a **honeypot** and
auto-rejected. This catches the "buyable-but-unsellable" class of scams
that contract-level checks miss.

### 1.2 Smart-money cross-check

Independent of the safety gate, the discovery layer surfaces how many
tracked smart-money wallets are currently holding or buying this token.
This is **positive signal only** — "smart money is buying" never
overrides a safety-gate failure. Smart-money metric:

```json
{
  "smart_money_holders": 7,
  "smart_money_net_buy_5m_sol": 23.4,
  "smart_money_top_wallet_winrate": 0.71
}
```

Governance may use these fields as inputs to position sizing, but
never as a bypass of safety-gate rejection.

---

## 2. Rug / Scam Pattern Catalog

Each pattern ships with a detection rule and a pre-committed response.

| Pattern | How it works | Detector | Response |
|--------|--------------|----------|----------|
| **Instant rug** | Dev dumps pre-loaded supply in block 1 via "different" wallets | Bundle detector (1.5): ≥3 wallets fresh-funded by same source buying in same block as creation | Reject before entry |
| **Slow rug** | Dev trickles allocation out over hours, draining liquidity | Post-entry monitor: dev wallet net-sell > X % of position / hour | Force exit; mark deployer blacklist |
| **Honeypot** | Contract permits buys, blocks sells (or taxes to 90 %+) | Honeypot simulation (1.1) | Reject before entry |
| **Liquidity pull** | LP never locked; dev yanks pool | LP-lock check (6) + on-chain LP-token owner monitor | Reject if not locked; force exit if ownership moves |
| **Copycat rug** | Fake clone of a pumping token with near-identical name | Contract-address cross-check against a canonical registry + string-similarity alarm | Reject; require operator approval to whitelist |
| **Mint abuse** | Dev mints new supply post-launch, crashes price | Mint-authority check (1) + post-launch supply-delta monitor | Reject pre-entry; force exit if supply grows |
| **Tax trap** | Hidden buy/sell tax pumps 50 %+ post-entry | Fee-delta monitor on each swap | Force exit on anomalous fee |

---

## 3. Sniping Strategy (Two-Phase Pattern)

Direct launch-sniping is optional (gated behind `mode=memecoin_aggressive`
+ explicit operator toggle). Default memecoin mode is **not** a sniper.

When enabled, sniping uses a two-phase pattern:

**Phase 1 — Insta-buy (block 0 or 1):**
- Position: ≤ 0.2 SOL (or configured `memecoin_phase1_size`).
- Slippage: 10–15 %.
- Safety gate: checks 1–6 run **before** the insta-buy; if any fails the
  insta-buy is cancelled.
- MEV: Jito bundle required. Private mempool preferred.

**Phase 2 — Confirmation add-on (30–60 s later):**
- Only if Phase 1 succeeded AND the token passes a re-run of the safety
  stack AND the post-launch behavior looks clean (no dev-wallet sells,
  organic trading volume, LP still in place).
- Position: configured per mode (default 2–5× Phase 1 size).

**Never:** all-in on Phase 1. Never: skip Phase 2 confirmation.

---

## 4. MEV / Slippage Defaults

| Setting | Default | Range | Notes |
|--------|---------|-------|-------|
| Slippage (new launches, Solana) | 12 % | 8–15 % | Failed tx cost < 1 pip of missed entry |
| Slippage (new launches, EVM) | 8 % | 5–12 % | EVM rotations are slower; lower works |
| Slippage (established memes, e.g. DOGE/SHIB/PEPE) | 1–3 % | 0.5–5 % | Treat as large-cap |
| Absolute slippage ceiling | 25 % | — | Never exceed; above this we are sandwich bait |
| Priority fee | adaptive | 0.0001–0.01 SOL | Driven by network congestion estimator |
| Jito bundle | ON | — | Mandatory for Solana sniping |
| Private mempool | ON for size > 2 SOL | — | Reduces sandwich surface on EVM chains |
| Max position per memecoin | 2 % of Memecoin bankroll | 0.5–5 % | Governance hard cap |

**Rule:** anything that would push slippage over 25 % or priority fee
over the adaptive cap is auto-rejected with a `MEV_RISK_REJECT` ledger
entry.

---

## 5. Position Sizing (Memecoin Bankroll Only)

Position size is a function of the Memecoin-dedicated bankroll, not the
global treasury. Memecoin bankroll is itself capped (default: 5 % of
total treasury; operator-adjustable; ledger-audited change).

| Memecoin bankroll | Max per snipe | Max concurrent positions |
|-------------------|---------------|--------------------------|
| < 5 SOL | 0.05–0.10 SOL | 2–3 |
| 5–20 SOL | 0.10–0.50 SOL | 3–5 |
| 20–100 SOL | 0.50–2.00 SOL | 5–10 |
| 100+ SOL | 1.00–5.00 SOL | 5–10 |

These are the **hard caps**. Governance may reduce further based on
neuromorphic risk signals or current drawdown velocity.

---

## 6. Win-Rate Reality (Strategy Math Baseline)

All memecoin strategies must be evaluated against a realistic baseline:

- **Win rate:** 15 – 30 %.
- **Winners:** +3× to +10× (higher possible but not assumed in EV math).
- **Losers:** −50 % to −100 %.
- **EV target per position:** > 0 after fees and expected loss rate.

A strategy that cannot show positive EV under these numbers is rejected
in backtest-review and never promoted to live.

---

## 7. Exit Strategy (Staged Take-Profit)

Default exit schedule (configurable per strategy):

| Trigger | Action |
|--------|--------|
| +2× | Sell 25 % |
| +3× | Sell 25 % |
| +5× | Sell 25 % (bankroll is now fully recovered) |
| Trailing stop on remaining 25 % | Let the runner go |
| Stop-loss | −40 % to −50 % (per strategy) |
| Smart-money follower net-sells ≥ X % of their holding | Sell 50 % immediately |
| Rug-detect fires post-entry (slow rug / mint abuse / tax trap) | Force exit, full position |

No "hold for moon." The tail runner is a separate decision governed by
trailing-stop math, not wishful thinking.

---

## 8. Modes

Memecoin trading ships as **three discrete modes**, each with its own
hard caps:

1. **Memecoin-Off** (default) — adapters loaded but trading disabled.
2. **Memecoin-Cautious** — only tokens with ≥ $50 k liquidity, ≥ 24 h
   age, LP locked ≥ 30 days, ≥ 2 smart-money holders. No Phase 1
   sniping. Position caps: 0.5 % of Memecoin bankroll per trade.
3. **Memecoin-Aggressive** — full safety stack still applies, Phase 1
   sniping enabled. Position caps: 2 % of Memecoin bankroll per trade.
   Requires operator toggle **and** clean neuromorphic risk signal
   before activation.

Mode transitions are operator-gated, ledger-audited, and subject to the
same promote-chain as any other governance change.

---

## 9. Burner-Wallet Policy (authority_lint W1)

- Dedicated key, dedicated seed, dedicated RPC endpoint set.
- Top-up comes from the main treasury via a signed, operator-approved
  transfer (ledger-audited, phased rate-limited).
- The burner never holds more than its own bankroll cap.
- The burner never has import paths to `execution.engine`,
  `governance.*`, or `mind.fast_execute`.
- Memecoin adapters have no import paths to the main wallet resolver.
- `authority_lint` rule **W1** scans the source tree and rejects any
  import edge that violates the above.

---

## 10. Glossary (shared vocabulary for all memecoin code & docs)

- **Slippage** — difference between quoted and executed price. Memecoin
  setting is deliberately wide (8–15 %) because transaction failure is
  more expensive than a moderately worse fill.
- **MEV** (Maximal Extractable Value) — profit extractable by reordering
  / including / censoring transactions. We defend with Jito bundles
  and private mempools.
- **Sandwich attack** — an MEV bot front-runs our buy, back-runs our
  sell, pocketing the slippage.
- **Frontrun** — MEV bot sees our pending tx, submits ahead with higher
  priority fee.
- **Backrun** — MEV bot immediately sells into our buy.
- **Honeypot** — contract allows buys but blocks/taxes sells.
- **Bundle** (launch bundle) — dev buys ≥ 15 % of supply across many
  fresh wallets in the launch block so it looks organic. Setup for
  an instant rug.
- **Rug pull** — dev drains liquidity / sells entire allocation, leaving
  holders with no exit.
- **LP lock** — liquidity-pool tokens are time-locked or burned so the
  dev cannot pull liquidity.
- **Bonding curve** — Pump.fun-style linear pricing mechanism. A token
  "graduates" to Raydium / PumpSwap at ~$30k–$35k market cap.
- **Graduate** — a Pump.fun token that crossed its bonding-curve
  threshold and has migrated to a real AMM pool.
- **Jito bundle** — Solana MEV-protection mechanism; transactions in a
  bundle are included atomically.
- **Dead-man (N5)** — every sensor must stamp proof-of-life at each
  evaluation; stale stamp = fail-closed.
- **Safety gate** — the 60-second pre-trade pipeline in §1.
- **Memecoin bankroll** — capital slice dedicated to memecoin trading,
  separated from the main treasury.

---

## 11. Integration Points (what lands in which phase)

| Component | Location | Phase |
|----------|----------|-------|
| Memecoin adapter base class | `execution/adapters/memecoin_base.py` | 2 |
| Jupiter / Raydium / PumpSwap adapters | `execution/adapters/{jupiter,raydium,pumpswap}.py` | 2 |
| Rugcheck / honeypot simulator client | `execution/safety/rugcheck.py` | 5 |
| Bundle detector | `execution/safety/bundle_detector.py` | 5 |
| Dev-wallet history client | `execution/safety/dev_wallet_history.py` | 5 |
| Memecoin signal plugin (volume / liquidity / social) | `mind/plugins/memecoin_signal.py` | 2 |
| Memecoin risk advisor (neuromorphic) | `governance/signals/memecoin_risk.py` | 4 |
| Memecoin Mode config + hard caps | `governance/modes/memecoin.py` | 4 |
| Burner-wallet policy + `authority_lint` W1 | `tools/authority_lint.py` + `core/wallets/` | 6 |
| Memecoin dashboard tab | `cockpit/app.py` + `cockpit/templates/memecoin.html` | PR #2 dashboard chain |

Nothing in this list may be built until Phase 0 has merged and Phase 1
(ledger + event_store) is solid.
