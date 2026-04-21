# DIX VISION v42.2 — Cockpit Dashboard Specification (Beyond Manifest)

**Goal:** an operator console that meets manifest §10/§11 *and* exceeds it — matching patterns from Google SRE, Hummingbot, Freqtrade, LSEG's chaos-engineering practice, and 2025–2026 professional trading dashboards.

---

## 1. What the manifest requires (baseline)

Per §10 + §11:

- mode / health / trading_allowed / active_hazards
- risk cache snapshot
- ledger chain verify
- prometheus metrics exposure

That's it — 4 categories. Everything else below is an additive enhancement.

---

## 2. What pro trading dashboards have in 2025-2026 (research synthesis)

**Sources:** Hummingbot Dashboard, Freqtrade UI, Grafana SRE templates, AWS chaos-engineering for LSEG, ZeonEdge alerting guide, Xoibit deterministic replay, Ember Trading Hub.

### 2.1 Four Golden Signals (Google SRE)
- **Latency** — fast_execute p50/p95/p99; hazard-detect latency; ledger-write latency.
- **Traffic** — trades/sec, ticks/sec, hazard events/sec, ledger events/sec.
- **Errors** — rejected-order rate, adapter error rate, hazard CRITICAL rate.
- **Saturation** — hazard-queue depth, ledger-queue depth, fast-risk-cache staleness.

### 2.2 RED + USE
- **Rate / Errors / Duration** per service.
- **Utilization / Saturation / Errors** per resource (SQLite, websockets, process mem/cpu).

### 2.3 SLO burn-rate alerts
Multi-window (1h/6h/24h) error-budget burn. If you burn >10% in 1h → page.

### 2.4 Strategy / form-tiles (Hummingbot pattern)
- Per **trading form** (SPOT / MARGIN / PERP / FUTURES / OPTIONS / DEX_SWAP / DEX_LP): tile with signal count, fill rate, realized PnL, open exposure.
- Per **adapter**: connection health, last-tick age, 1m throughput, rejects.

### 2.5 Replay + chaos (LSEG pattern)
- **Deterministic replay**: "Replay from sequence N" — rebuild state from ledger tail, show projector hash vs golden.
- **Chaos toggle**: "Inject adapter timeout" / "Inject feed silence" / "Inject ledger corruption" — disabled in prod, available in shadow.

### 2.6 Governance panel
- Current EXECUTION_CONSTRAINT_SET (human-readable).
- Last policy evaluation (what passed / what vetoed).
- Mode transitions timeline (NORMAL → SAFE → DEGRADED → HALTED) with reasons.

### 2.7 Security & auth panel
- Last-auth events, failed tokens, domain-violation attempts (`core.authority.AuthorityViolation`).
- Kill-switch state (ARMED / DISARMED / FIRED).

### 2.8 Forensic / audit panel
- Ledger hash chain state (OK / BROKEN + row).
- Event stream filter (MARKET / SYSTEM / GOVERNANCE / HAZARD / SECURITY).
- "Export last N events to JSONL" button.

### 2.9 Live charts (TradingView lightweight-charts CDN)
- Candle chart per active asset (driven by `mind/sources/market_streams/websocket_client`).
- Per-asset trade markers (buy/sell) overlaid on the candle.
- Portfolio equity curve.

---

## 3. Gap list — what we were going to miss

Ranked by impact:

| # | Gap | Severity | In baseline cockpit? |
|---|---|---|---|
| 1 | p99 latency on fast-path, not just average | HIGH | ❌ → **adding** |
| 2 | Error-budget burn-rate (SLO) | HIGH | ❌ → **adding** |
| 3 | Ledger chain status + last-event seq + chain break row | HIGH | partial → **upgrading** |
| 4 | Per-trading-form tiles (SPOT / MARGIN / PERP / FUTURES / OPTIONS / DEX_SWAP / DEX_LP) | HIGH | ❌ → **adding** |
| 5 | Per-adapter health + last-tick age | HIGH | ❌ → **adding** |
| 6 | Authority-violation counter + recent events | HIGH | ❌ → **adding** |
| 7 | Kill-switch state indicator (top-bar) | HIGH | ❌ → **adding** |
| 8 | Mode-transition timeline (NORMAL / SAFE / DEGRADED / HALTED) | MED | ❌ → **adding** |
| 9 | Open-orders + fills feed | MED | ❌ → **adding** |
| 10 | Portfolio / positions / exposure_usd / pnl_usd | MED | ❌ → **adding** |
| 11 | Prometheus metrics endpoint (`/metrics`) text-format | MED | ❌ → **adding** |
| 12 | Candlestick chart w/ trade markers | MED | ❌ → **adding** (CDN lightweight-charts, optional tab) |
| 13 | Replay-from-N panel (read-only preview) | LOW | ❌ → **adding** |
| 14 | Chaos / fault-injection panel (disabled in prod) | LOW | ❌ → **adding** (gated by `DIX_ENV=shadow`) |
| 15 | Hazard-queue + ledger-queue saturation gauges | LOW | ❌ → **adding** |
| 16 | Event-stream filter / export JSONL | LOW | ❌ → **adding** |
| 17 | Mobile-responsive layout | NIT | ❌ → **adding** |
| 18 | Keyboard shortcuts (`G`-goto panel, `R`-refresh now, `K`-kill-switch arm) | NIT | ❌ → **adding** |
| 19 | Dark / light theme toggle | NIT | ❌ → **adding** |
| 20 | WebSocket push instead of polling | NIT | — polling is fine at this scale; WS later |

---

## 4. Finalized dashboard layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DIX VISION v42.2      [MODE: NORMAL]  [TRADING: ✓]  [KS: ARMED]  [DOMAIN: INDIRA/DYON/GOV]  [TIME] │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─ Four Golden Signals ─────────────┐  ┌─ SLO Burn Rate ───────────────┐   │
│  │ fast p50/p95/p99                  │  │ 1h / 6h / 24h error budget    │   │
│  │ ticks/s, trades/s, hazards/s      │  │                               │   │
│  │ reject rate, adapter err rate     │  │                               │   │
│  │ queue depths (hazard / ledger)    │  │                               │   │
│  └───────────────────────────────────┘  └───────────────────────────────┘   │
│  ┌─ Trading Forms (per form tile) ──────────────────────────────────────┐   │
│  │ SPOT   MARGIN   PERP   FUTURES   OPTIONS   DEX_SWAP   DEX_LP         │   │
│  │ each:  signals | fills | open-exposure | realized PnL | adapters     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌─ Adapters ────────────────────────┐  ┌─ Positions / PnL ─────────────┐   │
│  │ binance (CEX) SPOT/MARGIN/PERP/..  │  │ asset | size | entry | mark  │   │
│  │ coinbase(CEX) SPOT/PERP/FUTURES    │  │ exposure_usd | pnl_usd        │   │
│  │ kraken  (CEX) SPOT/MARGIN/PERP/..  │  │                               │   │
│  │ uniswap (DEX) DEX_SWAP/DEX_LP      │  │                               │   │
│  │ raydium (DEX) DEX_SWAP/DEX_LP      │  │                               │   │
│  └───────────────────────────────────┘  └───────────────────────────────┘   │
│  ┌─ Hazard Feed ─────────────────────┐  ┌─ Mode Timeline ──────────────┐    │
│  │ severity × type × source × ts     │  │ NORMAL → SAFE → DEGRADED … │    │
│  │ color-coded, auto-scroll          │  │ with reason + ledger ref    │    │
│  └───────────────────────────────────┘  └──────────────────────────────┘    │
│  ┌─ Governance Constraints ──────────┐  ┌─ Security / Authority ───────┐    │
│  │ risk_cache.get() read-only view   │  │ AuthorityViolation count     │    │
│  │ last update_ts + last_source      │  │ recent kill-switch events    │    │
│  │ mode → constraint map             │  │ auth fails, origin IPs       │    │
│  └───────────────────────────────────┘  └──────────────────────────────┘    │
│  ┌─ Ledger Tail ─────────────────────────────────────────────────────────┐  │
│  │ seq | ts | stream | sub_type | source | hash_prefix | payload-preview │  │
│  │ filter bar: [MARKET][SYSTEM][GOVERNANCE][HAZARD][SECURITY]           │  │
│  │ [verify chain] [export last 1000 JSONL] [replay from seq N →]       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌─ Open Orders ───────────────────┐  ┌─ Fills ─────────────────────────┐  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│  ┌─ Candle chart (optional tab) ────────────────────────────────────────┐  │
│  │ asset picker · TradingView lightweight-charts · buy/sell markers     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
Footer: version · git-sha · uptime · foundation-hash ok/broken · help · logout
```

Keyboard:
- `g s` → system status, `g f` → forms, `g a` → adapters, `g h` → hazards, `g l` → ledger, `g c` → chart
- `R` → refresh now (bypasses 2s poll)
- `K` → arm/disarm kill-switch (confirmation dialog)
- `/` → global filter

Auth: bearer-token header *or* `?token=…` query (one-click from launcher).
Bind: `127.0.0.1` by default; override via `DIX_COCKPIT_BIND`.

Tech: single-page vanilla JS + Tailwind CDN + TradingView lightweight-charts CDN. No build step.

Endpoints (FastAPI):
```
GET  /                           → index.html (SPA)
GET  /static/*                   → css/js/favicon
GET  /health                     → {status}
GET  /api/status                 → system state + mode + trading + kill-switch
GET  /api/signals                → four golden + SLO burn
GET  /api/forms                  → per-trading-form rollup
GET  /api/adapters               → per-adapter meta + connection state
GET  /api/hazards?since=seq      → recent hazards (polling)
GET  /api/mode/timeline          → mode transitions
GET  /api/governance/constraints → risk_cache snapshot
GET  /api/security/events        → authority violations + kill switch events
GET  /api/ledger/tail?stream=... → last 100 events per stream
GET  /api/ledger/verify          → hash chain ok + break row
GET  /api/ledger/export?n=1000   → JSONL download
POST /api/ledger/replay          → replay preview (read-only; rebuilds projector hash in memory)
GET  /api/positions              → portfolio_manager snapshot
GET  /api/orders/open            → order_manager open orders
GET  /api/fills                  → fill_tracker recent fills
GET  /api/metrics                → Prometheus exposition-format
POST /api/kill-switch            → {arm|disarm|trigger} (CONTROL-domain only)
POST /api/chaos/inject           → SHADOW env only
```

---

## 5. What still would be out of scope

- Live order entry from the dashboard (manifest §5: execution is Indira's job; cockpit stays read-only + control-plane).
- Strategy backtest runner — separate tool; belongs in `mind/shadow_executor` CLI, not cockpit.
- Blockchain explorer — linked out to Etherscan/Solscan, not embedded.

---

## 6. Delivery plan

1. Implement endpoints (FastAPI + auth middleware + CSP header).
2. Ship single-file `cockpit/static/index.html` (Tailwind CDN, lightweight-charts CDN, vanilla JS).
3. Mount `cockpit.app` via `uvicorn cockpit:app --host 127.0.0.1 --port 8765`.
4. Launcher script opens `http://127.0.0.1:8765/?token=$DIX_COCKPIT_TOKEN` in default browser.
5. Enter test mode → record screencast of full one-click flow + each panel populated.
6. Ship in the final ZIP.

---

*Spec generated 2026-04-21. Sources: Google SRE 4 Golden Signals, Hummingbot Dashboard, Freqtrade UI, Grafana/Prometheus best-practices (2025-2026 reviews), AWS chaos-engineering for LSEG, Xoibit deterministic replay, ZeonEdge alerting guide.*
