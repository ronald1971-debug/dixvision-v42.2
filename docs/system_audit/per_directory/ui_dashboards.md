# ui/, dashboard2026/, dash_meme/, dashboard_backend/

## ui/ — 24 files (FastAPI harness)

### Purpose

The single-process composer of every engine. `ui/server.py`
(1728 lines) builds the engine bus, registers plugins, mounts the
React static bundles at `/dash2/` and `/meme/`, and exposes every
`/api/*` endpoint.

### Wiring

* Imports every engine and the dashboard control-plane router.
* Sets `HARNESS_APPROVER_ENV_VAR=1` at import time (line 114) — the
  only place in the codebase allowed to do so (B33 lint).
* Lazy-imports `UniswapXAdapter` (PR #178) so the launcher boots
  without `eth-account` installed.
* Mounts the cognitive chat surface via
  `intelligence_engine.cognitive.chat`.

### Static-analysis result

* 24 files, 17 with findings — **all 17 are ruff-format drift only**.
* No orphan modules. No semantic findings.

### Risks / gaps

* `ui/server.py` is large (1728 lines) and concentrates a lot of
  routing + composition. Splitting it into smaller routers
  (`ui/routes/{health,intent,feeds,governance,operator,cognitive}.py`)
  is a refactor candidate but not a bug — current size is workable.

### Verdict

**HEALTHY.** Single composer, single chokepoint, clean B-lint.

---

## dashboard2026/ — 183 files (operator cockpit `/dash2`)

### Purpose

React + Vite SPA. The full operator cockpit: modes, autonomy, all
asset surfaces (FX/stocks/crypto/options/NFT/memecoin/macro),
governance widgets, charts, AI (counterfactual/NLQ/earnings-RAG/
smart-money), order entry depth, position/PnL, market context,
testing harness, command palette, theme.

### Static-analysis result

* 183 files, 0 backend findings (frontend; ruff/vulture do not
  scan TS).
* Deep-read sample (`src/main.tsx`, `src/routes/`,
  `src/widgets/governance/*.tsx`) — wiring is correct, every API
  call goes through `src/api/client.ts` + `src/api/feeds.ts` (the
  feeds.ts schema mismatch was fixed in PR #181).
* Theme tokens introduced by PR #180 (B.0).

### Risks / gaps

* The dashboard polish queue (B.1..B.4 — density toggle, command
  palette upgrade, lightweight-charts unification, status pills)
  is queued but not blocking.

### Verdict

**HEALTHY.** Production cockpit; polish queued.

---

## dash_meme/ — 43 files (DIX MEME `/meme/`)

### Purpose

Separate React + Vite SPA, DEXtools-styled. Pages: PairExplorer,
PoolExplorer, BigSwap, Multichart, Multiswap, HotPairs, WalletInfo,
TradePage, CopyTradingPage (queued), SniperPage (queued). Same
backend, separate launcher (`scripts/windows/start_dixvision_meme.bat`).

### Static-analysis result

* 43 files, 0 backend findings.
* PR #181 shipped DEXtools-faithful layout; 7 Devin Review findings
  fixed in-thread (PoolExplorer age sort, feeds schema, defensive
  copy, wallet endpoint, dedup, etc.).

### Risks / gaps

* Full execution surface (TradePage manual / semi-auto / full-auto +
  CopyTradingPage + SniperPage) is queued — must route through the
  existing `/api/intent` chokepoint, not a parallel surface.

### Verdict

**HEALTHY scaffold; execution-surface polish queued.**

---

## dashboard_backend/ — 8 files

### Purpose

Backend control-plane surface for the React UI:
`engine_status_grid`, `mode_control_bar`, `decision_trace`,
`memecoin_control_panel`, `strategy_lifecycle_panel`, `router`.

### Static-analysis result

* 8 files, 4 with findings — **all 4 are ruff-format drift only**.
* No orphan modules. No semantic findings.

### Verdict

**HEALTHY.** Thin projection layer; no business logic.
