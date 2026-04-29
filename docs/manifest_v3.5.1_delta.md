# DIX VISION v42.2 — MANIFEST v3.5.1 DELTA

> Additive delta over v3.5 (`docs/manifest_v3.5_delta.md`, PR #56).
> Promotes SCVS rules **SCVS-03 / SCVS-05 / SCVS-06** from "named in
> Phase 1 §1.4" to "implemented + ledgered" by introducing the runtime
> source liveness manager.
>
> v3.5.1 is **additive only**. Every new node sits inside the
> `system_engine.scvs` boundary or extends an existing
> `SystemEventKind` / `HazardEvent` enum. No engine boundary is
> redrawn; no existing invariant is renumbered.

---

## 0. WHY v3.5.1 EXISTS

v3.5 shipped the **build-time** half of SCVS (registry + bidirectional
closure lint). The **runtime** half — "is the source actually emitting
data right now? if a critical source goes silent, who escalates?" —
was named (SCVS-03 / 05 / 06) but not yet wired.

| # | Concern | Status in v3.5.1 |
|---|---------|------------------|
| L1.a | No way to detect a registered source that has gone silent | **Closed** by `SourceManager` (this delta) |
| L1.b | No way to escalate a silent **critical** source to governance | **Closed** by `HAZ-13` emission on critical STALE transitions |
| L1.c | Liveness logic embedded across plugins instead of in one auditable seam | **Closed** by centralising the FSM in `system_engine.scvs.source_manager` |
| L1.d | Liveness thresholds undefined per category | **Closed** by `_DEFAULT_LIVENESS_MS_BY_CATEGORY` in the registry loader |

L2–L4 (CRL / authority matrix / constraint compiler) remain explicitly
deferred per v3.5 §0.

---

## 1. THE NEW INVARIANT + RULES + ARTEFACTS

### 1.1 INV-58 — Source Liveness FSM

**Statement.** Every `enabled: true` row in
`registry/data_source_registry.yaml` is in exactly one of three
runtime statuses at any caller-supplied `now_ns`:

| Status | Pre-condition |
|--------|---------------|
| `UNKNOWN` | No heartbeat ever recorded |
| `LIVE`    | A heartbeat exists and `now_ns - last_heartbeat_ns ≤ liveness_threshold_ns` |
| `STALE`   | A heartbeat exists and `now_ns - last_heartbeat_ns > liveness_threshold_ns` |

A `liveness_threshold_ms` of `0` disables the staleness check entirely
(used for `synthetic` replay sources).

**Determinism contract.** `SourceManager` owns no clock and no PRNG.
Every observation takes a caller-supplied `now_ns`; replay against
the same heartbeat trace + `now_ns` sequence must produce the same
status transitions byte-for-byte (INV-15 extension).

### 1.2 SAFE-57 — Critical-source fail-closed

**Statement.** When a source declared `critical: true` transitions to
`STALE`, `SourceManager` emits a `HAZ-13` `HazardEvent` with severity
`HIGH` alongside the `SOURCE_STALE` `SystemEvent`. Recovery emits
**no** hazard — the failing edge is the only escalation seam.

`HAZ-13` is the canonical code for `SCVS-06`. Governance owns the
downstream policy ("halt trading", "demote mode", etc.); the SCVS
layer only emits.

### 1.3 SAFE-58 — Liveness defaults are category-derived

Sources without an explicit `liveness_threshold_ms` inherit the
category default declared in `_DEFAULT_LIVENESS_MS_BY_CATEGORY`:

| Category     | Default | Rationale |
|--------------|---------|-----------|
| `market`     | 5 s     | Tick-level feeds; staleness here is a P&L hazard |
| `onchain`    | 60 s    | Block cadence + indexer batching |
| `news`       | 5 min   | Article publishing cadence |
| `social`     | 5 min   | Polling APIs (X / Reddit) |
| `macro`      | 24 h    | Government release cadence (CPI / NFP / etc.) |
| `regulatory` | 24 h    | EDGAR + filing pulls |
| `dev`        | 1 h     | GitHub issue / PR polling |
| `alt`        | 5 min   | Polymarket / app-store probes |
| `ai`         | 60 s    | Provider keep-alive ping |
| `synthetic`  | 0 (off) | Replay buffers don't heartbeat |

### 1.4 New event-kind values (additive)

Three new `SystemEventKind` values, mirrored in
`contracts/events.proto`:

| Python | Proto | Emitted on |
|--------|-------|-----------|
| `SOURCE_HEARTBEAT` | `SOURCE_HEARTBEAT = 12` | `UNKNOWN → LIVE` |
| `SOURCE_STALE`     | `SOURCE_STALE = 13`     | `LIVE → STALE`   |
| `SOURCE_RECOVERED` | `SOURCE_RECOVERED = 14` | `STALE → LIVE`   |

### 1.5 New hazard code (additive)

| Code     | Severity | Emitted on |
|----------|----------|-----------|
| `HAZ-13` | `HIGH`   | A `critical: true` source transitions `LIVE → STALE` |

---

## 2. NEW ARTEFACTS

* `system_engine/scvs/source_manager.py` — `SourceManager`, `SourceStatus`,
  `SourceLivenessReport`. Pure FSM driven by caller-supplied `now_ns`.
* `tests/test_scvs_phase2.py` — 19 tests covering classification,
  transition emission, idempotency, critical-source escalation, and
  INV-15 replay determinism.

## 3. SCOPE OF THIS DELTA

* **In:** runtime FSM, three new `SystemEventKind` values, `HAZ-13`,
  category-derived default thresholds, replay-determinism tests.
* **Out:** `consumes.yaml` runtime data-flow tracking (still build-time
  only; runtime data tracking lands with Phase 3 SCVS-04 schema
  enforcement); AI provider validation (SCVS-07, Phase 3); duplicate
  source detection (SCVS-08, Phase 3); stale data rejection inside
  the engines (SCVS-09, Phase 3); silent-fallback audit (SCVS-10,
  Phase 3).
* **Unchanged:** every existing engine boundary, every existing
  invariant (INV-01..INV-57), every existing lint rule, every
  existing event constructor. SCVS Phase 1 lint (SCVS-01 / SCVS-02)
  still runs first in CI; Phase 2 events are emitted at runtime only.
