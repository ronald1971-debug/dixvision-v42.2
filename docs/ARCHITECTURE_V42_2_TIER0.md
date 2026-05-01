# DIX VISION v42.2 — Tier-0 Architecture of Record

**Status:** binding architecture-of-record (spec only; implementation
ships one step per PR).
**Supersedes:** nothing. Extends: the Phase 0 manifest and the
neuromorphic triad spec.
**Governing axioms:** S1–S10 (safety), H1–H10 (hazard), N1–N8
(neuromorphic), and the new **T0-1 … T0-15** codified below.

This document is the single source of truth for the v42.2 Tier-0
architecture directive. It does **not** ship code. Each of Steps 0–15
below is tracked as its own implementation PR in strict phase order,
each with its own tests, its own authority-lint additions, and its own
operator-TOTP acceptance gate.

---

## 0. Architectural invariants (absolute, non-negotiable)

These restate the frozen contract from the manifest plus the new
operator directive. Any PR that contradicts any of these is rejected
at lint time.

1. **Indira owns the MARKET domain only.** Trade decisions, strategy
   selection, perception.
2. **Dyon owns the SYSTEM domain only.** Infrastructure, telemetry,
   dead-man, hazard emission.
3. **Governance is the sole authority for ALL transitions.** Modes,
   strategy lifecycle, kill-switch, policy evaluation.
4. **`SYSTEM_HAZARD` is the ONLY cross-domain signal channel.** Dyon
   is the sole producer; Governance is the sole consumer.
5. **The ledger is append-only, hash-chained, and authoritative.** No
   module may maintain mutable authoritative state outside this system.
6. **`FastRiskCache` is the ONLY runtime interface from governance →
   Indira.** No other runtime coupling is permitted.
7. **No direct calls from `mind.*` → `governance.kernel`.**
   `authority_lint` rule C1 enforces. Advisory signals (neuromorphic
   triad, fusion engine) travel through the event bus.
8. **No raw dicts across domain boundaries.** Typed schemas only
   (`core/contracts/*.py` Protocols + `HazardEvent`, `ExecutionEvent`,
   `SpikeSignalEvent`, `SystemAnomalyEvent`, `RiskSignalEvent`,
   `FusionResult`, `SnapshotFrame`).

**Core principle (new):** the system is **deterministic and
event-sourced**. All state must be:

- reconstructible from the ledger (N4 extended to T0-0),
- reproducible deterministically (T0-14),
- snapshot-accelerated (T0-0).

---

## 1. Step → Phase → PR mapping

Steps are **not** implemented in numeric order; they are implemented
in the order their dependencies permit. One step = one PR. No mega-PRs.

| Step | Short name | Phase | Notes |
|------|-----------|-------|-------|
| T0-0  | State reconstructor + snapshots | Phase 1 | Unblocks everything else |
| T0-1  | FastRiskCache consistency model | Phase 1 | Needs T0-4 |
| T0-4  | TimeAuthority | Phase 1 | No dependencies |
| T0-5  | Ledger storage tiers | Phase 1 | Needs T0-0 |
| T0-11 | Knowledge-store hardening | Phase 1 | No dependencies |
| T0-12 | Config schema (pydantic) | Phase 1 | No dependencies |
| T0-14 | Deterministic replay test | Phase 1 | Needs T0-0, T0-4, T0-5 |
| T0-7  | Strategy lifecycle FSM | Phase 2 | Needs T0-0 |
| T0-8  | Adapter circuit breakers | Phase 2 | Needs T0-4 |
| T0-3  | Fusion engine | Phase 2 | Needs T0-4 |
| T0-2  | Load controller + backpressure | Phase 3 | Needs T0-3, T0-4 |
| T0-6  | Governance policy split | Phase 4 | Needs T0-1 |
| T0-9  | Global kill switch | Phase 4 | Needs T0-8 |
| T0-10 | Metrics layer | Phase 7 | Needs T0-4, T0-5 |
| T0-13 | Chaos engine | Phase 7 | Needs T0-8, T0-9, T0-14 |
| T0-15 | Backward-compat guarantee | every PR | Not a standalone PR; it is a CI gate applied to every PR in the chain |

---

## 2. Step 0 — State Reconstructor + Snapshots  **(T0-0)**

**Files land:** `system/state_reconstructor.py`, `system/snapshots.py`,
`tests/test_state_reconstructor.py`, `tests/test_snapshots.py`.

**Contract:**

```python
class StateReconstructor(Protocol):
    def rebuild(self, at_timestamp_ns: int) -> SystemState: ...
    def rebuild_latest(self) -> SystemState: ...
```

- Rebuilds `SystemState` from the ledger + latest snapshot delta.
- Snapshots are taken every **N events** OR **T seconds** (config-driven;
  defaults: N=10_000, T=300 s) on a background thread, never inline.
- Snapshot payload: positions · risk state · active strategies · mode
  · knowledge-store index (if size-managed, see T0-11) · adapter
  connection state.
- Snapshot storage is **separate** from the ledger file (`data/snapshots/`).
- Boot sequence: load latest snapshot → replay ledger delta since
  snapshot → system ready. No module may hold mutable authoritative
  state outside the ledger / snapshot store.

**Acceptance:**

- `rebuild_latest()` on a live system produces a state identical to
  in-memory state (hash comparison of all tracked fields).
- Killing the process mid-run and re-booting produces a state
  bit-identical to the pre-kill state (deterministic replay).
- `tests/test_replay_determinism.py` passes (T0-14).

---

## 3. Step 1 — FastRiskCache consistency model  **(T0-1)**

**Files touched:** `system/fast_risk_cache.py` (or existing
`risk/fast_cache.py`), `core/contracts/risk.py` (existing `IRiskCache`
protocol extended).

**Structure:**

```python
@dataclass(frozen=True)
class RiskSnapshot:
    version: int           # monotonic
    updated_at_ns: int     # time_source.now_ns()
    state: RiskState       # frozen
```

**Rules:**

- **Single writer: governance only.** Lint rule: no writer import path
  from `mind/*` or `execution/*`.
- **Lock-free reads** via atomic snapshot swap (e.g. `threading.local`
  + `atomics` package, or immutable-dataclass + pointer swap).
- **Monotonic version required.** Any out-of-order observed version =
  bug; assert + hazard-emit.

**Enforcement (Indira side):**

```python
snap = risk_cache.snapshot()
if time_source.now_ns() - snap.updated_at_ns > STALENESS_THRESHOLD_NS:
    reject_trade(reason="STALE_RISK_CACHE")
```

`STALENESS_THRESHOLD_NS` is mode-dependent (default 500 ms for
USER_CONTROLLED, 250 ms for SEMI_AUTO, 100 ms for FULL_AUTO).

**Acceptance:**

- Unit tests prove monotonic version + atomic swap.
- Integration test: freezing governance for > threshold auto-rejects
  all trades (no exceptions).

---

## 4. Step 2 — Load Controller + Backpressure  **(T0-2)**

**Files land:** `system/load_controller.py`,
`tests/test_load_controller.py`.

**Responsibilities:**

- Control ingestion rate of raw market ticks into the plugin bus.
- Enforce max plugin-thread concurrency.
- Prevent CPU saturation (monitor via `os.getloadavg` + per-loop
  timing from `TimeAuthority`).

**Policies:**

- `max_plugin_threads` (config).
- `queue_size_limit` per plugin (config).
- Overflow strategy (config): `DROP_OLDEST` · `COALESCE` (merge
  consecutive ticks on same asset).
- Emits `LOAD_SHED_EVENT` every time shedding happens (ledger-audited).

**Integration:**

`market_graph` gains a `load_shedding_node` placed **before**
`plugin_bus_node`. The plugin bus does not observe the raw feed; it
observes the post-shed feed.

**Acceptance:**

- Synthetic high-rate feed test: p99 plugin-queue dwell < 2× mean
  tick interval; zero OOMs.

---

## 5. Step 3 — Fusion Engine  **(T0-3)**

**Files land:** `mind/fusion_engine.py`, `core/contracts/fusion.py`,
`tests/test_fusion_engine.py`.

**Rules:**

- All plugin outputs normalized to **`[-1, 1]`** before fusion.
- Timestamps aligned to the tick boundary (via `TimeAuthority`).
- Aggregation is **order-independent** and **deterministic** (sorted
  by plugin-id; associative combine; no floating-point dependence on
  iteration order — use `math.fsum` or equivalent).

**Output:**

```python
@dataclass(frozen=True)
class FusionResult:
    score: float                  # in [-1, 1]
    confidence: float             # in [0, 1]
    breakdown: dict[str, float]   # per-plugin contribution
    ts_ns: int
```

**Integration:**

`PluginBus` MUST delegate fusion to `FusionEngine`. Strategies consume
`FusionResult`, not individual plugin outputs, once T0-3 lands.

**Acceptance:**

- Re-running fusion on the same inputs in randomized plugin-return
  order produces bit-identical `FusionResult`.

---

## 6. Step 4 — Time Authority  **(T0-4)**

**Files land:** `system/time_source.py`, `tests/test_time_source.py`.

**API:**

```python
def now_ns() -> int: ...         # time.perf_counter_ns()
def monotonic_ns() -> int: ...   # guaranteed non-decreasing
def wall_ns() -> int: ...        # time.time_ns() — ledger-only, never hot path
```

**Rules:**

- Hot path uses `now_ns()` / `monotonic_ns()` **only**.
- `datetime.now()` and `time.time()` **banned** from hot path.
  `authority_lint` rule **T1** enforces via AST scan of
  `mind/*`, `execution/*`, `governance/*`, `system/*`.
- Ledger entries use `wall_ns()` for human-readable timestamps **and**
  `now_ns()` for sequencing.

**Acceptance:**

- `authority_lint` rule T1 passes on full tree.
- Monotonic guarantee unit test (10 M samples, zero regressions).

---

## 7. Step 5 — Ledger Storage Tiers  **(T0-5)**

**Files land:** `ledger/hot_store.py`, `ledger/cold_store.py`,
`ledger/indexer.py`, `tests/test_ledger_tiers.py`.

**Tiers:**

- **Hot store:** recent events (≤ 24 h by default), optimized for fast
  read + append. Memory-mapped file backed by `mmap` / rolling segment.
- **Cold store:** archival (> 24 h), compressed (`zstd`) segmented
  files, seek-by-index.
- **Indexer:** rolling index over both tiers; supports lookup by
  (stream, version) and by time range.

**Write path:**

- Append-only write queue (**non-blocking** for producer).
- Background thread flushes to hot store every `FLUSH_INTERVAL_MS`
  (default 50 ms) or when queue > `FLUSH_BATCH` (default 1_000 events).
- On shutdown: drain queue synchronously before exit.

**Acceptance:**

- Producer throughput ≥ 50k events/s with p99 append latency < 1 ms.
- Cold-store round-trip: write → archive → query by (stream, version)
  returns identical event.
- Shutdown drain test: no event loss on SIGTERM.

---

## 8. Step 6 — Governance Policy Layer  **(T0-6)**

**Files land:** `governance/policies/risk_policy.py`,
`governance/policies/mode_policy.py`,
`governance/policies/escalation_policy.py`,
`governance/policies/__init__.py`,
`tests/test_policies.py`.

**Restructure:** `GovernanceKernel` becomes an **orchestrator only** —
it routes inputs to policies and serializes their verdicts. All
decision logic moves into the policies.

- `RiskPolicy.evaluate(proposal, risk_snapshot, fusion_result) -> Verdict`
- `ModePolicy.evaluate(request, current_mode, two_person_gate) -> Verdict`
- `EscalationPolicy.evaluate(hazard_event) -> Action`

**Hard rule:** `governance/kernel.py` cannot contain decision logic
directly. `authority_lint` rule **G1** enforces via AST (no literal
thresholds, no `if risk > X`, no `if mode ==` branching inside kernel).

**Acceptance:**

- Existing governance tests still pass unchanged (T0-15 backward-compat).
- New policy unit tests + integration tests.
- Lint rule G1 passes.

---

## 9. Step 7 — Strategy Lifecycle FSM  **(T0-7)**

**Files land:** `mind/strategy_lifecycle.py`,
`tests/test_strategy_lifecycle.py`.

**States:**

```
PROPOSED → SANDBOX → PAPER → CANARY → ACTIVE → DEPRECATED
```

**Rules:**

- All transitions require **governance approval** (operator-TOTP-signed
  via the existing promote-chain).
- **30-day paper-trading required** before `PAPER → CANARY`.
- Every transition writes a ledger event (`STRATEGY_LIFECYCLE_TRANSITION`).
- Reverse transitions (`ACTIVE → DEPRECATED`, `CANARY → PAPER` on
  hazard) allowed; forward-skip transitions forbidden.

**Acceptance:**

- FSM unit tests covering every legal + illegal transition.
- Integration test: a strategy promoted end-to-end through sandbox →
  paper → canary → active produces a ledger trail that reconstructs
  to the same state (cross-check with T0-0).

---

## 10. Step 8 — Adapter Circuit Breakers  **(T0-8)**

**Files touched:** `execution/adapters/base.py` (extend existing base
class; do not replace), `tests/test_adapter_circuit_breaker.py`.

**Additions to base adapter:**

- Retry policy (exponential backoff with jitter; configurable
  `max_retries`).
- Timeout policy (per-call; `TimeAuthority`-measured).
- Circuit breaker (closed → half-open → open) with thresholds.
- On open: adapter auto-disables, emits `SYSTEM_HAZARD` severity
  MEDIUM (or HIGH for safety-critical adapters — Rugcheck, honeypot
  simulator).

**Acceptance:**

- Unit tests for each state transition.
- Integration test: synthetic adapter that fails on every call opens
  its breaker in N failures and emits exactly one `SYSTEM_HAZARD`.

---

## 11. Step 9 — Global Kill Switch  **(T0-9)**

**Files land:** `system/kill_switch.py`,
`tests/test_kill_switch.py`.

**Types:**

- **Global kill** — halts all trading everywhere.
- **Per-exchange kill** — halts a single venue.
- **Per-strategy kill** — halts a single strategy.

**Priority:** overrides ALL decisions, including Indira's fast path
and any pending governance verdict.

**Activation paths:**

- Operator cockpit button (double-click + TOTP per §15 of
  `PR2_SPEC.md`).
- Programmatic from `EscalationPolicy` (T0-6).
- Dyon dead-man trip (existing, extended to call kill switch).

**Acceptance:**

- Kill-switch assertion in a live-trading harness halts ALL pending
  orders within 1 tick.
- Programmatic kill + cockpit kill + dead-man kill all produce
  ledger trails that reconstruct to the same halted state.

---

## 12. Step 10 — Observability Metrics  **(T0-10)**

**Files land:** `system/metrics.py`, optional
`system/prometheus_exporter.py`, `tests/test_metrics.py`.

**Tracked families (minimum):**

- Latency per market-graph node (p50, p95, p99).
- Plugin execution time per plugin.
- Hazard-event frequency per severity.
- Decision distribution (approved / rejected / held) per mode.
- Ledger append latency, hot-store depth, cold-store roll rate.
- Dead-man age per sensor.
- Full Grafana set per `PR2_SPEC.md` §16 (ten dashboards).

**Rule:** metrics are **never** authoritative state. A metrics
subsystem crash must not block trading or governance.

**Acceptance:**

- `/metrics` Prometheus endpoint exports all families.
- Crashing the metrics exporter does not raise into any hot-path
  consumer.

---

## 13. Step 11 — Knowledge-Store Hardening  **(T0-11)**

**Files touched:** `mind/knowledge_store.py` (if exists) or
`trader_knowledge/` (per `INDIRA_WEB_AUTOLEARN_SPEC.md`),
`tests/test_knowledge_store_hardening.py`.

**Additions:**

- Size limits (`max_snippets`, `max_bytes`) — config-driven.
- LRU eviction on insert if over cap.
- Periodic compaction (merge small segments, drop tombstones).
- Snapshot inclusion (T0-0) so knowledge-store state is replayable.

**Acceptance:**

- Stress test inserting 10× the cap — store stays at cap, oldest
  entries evicted, LRU order preserved.
- Compaction round-trip preserves all live entries.

---

## 14. Step 12 — Config Schema Validation  **(T0-12)**

**Files land:** `system/config_schema.py`, `tests/test_config_schema.py`.

**Rule:**

- All YAML / TOML / JSON config files have a **pydantic** model (or
  equivalent strictly-typed validator).
- System **fails fast** on invalid config — no warning, no default
  fallback, no "best effort" parsing.
- Schema files live in `system/config_schema.py` (single source of
  truth) and are version-pinned.

**Acceptance:**

- Every config file in the repo has a schema + a passing load test.
- Mutation tests: injecting bad values (missing fields, wrong types,
  out-of-range numbers) produces a descriptive `ConfigValidationError`.

---

## 15. Step 13 — Chaos Engine  **(T0-13)**

**Files land:** `execution/chaos/chaos_engine.py`,
`tests/test_chaos_engine.py`, `tests/chaos_scenarios/*.yaml`.

**Scenarios (minimum):**

- Feed loss (websocket disconnect, silent data halt).
- Plugin crash (raise mid-evaluation).
- Delayed hazards (hazard emitted but consumer starvation).
- Corrupted risk cache (bad version, monotonic violation).
- Partial ledger write (power-cut simulation mid-append).
- Adapter circuit-breaker storm (50 % of adapters failing simultaneously).

**Rule:** chaos engine runs **only** in sandbox / shadow / paper
environments. `authority_lint` rule **X1**: `execution.chaos.*` cannot
be imported from any live-path module.

**Acceptance:**

- Every scenario has an expected ledger trail + state outcome; the
  test asserts both.
- Chaos run on paper desk: system survives every scenario without
  entering undefined behavior (always fail-closed).

---

## 16. Step 14 — Deterministic Replay  **(T0-14)**

**Files land:** `tests/test_replay_determinism.py` (initial test file;
scales as T0-0, T0-3, T0-6, T0-7 land).

**Rule:** same input → identical output, bit-level where possible.
Where floating-point non-determinism is unavoidable (e.g. external
library), the test tolerates a documented epsilon with an explicit
comment.

**Scope:**

- Replay of a recorded ledger produces the same sequence of
  governance verdicts, strategy outputs, and state snapshots.
- Replay is re-runnable by CI on every PR (fast suite, ≤ 60 s).

**Acceptance:**

- CI test passes on main and on every PR.
- Test harness can load a ledger snapshot from a fixture and replay
  it in under 60 s.

---

## 17. Step 15 — Backward Compatibility (cross-cutting)  **(T0-15)**

**Not a standalone PR.** It is a CI gate applied to every Step PR.

**Non-breaking contract:**

- `core/contracts/*.py` Protocols (`IRiskCache`, `IRiskConstraints`,
  `ISystemHazardEvent`, `IHazardEmitter`, `IGovernanceHazardSink`,
  `HazardEvent`, `ExecutionEvent`) remain backward-compatible.
- Additive changes only. Removing or renaming a field requires a
  versioned Protocol (`IRiskCacheV2`) + deprecation window.
- Existing modules must **integrate**, not be replaced. Rewrite =
  rejected at review.

**CI gate:**

- Every PR runs a contract-diff check (`tools/contract_diff.py`,
  lands with T0-15's first PR) against `origin/main`. Breaking
  change = CI fail.

---

## 18. Final Guarantees

After all Steps land, the system **must** guarantee:

1. **Determinism** — replay produces identical decisions.
2. **Safety** — stale risk cache halts trading; kill switch overrides
   everything.
3. **Isolation** — zero cross-domain leakage (`authority_lint`
   C1 / C2 / C3 / G1 / T1 / W1 / W2 / X1 all clean).
4. **Performance** — p50 < 1 ms, p99 < 5 ms on the decision path
   (Indira fast path, measured by `TimeAuthority`).
5. **Auditability** — full state reconstructible at any timestamp
   via `StateReconstructor.rebuild(at_timestamp_ns)`.
6. **Resilience** — survives every chaos scenario (T0-13) without
   undefined behavior.

---

## 19. What v42.2 still does NOT do (intentional)

- No multi-machine scaling (Phase 9+).
- No GPU in the trading loop.
- No merging of governance files (kernel + policies stay separated
  per T0-6).
- No synchronous hazard handling (async bus is the only path).
- No manager-pattern abstractions — typed contracts only.
- No autonomous runtime self-modification beyond bounded E1–E5
  (see `INDIRA_WEB_AUTOLEARN_SPEC.md` §10).

---

## 20. New authority-lint rules introduced by this spec

Lands in the step-PR that first needs the rule. Each rule is unit-tested.

| Rule | Scope | Forbids |
|------|-------|---------|
| **G1** | `governance/kernel.py` | Literal thresholds, decision logic (moves to policies in T0-6) |
| **T1** | `mind/*`, `execution/*`, `governance/*`, `system/*` | `datetime.now()`, `time.time()` in hot path |
| **X1** | live-path modules | Imports from `execution.chaos.*` |
| **C3** | `mind/autolearn/*` | Imports from `execution.*`, `governance.*`, wallet resolvers (per `INDIRA_WEB_AUTOLEARN_SPEC.md`) |
| **W1** | memecoin adapters | Imports from main-wallet resolver (per `MEMECOIN_TRADING_SPEC.md`) |
| **W2** | sniper-bot gateways | Imports from non-tier-2 burner (per `DEX_AND_BOT_ADAPTER_ROADMAP.md`) |

Rules C1 and C2 already exist (manifest + neuromorphic triad) and
are extended, not replaced.

---

## 21. PR rhythm

- One Step = one PR = one operator-TOTP acceptance.
- Every PR ships with: acceptance tests · authority-lint rule
  additions (if any) · backward-compat contract-diff green ·
  Grafana-metric additions (if applicable) · chaos-scenario coverage
  (if applicable).
- No Step jumps its phase. No Step merges without operator approval.
- PRs that touch more than one Step are rejected — they get split.
