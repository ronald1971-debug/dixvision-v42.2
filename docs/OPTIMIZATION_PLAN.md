# DIX VISION v42.2 — Optimization Plan (2026 targets)

**Scope:** this doc is the cross-cutting optimization roadmap applied to every phase PR. Each phase captures baseline numbers (pyinstrument / scalene) for the functions it touches; actual switches to the 2026 stack land as dedicated, test-gated PRs after the manifest is complete and measurable.

Principle: **simplify without losing anything** (operator directive). Every optimization lands only after:
1. A baseline measurement.
2. A test proving semantic equivalence.
3. A measured improvement > threshold (≥ 2× for hot path, ≥ 1.3× for cold paths).

## 1. Python 3.13 free-threaded (no-GIL)

**Target:** true-parallel CPU work on background workers without process-level fan-out.

**Where it fits us:**
- Ledger appender (SHA-chaining + sqlite write) runs off the hot path; a free-threaded build lets the chain compute in parallel with the fast path's atomic reference read.
- Weekly scout + DYON engine already run on their own threads — they currently compete for the GIL with the cockpit event loop.
- Hot path (`fast_execute_trade`, `fast_risk_cache.get()`) stays single-writer + read-only so the no-GIL win is additive, never a correctness risk.

**What we do:**
- CI matrix adds `python3.13t` once released as stable.
- Benchmark suite records per-phase GIL contention (scalene `--cpu` profile).
- Initial adopters: `state.ledger.event_store.append_event`, `system_monitor.weekly_scout.discover`, `cockpit.llm` parallel provider fan-out.
- Hot path **stays single-thread** — we gain in parallelism on workers, not in trade hot path.

## 2. msgspec (Pydantic alternative)

**Target:** ~3× faster JSON serialization + strict schema validation at wire boundaries.

**Where it fits us:**
- Cockpit API (`/api/status`, `/api/risk`, `/api/charters`, `/api/ai`, `/api/autonomy/*`, `/api/operator/*`, `/api/custom-strategies/*`, `/api/weekly-scout/*`) currently hand-builds dicts.
- WebSocket frames to the mobile PWA.
- Sandbox pipeline patch records.

**What we do:**
- Replace hand-built response dicts with `msgspec.Struct` definitions colocated with each route.
- Validation at the API edge — no untyped dicts cross the wire.
- `msgspec.json.encode` / `decode` replaces `json.dumps` / `json.loads` in `cockpit/app.py` and `mobile_pwa/` bridge.
- Dependencies: `msgspec>=0.19` (pure C, no build deps on CPython wheels).

## 3. orjson (ledger payloads)

**Target:** 3–5× faster JSON encoding on the ledger write path.

**Where it fits us:**
- `state.ledger.event_store.LedgerEvent.compute_hash` uses `json.dumps(..., sort_keys=True)` for the hash preimage. orjson supports sorted keys and is ~5× faster. Must keep sort-key semantics exact so hash-chain compat is preserved.

**Guard:** hash-chain compatibility test — before/after sha256 over 10k random events must match byte-for-byte.

## 4. LMAX Disruptor pattern (hazard bus)

**Target:** >1M events/sec, lock-free read, single-writer contract already matches us.

**Where it fits us:**
- `execution.hazard.async_bus.HazardBus` is currently a `queue.Queue`. Queue uses a mutex per put/get → contention under burst.
- Dyon is the sole producer, Governance is the sole consumer (axioms H2 + H3). This is the exact shape Disruptor was designed for.

**What we do:**
- Replace internal `queue.Queue` with a preallocated ring buffer (power-of-two size, padded cursors, memory-ordered loads/stores via `threading.Lock` + `atomic` shim).
- Behaviour preserved: `emit()` stays non-blocking; overflow drops the newest event and emits an error per axiom H5.
- Measure: 1M-event burst throughput + p99 emit latency.

## 5. Polars + Arrow (backtest / forward-test)

**Target:** 5–9× faster joins + zero-copy to numpy for SNN feature extraction.

**Where it fits us:**
- Backtest engine (Phase 2 / PR #2d) — candle joins + OFI rollups + walk-forward splits.
- Forward-test (paper) — same pipeline, streaming.
- Neuromorphic feature extractor — 64-step rolling window; Polars Series → numpy view feeds snntorch / ONNX.

**What we do:**
- New dep `polars>=1.6` (rust-backed).
- Backtest data layer redesigned around `pl.LazyFrame` with sink to parquet for reproducibility.
- CSV export path unchanged for operator visibility.

## 6. mmap'd state snapshots

**Target:** deterministic crash-recovery at > 100 MB state size without pickle overhead.

**Where it fits us:**
- `system.state.StateManager` currently keeps an in-memory dataclass. Crash recovery today replays the ledger.
- Large state (strategy weights, open positions, autonomy envelopes, wallet ledger) would benefit from an mmap-backed `struct`-packed layout.

**Not urgent** — ledger replay is correct and audit-friendly. This is a P3 optimization.

## 7. Profiling harness (every PR)

**Target:** catch hot-path regressions in CI.

**What we do:**
- `scripts/profile_hot_path.py` runs pyinstrument over `fast_execute_trade` synthetic workload + scalene over `hazard_bus` burst.
- PR template requires baseline numbers for changed hot-path functions.
- CI compares against the `main` baseline; regression > 10% fails the build.

## 8. Per-phase application

| Phase | Primary optimization candidates |
|---|---|
| 0 | Baseline (this PR) — no switches yet |
| 1 (ledger + memory) | orjson for event payload hashing, msgspec for cockpit status/risk endpoints |
| 2 (INDIRA market engine) | Polars for backtest data, Disruptor pattern for signal event bus |
| 3 (DYON) | free-threaded worker pool for telemetry scrapers |
| 4 (governance) | msgspec constraint serialization; advisory RISK_SIGNAL_EVENT via shared ring buffer |
| 5 (translation + execution) | fast_execute hot-path micro-opts (inline `allows_trade` branchless); baseline already < 500 µs — target < 250 µs |
| 6 (enforcement) | authority_lint AST caching (currently O(files × rules); cache parse trees) |
| 7 (observability) | msgspec + orjson for Prometheus exposition, zero-copy OpenTelemetry spans |
| 8 (Windows production) | signed release tag, build-time link optimization |

## 9. Non-goals

- No rewrites of decision logic for speed. Correctness and auditability beat latency everywhere that isn't the frozen hot path.
- No speculative parallelism on ledger writes — single-writer is an axiom.
- No runtime topology change in neuromorphic layer (axiom N8).

## 10. Measurement

- pyinstrument: per-function wall clock, call stacks.
- scalene: CPU + memory + GIL time per line.
- Output stored as `.profile/phase_N_{function}_{commit}.json` (not checked in) with a CI summary table posted to the PR.

This plan is the reference every phase PR links back to. Specific numbers and implementations land in the phase PR they apply to, gated by tests and the two-person hardware-key override for anything that touches frozen hot-path code.
