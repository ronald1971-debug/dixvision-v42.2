# DIX VISION v42.2 — COMPILED POLYGLOT SYSTEM

_Architecture-of-record. Source of truth. Any contradiction between a
conversation message and this file is resolved in favor of this file._

---

## ROOT ARCHITECTURE

```
dix-vision/
├── mind/           (PYTHON)
├── execution/      (RUST)
├── system/         (RUST)
├── governance/     (PYTHON)
├── state/ledger/   (GO)
├── dashboard/      (TYPESCRIPT)
├── contracts/      (PROTOBUF)
├── tools/          (PYTHON + RUST)
├── tests/          (MULTI-LANGUAGE)
└── bootstrap/      (RUST)
```

---

## MIND (PYTHON — Indira / MARKET LOGIC)

```
mind/
  engine.py
  fusion_engine.py
  plugin_bus.py
  plugin_registry.py
  knowledge_store.py
  strategy_lifecycle.py
  strategy_sandbox.py
  intent_producer.py
  drift_monitor.py
  feedback_cleaner.py
```

**Role:** trading logic · signal generation · AI decision layer.

---

## EXECUTION (RUST — Dyon / RUNTIME SAFETY)

```
execution/
  adapters/
    base.rs
    binance.rs
    coinbase.rs
    kraken.rs
  hazard/
    sensor_array.rs
    event_emitter.rs
    async_bus.rs
  chaos/
    chaos_engine.rs
  feedback.rs
  runtime_monitor.rs
```

**Role:** exchange execution · hazard detection · system failure safety.

---

## SYSTEM (RUST — CONTROL PLANE)

```
system/
  fast_risk_cache.rs
  state_reconstructor.rs
  snapshots.rs
  time_source.rs
  kill_switch.rs
  load_controller.rs
  config/
    config_manager.rs
    config_watcher.rs
    config_version_manager.rs
    feature_flags.rs
    fallback_manager.rs
  metrics.rs
  config_schema.rs
```

**Role:** system state · risk authority · runtime control.

---

## GOVERNANCE (PYTHON — POLICY ENGINE)

```
governance/
  kernel.py
  policies/
    risk_policy.py
    mode_policy.py
    escalation_policy.py
```

**Role:** rule enforcement · system decisions · approvals.

---

## LEDGER (GO — EVENT SOURCE OF TRUTH)

```
state/ledger/
  hot_store.go
  cold_store.go
  indexer.go
  event_types.go
  integrity.go
```

**Role:** immutable event log · replay system · audit trail.

---

## DASHBOARD (TYPESCRIPT — UI)

```
dashboard/src/
  components/
  views/
  panels/
    DecisionTrace.tsx
    RiskView.tsx
    PortfolioView.tsx
    SystemHealth.tsx
    GovernancePanel.tsx
  hooks/
  services/
    websocket.ts
    api.ts
  App.tsx
```

**Role:** real-time visualization · operator control UI.

---

## CONTRACTS (PROTOBUF)

```
contracts/
  execution.proto
  market.proto
  governance.proto
  system.proto
  ledger.proto
```

**Role:** cross-language communication standard.

---

## TOOLS (VALIDATION LAYER)

```
tools/
  authority_lint.py
  contract_diff.py
  replay_validator.py
  config_validator.py
```

**Role:** enforce architecture rules · CI safety checks.

---

## TESTS (MULTI-LANGUAGE)

```
tests/
  test_replay_determinism.py
  test_domain_isolation.py
  test_latency_slo.py
  test_hazard_flow.py
  test_ledger_integrity.py
  test_chaos_engine.rs
  test_governance.go
```

**Role:** validation · system correctness.

---

## BOOTSTRAP (SYSTEM START)

```
bootstrap/
  kernel_boot.rs
  system_init.rs
  dependency_resolver.rs
```

**Role:** deterministic system startup sequence.

---

## LANGUAGE MAP

| Component | Language |
|---|---|
| `mind/` | Python |
| `execution/` | Rust |
| `system/` | Rust |
| `governance/` | Python |
| `state/ledger/` | Go |
| `dashboard/` | TypeScript |
| `contracts/` | Protobuf |

---

## CORE GUARANTEES

1. No cross-domain imports.
2. Ledger is the single source of truth.
3. Deterministic replay required.
4. Typed communication only.
5. Kill switch always overrides execution.
6. Event-sourced architecture enforced.

---

## MIGRATION STATUS (v42.2 → polyglot v42.2)

Phase 1 delivered the Tier-0 control plane in **Python** as the
reference implementation. The polyglot plan keeps `mind/` and
`governance/` as Python and ports the hot-path / safety-critical
modules to the languages specified above, one PR per module, with
benchmarks justifying each port.

| Module | Reference (Python) | Target | Port PR | Status |
|---|---|---|---|---|
| `contracts/*.proto` + workspace scaffold | — | Protobuf + multi-lang | **#14** | ✅ CI green, awaiting merge |
| `rust/` workspace (execution + system + bootstrap crates) | — | Rust (Cargo) | **#15** | ✅ CI green, awaiting merge |
| `system/time_source` | PR #7 | Rust (+ PyO3) | **#16** | ✅ CI green, awaiting merge |
| `system/fast_risk_cache` | PR #12 | Rust (+ PyO3) | **#17** | ✅ CI green, awaiting merge |
| `system/metrics` (in-memory sink, T0-10) | Phase 7 | Rust (+ PyO3) | **#18** | ✅ CI green, awaiting merge |
| `execution/adapters/base` (circuit-breaker, T0-8) | Phase 2 | Rust (+ PyO3) | **#19** | ✅ CI green, awaiting merge |
| `system/config_schema` | PR #8 | Rust | TBD (≥#24) | pending |
| `mind/knowledge_store` | PR #9 | **stays Python** | — | — |
| `system/state_reconstructor` + `snapshots` (T0-0) | PR #10 | Rust | **#26** | blocked on PR #10 merge |
| `state/ledger/{hot_store, cold_store, indexer}` (T0-5) | PR #11 | Go (+ gRPC facade) | **#27** | blocked on PR #11 merge |
| `tests/test_replay_determinism.py` | PR #13 | Python (multi-lang harness) | — | — |
| `mind/fusion_engine` | Phase 2 | **stays Python** | — | — |
| `mind/strategy_lifecycle` | Phase 2 | **stays Python** | — | — |
| `execution/adapters/{binance,coinbase,kraken}` | Phase 2 | Rust | TBD | pending |
| `execution/hazard/*` | Phase 2 | Rust | **#20** | in progress |
| `execution/chaos/chaos_engine` (T0-13) | Phase 7 | Rust | **#21** | pending |
| `system/kill_switch` (T0-9) | Phase 4 | Rust | **#22** | pending |
| `system/load_controller` (T0-2) | Phase 3 | Rust | **#23** | pending |
| `system/config_*` + `feature_flags` + `fallback_manager` (T0-12/T0-16) | Phase 7 | Rust | **#24** | pending |
| `system/metrics` Prometheus text exporter | Phase 7 | Rust | **#25** | pending |
| `bootstrap/kernel_boot` | new | Rust | **#28** | pending |
| `dashboard/*` (5 panels) | — | TypeScript | TBD | pending |
| `tools/{contract_diff,replay_validator,config_validator}` | — | Rust | TBD | pending |
| multi-language tests (`test_chaos_engine.rs`, `test_governance.go`, …) | — | mixed | TBD | pending |

Each port PR:
1. Introduces the new-language implementation.
2. Publishes a stable FFI/gRPC surface matching the `.proto` contract.
3. Deletes the Python file in the same commit it lands.
4. Migrates the module's tests to the target language (plus integration
   tests in Python where cross-domain).
5. Carries a reproducible benchmark showing the latency / safety /
   correctness improvement that justified the port.

---

## SAFETY GATES (HARDCODED IN CODE)

- **TOTP** — kill-switch override, live-mode entry.
- **Two-person** — fast-path risk amend.
- **30-day paper** — strategy `PAPER → CANARY` gate.
- **Sandbox promote** — every plugin/strategy starts in sandbox.
- **Operator acceptance** — autolearn ingest, plugin activation, kill-switch clear.

These gates live in code. They are never bypassed, in any language.

---

## PERFORMANCE ENVELOPE

- p50 < 1 ms per tick through `market_graph`
- p99 < 5 ms per tick through `market_graph`
- `FastRiskCache` reads are lock-free. Writers never hold a lock
  across a projector apply.

---

## INTENTIONAL NON-GOALS (v42.2)

- No multi-machine scaling (Phase 9+).
- No GPU in the trading loop.
- No merging of governance files.
- No synchronous hazard handling.
- No manager-pattern abstractions.

---

_Updates to this manifest are proposed in the PR that implements them._

---

## BUILD STATE — CANONICAL SUMMARY

_This section is refreshed each time a port PR lands in CI-green
state. The most recent revision reflects the state after **PR #19**._

### Shipped, CI-green, awaiting operator merge

| PR | Title | Crate / module touched | Tests | Key notes |
|---|---|---|---|---|
| **#14** | `feat(polyglot):` architecture-of-record + `contracts/*.proto` scaffold | `docs/DIX_VISION_v42_2_COMPILED.md`, `contracts/` (5 `.proto` files), `pyproject.toml` | N/A | Source-of-truth manifest; Python protobuf codegen wired into CI. |
| **#15** | `feat(polyglot):` Rust workspace scaffold | `rust/Cargo.toml` + empty `execution`, `system`, `bootstrap` crates + `rust.yml` CI workflow | crate-version parse, placeholder unit | Workspace lints (`unsafe_code = "deny"`, `clippy::panic = "deny"`, etc.); `#![forbid(unsafe_code)]` at every crate root. |
| **#16** | `feat(polyglot/T0-4):` port `system/time_source` to Rust + PyO3 | `rust/system/src/time_source.rs`, `rust/py_system/` (PyO3 seam), `system/time_source.py` (dual-backend wrapper) | 9 Rust + parity | Canary port; proved the maturin → cdylib → `import dixvision_py_system` pipeline end-to-end. |
| **#17** | `feat(polyglot/T0-1):` port `system/fast_risk_cache` to Rust + PyO3 | `rust/system/src/fast_risk_cache.rs` | 15 Rust + 15 parity | Lock-free atomic-swap read path; `version_id` monotonic across both backends. |
| **#18** | `feat(polyglot/T0-10):` port `system/metrics` to Rust + PyO3 | `rust/system/src/metrics.rs` | 12 Rust + 20 parity | In-memory sink; process-global singleton via `OnceLock<Mutex>`; Prometheus exporter deferred to PR #25. Follow-up fix `717ec35` restores `MetricsSink()` no-arg constructor. |
| **#19** | `feat(polyglot/T0-8):` circuit-breaker primitive in Rust + PyO3 | `rust/execution/src/circuit_breaker.rs`, name-keyed registry in `rust/py_system/src/lib.rs` | 11 Rust + 10 parity | Three-state FSM (Closed/Open/HalfOpen), generic `MonotonicClock` trait for test injection, `parking_lot::Mutex<Inner>` for concurrency. Follow-up fix `78f264c` adds crate-inner `#![allow(clippy::useless_conversion)]` for clippy 1.95 / PyO3 0.22 macro interaction. |

### In progress

| PR | Scope |
|---|---|
| **#20** | `execution/hazard/*` — bounded non-blocking queue + pure severity classifier (tight scope; dispatch loop stays in Python). |

### Test / lint gate as of PR #19

- **Rust:** `cargo fmt --all --check` ✅, `cargo clippy --all-targets --all-features -- -D warnings` ✅ (rustc 1.83 local + 1.95 CI), `cargo test --all --all-features --locked` ✅ (50 tests across 4 crates).
- **Python:** 153 tests pass on pure-Python reference, 10 parity tests green when Rust wheel is built in CI.
- **Authority lint:** `python tools/authority_lint.py` → 0 violations (TimeAuthority + domain-isolation invariants hold).
- **CI:** `rust` workflow green on every PR in the stack.

### Polyglot tree — live vs. planned

```
dix-vision/
├── mind/           PYTHON   (reference, stays Python per manifest)
├── execution/      RUST     ████░░░░░░░░  circuit-breaker shipped (PR #19);
│                               hazard/* in progress (PR #20);
│                               adapters/{binance,coinbase,kraken} + chaos
│                               pending.
├── system/         RUST     ██████░░░░░░  time_source, fast_risk_cache,
│                               metrics shipped (PRs #16-#18);
│                               kill_switch/load_controller/config_*
│                               pending.
├── governance/     PYTHON   (reference, stays Python per manifest)
├── state/ledger/   GO       ░░░░░░░░░░░░  blocked on PR #11 merge.
├── dashboard/      TS       ░░░░░░░░░░░░  blocked on ledger + system websocket
│                               contract.
├── contracts/      PROTOBUF ████████████  5 .proto files scaffolded (PR #14).
├── tools/          PY+RUST  █░░░░░░░░░░░  authority_lint.py shipped;
│                               contract_diff/replay_validator/config_validator
│                               pending.
├── tests/          MIXED    ████░░░░░░░░  Python replay-determinism + parity
│                               suites green; multi-lang suites pending.
└── bootstrap/      RUST     █░░░░░░░░░░░  crate scaffold (PR #15);
                                kernel_boot.rs pending (PR #28).
```

### Next four PRs (autonomous execution plan)

1. **PR #20** — `execution/hazard/*` primitive (severity classifier + bounded queue).
2. **PR #21** — `execution/chaos/chaos_engine` (T0-13).
3. **PR #22** — `system/kill_switch` (T0-9; TOTP-gated in code).
4. **PR #23** — `system/load_controller` (T0-2; admission-control hot path).

All four are stacked on the PR #19 Rust branch (`rust/` + `py_system/` seam are prerequisites). Merge-forward is unblocked once the operator clicks merge on the stack #15 → #16 → #17 → #18 → #19.

### Invariants that have stayed unchanged through the polyglot pivot

- **No cross-domain imports.** The `rust/py_system/` crate is still the **only** place allowed to bridge Python ↔ Rust. All other `system/*.py` imports go through the module's Python wrapper, which selects between pure-Python reference and Rust backend at import time.
- **Operator gates stay hardcoded.** TOTP for kill-switch clear, two-person for fast-path risk amends, 30-day paper for PAPER→CANARY, sandbox for every plugin/strategy. Moving the runtime to Rust does not relax any gate.
- **Dual-backend parity.** Every Rust port ships with a parity test suite the Python reference passes unchanged. A port is only accepted when **both backends are green** on the same invariant contract.
- **Deterministic replay + event-sourced ledger.** Will survive the Go port of `state/ledger/*` because the gRPC facade is part of the contract, not an implementation detail.
