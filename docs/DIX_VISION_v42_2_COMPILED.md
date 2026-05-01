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

| Module | Reference (Python) | Target | Port PR |
|---|---|---|---|
| `system/time_source` | PR #7 | Rust (+ PyO3) | TBD |
| `system/config_schema` | PR #8 | Rust | TBD |
| `mind/knowledge_store` | PR #9 | **stays Python** | — |
| `system/state_reconstructor` + `snapshots` | PR #10 | Rust | TBD |
| `state/ledger/{hot_store, cold_store, indexer}` | PR #11 | Go (+ gRPC facade) | TBD |
| `system/fast_risk_cache` | PR #12 | Rust (+ PyO3 / shared-mem) | TBD |
| `tests/test_replay_determinism.py` | PR #13 | Python (multi-lang harness) | — |
| `mind/fusion_engine` | Phase 2 | **stays Python** | — |
| `mind/strategy_lifecycle` | Phase 2 | **stays Python** | — |
| `execution/adapters/*` | Phase 2 | Rust | TBD |
| `execution/hazard/*` | Phase 2 | Rust | TBD |
| `execution/chaos/chaos_engine` | Phase 7 | Rust | TBD |
| `system/kill_switch` | Phase 4 | Rust | TBD |
| `system/load_controller` | Phase 3 | Rust | TBD |
| `system/metrics` | Phase 7 | Rust | TBD |
| `bootstrap/` | new | Rust | TBD |

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
