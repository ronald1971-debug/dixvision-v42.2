# DIX VISION — Causal Contract

> **Architectural reference.** This document is *non-normative prose*.
> The authoritative contracts live in:
>
> - The manifest deltas (`docs/manifest_v3.*_delta.md`) for invariants
>   `INV-*`.
> - `tools/authority_lint.py` for the static rule set (`B*`).
> - The runtime guard modules (`core/contracts/event_provenance.py`,
>   `core/contracts/learning_evolution_freeze.py`,
>   `execution_engine/execution_gate.py`) for the runtime asserts.
>
> When this document and any of those sources disagree, **the source
> wins** and this document is wrong. Open a PR to fix it.

The purpose of this document is to give a single place that
cross-references every causal restriction in the system —
*"who is allowed to cause what"* — so an operator, an auditor, or a new
agent can answer that question without having to grep three different
trees.

---

## 1. Why "causal contract"?

The system is built around an asymmetry: **only one designated module
may construct any given typed event**. Decoupling *who decides*, *who
emits*, and *who acts* is what gives DIX VISION its three guarantees:

1. **Determinism** — replay of the same input event sequence produces
   the same output event sequence (`INV-15`).
2. **Auditability** — every typed event carries
   `produced_by_engine`, so the audit ledger names the cause as well
   as the effect (`INV-69`).
3. **Reproducibility** — a frozen LearningEvolutionFreezePolicy lets
   us pin the system to a specific decision regime and replay it
   bit-for-bit (`INV-70`).

Each of these is enforced **twice**: once at lint time (so a faulty
PR fails CI) and once at runtime (so a faulty deployment refuses to
emit). The point of dual enforcement is that *neither layer is the
last line of defence*; if one is bypassed (e.g. a dynamic import that
the linter cannot see), the other catches it.

---

## 2. The dominant loop

```
                          ┌──────────────────────┐
                          │   Market Data        │
                          │   (broker / sensor)  │
                          └─────────┬────────────┘
                                    │  MarketEvent
                                    ▼
   ┌────────────────┐     ┌──────────────────────┐
   │  System Engine │────▶│  Intelligence Engine │
   │  (Dyon)        │     │  (Indira)            │
   │  hazards /     │     │  signals + meta-     │
   │  drift / state │     │  controller          │
   └────────┬───────┘     └─────────┬────────────┘
            │                       │  SignalEvent +
            │  SystemEvent           │  ExecutionDecision
            │  (telemetry)           ▼
            │             ┌──────────────────────┐
            │             │  Governance Engine   │
            │             │  approves / rejects /│
            │             │  constrains          │
            │             └─────────┬────────────┘
            │                       │  PolicyDecision
            │                       ▼
            │             ┌──────────────────────┐
            │             │  Execution Engine    │
            │             │  orders + fills +    │
            │             │  lifecycle FSM       │
            │             └─────────┬────────────┘
            │                       │  ExecutionEvent
            ▼                       ▼
   ┌────────────────────────────────────────────┐
   │  State + Ledger (hash-chained, replay)     │
   └─────────────────┬──────────────────────────┘
                     │  read-only
                     ▼
              ┌────────────────────┐
              │  Learning Engine   │
              │  (offline reward   │
              │  attribution)      │
              └─────────┬──────────┘
                        │  LearningUpdate (proposals only)
                        ▼
              ┌────────────────────┐
              │  Evolution Engine  │
              │  (sandbox; PatchProposal proposals only)
              └────────────────────┘
```

A more thorough walkthrough lives in
[`canonical_pipeline.md`](canonical_pipeline.md).

---

## 3. The causal contract — rule by rule

Each row below is an invariant of the system. The "lint" column points
to the static check; the "runtime" column points to the assertion that
runs in production.

### 3.1 Triad Lock (Decider / Executor / Approver are mutually exclusive)

| Aspect | Reference |
|---|---|
| Invariant | `INV-56` |
| Lint | `B20`, `B21`, `B22` in `tools/authority_lint.py` |
| Runtime | The `ExecutionEngine.execute(intent)` chokepoint in `execution_engine/engine.py` rejects any caller that did not pass through the `AuthorityGuard`. The legacy `process(SignalEvent)` path was deleted in HARDEN-05; that deletion is the runtime expression of the lock. |
| What it forbids | A single module being both the proposer and the acceptor of an order. |

### 3.2 Execution Gate origin restriction

| Aspect | Reference |
|---|---|
| Invariant | `INV-68` (HARDEN-01) |
| Lint | `B25` in `tools/authority_lint.py:740` — only `intelligence_engine.*` and the dev-harness may construct `ExecutionIntent`. |
| Runtime | `execution_engine.execution_gate.AuthorityGuard.validate(intent)` (`execution_engine/engine.py:84`) — runs on every call into `ExecutionEngine.execute()` and refuses intents whose origin is not in the allowlist. |
| What it forbids | Any module other than the intelligence engine generating a tradeable intent. |

### 3.3 Operator-approval edge restriction (cognitive-origin signals)

| Aspect | Reference |
|---|---|
| Invariant | Wave-03 PR-5 |
| Lint | `B26` in `tools/authority_lint.py:787` — only the approval edge may stamp `produced_by_engine = "intelligence_engine.cognitive"`. |
| Runtime | `core/contracts/event_provenance.py::assert_event_provenance(event, strict=True)` runs on receivers; rejects any `SignalEvent` whose `produced_by_engine` does not match the producer set for cognitive-origin signals. |
| What it forbids | A LangGraph turn smuggling a SignalEvent onto the bus without an operator acknowledging it first. |

### 3.4 Learning subsystem authority

| Aspect | Reference |
|---|---|
| Invariant | `INV-71` (HARDEN-06) |
| Lint | `B27` in `tools/authority_lint.py:859` — only `learning_engine.*` may construct `LearningUpdate`. |
| Runtime | `core/contracts/learning_evolution_freeze.py::assert_unfrozen(policy)` (`core/contracts/learning_evolution_freeze.py:93`) — the `LearningEvolutionFreezePolicy` is consulted before any `LearningUpdate` is applied; if frozen, the update is rejected. |
| What it forbids | An adaptive mutation slipping into the live decision regime when the system is supposed to be locked. |

### 3.5 Evolution subsystem authority

| Aspect | Reference |
|---|---|
| Invariant | `INV-71` (HARDEN-06), symmetric to 3.4. |
| Lint | `B28` in `tools/authority_lint.py:906` — only `evolution_engine.*` may construct `PatchProposal`. |
| Runtime | Same `LearningEvolutionFreezePolicy` chokepoint as 3.4; `PatchProposal` only enters production after the operator-approval edge consumes it. |
| What it forbids | A sandbox-mutated component being deployed without the freeze policy + operator approval. |

### 3.6 Trader-modeling authority (Wave-04 PR-1)

| Aspect | Reference |
|---|---|
| Invariant | `INV-71` (Wave-04 PR-1, completes the symmetry started by HARDEN-06) |
| Lint | `B29` in `tools/authority_lint.py:950` — only `intelligence_engine.trader_modeling.*` may construct `TraderObservation`. |
| Runtime | `assert_event_provenance` rejects any `TraderObservedEvent` whose `produced_by_engine` is not in the trader-modeling producer set. |
| What it forbids | A market-data adapter (e.g. Binance) being wired into the *trader* event channel by mistake — a structural class confusion that would let venue ticks drive trader-philosophy extraction. |

### 3.7 Per-event provenance (HARDEN-03)

| Aspect | Reference |
|---|---|
| Invariant | `INV-69` (HARDEN-03) |
| Lint | None at construction time (the constructor cannot know the caller); see 3.2 / 3.3 / 3.4 / 3.5 / 3.6 for class-specific construction restrictions. |
| Runtime | `assert_event_provenance(event, strict=True)` in `core/contracts/event_provenance.py:116` runs on every typed event at receive time. |
| What it forbids | A typed event with an empty or wrong `produced_by_engine` — i.e. a forged or anonymous event entering the audit ledger. |

### 3.8 Triad Lock — Governance is order-blind

| Aspect | Reference |
|---|---|
| Invariant | `INV-56` |
| Lint | `B20` in `tools/authority_lint.py:558` — `governance_engine.*` may not import from `execution_engine.*`. |
| Runtime | None needed; the import containment is sufficient because Python cannot synthesise the dependency at runtime without an explicit `__import__`. |
| What it forbids | Governance acquiring a static dependency on the broker / fill model — i.e. governance becoming a function of order outcomes. |

### 3.9 LangGraph / LangChain import containment

| Aspect | Reference |
|---|---|
| Invariant | `INV-67` |
| Lint | `B24` in `tools/authority_lint.py:680` — the non-deterministic graph orchestrator may only be imported from `intelligence_engine.cognitive.*` and the dashboard. |
| Runtime | `AuditLedgerCheckpointSaver` (Wave-03 PR-2) writes the LangGraph state into the same hash-chained audit ledger as everything else, so even non-deterministic cognitive turns end up in deterministic ledger rows. |
| What it forbids | A third-party agent framework being plumbed into the hot path. |

### 3.10 Registry-driven AI providers

| Aspect | Reference |
|---|---|
| Invariant | Dashboard-2026 wave-01 |
| Lint | `B23` in `tools/authority_lint.py:1001` — AI provider configuration must come from the SCVS registry, not be hard-coded. |
| Runtime | The cognitive router queries the registry by capability (`INDIRA_REASONING`, `INDIRA_MULTIMODAL_RESEARCH`, …) and falls through on `TransientProviderError` so a missing key is degraded into a different provider, not a crash. |
| What it forbids | A hard-coded model name or endpoint — i.e. a configuration change requiring a code release. |

---

## 4. Decision-trace contract (the *why* of every action)

Every action observed by the system is replayable via the
`DecisionTrace` contract in `core/contracts/decision_trace.py`. A
trace is anchored by a deterministic 16-hex-char `trace_id` (a hash of
`(symbol, ts_ns, plugin_chain)` — see `compute_trace_id`) and carries
a structured projection of the lenses the decider had at decision
time:

| Lens | Field | Source |
|---|---|---|
| Confidence breakdown | `confidence_breakdown` | `intelligence_engine.confidence_engine` |
| Pressure summary | `pressure_summary` | `core.coherence.performance_pressure` (INV-50) |
| Safety modifier | `safety_modifier` | `system_engine.dyon` (composite) |
| Active hazards | `active_hazards` | `system_engine.dyon.hazards` |
| Throttle applied | `throttle_applied` | `governance_engine.throttle` (INV-64) |
| Execution outcome | `execution_outcome` | `execution_engine.engine` (post-fill) |
| **Why layer** (Wave-04 PR-5) | `why` | `core.contracts.decision_trace.WhyLayer` — pointer-only references back to `PhilosophyProfile`, `EntryLogic`, `ExitLogic`, `RiskModel`, `Timeframe`, `MarketCondition`, and the parent `ComposedStrategy` (PR-4). |

The `why` layer is a pointer-only projection — it captures
`component_id`s, never the components themselves — so the trace stays
small and replay-stable even when the underlying registry rows are
revised.

The decision-trace builder (`core/coherence/decision_trace.py`) is
**pure**: no clock reads, no PRNG, no I/O. Same inputs → byte-
identical bytes (`INV-15`, `INV-65`).

---

## 5. How to extend this contract

When you add a new typed event class:

1. **Pick a producer set.** Decide which module(s) may construct it.
2. **Add a `B*` lint rule.** Mirror `B25` / `B27` / `B28` / `B29` —
   the rule should reject any importer outside the producer set that
   names the class as a callable.
3. **Add a producer set entry in `core/contracts/event_provenance.py`.**
   The receiver assertion will then refuse events whose
   `produced_by_engine` is not in the set.
4. **Add a row to this file.** Cite both the lint rule and the
   runtime assertion. If a row would have only one of the two, that
   is a smell — investigate before writing.
5. **Reference the invariant in the manifest delta** so the audit
   trail of "why this rule exists" survives PR archaeology.

**Pre-flight checklist before merging a PR that adds a new event:**

- [ ] Lint rule exists in `tools/authority_lint.py`.
- [ ] Producer set entry exists in `core/contracts/event_provenance.py`.
- [ ] Both are referenced from a row in this file.
- [ ] Runtime assertion is exercised in a test (positive + negative).
- [ ] `python3 tools/authority_lint.py --strict .` passes locally.
- [ ] `python3 -m pytest -q` passes locally.

---

## 6. Where the ledger sits

The hash-chained audit ledger lives behind `system_engine.ledger`. Its
structure is:

- **Hot ring** — bounded in-memory ring buffer for the last N events
  (sized to fit the replay-determinism acceptance suite).
- **Cold facade** — append-only on-disk store; never rewritten, never
  deleted from the live path.
- **Indexer** — a `SystemEvent` projector that lets the dashboard /
  Decision-Trace widget query by `trace_id`, `symbol`, or `ts_ns`
  range.

Every typed event listed in §3 above lands in this ledger with its
`produced_by_engine` stamped. The ledger is the source of truth for
forensic replay.

---

## 7. What to do if you find a gap

If you find a row where this document or `tools/authority_lint.py` or
`core/contracts/event_provenance.py` disagrees with the others:

1. **Stop.** Do not assume any of the three is right.
2. Trace the `INV-*` to its manifest delta. The manifest is the
   architectural authority.
3. If the lint rule is wrong → file a `tools/authority_lint.py` PR.
4. If the runtime assertion is wrong → file an
   `event_provenance.py` PR.
5. If this document is wrong → file a `docs/CAUSAL_CONTRACT.md` PR.
6. If the manifest is wrong → file a `docs/manifest_v3.*_delta.md`
   PR. Do this last — it changes the contract itself.
