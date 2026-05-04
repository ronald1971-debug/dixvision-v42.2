# core/ — shared contracts + coherence projections (37 files)

## Purpose

Pure-data layer used by every engine. Two responsibilities:

1. **Frozen event contracts** — `core/contracts/*` defines every typed
   event that crosses an engine boundary (`SignalEvent`,
   `ExecutionIntent`, `ExecutionEvent`, `HazardEvent`, `SystemEvent`,
   `OperatorRequest`, `DecisionTrace`, `MarketTick`, `NewsItem`,
   `LearningUpdate`, `OperatorConsent`, `MemecoinLaunch`, etc.). All
   are `@dataclass(frozen=True)` with `__post_init__` validators so
   construction itself enforces the invariant (INV-65).
2. **Read-only projections** — `core/coherence/{belief_state,
   pressure_vector, decision_trace, performance_pressure,
   system_intent}.py`. Pure functions; no clock reads, no I/O. Used by
   intelligence + the meta-controller.

## Wiring

* Imported by every engine. `core/` itself imports nothing from
  `*_engine/` (B1 lint enforces this — verified post-audit).
* `core/cognitive_router/` selects AI providers by capability +
  registry preference; consumed by `intelligence_engine/cognitive/`
  and the chat surface in `ui/server.py`.
* `core/constraint_engine/` is a pure rule-graph oracle (INV-61).
  Compiled once at startup; queried per-tick from the meta-controller.

## Static-analysis result

* 37 files, 18 with findings — **all 18 are ruff-format drift only**
  (FORMAT rule, `would be reformatted`).
* No semantic findings (mypy, vulture, authority lint clean).
* No orphan modules. Every `core/contracts/*` row is imported by at
  least one engine.

## Deep-read observations

* `core/contracts/decision_trace.py` — `DECISION_TRACE_VERSION = 3`
  (Paper-S1 bumped from 2). Three optional fields
  `signal_trust / signal_source / validation_score` with strict
  invariants. PR #183 fixed a round-trip projection bug here.
* `core/contracts/operator_consent.py` — typed envelope
  (PR #169) with two-phase validate/commit so a downstream
  rejection does not burn the operator nonce.
* `core/contracts/strategy_registry.py` + `mode_effects.py` —
  declarative tables for the governance FSM. The B36 lint rule (PR
  #174) enforces that mutation must go through the FSM mutator.
* `core/contracts/event_provenance.py` — every typed event carries
  `produced_by_engine` (PR #80, INV-69). Receivers assert.

## Risks / gaps

* None blocking.
* `core/contracts/launches.py` + `news.py` carry default `lang="en"`
  and default `signal_source=""`; the empty-string default leaks into
  `DecisionTrace.signal_source` when a producer forgets to override.
  PR #183 normalised this in `build_decision_trace`. Other consumers
  should follow the same pattern.

## Verdict

**HEALTHY.** Drift is cosmetic. Contracts are well-typed and
single-source-of-truth.
