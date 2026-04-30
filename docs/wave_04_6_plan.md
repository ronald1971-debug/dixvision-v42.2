# Wave-04.6 plan — Mode-Effect Table + StrategyRegistry + Closed Learning Loop

**Status:** parked, next priority after FRED PR #108 merges.
**Author:** drafted in session ending PR #108; ratified by reviewer #3 ordering.
**Predecessors:** Wave-Live (#103-#107), Wave-04.5 PR-1 / PR-2 (#102 / #108).
**Successors:** Wave-News-Fusion, then B30 lint, then Rust dual-backend resolution.

This document is the canonical breakdown of Wave-04.6 so the next session
can start cleanly even if the current autonomy budget runs out.

---

## 0. What reviewer #3 got right and what was stale

> *"The build plan's Mode FSM has 4 states. The design spec requires 7."*

**Stale.** Verified `core/contracts/governance.py:39-53`: `SystemMode` already
enumerates **all 7 states** — `SAFE`, `PAPER`, `SHADOW`, `CANARY`, `LIVE`,
`AUTO`, `LOCKED`. `governance_engine/control_plane/state_transition_manager.py`
already enforces the forward chain `SAFE → PAPER → SHADOW → CANARY → LIVE →
AUTO` with backward de-escalation, kill-to-LOCKED from any state, and
`LOCKED → SAFE` recovery.

The reviewer's underlying concern still holds, just one layer down: the
**mode-effect table** — what each engine does *differently* in each mode —
is incomplete. `PolicyEngine` largely treats SHADOW / CANARY / AUTO the same
as their neighbours; the only distinguishing rule is the `LIVE / AUTO`
operator-authorisation gate. SHADOW does not formally suppress execution,
CANARY does not formally cap size, AUTO does not formally relax per-trade
oversight to exception-only. Those are the missing rules.

> *"That is a replay buffer with no training step."*

**Confirmed real.** `learning_engine/update_emitter.py` produces
`SystemEvent(UPDATE_PROPOSED)`; `governance_engine/engine.py:203-211` writes
a `UPDATE_PROPOSED_AUDIT` row to the ledger and returns. Nothing validates,
ratifies, or applies the update. The bridge is built; the receiver is a
sink, not a consumer.

> *"StrategyRegistry — DRAFT → VALIDATING → APPROVED → RETIRED."*

**Confirmed missing.** No `StrategyRegistry` class, no per-strategy
lifecycle state, no canonical persistent store. Strategies exist only as
plugin instances loaded from `registry/plugins.yaml`.

---

## 1. Goal

Close the loop between *proposing* parameter / strategy changes and
*ratifying* them, while ensuring each `SystemMode` has materially
distinct behaviour across all engines.

**Acceptance criteria:**

1. SHADOW, CANARY, AUTO each have at least one observably-distinct
   behaviour vs. their forward and backward neighbours, encoded in the
   PolicyEngine decision table or in a new `mode_effect_table` constant
   referenced by every engine that conditions on mode.
2. A persistent `StrategyRegistry` stores every strategy with its
   lifecycle state (`DRAFT | VALIDATING | APPROVED | RETIRED`) and
   audit-ledger anchored transitions.
3. Every `UPDATE_PROPOSED` reaches an `UpdateValidator` that either
   ratifies the update (writes `UPDATE_RATIFIED` and applies) or
   rejects it (writes `UPDATE_REJECTED`). The applier is the only writer
   of strategy parameters at runtime.
4. Determinism preserved: replaying the same event sequence produces the
   same registry state, same ledger rows, same parameter values.

---

## 2. PR breakdown

### PR-A — Mode-effect table (`docs/MODE_EFFECTS.md` + `core/contracts/mode_effects.py`)

Single canonical Python module that exports a frozen mapping
`MODE_EFFECTS: dict[SystemMode, ModeEffect]` where `ModeEffect` is a
slotted frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class ModeEffect:
    signals_emit: bool          # IntelligenceEngine emits SignalEvent
    executions_dispatch: bool   # ExecutionEngine dispatches ExecutionIntent
    size_cap_pct: float | None  # PolicyEngine clamps notional to this %
    learning_emit: bool         # UpdateEmitter is unfrozen
    learning_apply: bool        # UpdateApplier ratifies UPDATE_PROPOSED
    operator_auth_required: bool  # forward transitions require operator
    oversight_kind: Literal["per_trade", "exception_only", "none"]
```

Reference values (one row per mode):

| mode    | signals | exec   | size_cap | learn_emit | learn_apply | op_auth | oversight       |
| ------- | ------- | ------ | -------- | ---------- | ----------- | ------- | --------------- |
| LOCKED  | False   | False  | 0%       | False      | False       | n/a     | none            |
| SAFE    | False   | False  | 0%       | False      | False       | False   | per_trade       |
| PAPER   | True    | paper  | 0%       | True       | False       | False   | per_trade       |
| SHADOW  | True    | False  | 0%       | True       | False       | False   | per_trade       |
| CANARY  | True    | True   | 1%       | True       | True        | True    | per_trade       |
| LIVE    | True    | True   | None     | True       | True        | True    | per_trade       |
| AUTO    | True    | True   | None     | True       | True        | True    | exception_only  |

Tests: golden table-hash check (canonical-sorted SHA-256 over the ModeEffect
tuple values, like PolicyEngine's `_hash_decision_table`); per-mode
property tests; lint rule **B31** that requires every engine module which
reads `current_mode()` to import from `mode_effects` rather than
hard-coding mode comparisons.

Risk: medium. Touches every engine that conditions on mode (intelligence,
execution, learning). Keep the actual *behaviour* changes in PR-B
onwards; PR-A is just the table + lint rule.

LoC estimate: ~400 incl. tests.

### PR-B — Wire SHADOW = signals-on-execution-off

Smallest behaviour change. In `execution_engine/engine.py`'s execute
chokepoint, gate dispatch on `MODE_EFFECTS[mode].executions_dispatch`.
SHADOW now emits `ShadowExecution` audit rows but never reaches a broker
adapter. Tests: feed a SignalEvent in SHADOW, assert no broker call but
ledger row present.

LoC estimate: ~150 incl. tests.

### PR-C — Wire CANARY size cap

In `governance_engine/control_plane/policy_engine.py`, add a
`permit_execution_intent` gate that clamps `intent.notional_pct` to
`MODE_EFFECTS[mode].size_cap_pct`. CANARY now caps to 1%. Tests: 5%
intent in CANARY → clamped to 1% with `CLAMP_AUDIT` row.

LoC estimate: ~200 incl. tests.

### PR-D — StrategyRegistry contract + storage

Create `core/contracts/strategy_registry.py`:

```python
class StrategyLifecycle(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"

@dataclass(frozen=True, slots=True)
class StrategyRecord:
    strategy_id: str
    version: int
    lifecycle: StrategyLifecycle
    parameters: Mapping[str, str]
    composed_from: tuple[str, ...]   # component IDs (Wave-04 PR-3)
    why: tuple[str, ...]             # DecisionTrace.why refs
    created_ts_ns: int
    last_transition_ts_ns: int
```

And `governance_engine/strategy_registry.py` with the FSM:

* `DRAFT → VALIDATING` — when CompositionEngine produces a candidate.
* `VALIDATING → APPROVED` — when N consecutive PAPER/SHADOW evaluations
  exceed score threshold.
* `VALIDATING → RETIRED` — on validation failure.
* `APPROVED → RETIRED` — on operator request or three rejected
  `UPDATE_PROPOSED` in a row for the same strategy_id.

Every transition writes a `STRATEGY_LIFECYCLE` ledger row. Determinism:
the registry is replayed by walking the ledger.

LoC estimate: ~500 incl. tests.

### PR-E — UpdateValidator + UpdateApplier (close the learning loop)

Replace the current `engine.py:203-211` audit-only branch with:

```
SystemEvent(UPDATE_PROPOSED)
  ↓
UpdateValidator
  ├── reject  → UPDATE_REJECTED ledger row
  └── ratify  → UPDATE_RATIFIED ledger row → UpdateApplier.apply()
                                              ↓
                                              StrategyRegistry.record_change()
```

Validator rules (deterministic):

* `MODE_EFFECTS[mode].learning_apply` must be `True` (PAPER and SHADOW
  reject).
* The `strategy_id` must exist in StrategyRegistry with lifecycle
  `APPROVED`.
* The `parameter` must be in the strategy's declared mutable-parameter
  whitelist (registry-driven; SCVS row).
* The `(old_value, new_value)` pair must satisfy a strategy-specific
  bound (e.g. `risk_per_trade ∈ [0.001, 0.05]`).

Tests: golden trace test where a sequence of `UPDATE_PROPOSED` flows
through a PAPER → CANARY → LIVE transition and only the LIVE-time
proposals are applied.

LoC estimate: ~600 incl. tests.

### PR-F — AUTO mode oversight relaxation

In `governance_engine/control_plane/operator_interface_bridge.py`,
flip per-trade approval gating to `MODE_EFFECTS[mode].oversight_kind`.
AUTO now requires operator only on policy-breach hazards, not per
trade. Tests: 100 routine LIVE intents in AUTO with no operator
present → all approved; one HAZ-12 hazard → operator-required gate
fires.

LoC estimate: ~250 incl. tests.

---

## 3. Order and budget

| order | PR  | risk   | LoC | depends on            |
| ----- | --- | ------ | --- | --------------------- |
| 1     | A   | medium | 400 | —                     |
| 2     | B   | low    | 150 | A                     |
| 3     | C   | low    | 200 | A                     |
| 4     | D   | medium | 500 | A                     |
| 5     | E   | high   | 600 | A, D                  |
| 6     | F   | low    | 250 | A                     |

PR-E is the highest-risk change in the project history because it is the
first time a non-operator actor (the learning engine) is permitted to
mutate a strategy parameter at runtime. **Reviewer time should be
~3× normal per the warning in the v3 audit.** Mandatory hand-review by
operator before merge; CI alone is insufficient.

---

## 4. Out of scope (explicitly punted to follow-on waves)

* **Multi-engine concurrent updates.** Wave-04.6 assumes a single update
  in flight per strategy at a time. Concurrent-update arbitration is a
  Wave-04.7 problem.
* **Multi-tenant StrategyRegistry.** Single-tenant only.
* **Rollback of applied updates.** PR-E ratifies; un-ratifying is
  Wave-04.7.
* **News-driven update validation** (`UpdateValidator` reads BeliefState).
  That is Wave-News-Fusion's contribution to PR-E, scheduled after.

---

## 5. Pre-flight checks before starting Wave-04.6

* [ ] PR #108 (FRED HTTP) merged to main.
* [ ] `pytest -q` passes 1874+ tests on main.
* [ ] `python -m tools.authority_lint --strict` clean on main.
* [ ] `python -m tools.scvs_lint --strict` clean on main.
* [ ] Confirm reviewer #3's strict ordering still preferred (Wave-04.6
  before Wave-News-Fusion before BLS) — this document's premise.

---

## 6. After Wave-04.6 — strict next-up

1. **B30 lint rule** — Unify-Intelligence-into-BeliefState; bans
   intelligence streams (cognitive, news, trader) from writing to the
   meta-controller except via BeliefState.
2. **Wave-News-Fusion PR-1** — `news_projection.py`, deterministic
   NewsItem → BeliefState delta.
3. **Wave-News-Fusion PR-2** — `EventGuard`, Hazard.NEWS_SHOCK during
   high-urgency events.
4. **Resolve Rust dual-backend** — delete crates + ship
   `tools/rust_revival_reminder.py` (CI warns at day 25 of 30, opens
   GitHub issue at day 30, Devin schedule as backstop).
5. **Wave-04.5 PR-3 (BLS HTTP)** — last ingestion adapter.
6. **Wave-Stress-Tests** — adversarial suite per reviewer #3 §4.

---

*Document end. Anchor for next session: `docs/wave_04_6_plan.md`.*
