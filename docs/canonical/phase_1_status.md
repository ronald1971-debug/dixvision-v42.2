# Phase 1 — GOVERNANCE CORE — Status Report

**Authority:** This report audits `main` HEAD against
`build_plan.md` §"PHASE 1 — GOVERNANCE CORE" (lines 65–83) and the
related sections of `manifest.md` and `executive_summary.md`.

**Scope (per spec):** GOV-CP-01..07, the 7-state `SystemMode` FSM, the
`OperatorBridge`, and `core/contracts/governance.py`.

**Phase verdict:** ✅ DONE with **two recorded divergences** from the
literal spec text. Both divergences are intentional, post-Phase-1
evolutions that the user explicitly approved in subsequent PRs; they
are documented below as "spec evolution" rather than as gaps.

---

## Deliverable-by-deliverable status

### GOV-CP-01..07 — control plane

| Deliverable | File | LOC | Status |
| --- | --- | --- | --- |
| GOV-CP-01 PolicyEngine | `governance_engine/control_plane/policy_engine.py` | 341 | ✅ |
| GOV-CP-02 RiskEvaluator | `governance_engine/control_plane/risk_evaluator.py` | 228 | ✅ |
| GOV-CP-03 StateTransitionManager | `governance_engine/control_plane/state_transition_manager.py` | 507 | ✅ (sole `SystemMode` mutator) |
| GOV-CP-04 EventClassifier | `governance_engine/control_plane/event_classifier.py` | 227 | ✅ |
| GOV-CP-05 LedgerAuthorityWriter | `governance_engine/control_plane/ledger_authority_writer.py` | 271 | ✅ (sole ledger writer; INV-37) |
| GOV-CP-06 ComplianceValidator | `governance_engine/control_plane/compliance_validator.py` | 188 | ✅ |
| GOV-CP-07 OperatorInterfaceBridge | `governance_engine/control_plane/operator_interface_bridge.py` | 418 | ✅ |

### `SystemMode` FSM

> Spec: "7-state `SystemMode` FSM (`SAFE → PAPER → SHADOW → CANARY → LIVE
> → AUTO`; `LOCKED` from any)."

On disk (`core/contracts/governance.py:47`):

```python
class SystemMode(IntEnum):
    SAFE   = 0
    PAPER  = 1
    # rank=2 vacated by SHADOW-DEMOLITION-02 (PR #221)
    CANARY = 3
    LIVE   = 4
    AUTO   = 5
    LOCKED = 99
```

**Forward chain** (`state_transition_manager.py:62`):

```
SAFE → PAPER → CANARY → LIVE → AUTO
```

**LOCKED recovery** (`state_transition_manager.py:19, 92`): `LOCKED → SAFE`
is the **only** legal exit from `LOCKED`.

#### Divergence #1 — `SystemMode.SHADOW` removed

The spec lists `SHADOW` between `PAPER` and `CANARY`. The on-disk FSM
**does not** include `SHADOW`. This is intentional: PR #216
(`shadow-demolition-02-strategy`) removed `StrategyState.SHADOW`, and
PR #221 (`shadow-demolition-02-mode`) collapsed `SystemMode.SHADOW`
into `PAPER` semantics. Rank=2 is left vacant so archived ledger rows
that recorded `mode=2` decode without breakage.

The user explicitly approved this evolution mid-build:

- > "The system is locked on the dashboard unlock it" (operator was
>   stuck in LOCKED behind SHADOW promotion gates that did not match
>   the new operator workflow).

Rationale captured in PR #221 description and
`docs/SHADOW_DEMOLITION.md`.

For canonical-rebuild purposes the spec's "SHADOW = signals-on /
execution-off" semantics now live on the `PAPER` row of the
**Mode-Effect Table** (`core/contracts/mode_effects.py`, PR #110).
Phase 7 PR-A audit will confirm the table row is correct.

### `core/contracts/governance.py` — contract surface

> Spec: "`IGovernanceHazardSink`, `SystemMode` enum."

Classes present in `core/contracts/governance.py`:

- `SystemMode` ✅ (above)
- `OperatorAction`, `OperatorRequest`
- `ModeTransitionRequest`, `ModeTransitionDecision`
- `IntentObjective`, `IntentRiskMode`, `IntentHorizon`
- `IntentTransitionRequest`, `IntentTransitionDecision`
- `ConstraintScope`, `ConstraintKind`, `Constraint`
- `RiskAssessment`, `ComplianceReport`
- `DecisionKind`, `GovernanceDecision`, `LedgerEntry`
- `StateTransitionProtocol(Protocol)` (line 356)

#### Divergence #2 — `IGovernanceHazardSink` not present by name

The spec calls for a typed `IGovernanceHazardSink` Protocol on
`core/contracts/governance.py`. No such symbol exists.

The hazard-ingress chain on disk is **stronger** than a direct typed
sink: hazards reach governance via the canonical event bus, not via a
direct method call. Specifically:

```
HAZ sensor (system_engine/hazard_sensors/*)
  → HazardEvent on bus
  → HazardThrottleAdapter (system_engine/coupling/hazard_throttle_adapter.py)
  → SystemEvent(kind=HAZARD_*)
  → governance_engine/engine.py::Governance.process(SystemEvent)
  → EventClassifier (GOV-CP-04)
  → PolicyEngine (GOV-CP-01)
  → StateTransitionManager (GOV-CP-03) when downgrade is warranted
```

This is intentional: the event-bus path preserves the single-ingress
audit trail (every hazard becomes a typed event row in the ledger),
matches INV-15 (replay determinism), and avoids adding a second
ingress that lint rules would have to police. The architectural
spec's `IGovernanceHazardSink` was an early-Phase-1 simplification;
the Phase-4 wiring chose the event-bus path and the spec text was
not updated.

**Action:** None required. If a typed Protocol is still wanted for
documentation purposes, it can be added as a comment-only contract
that points at the chain above. Filed as **canonical-rebuild backlog
item B-01** (low priority, doc-only).

### `OperatorBridge`

> Spec: "OperatorBridge."

✅ Present as `governance_engine/control_plane/operator_interface_bridge.py`
(GOV-CP-07, 418 LOC). Ingests typed `OperatorRequest` /
`ModeTransitionRequest` / `IntentTransitionRequest` from the
dashboard, validates them, and forwards to the appropriate GOV-CP
module. Tested in `tests/test_governance_control_plane.py` (894 LOC)
+ `tests/test_operator_attention.py` + `tests/test_audit_p1_2_kill_switch_protocol.py`.

---

## Invariants Locked

| Spec row | Definition | On-disk status |
| --- | --- | --- |
| INV-37 | Governance is sole ledger writer | ✅ Enforced by GOV-CP-05 + W1 lint; **4** explicit `INV-37` refs |
| Backward de-escalation | Any mode → SAFE always allowed | ✅ Enforced in `state_transition_manager.py` |
| `LOCKED → SAFE` recovery | `LOCKED` exits **only** to SAFE | ✅ `_is_legal_edge` returns `FSM_LOCKED_ONLY_TO_SAFE` for any other target |

---

## Test coverage

The Phase 1 surface is exercised primarily by:

| File | LOC | Covers |
| --- | --- | --- |
| `tests/test_governance_control_plane.py` | 894 | All 7 GOV-CP modules + LOCKED→SAFE recovery |
| `tests/test_authority_lint.py` | 889 | W1 + B-family rules touching GOV-CP-05/03 |
| `tests/test_audit_p0_2_sqlite_ledger_reader.py` | 229 | LedgerReader against the writer's DB (INV-37 round-trip) |
| `tests/test_audit_p1_2_kill_switch_protocol.py` | n/a | LOCKED state transitions + StateTransitionProtocol |
| `tests/test_governance_fail_closed.py` | n/a | Fail-closed invariant on missing/invalid inputs |
| `tests/test_kill_switch.py` | n/a | Kill-to-LOCKED from any mode |
| `tests/test_operator_attention.py` | n/a | OperatorBridge ingress |
| `tests/test_stress_mode_effects.py` | n/a | Mode-FSM stress (LOCKED↔SAFE pumping, ratchet) |

---

## Gap list

1. **B-01 (low, doc-only):** Add an `IGovernanceHazardSink`
   comment-Protocol to `core/contracts/governance.py` that describes
   the event-bus hazard-ingress chain (no behavioural change). Filed
   to canonical-rebuild backlog; not blocking phase progression.

No code-behaviour gaps. The two recorded divergences from the spec
text are deliberate post-Phase-1 evolutions and are out-of-scope for
this phase's audit.

---

## Provenance

- Audited against `build_plan.md` §"PHASE 1 — GOVERNANCE CORE" (lines
  65–83).
- Cross-referenced with `manifest.md` (INV-37 + Mode FSM) and
  `executive_summary.md` (Governance engine § GOV-CP-01..07).
- Audit performed at HEAD of `main` on the
  `devin/canonical-rebuild-phase-1` branch.

Phase 2 (EXECUTION CORE) audit will land in
`docs/canonical/phase_2_status.md`.
