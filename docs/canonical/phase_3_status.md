# Phase 3 — INDIRA (INTELLIGENCE ENGINE) — Status Report

**Authority:** Audits `main` HEAD against `build_plan.md`
§"PHASE 3 — INDIRA (INTELLIGENCE ENGINE)" (lines 110–127).

**Scope (per spec):** Signal pipeline, microstructure, strategy
runtime, learning interface.

**Phase verdict:** ✅ DONE. All 9 deliverables present, both
invariants (B1, INV-15) attributed and enforced.

**Out-of-scope reminder:** the build_plan.md Phase 3 section lists
**only** `microstructure_v1` as the intelligence plugin. The 31-layer
learning architecture, 300 trader archetypes, and 5 AGT-XX agents
that appear in the executive summary and full feature spec are
**Phase 10** scope, not Phase 3. They are tracked separately in
`docs/canonical/phase_10_status.md` (forthcoming).

---

## Deliverable-by-deliverable status

### Core pipeline + learning interface

| Deliverable | File | LOC | Status |
| --- | --- | --- | --- |
| IND-SP-01 signal pipeline | `intelligence_engine/signal_pipeline.py` | 119 | ✅ |
| IND-LI-01 learning interface | `intelligence_engine/learning_interface.py` | 164 | ✅ |
| Plugin package marker | `intelligence_engine/plugins/__init__.py` | 13 | ✅ |
| IND-L02 microstructure v1 | `intelligence_engine/plugins/microstructure/microstructure_v1.py` | 105 | ✅ |

### Strategy runtime (5 modules)

| Deliverable | File | LOC | Status |
| --- | --- | --- | --- |
| IND-ORC-01 orchestrator (regime+lifecycle gating) | `strategy_runtime/orchestrator.py` | 123 | ✅ |
| IND-SCH-01 scheduler (bar-aligned cadence) | `strategy_runtime/scheduler.py` | 82 | ✅ |
| IND-REG-01 regime detector (runtime regime tags) | `strategy_runtime/regime_detector.py` | 147 | ✅ |
| IND-SLM-01 strategy lifecycle FSM | `strategy_runtime/state_machine.py` | 212 | ✅ |
| IND-CFR-01 conflict resolver | `strategy_runtime/conflict_resolver.py` | 138 | ✅ |

`IND-SLM-01` had `SHADOW` removed from its FSM in PR #216
(`shadow-demolition-02-strategy`). The Phase-3 spec listed `SHADOW`
implicitly via the strategy lifecycle. Per the user's explicit
instruction, the strategy FSM now goes `PROPOSED → CANARY` directly.
This is the same divergence already recorded against Phase 1
(SystemMode) and is documented again here so the strategy-level
removal is auditable.

`IND-CFR-01` shipped with a Devin Review fix in PR #31:
`ConflictResolver` collapses balanced BUY/SELL to HOLD with default
`min_net_score=0` (BUG_0001). On disk this fix is live.

---

## Invariants Locked

| Invariant | Definition | Refs | Enforcement |
| --- | --- | --- | --- |
| B1 | No engine-to-engine imports; only `core/contracts/` shared | **58** | `tools/authority_lint.py` rule B1; one of the most heavily-tested rules |
| INV-15 | Signal pipeline is deterministic | **141** | `signal_pipeline.py` consumes only typed inputs; T1 chokepoint bans `time.time()` calls; replay determinism suite asserts identical inputs → identical outputs |

---

## Test coverage

The Phase-3 surface is exercised primarily through end-to-end suites
that exercise the engine path rather than per-module units:

| File | Covers |
| --- | --- |
| `tests/test_authority_lint.py` | B1 rule unit tests |
| `tests/test_authority_symmetry.py` | B27/B28 + B1-symmetric authority |
| `tests/test_decision_trace.py` | DecisionTrace contract (downstream of signal pipeline) |
| `tests/test_decision_trace_why_layer.py` | Why-Layer references back to signal pipeline outputs |
| `tests/test_execution_engine_learning_loop.py` | Learning interface (IND-LI-01) round-trip |
| `tests/test_audit_wire_3_feedback.py` | FeedbackCollector + IND-LI-01 wiring (PR #192) |
| `tests/test_b30_belief_state_unify.py` | B30 — single BeliefState ingress for intelligence |
| `tests/test_governance_control_plane.py` | Strategy-state FSM transitions through governance |

---

## Spec-vs-disk divergence — out-of-scope reminder

The build_plan.md Phase 3 deliverable list ends at
`conflict_resolver.py` (line 123). It does **not** list the following
features that appear in `executive_summary.md` and
`full_feature_spec.md`:

- 31-layer learning architecture (FAISS RAG, RAL replay, neural HMM
  orderflow, MoE domain adaptation, federated learning, …) — **Phase
  10** scope.
- 300 trader archetypes (`registry/trader_archetypes.yaml`) — **Phase
  10** scope.
- AGT-XX agents (scalper/swing/macro/LP/adversarial) — **Phase 10**
  scope.
- Macro regime engine, portfolio brain, cross-asset coupling — **Phase
  10.5..10.8** scope.

This is correct phase isolation. The Phase 3 audit is clean. The
"only one plugin in `plugins/`" observation reported in earlier
direction-check messages is a Phase 10 gap, not a Phase 3 gap.

---

## Gap list

**None for Phase 3.** All 9 deliverables ship and both invariants are
enforced. Plugin-count concerns are filed against Phase 10.

---

## Provenance

- Audited against `build_plan.md` §"PHASE 3 — INDIRA (INTELLIGENCE
  ENGINE)" (lines 110–127).
- Cross-referenced with `manifest.md` (Indira engine §) and
  `executive_summary.md` (Intelligence engine § IND-SP/LI/L/ORC/SCH/REG/SLM/CFR).
- Audit performed at HEAD of `main` on the
  `devin/canonical-rebuild-phase-3` branch.

Phase 4 (DYON / SYSTEM ENGINE) audit lands in
`docs/canonical/phase_4_status.md`.
