# Manifest delta — v3.6.3 (BEHAVIOR-P5 / evolution wiring)

This delta closes the structural-mutation half of the closed feedback
loop. The five Phase 4 patch-pipeline stages
(``sandbox`` → ``static_analysis`` → ``backtest`` → ``shadow`` →
``canary``) now compose into a single deterministic orchestrator, and
every transition in the pipeline emits a canonical ``SystemEvent``
row to the audit ledger so the dashboard's Strategy-Lifecycle widget
(``DASH-SLP-01``) and the eventual Indira / Dyon chat widgets can
render reviewable cards directly off the ledger.

The orchestrator is **glue, not policy** — it never decides whether a
patch is good. It calls the existing per-stage gates in canonical
order, funnels each gate's verdict back through the
``PatchApprovalBridge``, and short-circuits to ``REJECTED`` on the
first failing gate. Governance retains exclusive authority for every
``APPROVED`` / ``REJECTED`` / ``ROLLED_BACK`` transition (SAFE-69).

Nothing in this delta enables Evolution to mutate runtime engines on
its own — the sandbox / static-analysis / backtest / shadow / canary
gates still require caller-supplied evidence. Wiring a real sandbox
runner is a separate PR; this delta is the deterministic spine.

## New invariants

### INV-66 — Patch pipeline orchestrator is pure / deterministic / no I/O

The orchestrator
(:class:`evolution_engine.patch_pipeline.orchestrator.PatchPipelineOrchestrator`)
is a pure function of its inputs:

* No clock reads — every per-stage ``ts_ns`` is derived from the
  caller's base ``ts_ns`` plus a fixed integer offset per
  :class:`PatchStage`. Replays produce monotonically-increasing,
  byte-identical event timestamps.
* No PRNG, no IO, no mutation of caller-owned objects.
* The three new ``SystemEvent`` payloads
  (``PATCH_PROPOSED`` / ``PATCH_STAGE_VERDICT`` / ``PATCH_DECISION``)
  use ``json.dumps(sort_keys=True, separators=(",", ":"))`` so the
  resulting ledger rows are byte-identical across replays of the same
  inputs (consistent with INV-15 + INV-65).
* The reverse projections
  (``proposal_from_system_event`` / ``verdict_from_system_event`` /
  ``decision_from_system_event``) round-trip losslessly so the
  ledger surface is replay-faithful end-to-end.

Polyglot ports (Phase E9) MUST follow these exact projection rules
to stay replay-compatible across language boundaries.

## New safety rules

### SAFE-69 — Governance is the sole authority for terminal patch transitions

The orchestrator NEVER calls
:meth:`PatchPipelineProtocol.transition` directly for the terminal
stages (``APPROVED`` / ``REJECTED`` / ``ROLLED_BACK``). Every
transition into a terminal stage goes through
:class:`PatchApprovalBridge` so the governance bridge stays the only
entrypoint that mutates patch state to a terminal value
(Build Compiler Spec §1.1).

The orchestrator depends on
:class:`core.contracts.patch.PatchApprovalBridgeProtocol` — not the
concrete bridge class — so the L2 cross-engine seam (offline →
runtime imports forbidden) stays clean. Symmetric to the existing
``PatchPipelineProtocol`` discipline (governance → evolution).

## New module surface

```
core/contracts/
  patch.py                    # +PatchApprovalDecision  (was governance-only)
                              # +PatchApprovalBridgeProtocol  (new)

evolution_engine/patch_pipeline/
  events.py                   # NEW: pure projection helpers for the
                              #      three new SystemEventKind values
  orchestrator.py             # NEW: PatchPipelineOrchestrator,
                              #      StageEvidence, PatchPipelineRun
```

Three new ``SystemEventKind`` values are the **only** ledger surfaces
added; every existing event variant is unchanged.

| Sub-kind                | Source                                 | Emitted when                                     |
|-------------------------|----------------------------------------|--------------------------------------------------|
| ``PATCH_PROPOSED``      | ``evolution.patch_pipeline.proposal``  | A ``PatchProposal`` is registered with the bridge. |
| ``PATCH_STAGE_VERDICT`` | ``evolution.patch_pipeline.verdict``   | A pipeline stage records a verdict (one per stage executed). |
| ``PATCH_DECISION``      | ``governance.patch_pipeline.decision`` | A terminal decision is driven by the bridge.     |

The proto file (``contracts/events.proto``) has the matching enum
values 17 / 18 / 19 so the polyglot port (Phase E9) inherits the new
wire types mechanically. Devin Review's "proto must stay in sync with
Python" rule (raised on PR #64) is followed.

## Stage evidence contract

The orchestrator never *fabricates* gate inputs. The caller supplies
a single :class:`StageEvidence` payload covering all five gates:

| Field                    | Stage           | Used by                  |
|--------------------------|-----------------|--------------------------|
| ``sandbox_touchpoints``  | sandbox         | ``SandboxStage``         |
| ``static_findings``      | static_analysis | ``StaticAnalysisStage``  |
| ``backtest_summary``     | backtest        | ``BacktestStage``        |
| ``shadow_samples``, ``shadow_matches`` | shadow | ``ShadowStage`` |
| ``canary_orders``, ``canary_rejects``, ``canary_realised_pnl`` | canary | ``CanaryStage`` |

Missing or invalid evidence at any stage short-circuits the run to
``REJECTED`` with the failing stage's verdict surfaced in
:attr:`PatchApprovalDecision.reason` (e.g.
``"sandbox_failed:forbidden: subprocess.run"``).

## Test discipline

The closed-loop ``test_phase5_closed_loop.py`` suite (32 cases)
remains untouched. The new orchestrator + ledger surface ships with
``tests/test_patch_pipeline_orchestrator.py`` (17 new cases):

* Happy-path drives ``PROPOSED`` → ``APPROVED`` with 7 ledger events
  in canonical order.
* Each gate (sandbox / static / backtest / shadow / canary) has its
  own short-circuit-to-``REJECTED`` test.
* Replay byte-identical determinism for the full event tuple.
* Governance authority preserved
  (``bridge.decisions[0] is run.decision``).
* Lossless round-trip for all three new event projections.
* Duplicate ``patch_id`` rejection.

All linters (ruff, ``authority_lint``, ``scvs_lint``,
``authority_matrix_lint``, ``constraint_lint``) clean.
