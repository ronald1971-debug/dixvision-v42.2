# governance_engine/ — policy enforcement + audit (21 files)

## Purpose

INV-56 Triad Lock approver. The **only** authority that:

* mutates the `SystemMode` FSM (LOCKED / SAFE / PAPER / SHADOW /
  CANARY / LIVE / AUTO),
* approves `SignalEvent → ExecutionIntent`,
* signs the `governance_decision_id`,
* writes the audit ledger row,
* enforces drift / promotion / hazard policies.

Sub-packages:

* `engine.py` (491 lines) — `GovernanceEngine.process(...)` is the
  single mutator (B32 lint enforces).
* `control_plane/state_transition_manager.py` — Mode FSM single
  mutator (GOV-CP-03).
* `control_plane/promotion_gates.py` — hash-anchored
  SHADOW→CANARY→LIVE→AUTO (PR #124).
* `control_plane/drift_oracle.py` — continuous AUTO-mode gate
  (PR #125 + PR #145).
* `control_plane/policy_engine.py` — O(1) decision table (PR #55).
* `control_plane/operator_attention.py` — AUTO-mode oversight
  relaxation (PR #115).
* `control_plane/decision_signer.py` — HMAC governance-decision
  signature (PR #170/#171).
* `control_plane/policy_hash_anchor.py` — policy-hash drift sentry
  (PR #172, #173).
* `control_plane/update_validator.py` + `update_applier.py` — closed
  learning loop (PR #114).
* `harness_approver.py` — env-gated harness shim (PR #166, B33 lint).
* `strategy_registry.py` — governance-side strategy FSM (PR #113).
* `services/patch_pipeline_bridge.py` — orchestrates patch pipeline
  + ledger surface (PR #65).

## Wiring

* `ui/server.py` instantiates the engine, registers the FSM mutator
  with the ledger writer, and exposes `/api/mode/*`,
  `/api/intent/*`, `/api/promotion/*`, `/api/drift/*`,
  `/api/operator/consent`.
* The `B33`, `B35`, `B36` lint rules enforce that *no other module*
  may approve, propose, or mutate policy outside this engine.

## Static-analysis result

* 21 files, 17 with findings — **all 17 are ruff-format drift only**
  (FORMAT rule).
* No orphan modules. No semantic findings.

## Deep-read observations

* `engine.py` — `process(...)` returns `()` for HAZARD events
  because FSM mutation happens inside `_handle_hazard`. PR #173
  extended `PolicyDriftSentry` with an `on_hazard` callback so the
  harness can record hazards on the audit ring even when
  `process()` returns empty.
* `harness_approver.py` — `HARNESS_APPROVER_ENV_VAR` is set in
  `ui/server.py` at import time (line 114). Engines / adapters that
  *also* import this module trigger B33 lint. Verified clean.
* `control_plane/operator_attention.py` — relaxes oversight
  *thresholds*, never bypass. Always preserves the audit trail.
* `control_plane/promotion_gates.py` — gates are hash-anchored to
  the committed policy hash so a rebase that silently flips a gate
  cannot promote.

## Risks / gaps

* None blocking.
* The Paper-S5 wiring (external-signal cap into the governance gate)
  is the next planned change here. Currently SignalTrust is a
  **contract field only**; `external_signal_trust.yaml` exists but
  is not yet read by `governance_engine.engine.process` — that lands
  in Paper-S5 alongside the operator UI toggle (Paper-S6).

## Verdict

**HEALTHY.** This is the system's most heavily linted package. All
new authority gates land here.
