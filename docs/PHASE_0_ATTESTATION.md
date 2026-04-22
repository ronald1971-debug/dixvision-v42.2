# Phase 0 Attestation (DIX VISION v42.2)

**Branch:** `devin/1776820789-phase0-audit`
**Scope:** Build Plan Phase 0 — bootstrap + contract lock + immutable core.
**Runtime guard:** `tests/test_phase0_attestation.py` (9 tests, all green).

## Deliverables (Build Plan §1.1 – §1.4)

| # | Deliverable | File | Status |
|---|---|---|---|
| 1.1 | Immutable core: foundation.py + hash | `immutable_core/foundation.py`, `immutable_core/foundation.hash` | ✅ hash regenerated via `scripts/generate_hash.py` → `f0bb648a…` |
| 1.1 | Genesis record with actual hash | `immutable_core/genesis.json` | ✅ placeholder replaced with real hash |
| 1.1 | Kill-switch (stdlib only) | `immutable_core/kill_switch.py` | ✅ unchanged — already hardened |
| 1.1 | System identity | `immutable_core/system_identity.py` | ✅ unchanged |
| 1.1 | Safety axioms | `immutable_core/safety_axioms.lean` | ✅ expanded from TODO stub → S1..S10 specification |
| 1.1 | Hazard axioms (NEW) | `immutable_core/hazard_axioms.lean` | ✅ H1..H10 specification of SYSTEM_HAZARD contract |
| 1.1 | Neuromorphic axioms (NEW) | `immutable_core/neuromorphic_axioms.lean` | ✅ N1..N8 specification of sensory-only triad |
| 1.2 | Contracts directory | `core/contracts/*.py` | ✅ 8 existing + NEW `risk.py` (IRiskCache, IRiskConstraints, ISystemHazardEvent, IHazardEmitter, IGovernanceHazardSink) |
| 1.3 | Component registry + lock | `core/registry.py` + `bootstrap_kernel.py` | ✅ `registry.lock()` now called at the end of the boot sequence (Step 11) — post-boot registration denied |
| 1.4 | Bootstrap kernel | `bootstrap_kernel.py` | ✅ foundation check, governance boot gate, ledger init, registry lock, BOOT_COMPLETE audit event |

## Gaps found + resolved

1. **`foundation.hash` was stale.** Recorded `c43785d6…`; actual SHA-256 of `foundation.py` is `f0bb648a…`. Regenerated.
2. **`genesis.json.foundation_hash` held a literal placeholder string** (`"RUN_scripts/generate_hash.py"`). Replaced with the real hex hash.
3. **`bootstrap_kernel.py` never called `registry.lock()`.** Lock method existed in `core/registry.py` but was unwired — a rogue component could register factories at runtime, bypassing the Phase 0 contract lock. Fixed: Step 11 of the boot sequence now locks the registry after all components are resolved.
4. **No risk contract.** Build Plan §1.2 requires a SYSTEM_HAZARD schema contract. Added `core/contracts/risk.py` with four `@runtime_checkable` Protocols (`IRiskCache`, `IRiskConstraints`, `ISystemHazardEvent`, `IHazardEmitter`, `IGovernanceHazardSink`).
5. **`safety_axioms.lean` was an empty TODO.** Expanded into the S1..S10 specification. (A Lean4 proof encoding is a future phase; the text form is the authoritative spec the runtime must obey.)
6. **No hazard axioms.** Added `hazard_axioms.lean` (H1..H10) codifying the SYSTEM_HAZARD channel contract (single channel, Dyon sole producer, governance sole consumer, non-blocking, overflow fails closed, severity→response mapping, ledger durability, two-person override gate).

## Neuromorphic triad — Phase 0 scope

Per the operator's locked rule ("Neuromorphic components may observe, detect, and advise. They may never decide, execute, or modify system state. Their outputs are events. Their models are immutable at runtime. Their existence is audited."):

- `immutable_core/neuromorphic_axioms.lean` — N1..N8 axioms.
- `docs/NEUROMORPHIC_TRIAD_SPEC.md` — full spec.
- `mind/plugins/neuromorphic_signal.py` — Indira-side microstructure sensor stub (rule-based detector emits `SPIKE_SIGNAL_EVENT`; Phase 2 replaces with SNN).
- `execution/monitoring/neuromorphic_detector.py` — Dyon-side anomaly sensor stub (rule-based detector emits `SYSTEM_ANOMALY_EVENT`; Phase 3 replaces with LSM; dead-man `check_self()` wired per N5).
- `governance/signals/neuromorphic_risk.py` — Governance-side risk-acceleration sensor stub (rule-based detector emits `RISK_SIGNAL_EVENT`; Phase 4 replaces with SNN over risk features; advisory-only per N7).
- `tools/authority_lint.py` — NEW rule **C2** forbids the three neuromorphic files from importing any of `governance.kernel`, `governance.policy_engine`, `governance.constraint_compiler`, `governance.mode_manager`, `governance.patch_pipeline`, `mind.fast_execute`, `execution.engine`, `execution.adapter_router`, `execution.adapters`, `security.operator`, `security.wallet_policy`, `security.wallet_connect`, `core.registry`. Clean: 0 violations.
- `tests/test_neuromorphic_triad.py` — 12 tests covering event emission, ledger audit (N4), dead-man (N5), forbidden-imports static scan (N1/N6), and stable type tuples.

## Regression guards (runtime-enforced)

`tests/test_phase0_attestation.py` (9 tests, all green):

- `test_foundation_hash_file_matches_foundation_py` — SHA-256 drift fails the suite.
- `test_genesis_json_foundation_hash_matches_hash_file` — placeholder drift fails.
- `test_safety_axioms_lean_has_content` — S1..S10 must be present.
- `test_hazard_axioms_lean_exists_with_h1_h10` — H1..H10 must be present.
- `test_contracts_module_exports_risk_protocols` — `IRiskCache`, `IRiskConstraints`, `ISystemHazardEvent`, `IHazardEmitter`, `IGovernanceHazardSink` exported.
- `test_hazard_event_satisfies_system_hazard_contract` — `HazardEvent` is `isinstance(ISystemHazardEvent)`.
- `test_risk_cache_satisfies_risk_contract` — `FastRiskCache` is `isinstance(IRiskCache)` and returns `IRiskConstraints`.
- `test_registry_lock_prevents_post_boot_registration` — locked registry rejects new factories.
- `test_bootstrap_kernel_calls_registry_lock` — wiring-regression static guard on `bootstrap_kernel.py`.

## Simplification notes (zero behaviour change)

Candidates identified during Phase 0 audit. None applied in this PR — each requires its own dedicated removal PR with a test proving the code path is unused.

1. **Duplicate foundation env-var reads.** `immutable_core/foundation.py` honours `DIX_STRICT_INTEGRITY` and `bootstrap_kernel.py:50` reads `env == "prod"`. The two flags can collide. Proposal: canonical path is `FoundationIntegrity(_strict=...)`; bootstrap stops its own env check.
2. **`_stub()` helper in `core/registry.py`** is only triggered by `resolve()`; the call graph has no producer that both locks the registry and then asks for an unregistered component. Proposal: test-gated removal or convert to explicit `KeyError`.
3. **Two places instantiate `StateManager`** in tests (`test_round10_fixes.py::...halt` and `...safe_mode`). Candidate for a `pytest` fixture to shrink duplication.
4. **`cockpit/llm.py`** — AI router has an overlap between `ollama_local_enabled` + `ollama_local` (two flags, one effect). Not Phase 0 scope but logged for Phase 7.

## Optimization notes

Full distillation lives in `docs/OPTIMIZATION_PLAN.md`. Headlines (2026 production-grade targets):

- **Python 3.13 free-threaded (no-GIL)** for true-parallel background workers (ledger writer, weekly scout, audit logger). Opt-in via `python3.13t`; single-writer hot path stays single-thread.
- **msgspec** for cockpit JSON serialization (reported ~3.3× faster than Pydantic v2; strict schemas replace ad-hoc dicts).
- **orjson** for ledger event payload encoding (3–5× faster than stdlib `json`).
- **LMAX Disruptor pattern** for the hazard bus (ring-buffer, single-writer, lock-free read) — target > 1M events/sec sustained.
- **Polars + Arrow** for backtest / forward-test data processing (reported 5–9× faster joins vs pandas; zero-copy to numpy for the SNN feature extractor).
- **mmap'd state snapshots** for crash-recovery checkpoint rather than pickle; deterministic re-open.
- **pyinstrument + scalene** as baseline profiling harness; hot-path regressions caught in CI before merge.

None of these are implemented in Phase 0. Each phase PR will add baseline numbers for the functions it touches; the actual switch to the 2026 stack is a later phase once the manifest is complete and measurable.

## Attestation

Phase 0 deliverables are verified and locked in this PR:

- Foundation hash and genesis hash agree.
- Safety + hazard + neuromorphic axioms specs present.
- Risk contracts exported from `core.contracts`.
- Registry lock wired into bootstrap sequence.
- Neuromorphic triad stubs + axioms + authority_lint rule + test suite.
- Simplification + optimization notes captured.

Ready to proceed to Phase 1 (ledger + memory audit) once this PR merges.
