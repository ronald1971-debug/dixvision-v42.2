# Phase 0 — BOOTSTRAP CORE — Status Report

**Authority:** This report audits the on-disk state of `main` against the
Phase 0 deliverables and invariants enumerated in the canonical
`build_plan.md` (DIX VISION v42.2, "Full Extended Edition · No Omissions").
Where the on-disk implementation diverges from the spec, the divergence is
recorded here; the spec is the authority and any divergence becomes a
gap-closing item for a follow-up PR.

**Scope (per build_plan.md §"PHASE 0 — BOOTSTRAP CORE"):** core contracts,
ledger stub, registry, time source, event bus, six engine shells, full
authority-lint baseline rule set, CI workflow.

**Phase verdict:** ✅ DONE. All 9 deliverables are present and exercised by
tests; INV-15 (replay determinism) is enforced; the INV-01..14 bucket is
covered by the four families the spec calls out (engine sealing, ledger
append-only, governance-only mode writes, event provenance) — each via
the enforcement points listed below.

---

## Deliverable-by-deliverable status

### D1 — `core/contracts/engine.py` Protocols

> Spec: "`Engine` / `RuntimeEngine` / `OfflineEngine` / `Plugin` Protocols."

- ✅ Present at `core/contracts/engine.py` (220 LOC).
- All four `Protocol` classes exported:
  - `class Plugin(Protocol)`
  - `class Engine(Protocol)`
  - `class RuntimeEngine(Engine, Protocol)`
  - `class OfflineEngine(Engine, Protocol)`
- Instantiation and structural-typing tests live in
  `tests/test_engine_contracts.py` (120 LOC).

### D2 — Canonical 4-event bus (`events.py` + `events.proto`)

> Spec: "`SignalEvent` · `ExecutionEvent` · `SystemEvent` · `HazardEvent`,
> defined once in `core/contracts/events.py`, mirrored in
> `contracts/events.proto`."

- ✅ Present at `core/contracts/events.py` (339 LOC) and
  `contracts/events.proto` (184 LOC).
- Python: `class SignalEvent`, `class ExecutionEvent`,
  `class SystemEvent`, `class HazardEvent` (frozen dataclasses).
- Proto: `message SignalEvent`, `message ExecutionEvent`,
  `message SystemEvent`, `message HazardEvent` mirroring the Python
  definitions field-for-field.
- Round-trip tests cover the Python ↔ wire boundary.

### D3 — Six engine shells

> Spec: "Six engine shells (one per engine)."

| Engine | `__init__.py` | `engine.py` |
| --- | --- | --- |
| `intelligence_engine/` | ✅ | ✅ |
| `execution_engine/` | ✅ | ✅ |
| `system_engine/` | ✅ | ✅ |
| `governance_engine/` | ✅ | ✅ |
| `learning_engine/` | ✅ | ✅ |
| `evolution_engine/` | ✅ | ✅ |

All six shells present; each implements the `RuntimeEngine` /
`OfflineEngine` Protocol from D1.

### D4 — Declarative registry

> Spec: "`registry/engines.yaml`, `registry/plugins.yaml` — declarative
> truth."

- ✅ `registry/engines.yaml` (65 LOC) — engine declarations with tier and
  protocol assertions.
- ✅ `registry/plugins.yaml` (199 LOC) — plugin declarations.
- Loaded by `tools/authority_lint.py` so the `engines.yaml` / `plugins.yaml`
  declarations remain in sync with the on-disk Protocols.

### D5 — Authority lint baseline rule set

> Spec: "`tools/authority_lint.py` — full rule set (T1, C2, C3, W1, L1, L2,
> L3, B1)."

`tools/authority_lint.py` (1869 LOC) ships **all 8 baseline rules** plus the
later-phase rules added in subsequent PRs. Baseline coverage:

| Rule | Status |
| --- | --- |
| T1 — typed time-source chokepoint | ✅ |
| C2 — engine-cross-import bans | ✅ |
| C3 — domain-isolation contracts | ✅ |
| W1 — write-path single-writer | ✅ |
| L1 — engine sealing (no-private-cross-import) | ✅ |
| L2 — registry truth | ✅ |
| L3 — Protocol-bound shells | ✅ |
| B1 — typed-event-only bus | ✅ |

Beyond Phase 0, `authority_lint.py` already carries B7..B36 + C-family
extensions added during Phases 1–8. None of those are required for Phase 0
but they remain co-resident in this single linter file.

### D6 — Ledger reader stub

> Spec: "`state/ledger/reader.py` — ledger read stub."

- ✅ Present (215 LOC). Already evolved past "stub": ships
  `LedgerCursor`, `LedgerReader.read()`, `LedgerReader.tail()`,
  `authority_entries()`, `authority_count()`, plus a SQLite read-only
  connection helper. The Phase 0 spec asked for a stub; the on-disk
  implementation is an over-deliver that remains backward-compatible.

### D7 — `state/__init__.py`

- ✅ Present.

### D8 — Phase-0 tests

> Spec: "`tests/` — engine instantiation + lint rule unit tests."

- ✅ `tests/test_engine_contracts.py` (120 LOC) — engine-Protocol
  instantiation + structural-typing assertions.
- ✅ `tests/test_authority_lint.py` (889 LOC) — exhaustive per-rule unit
  tests for every rule listed in D5 (and beyond).

### D9 — CI workflow

> Spec: "`.github/workflows/ci.yml` — ruff + authority_lint + pytest."

- ✅ `.github/workflows/ci.yml` runs:
  - `ruff check .`
  - `python tools/authority_lint.py --strict .`
  - `pytest -q`

Workflow has been extended (build, codegen drift, total-validation
advisory) without dropping any of the three Phase 0 gates.

---

## Invariants Locked

The build_plan.md spec lists **two** Phase-0 invariant rows:

| Row | Spec text | On-disk status |
| --- | --- | --- |
| `INV-01..14` | "Engine sealing, ledger append-only, governance-only mode writes, event provenance" | ✅ Enforced via the four families below |
| `INV-15` | "Replay determinism — same input event sequence → same output event sequence" | ✅ Enforced; **183** explicit `INV-15` references across `core/`, `governance_engine/`, `system_engine/`, `intelligence_engine/`, `execution_engine/`, `tools/`, and `tests/` |

### INV-01..14 enforcement attribution

The spec rolls these into one bucket, naming **four families**. Each family
maps to a concrete enforcement point on disk:

1. **Engine sealing.** Enforced by `tools/authority_lint.py` rules **L1**
   (engine-private cross-import ban), **L2** (registry truth: only declared
   engines exist), **L3** (Protocol-bound shells), and the Protocol
   structure in `core/contracts/engine.py`. Tested in
   `tests/test_authority_lint.py` and `tests/test_engine_contracts.py`.
2. **Ledger append-only.** Enforced by
   `governance_engine/control_plane/ledger_authority_writer.py` (GOV-CP-05),
   the SQLite schema in `state/ledger/` (append-only table; sqlite primary
   key auto-increment), and lint rule **W1** (single-writer to
   `state/ledger/`). Tested in
   `tests/test_ledger_authority_writer.py`.
3. **Governance-only mode writes.** Enforced by
   `governance_engine/control_plane/state_transition_manager.py`
   (GOV-CP-03; sole `SystemMode` mutator) and lint rule **B32**
   (Mode-FSM single-mutator). Tested in
   `tests/test_state_transition_manager.py` and
   `tests/test_authority_lint_b32.py`.
4. **Event provenance.** Enforced by `produced_by_engine` on every typed
   event (`core/contracts/events.py` field + receiver assertions added in
   PR #80) and lint rule **B23/B24** (event-author identity). Tested in
   `tests/test_event_provenance.py` and
   `tests/test_authority_lint_b23.py`.

### INV-15 enforcement

`INV-15` is referenced **183** times across the runtime tier. Primary
chokepoints:
- `system/time_source.py` (T1 chokepoint; no `time.time()`/`time_ns()` in
  hot paths).
- `core/coherence/*` (BeliefState / PressureVector projections — pure
  functions of inputs).
- `governance_engine/control_plane/*` (deterministic FSM).
- `tests/test_replay_determinism.py` (input-equivalence acceptance suite).

---

## Gap list

**None.** Phase 0 is feature-complete relative to its build_plan.md spec.
Any future Phase-0 work (e.g. additional Protocol-class generics, lint-rule
hardening) belongs to a later phase and is recorded against that phase's
status doc.

---

## Provenance

- Audited against `build_plan.md` §"PHASE 0 — BOOTSTRAP CORE" (lines 44–62).
- Cross-referenced with `manifest.md` (invariants registry) and
  `executive_summary.md` (engine model § Architectural Invariants).
- Audit performed at HEAD of `main` on the
  `devin/canonical-rebuild-phase-0` branch.

When Phase 1 (GOVERNANCE CORE) audit completes its gap list will land in
`docs/canonical/phase_1_status.md` alongside this report.
