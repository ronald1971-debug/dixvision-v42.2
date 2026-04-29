# Manifest v3.5.2 — SCVS Phase 3 (per-packet validation + silent-fallback audit)

This delta closes the remaining five SCVS rules from v3.5 §1.4 — schema enforcement (SCVS-04), AI provider validation (SCVS-07), duplicate-source warning (SCVS-08), per-packet stale-data rejection (SCVS-09), and silent-fallback audit (SCVS-10). After this delta, all ten SCVS rules from the v1.0 spec are implemented.

## 0. Why v3.5.2 exists

v3.5 (PR #56) closed the *bidirectional closure* between the source registry and per-module `consumes.yaml`. v3.5.1 (PR #57) closed *source liveness* — heartbeats, the LIVE/STALE FSM, and the `HAZ-13` critical-source escalation. Neither closed three remaining gaps in operator concerns L1.e–L1.g:

* **L1.e** — a packet from a *live* source could still be malformed (wrong shape) and silently flow into the engine.
* **L1.f** — AI provider responses (category `ai`) had no parity with on-the-wire validation, despite being first-class registered sources.
* **L1.g** — fallback substitutions could occur with no audit row, recreating the "silent failure" pattern the SCVS spec exists to forbid.

v3.5.2 closes all three.

## 1. Specification deltas

### 1.1 INV-59 — per-packet contract conformance

> Every packet attributed to a registered source MUST be validated against
> the contract declared in `data_source_registry.yaml` (column `schema`).
> Packets failing the contract MUST be rejected before any engine consumes
> them, and the rejection MUST be observable through the validator's return
> value (not a silent drop).

The validation is a pure function (`SchemaGuard.validate`) — caller-supplied `now_ns` and `packet_ts_ns`, no clock, no PRNG, INV-15 deterministic. Engines opt in by routing inbound packets through the guard before further processing.

### 1.2 SAFE-59 — staleness is a per-packet concern *and* a per-source concern

| Layer            | Rule    | Owner            | Trigger                                                 |
|------------------|---------|------------------|---------------------------------------------------------|
| Per-source       | SCVS-05 | `SourceManager`  | `now_ns − last_heartbeat_ns > liveness_threshold_ns`    |
| Per-packet       | SCVS-09 | `SchemaGuard`    | `now_ns − packet_ts_ns > max_age_ns`                    |

Both must be active for full coverage — the source-level FSM (Phase 2) catches a *quiet* upstream; the per-packet guard catches a *talkative-but-stale* upstream (e.g. a feed that keeps streaming yesterday's data). They are independent and deliberately overlapping.

### 1.3 SAFE-60 — AI providers escalate through the same `HAZ-13` seam

A failing **critical** AI source (rule SCVS-07) emits the *same* `HAZ-13` hazard the Phase 2 critical-source FSM uses (rule SCVS-06). This avoids splitting governance escalation across multiple hazard codes for what is, semantically, the same operator concern: "a critical source is not delivering data the system can use."

### 1.4 New `SystemEventKind` value

| Value | Numeric ID | Phase / Rule | Emitter |
|-------|-----------:|---------------|---------|
| `SOURCE_FALLBACK_ACTIVATED` | 15 | Phase 3 / SCVS-10 | Any engine that swaps to a fallback source — emitted via `system_engine.scvs.fallback_audit.make_fallback_event(...)` so the shape is canonical. |

### 1.5 Per-rule scope summary

| Rule    | Phase | Owning module                                     | Severity        |
|---------|------:|---------------------------------------------------|-----------------|
| SCVS-01 |     1 | `system_engine.scvs.lint`                         | BUILD-FAIL      |
| SCVS-02 |     1 | `system_engine.scvs.lint`                         | BUILD-FAIL      |
| SCVS-03 |     2 | `system_engine.scvs.source_manager`               | runtime FSM     |
| SCVS-04 |     3 | `system_engine.scvs.schema_guard`                 | per-packet gate |
| SCVS-05 |     2 | `system_engine.scvs.source_manager`               | runtime FSM     |
| SCVS-06 |     2 | `system_engine.scvs.source_manager`               | HAZ-13          |
| SCVS-07 |     3 | `system_engine.scvs.ai_validator`                 | per-call gate / HAZ-13 if critical |
| SCVS-08 |     3 | `system_engine.scvs.lint.find_redundant_sources`  | WARN (non-fatal)|
| SCVS-09 |     3 | `system_engine.scvs.schema_guard`                 | per-packet gate |
| SCVS-10 |     3 | `system_engine.scvs.fallback_audit`               | audit emitter   |

## 2. New artefacts

* `system_engine/scvs/schema_guard.py` — pure per-packet validator: source existence, enabled, packet shape, schema match, monotone timestamp, staleness threshold.
* `system_engine/scvs/ai_validator.py` — pure AI provider validator: latency, structure, empty-output detection, plus `HAZ-13` escalation for critical AI sources.
* `system_engine/scvs/fallback_audit.py` — single canonical constructor for `SOURCE_FALLBACK_ACTIVATED` events; rejects self-fallback and missing reasons.
* `system_engine/scvs/lint.py` — extended with `find_redundant_sources(...)` for SCVS-08 (WARN-only, non-fatal).
* `tools/scvs_lint.py` — surfaces SCVS-08 warnings alongside the SCVS-01 / SCVS-02 violations; warnings do not fail the build.
* `tests/test_scvs_phase3.py` — 29 new tests covering schema accept/reject paths, staleness, AI latency / structure / empty / non-AI category, critical vs non-critical hazard emission, duplicate detection, fallback-event guard rails, and replay determinism.

## 3. Scope

### In

* Pure validators for SCVS-04 / SCVS-07 / SCVS-09.
* Build-time WARN for SCVS-08 (does not fail CI).
* Canonical event constructor for SCVS-10.
* New `SystemEventKind.SOURCE_FALLBACK_ACTIVATED` + proto mirror.

### Out (deferred, in committed order)

* `authority_matrix.yaml` — single conflict-resolution table. Next.
* Constraint compiler layer (unify INV / SAFE / HAZ / PERF). After authority.
* Cognitive Router Layer — multi-AI routing + arbitration. After compiler.
* Wave 5 — Phase 10.6 Strategic Execution (Almgren-Chriss + market impact). After CRL.

### Unchanged

* Phase 1 + Phase 2 surfaces (`source_registry`, `consumption_tracker`, `lint`, `source_manager`).
* All other engines — Phase 3 modules are opt-in. No engine yet routes through them.
