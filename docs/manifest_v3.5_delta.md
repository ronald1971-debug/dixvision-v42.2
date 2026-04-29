# DIX VISION v42.2 — MANIFEST v3.5 DELTA

> Additive delta over v3.4 (`docs/manifest_v3.4_delta.md`, PR #50).
> Introduces the **Source & Consumption Validation System** (SCVS),
> approved by operator review of the v3.4 build under
> "spec entropy is rising again" and the explicit
> "**FINAL RULE (NON-NEGOTIABLE)**" clause:
>
> > If a source is declared but not used → system is INVALID.
> > If a module uses data but does not declare it → system is INVALID.
>
> v3.5 is **additive only**. Build Compiler Spec §1.0–§1.1 freeze
> rules apply: no engine renames, no domain collapses, no module
> removals. Every v3.5 node sits inside the `system_engine` boundary
> (registry loader + lint), the `registry/` boundary (data declaration),
> or the `tools/` boundary (CI entry point).
>
> Resolution rule: **v3.5 wins over v3.4 wins over v3.3 wins over v3.2
> wins over v3.1 wins over v3** when in conflict.

---

## 0. WHY v3.5 EXISTS

The post-v3.4 architecture review surfaced four adjacent concerns:

| # | Concern | Status in v3.5 |
|---|---------|----------------|
| L1 | No source enforcement: external feeds (CEX, DEX, news, social, on-chain, macro, regulatory, dev, alt, AI, synthetic) are referenced in spec but never validated to be live, registered, or consumed | **Closed** by SCVS Phase 1 (this delta) |
| L2 | No multi-AI routing: ChatGPT / Gemini / Grok / DeepSeek / local models are admissible but no arbitration engine selects among them | Deferred — Cognitive Router Layer (CRL) follow-on |
| L3 | No single authority resolution table: precedence rules between governance, hazard interrupts, and the FastRiskCache live in 3+ files | Deferred — `authority_matrix.yaml` follow-on |
| L4 | No constraint compiler: invariants live in English (manifests), YAML (registry), Python (lint), Lean (proofs), and tests, with no single rule-graph compilation | Deferred — constraint engine follow-on |

v3.5 closes L1 only. The deferred items (L2–L4) are explicitly named
here so they cannot drift into "spec inflation > implementability."

---

## 1. THE NEW INVARIANT + RULES + ARTEFACTS

### 1.1 INV-57 — Source & Consumption Closure

**Statement.** Every external data input the system can ingest must be
declared exactly once in `registry/data_source_registry.yaml`. Every
module that consumes data must declare its inputs in a sibling
`consumes.yaml`. The relationship between the two is bidirectionally
closed:

* every `enabled: true` source is referenced by at least one
  `consumes.yaml` (no unused live source);
* every `source_id` in any `consumes.yaml` exists in the registry
  (no phantom consumption).

**Why.** Sources without consumers are dead weight that mask integration
gaps; consumers without registered sources hide undocumented external
dependencies. Either condition makes the runtime non-replayable
because the data dependency graph is incomplete.

**Determinism contract.** The registry + the set of `consumes.yaml`
files are the *only* sources of truth for what the runtime may read.
Adding or removing a source is a registry edit (governed via the
patch pipeline), not a code edit, so replay determinism (INV-15)
holds: same registry → same admissible sources.

### 1.2 SAFE-54 — `enabled: false` is a registration-only state

A source row with `enabled: false` is a *placeholder* for an adapter
that is not yet wired. SCVS-01 deliberately exempts these rows from
the "must be consumed" rule so that the registry can enumerate every
intended source up front without forcing every adapter to ship in the
same PR.

When a source flips to `enabled: true`, SCVS-01 immediately requires a
matching `consumes.yaml` entry — the lint catches the activation seam.

### 1.3 SAFE-55 — Bidirectional closure is a CI build-fail rule

`tools/scvs_lint.py` runs on every CI invocation
(`.github/workflows/ci.yml`). Any SCVS-01 / SCVS-02 violation exits
non-zero and fails the build before tests run.

### 1.4 SAFE-56 — Phase 1 surface is bounded; Phase 2/3 named

| Rule | Phase | Scope |
|------|-------|-------|
| SCVS-01 | Phase 1 | No unused live source |
| SCVS-02 | Phase 1 | No phantom consumption |
| SCVS-03 | Phase 2 | Runtime data-flow required (heartbeat) |
| SCVS-04 | Phase 3 | Schema enforcement on incoming packets |
| SCVS-05 | Phase 2 | Source liveness threshold |
| SCVS-06 | Phase 2 | Critical source fail-closed |
| SCVS-07 | Phase 3 | AI provider validation (latency + structure) |
| SCVS-08 | Phase 3 | Duplicate source detection |
| SCVS-09 | Phase 3 | Stale data rejection |
| SCVS-10 | Phase 3 | No silent fallback |

Phase 2 will introduce `system_engine/scvs/source_manager.py` with the
heartbeat + liveness state. Phase 3 will introduce schema-bound runtime
validators. Both will land as separate small PRs — see `build_plan.md`.

---

## 2. NEW ARTEFACTS

### 2.1 `registry/data_source_registry.yaml`

Skeleton enumerating every category from the v3.5 source matrix
(market CEX/DEX, news, social, on-chain, macro, regulatory, dev, alt,
AI, synthetic). Every row ships `enabled: false` until its adapter is
wired.

### 2.2 `system_engine/scvs/`

* `source_registry.py` — strict YAML loader → frozen `SourceRegistry`.
* `consumption_tracker.py` — strict `consumes.yaml` loader + recursive
  discovery walker.
* `lint.py` — pure SCVS-01 / SCVS-02 validator.
* `__init__.py` — public surface re-export.

### 2.3 `tools/scvs_lint.py`

CI entry point. Loads the canonical registry, walks the engine roots,
prints violations, exits non-zero on failure.

### 2.4 `consumes.yaml` files

Phase 1 ships zero in-tree `consumes.yaml` declarations because zero
sources are `enabled: true`. As adapters wire in, the consuming
module's `consumes.yaml` lands in the same PR.

---

## 3. SCOPE OF THIS DELTA

* **In:** registry skeleton, loader, tracker, lint, CI wiring,
  Phase-1 tests.
* **Out:** runtime liveness (Phase 2), schema enforcement (Phase 3),
  AI router (CRL follow-on), authority matrix, constraint compiler.
* **Unchanged:** every existing engine boundary, every existing
  invariant (INV-01..INV-56), every existing lint rule.
