# Manifest v3.5.3 — Authority Matrix (single conflict-resolution table)

This delta closes operator concern **L3** from the v3.5 critique:

> "🔧 1. One authority resolution table (mandatory). A single file:
> `authority_matrix.yaml` defining who wins in conflict, precedence
> order, allowed overrides, deterministic resolution rules."

It is the third of six remaining items in the locked sequence
`scvs-2 → scvs-3 → authority → compiler → crl → wave-5`.

## 0. Why v3.5.3 exists

INV-56 (the Triad Lock, PR #50) **structurally** prevented Governance
from constructing `SignalEvent` / `ExecutionEvent` directly (lint rules
B20 / B21 / B22). It did not, however, capture the rest of the
authority surface:

* What is the precedence between Indira and Dyon when both have
  something to say about the same tick?
* Which authority owns `mode_fsm` transitions vs. plugin lifecycle
  vs. AI-provider arbitration?
* Where are the "exceptional" override edges (e.g. operator KILL,
  hazard-driven LOCKED) and why are they legal?
* When does the ledger trump in-memory state?

Until v3.5.3 those answers lived across a manifest, three READMEs, and
several invariant rows. The matrix concentrates them into one file
that the CI lint validates on every push.

## 1. Specification deltas

### 1.1 INV-60 — single authority resolution table

> The control plane MUST have exactly one canonical authority matrix
> at `registry/authority_matrix.yaml`. Every authority that can express
> intent over state MUST be registered there. Conflict rows MUST name
> the winner explicitly or mark the row as `deferred` with a documented
> reason. Override edges MUST route through Governance.

The runtime never branches on free strings — every reference resolves
through `system_engine.authority.matrix.AuthorityMatrix`, which fails
the build if a referenced actor is unknown.

### 1.2 The matrix

| Section       | Purpose |
|---------------|---------|
| `actors`      | Every authority on the control plane (governance / intelligence / execution / system / learning / evolution / operator / ledger). Each carries `module`, `role`, and the invariants it owes. |
| `precedence`  | Total ordering used as a tie-breaker when no explicit conflict row matches. |
| `conflicts`   | 11 documented decision points (governance vs engine, hazard vs cache, source liveness vs cached data, schema vs runtime, operator vs strategy, mode FSM, plugin lifecycle, ledger vs state, silent-fallback attempt, signal direction, AI provider disagreement → deferred to CRL). |
| `overrides`   | 3 legal exceptional edges — operator kill switch, hazard → LOCKED, governance constraint on signal — all routed through Governance. |

### 1.3 Loader semantics

* Strict structural validation (every reference resolves; precedence
  covers every actor; no override goes around Governance).
* Pure / deterministic — no clock, no PRNG.
* `AuthorityMatrix.resolve(a, b)` returns the higher-precedence actor
  between two ids. Used in tests; runtime call sites will adopt it
  when the constraint compiler layer (next PR) lands.

### 1.4 Deferred rows

`CONF-05` (AI provider disagreement) is intentionally `deferred`. It
will be replaced with the arbitration policy the Cognitive Router
Layer codifies (majority vote / weighted confidence / governance
override). Recording the deferral keeps future readers from assuming
an implicit policy.

## 2. New artefacts

* `registry/authority_matrix.yaml` — the canonical matrix.
* `system_engine/authority/__init__.py` — public surface.
* `system_engine/authority/matrix.py` — strict loader + `AuthorityMatrix`,
  `AuthorityActor`, `ConflictRow`, `AuthorityOverride` immutable dataclasses.
* `tools/authority_matrix_lint.py` — CI entry point.
* `tests/test_authority_matrix.py` — 18 tests covering canonical-file
  invariants and 11 loader rejection paths.

## 3. CI integration

`.github/workflows/ci.yml` now runs `python tools/authority_matrix_lint.py`
between `tools/scvs_lint.py` and `pytest`. The lint exits non-zero on
any structural inconsistency.

## 4. Scope

### In

* The matrix file + loader + lint + tests.
* Documentation only (no engine yet calls `AuthorityMatrix.resolve` at
  runtime).

### Out (deferred, in committed order)

* **Constraint compiler layer** — next PR. Will compile INV / SAFE /
  HAZ / PERF rules into a single runtime-evaluable rule graph using
  the matrix as its authority oracle.
* **Cognitive Router Layer** — replaces `CONF-05` (deferred row).
* **Wave 5 — Strategic Execution** — Phase 10.6 Almgren-Chriss /
  market impact.

### Unchanged

* INV-56 / B20 / B21 / B22 (Triad Lock) — still the structural floor.
* SCVS Phase 1–3 surfaces.
* All other engines.
