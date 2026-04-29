# DIX VISION v42.2 — MANIFEST v3.4 DELTA

> Additive delta over v3.3 (`docs/manifest_v3.3_delta.md`, PR #39).
> Captures the **Triad Lock** invariant (INV-56) and the canonical
> pipeline document, approved under operator decisions on
> "Core Loop dominance" and "Governance Risk".
>
> v3.4 is **additive only**. Build Compiler Spec §1.0–§1.1 freeze
> rules apply: no engine renames, no domain collapses, no module
> removals, no event-type explosion. Every v3.4 node sits inside
> an existing engine boundary.
>
> Resolution rule: **v3.4 wins over v3.3 wins over v3.2 wins over
> v3.1 wins over v3** when in conflict.

---

## 0. WHY v3.4 EXISTS

v3.3 closed 5 self-correction gaps. Operator review of the post-v3.3
build surfaced two adjacent concerns that are not new spec but were
not yet **enforced in code**:

1. **Core loop dominance** — many subsystems exist, but no document
   names the canonical runtime pipeline as dominant. The reader has
   to reverse-engineer the loop from PRs #28 → #48.
2. **Governance risk** — INV-08 / INV-37 / B1 forbid governance from
   importing the execution engine, but no rule names the triad
   itself, and no rule prevents a non-execution module from
   *constructing* an `ExecutionEvent` directly.

The existing import-isolation rules (B1 / B7 / B17) are correct.
v3.4 makes the triad explicit at the rule level and adds two
construction-level locks so the invariant is enforced at the
event-creation seam, not only at the module-boundary seam.

| # | Concern | Layer | Failure mode if ignored |
|---|---------|-------|------------------------|
| K1 | Triad lock not named in rule set | Authority lint | A future contributor adds a "convenience" import from `governance_engine` into `execution_engine` — passes review because nobody cites the rule |
| K2 | `ExecutionEvent` constructor is open | Typed bus | A non-execution module synthesises a fill — the ledger looks identical to a real fill |
| K3 | `SignalEvent` constructor is open | Typed bus | A non-Indira module synthesises a signal — the meta-controller and reward shaping treat it as real |
| K4 | Canonical pipeline is not documented | Architecture | New contributors / agents cannot tell which sequence is dominant |

v3.4 closes K1–K4 with one new invariant + three lint rules + one
canonical-pipeline document. It changes no runtime behaviour.

---

## 1. THE 1 NEW INVARIANT + 3 LINT RULES + 1 DOC

### 1.1 INV-56 — Triad Lock

**Statement.** The runtime trades are produced by exactly three
engines, each with one role:

| Role           | Engine                                                  |
|----------------|---------------------------------------------------------|
| **Decider**    | `intelligence_engine` (Indira) — signals + meta-controller |
| **Executor**   | `execution_engine` — orders + fills + lifecycle FSM     |
| **System / Health** | `system_engine` (Dyon) — hazards / drift / state   |
| **Approver**   | `governance_engine` — approves / rejects / constrains;  |
|                | **never trades**                                        |

The triad is enforced at three seams:

* **Imports.** B1 (general) + B17 (shadow-policy) + **B20 (new)**
  forbid governance from importing any execution-engine surface.
* **Event construction.** **B21 (new)** restricts `ExecutionEvent(...)`
  construction to `execution_engine/**`. **B22 (new)** restricts
  `SignalEvent(...)` construction to `intelligence_engine/**` (with
  a single named exemption for the `ui/` dev harness).
* **Typed-bus emission.** Already enforced by INV-08 (cross-engine
  isolation) + the `SystemEvent` envelope rule from v3.

The triad lock explicitly does **not** add new authority surfaces,
new event kinds, or new runtime adapters. It is a **rule-level**
invariant only.

### 1.2 B20 — Triad Lock: Governance is order-blind

**Path:** `tools/authority_lint.py` (`_check_b20`).
**Spec ID:** TRIAD-LINT-01.

`governance_engine.*` may not import any `execution_engine.*`
surface. Complements B1 by giving the triad-lock case a named rule
with a triad-specific error message. B1 still fires too — both
messages are emitted on the same offending import for clarity.

### 1.3 B21 — Triad Lock: `ExecutionEvent` constructor lock

**Path:** `tools/authority_lint.py`
(`_check_triad_event_constructions`).
**Spec ID:** TRIAD-LINT-02.

A direct `ExecutionEvent(...)` call is permitted only inside
`execution_engine/**`. `tests/**` and `contracts/**` are exempt.
The check walks `ast.Call` nodes whose `func` is a plain `Name` with
`id == "ExecutionEvent"`; aliased / attribute-access imports are
out of scope for v3.4 (lift later if needed).

### 1.4 B22 — Triad Lock: `SignalEvent` constructor lock

**Path:** `tools/authority_lint.py`
(`_check_triad_event_constructions`).
**Spec ID:** TRIAD-LINT-03.

A direct `SignalEvent(...)` call is permitted only inside
`intelligence_engine/**` and inside the `ui/` developer harness
(`ui/server.py` exposes a synthetic-signal POST endpoint that flows
through Intelligence → Execution; this is not a production signal
producer, but moving it out is a follow-on refactor). `tests/**`
and `contracts/**` are exempt.

### 1.5 `docs/canonical_pipeline.md`

**Path:** `docs/canonical_pipeline.md` (new).
**Spec ID:** TRIAD-DOC-01.

A single-page architecture document that declares the canonical
runtime pipeline and the dominance rule. It is **non-normative
prose** that points at the locked invariants (INV-08, INV-15,
INV-17, INV-37, INV-48..56) for the actual contracts. It does not
define new behaviour.

---

## 2. SAFE-RULE TABLE

| ID       | Statement                                                   |
|----------|-------------------------------------------------------------|
| SAFE-51  | A non-execution module that constructs an `ExecutionEvent` directly is a B21 violation. There is no "convenience" exemption. |
| SAFE-52  | A non-decider module that constructs a `SignalEvent` directly is a B22 violation. The `ui/` dev-harness exemption is named and time-boxed. |
| SAFE-53  | `governance_engine` may not import any `execution_engine` surface. Both B1 and B20 fire on violation. |

---

## 3. WHAT v3.4 DOES NOT CHANGE

* No new `SystemEventKind` values.
* No new proto messages.
* No new YAML registry files.
* No new runtime modules.
* No changes to existing engine APIs.
* No changes to test fixtures or test contracts.
* No changes to dashboard widgets.
