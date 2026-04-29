# Manifest v3.5.4 — Constraint Engine (single rule-graph oracle)

This delta closes operator concern **L4** from the v3.5 critique:

> "🔧 2. One constraint compiler layer. Something like:
> `/core/constraint_engine/`. That compiles invariants (INV),
> governance rules (GOV), safety rules (SAFE), performance constraints
> (PERF) → into ONE runtime-evaluable rule graph."

It is the fourth of six items in the locked sequence
`scvs-2 → scvs-3 → authority → constraint → crl → wave-5`.

## 0. Why v3.5.4 exists

Before this PR, rules existed in five surfaces:

* English in `docs/manifest_*.md`
* YAML in `registry/data_source_registry.yaml` and `consumes.yaml`
* Python branches inside each engine
* Lean/proof annotations in invariant specs
* Test names like `test_inv_15_*`

There was no single artefact that the runtime could query to ask "what
fires for these facts?". v3.5.4 introduces that artefact:

```
registry/constraint_rules.yaml  ─┐
registry/authority_matrix.yaml ─┼→ core.constraint_engine.compile_rules(...)
                                 └→ RuleGraph (immutable; .evaluate(facts))
```

The graph is a *pure oracle*. It does not mutate state, emit hazards,
or write to the ledger. Engines named in the authority matrix remain
the only actors that act — they query the graph for *which rules fire
for these facts*, and use that as input to their own decisions.

## 1. Specification deltas

### 1.1 INV-61 — single rule-graph oracle

> The control plane MUST have exactly one canonical constraint graph at
> `registry/constraint_rules.yaml`. Every INV / SAFE / HAZ / SCVS / GOV
> / PERF rule that the runtime evaluates MUST appear there. Each rule
> MUST declare its `owner` from the authority matrix, its `severity`
> and `action` from closed enumerations, and may declare a `when`
> predicate over a typed fact mapping. The graph MUST be a DAG.

### 1.2 The rule schema

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | Globally unique (e.g. `SCVS-04`, `INV-15`). |
| `kind` | yes | One of `INV`, `SAFE`, `HAZ`, `SCVS`, `GOV`, `PERF`. |
| `severity` | yes | One of `BLOCK`, `HIGH`, `WARN`, `AUDIT`. |
| `action` | yes | One of `REJECT`, `HALT`, `HAZARD_EMIT`, `WARN`, `AUDIT`. |
| `owner` | yes | Must resolve to an actor declared in `authority_matrix.yaml`. |
| `description` | yes | Human-readable. |
| `depends_on` | no | List of rule ids; must form a DAG. |
| `when` | no | Optional predicate over the fact mapping. |
| `notes` | no | Free-form. |

### 1.3 The expression DSL

The `when` clause uses a small, side-effect-free grammar (parser at
`core/constraint_engine/expr.py`):

```
expr ::= or_expr
or   ::= and ("or" and)*
and  ::= unary ("and" unary)*
unary::= "not" unary | atom
atom ::= "(" expr ")" | cmp
cmp  ::= operand op operand
op   ::= "==" | "!=" | "<" | "<=" | ">" | ">="
operand ::= number | identifier
```

Deliberate non-features:

* No function calls.
* No state reads (other than the supplied fact mapping).
* No unbounded loops or recursion.
* String facts compare with `==` / `!=` only; ordered ops require
  numeric facts.

Those restrictions are what let the compiler validate every rule
statically without ever evaluating it.

### 1.4 The compiler

`core.constraint_engine.compile_rules(...)` enforces:

* every `owner` resolves to an authority-matrix actor;
* every dependency points at a declared rule;
* the dependency graph is a DAG (Kahn's algorithm with deterministic
  ordering by rule id);
* `kind` / `severity` / `action` come from the closed enumerations;
* every `when` clause parses against the DSL.

Output is a frozen `RuleGraph` exposing:

* `rules`, `order`, `by_id` — direct introspection;
* `evaluate(facts)` — pure; returns rules whose `when` predicate fires,
  in topological order;
* `rules_owned_by(actor_id)` / `rules_of_kind(kind)` — convenience
  views used by lint/test tooling.

INV-15: pure, deterministic, no clock or PRNG.

## 2. New artefacts

* `registry/constraint_rules.yaml` — 26 rules across all six families,
  every owner cross-checked against the authority matrix.
* `core/constraint_engine/expr.py` — tokenizer + recursive-descent
  parser + pure evaluator + `free_idents(...)`.
* `core/constraint_engine/compiler.py` — `RuleKind` / `RuleSeverity` /
  `RuleAction` enumerations + `CompiledRule` + `RuleGraph` +
  `compile_rules(...)`.
* `core/constraint_engine/__init__.py` — public surface.
* `tools/constraint_lint.py` — CI entry point.
* `tests/test_constraint_engine.py` — 47 tests covering DSL parsing,
  evaluator truth tables, canonical-file invariants, and 14 loader
  rejection paths.

## 3. CI integration

`.github/workflows/ci.yml` now runs `python tools/constraint_lint.py`
between `tools/authority_matrix_lint.py` and `pytest`. The lint exits
non-zero on any structural inconsistency (unknown owner, dangling
dependency, dependency cycle, unknown enum value, invalid `when`
expression).

## 4. Scope

### In

* The rules file + compiler + DSL + lint + tests.
* INV-61 spec.
* Documentation only — engines do not yet route runtime decisions
  through the rule graph.

### Out (deferred, in committed order)

* **Cognitive Router Layer** — replaces the deferred AI-arbitration
  conflict row from v3.5.3.
* **Wave 5 — Strategic Execution** — Phase 10.6 Almgren-Chriss /
  market impact.
* **Engine adoption of `RuleGraph.evaluate`** — each engine wires it
  into its own decision path as it lands; no surface today.

### Unchanged

* Authority matrix (v3.5.3 + INV-60).
* Triad Lock (v3.4 + INV-56 + B20/B21/B22).
* SCVS Phase 1–3 surfaces.
* All other engines.
