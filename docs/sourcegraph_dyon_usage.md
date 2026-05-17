# Sourcegraph `sg` CLI — Dyon System Intelligence Usage Reference

**Scope:** C-27 (OFFLINE_ONLY tool). Hermetic stdlib-AST analyzer with a
Sourcegraph-shape CLI surface. No Sourcegraph server is run in
production; this is a developer + CI tool only.

**Adapter:** <ref_file file="/home/ubuntu/repos/dixvision-v42.2/tools/codebase_intelligence.py" />

## When to reach for it

Dyon system intelligence flows use this tool when they need
codebase-wide answers that authority_lint cannot give:

1. **Cross-reference search** — "who calls `create_execution_intent`?"
2. **Authority violations** — "find every place a RUNTIME_SAFE symbol
   calls into an OFFLINE_ONLY symbol".
3. **Dependency graph** — "what does `governance_engine` actually
   import?" (full edge list, not just rule violations).

For lint enforcement, prefer `tools/authority_lint.py`. For
codebase-intelligence queries, use this tool.

## CLI shape (Sourcegraph parity)

| `sg` command        | DIX adapter equivalent                      |
|---------------------|----------------------------------------------|
| `sg search --refs`  | `CodebaseIntelligence.find_refs(symbol=…)`   |
| `sg code-nav refs`  | `CodebaseIntelligence.find_callers(callee)`  |
| `sg search --syms`  | `CodebaseIntelligence.symbol_search(query=…)`|
| `sg dependency-graph` | `CodebaseIntelligence.dependency_graph()` |
| (custom)            | `CodebaseIntelligence.authority_violations(tier_map=…)` |

## Example — Dyon "who calls governance" probe

```python
from tools.codebase_intelligence import CodebaseIntelligence

ci = CodebaseIntelligence(
    root="/home/ubuntu/repos/dixvision-v42.2",
    exclude=("tests/", ".venv/", "node_modules/"),
)
callers = ci.find_callers("create_execution_intent")
for site in callers:
    print(f"{site.module}:{site.location} {site.caller!r} → {site.callee!r}")
```

Every emitted row carries the **caller's bare name** + **callee's
fully-qualified attribute chain** + the **module-relative location**.
The output is sorted alphabetically so Dyon can diff successive runs.

## Example — tier-violation report

```python
from tools.codebase_intelligence import CodebaseIntelligence

ci = CodebaseIntelligence(root=".")
violations = ci.authority_violations(
    tier_map={
        "fast_execute": "RUNTIME_SAFE",
        "compose_patch": "OFFLINE_ONLY",
    }
)
for v in violations:
    print(
        f"{v.caller_module}::{v.caller_symbol} "
        f"({v.caller_tier}) → {v.callee_symbol} ({v.callee_tier})"
    )
```

## Authority constraints

* **OFFLINE_ONLY** — never wired to the hot path. CI / sandbox use only.
* **No clock, no PRNG, no network.** All reads are local filesystem
  walks; all output is deterministic.
* **No top-level Sourcegraph imports.** The optional
  `sg_binary_factory()` lazy-binds the `sg` CLI binary; the in-memory
  analyzer never touches it.
* **No `subprocess` calls** from the analyzer. The
  `sg_binary_factory` only locates the binary via `shutil.which` and
  returns a record; callers must subprocess it themselves.

## Determinism (INV-15)

* All result tuples are emitted in canonical sort order
  (alphabetical by symbol, then by location).
* Paths are emitted relative to the supplied `root` — absolute
  filesystem locations never leak into output.
* Three-run byte-identical replay pinned by tests.

## Optional live `sg` binary

`sg_binary_factory()` is the lazy seam that opts into the **live**
Sourcegraph CLI. Install via `npm install -g @sourcegraph/sg` or
`brew install sourcegraph/sg/sg`. Production deployment does **not**
ship `sg`; the in-memory analyzer is sufficient for CI.
