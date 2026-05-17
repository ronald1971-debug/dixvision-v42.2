# TOTAL VALIDATION SPEC

Authoritative, machine-enforced 13-phase audit of the v42.2 codebase
covering filesystem, declared features, invariants, dependency graph,
AST, runtime telemetry and runtime-topology drift. The spec ships in
**advisory mode** so the CI surface is non-blocking while the initial
remediation backlog is worked through; once that backlog is clean the
workflow flips to ``--strict``.

## Validity condition

System is VALID iff:

```
FILE_COVERAGE              == 100%
FEATURE_COVERAGE           == 100%
INVARIANT_COVERAGE         == 100%
SOURCE_COVERAGE            == 100%
DEAD_FILES                 == 0
UNMAPPED_DECLARATIONS      == 0
AMBIGUITY                  == 0
DEPENDENCY_GRAPH_VALID     == true
AST_VALIDATION             == true
RUNTIME_TELEMETRY_MATCH    == true
TOPOLOGY_DRIFT_VALID       == true
```

## Authoritative sources

| ID  | Type               | Path                                                  |
| --- | ------------------ | ----------------------------------------------------- |
| S1  | manifest           | `docs/manifest_v3.6.4_delta.md` (+ all delta history) |
| S2  | executive_summary  | `docs/system_audit/build_plan_stage.md`               |
| S3  | build_plan         | `docs/system_audit/build_plan_stage.md`               |
| S4  | directory_tree     | `docs/directory_tree.md`                              |
| S5  | registry           | `registry/`                                           |
| S6  | source_code        | repo root                                             |
| S7  | contracts          | `core/contracts/`                                     |
| S8  | invariants         | `tools/authority_lint.py`                             |
| S9  | tests              | `tests/`                                              |
| S10 | workflows          | `.github/workflows/`                                  |
| S11 | metrics            | `system_engine/metrics.py`                            |
| S12 | runtime_logs       | `analysis/runtime_logs.txt` (optional)                |

## Phases

| #   | Name                        | Artifact                            |
| --- | --------------------------- | ----------------------------------- |
| 0   | source_ingestion            | `analysis/source_index.csv`         |
| 1   | file_index                  | `analysis/file_index.csv`           |
| 2   | feature_extraction          | `analysis/feature_index.csv`        |
| 3   | file_analysis               | `analysis/tracking_table.csv`       |
| 4   | feature_coverage            | `analysis/feature_coverage.csv`     |
| 5   | source_coverage             | `analysis/source_coverage.csv`      |
| 6   | invariant_validation        | `analysis/invariant_coverage.csv`   |
| 7   | file_usage                  | `analysis/file_usage.csv`           |
| 8   | declaration_consistency     | `analysis/declaration_map.csv`      |
| 9   | dependency_graph            | `analysis/dependency_graph.json`    |
| 10  | ast_validation              | `analysis/ast_validation.json`      |
| 11  | runtime_telemetry           | `analysis/runtime_validation.json`  |
| 12  | topology_drift              | `analysis/topology_drift.json`      |
| 13  | summary                     | `analysis/coverage_summary.json`    |

## Phase 12 — topology drift (PR-RT-5)

Pins the Runtime Topology Authority invariant::

    declared_topology == active_topology ∪ DECLARED_BUT_DORMANT_ALLOWLIST

Every declared node in `ui.harness.runtime_registrar._DECLARED_NODE_SPECS`
must either be statically wired (its `state_attr` is assigned somewhere
in `ui/server.py` or `ui/harness/boot_manager.py`) or be on the
allowlist `DECLARED_BUT_DORMANT_ALLOWLIST` (which starts empty).

Any node that is declared, not statically wired and not on the
allowlist is **silent runtime topology drift** — the declared
architecture has diverged from what the harness actually constructs.
In strict mode this downgrades the pipeline to FAIL.

## Running

```bash
# advisory (default): writes artifacts, never blocks
python tools/total_validation.py
python tools/enforce.py

# strict: any gap → exit 1
python tools/total_validation.py --strict
python tools/enforce.py --strict
```

## CI integration

`.github/workflows/total_validation.yml` runs the advisory pipeline on
every push and pull request, uploads `analysis/` as a build artifact,
and renders `coverage_summary.json` into the PR step summary.

## Flipping to strict

Once the remediation backlog (declared-not-implemented features, dead
files, dependency-graph violations, missing invariant tests) is clean,
flip the two `python tools/...` lines in
`.github/workflows/total_validation.yml` to `--strict` and the build
will block on any future regression.

## What it currently surfaces

On the current `main`, advisory output looks like::

    feature_coverage      ~39%   (declared-not-implemented backlog)
    invariant_coverage    ~95%   (a handful of INV-XX still unwired)
    dead_files            ~26    (mostly system_audit/ generated dumps)
    unmapped_declarations ~116
    ambiguity             ~53    (tested but no source impl found via grep)
    dep_graph_valid       false  (cross-domain edges not yet legalised)

Each row in `analysis/declaration_map.csv` and
`analysis/feature_coverage.csv` is a direct remediation target.
