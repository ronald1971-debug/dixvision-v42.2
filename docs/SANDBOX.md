# Sandbox-first patch pipeline

Manifest §15 requires shadow-trading validation before any code goes live.
DIX VISION enforces this as a **hard, gated pipeline** rather than prose:
every code change -- proposed by DYON, drafted by DEVIN, or submitted by
the operator -- must traverse the same stages, and the CI gate refuses to
merge a PR unless every changed `.py` file has a `PATCH_SANDBOX_PASS`
ledger event at the head SHA.

---

## Stages

```
    PROPOSED
       |
       v
    SANDBOX_IMPORT        isolated subprocess, -I -S, no network
       |
       v
    AUTHORITY_LINT        tools/authority_lint.py == 0 violations
       |
       v
    UNIT_TESTS            pytest -q on sandbox copy
       |
       v
    DEP_SCAN              typosquat / advisory block-list
       |
       v
    SHADOW_TEST           arbiter replay vs ledger tape (cold-path only)
       |
       v
    CANARY                gated small-size live window, auto-rollback
       |
       v
    GOVERNANCE_APPROVED   explicit human click in cockpit
       |
       v
    LIVE
```

Every stage emits a `GOVERNANCE/PATCH_*` ledger event; failures emit
`GOVERNANCE/PATCH_REJECTED` with the captured stderr (last 500 bytes).

---

## Who runs what

| Stage                 | Triggered by                            | Implementation                         |
|-----------------------|-----------------------------------------|----------------------------------------|
| SANDBOX_IMPORT        | CI on every PR + cockpit `/api/patch`   | `tools/sandbox_runner.sandbox_import`  |
| AUTHORITY_LINT        | CI + pipeline                           | `tools/authority_lint.py`              |
| UNIT_TESTS            | CI + pipeline (optional on PR file-scope) | `pytest -q`                          |
| DEP_SCAN              | CI + pipeline                           | `tools/sandbox_runner.sandbox_dep_scan`|
| SHADOW_TEST           | worker (cold path)                      | strategy arbiter replay                |
| CANARY                | operator in cockpit                     | 1% notional, <= 15 min window          |
| GOVERNANCE_APPROVED   | operator in cockpit                     | click + (for wallet changes) 2-person  |
| LIVE                  | post-merge                              | adapter registry / router              |

---

## CI enforcement

`.github/workflows/sandbox.yml` runs on every pull request:

```yaml
- name: sandbox pipeline (mandatory)
  run: |
    git fetch origin ${{ github.base_ref }} --depth=1 || true
    python -m governance.patch_pipeline --check-pr \
      --base origin/${{ github.base_ref }}
```

The CLI walks every `.py` file in `git diff --name-only base..HEAD`, runs
SANDBOX_IMPORT + AUTHORITY_LINT + DEP_SCAN on each, and exits non-zero on
the first failure. Protect the `main` branch with "Require status checks"
on the `sandbox-gate` job to make it merge-blocking.

---

## Hot-path exceptions (manifest §1 + §5)

`mind.indira.fast_execute_trade` and `risk.fast_risk_cache` are frozen-
function locked. They cannot be patched except through an explicit
`GOVERNANCE/FAST_PATH_AMENDED` event plus a hardware-key two-person
signature. The manifest file itself (`DIX VISION v42.2 ...`) is immutable;
only addenda ever land.

---

## Running the pipeline locally

```bash
# Single file:
python -m governance.patch_pipeline --path mind/strategies/my_new_strategy.py

# Every changed file in this branch:
python -m governance.patch_pipeline --check-pr --base origin/main

# Or the raw sandbox of one module (JSON output):
python tools/sandbox_runner.py mind/strategies/my_new_strategy.py
```

The sandbox runs the target under `python -I -S -W error` in a temp repo
copy with `HTTPS_PROXY=127.0.0.1:1` (outbound network denied) and a 60 s
wall-clock cap.
