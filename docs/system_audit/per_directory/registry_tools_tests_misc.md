# registry/, tools/, tests/, state/, scripts/, docs/, .github/

## registry/ — 10 files (YAML)

### Purpose

Single source of truth for runtime policy. Every entry is consumed
by at least one Python module — verified by
`docs/system_audit/_tools/registry_coverage.py` (0 orphans).

### Files

* `engines.yaml`, `plugins.yaml`, `data_sources.yaml`,
  `data_source_registry.yaml`, `mode_effects.yaml`,
  `promotion_gates.yaml`, `external_signal_trust.yaml` (Paper-S1),
  `causal_contract.yaml`, `agent_context_keys.yaml`, `pressure.yaml`.

### Verdict

**HEALTHY.** All 10 referenced. Hash-anchored via PolicyHashAnchor.

---

## tools/ — 8 files

### Purpose

Static linters that enforce architectural invariants at CI time:

* `authority_lint.py` — B1, B7-B36 rules (Triad Lock, dashboard
  authority, mode-mutation, no-implicit-approval, policy-mutation).
* `authority_matrix_lint.py` — INV-60 single-table conflict
  resolution.
* `constraint_lint.py` — INV-61 rule-graph oracle.
* `scvs_lint.py` — INV-57 bidirectional source/consumption closure.
* `rust_revival_reminder.py` — 30-day Rust-port reminder
  (workflow-based).

### Static-analysis result

* 8 files, 4 with findings — **all 4 are ruff-format drift only**.

### Verdict

**HEALTHY.** Lint coverage is broad and load-bearing.

---

## tests/ — 160 files

### Purpose

Pytest suite. 2704 tests pass (verified post-PR #183).

### Static-analysis result

* 160 files, 101 with findings — **all 101 are ruff-format drift
  only**. No semantic findings; no orphans.
* The orphan-module scanner classified 160 of these as `kind=test`
  (entry points discovered via pytest collection); none are unused.

### Verdict

**HEALTHY.** 2704-test green suite; drift-only formatting noise.

---

## state/ — 3 files

### Purpose

Persistence + memory tensor (canonical-tree skeleton; full
`memory_tensor/` and `databases/` queued).

### Static-analysis result

* 3 files, 1 with findings — ruff-format drift only.

### Verdict

**SKELETON.** Phase 10.13 + the full DB layer remain queued.

---

## scripts/ — 6 files

### Purpose

Operator scripts: credential check, Windows launcher (start, stop,
install_shortcut, install_shortcut_meme, start_meme).

### Static-analysis result

* 6 files, 1 with findings — ruff-format drift only.

### Verdict

**HEALTHY.** Launcher is the single boot surface.

---

## docs/ — 25 files

### Purpose

Manifests, deltas (v3.1..v3.6.4), CAUSAL_CONTRACT, build_status,
directory_tree, this audit.

### Verdict

**HEALTHY.** `build_status.md` is stale (last regenerated at PR
#172); replaced functionally by `build_plan_stage.md` in this audit.

---

## .github/ — 3 files

* `workflows/ci.yml` — lint + test + Devin Review hook.
* `workflows/dashboard2026.yml` — TS build.
* `workflows/rust_revival_reminder.yml` — 30-day reminder.

### Verdict

**HEALTHY.** Two CI checks (`lint-and-test` + `Devin Review`) gate
every PR.

---

## immutable_core/, enforcement/ — 3 files

* `immutable_core/{kill_switch.py, system_identity.py}` (PR #138).
* `enforcement/__init__.py` — re-export facade (PR #137).

### Verdict

**HEALTHY.** Skeleton consistent with the canonical tree; the
`safety_axioms.lean` / `hazard_axioms.lean` Lean-4 layer is
canonical-tree but deferred (Phase 0 stub).
