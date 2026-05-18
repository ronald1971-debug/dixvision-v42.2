"""Generate ``docs/build_status.md`` from a live filesystem walk.

PR-P0-4: the prior ``docs/build_status.md`` was generated against an
aspirational v3.3 ``docs/directory_tree.md`` and stuck at "131 on disk,
361 missing, 26% coverage" even though the codebase has since grown
past 1100 files implementing the v3.4 / v3.5.* / v3.6.* manifest
deltas. The old file is preserved at
``docs/archive/build_status_v3.3_stale.md``.

This module walks the actual repository tree (skipping vendor / build /
cache directories) and emits a fresh per-package implementation count
plus a static reconciliation table against the manifest delta chain
(v3.1 - v3.6.4). It is **not** invoked by the CI gate (strict gating
lives in ``tools/enforce.py``); it is a documentation regenerator only.

Run as a script:

    python tools/build_status_generator.py [--check]

``--check`` mode re-renders the doc in-memory and exits non-zero if the
file on disk differs (useful for a future pre-commit hook). Default
mode writes ``docs/build_status.md`` in place.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = frozenset(
    {
        ".git",
        ".github",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".venv",
        "venv",
        "target",
        "analysis",
        "dixvision.egg-info",
    }
)

SKIP_EXT = frozenset({".pyc"})


def _walk_repo(root: Path) -> dict[str, dict[str, int]]:
    """Walk ``root`` and group counts by ``top-level-package -> ext -> n``."""

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel = os.path.relpath(current, root)
        top = "<root>" if rel == "." else rel.split(os.sep)[0]
        for fname in files:
            if any(fname.endswith(ext) for ext in SKIP_EXT):
                continue
            ext = os.path.splitext(fname)[1] or "<noext>"
            counts[top][ext] += 1
    return counts


def _classify(top: str) -> str:
    """Map a top-level directory to its canonical role."""

    roles = {
        "core": "contracts",
        "contracts": "contracts",
        "immutable_core": "contracts",
        "system_engine": "runtime",
        "execution_engine": "runtime",
        "governance_engine": "runtime",
        "intelligence_engine": "runtime",
        "learning_engine": "offline",
        "evolution_engine": "offline",
        "sensory": "ingest",
        "simulation": "offline",
        "simulation_engine": "offline",
        "state": "runtime",
        "registry": "registry",
        "ui": "transport",
        "dashboard_backend": "transport",
        "dashboard2026": "frontend",
        "dash_meme": "frontend",
        "tools": "tooling",
        "scripts": "tooling",
        "tests": "tests",
        "docs": "docs",
        "enforcement": "runtime",
        "opponent_model": "runtime",
        "system": "compat-shim",
    }
    return roles.get(top, "other")


# ---------------------------------------------------------------------------
# Manifest delta reconciliation table — pinned to the v3.1 -> v3.6.4 chain.
# Each entry: (version, summary, canonical_paths, status_label).
# ``status_label`` is the human verdict given the live tree. Status values:
#   - "landed"     : at least one canonical path is present on disk
#   - "relocated"  : the surface exists but under a different path than the
#                    original delta declared (note the live path)
#   - "partial"    : surface exists but with named subpaths still pending
#   - "pending"    : nothing on disk matches the surface
# This table is hand-curated against the manifest delta chain and is the
# reconciliation evidence the previous build_status.md lacked.
# ---------------------------------------------------------------------------
_V31_SUMMARY = "System Intent + Opponent + Reflexive Sim + Genetics + Regret + Debate Round"
_V31_PATHS = (
    "core/coherence/system_intent.py",
    "opponent_model/behavior_predictor.py",
    "evolution_engine/genetic",
)
_V31_NOTES = (
    "PR #36; some files relocated (`system_intent_engine.py` ->"
    " `system_intent.py`, debate_round under `agents/`)."
)

_V32_SUMMARY = (
    "Meta-Controller fallback + Regime hysteresis + Entropy"
    " + agent_context + SimulationOutcome + Archetype lifecycle"
    " + PolicyEngine"
)
_V32_PATHS = (
    "intelligence_engine/meta_controller/policy/execution_policy.py",
    "registry/pressure.yaml",
    "registry/trader_archetypes.yaml",
    "intelligence_engine/strategy_runtime/archetype_lifecycle.py",
    "governance_engine/control_plane/policy_engine.py",
)
_V32_NOTES = (
    "PR #38 + Phase 6.T1a-e (PRs #40-#48). Regime hysteresis file"
    " folded into `intelligence_engine/regime/`."
)

_V33_SUMMARY = (
    "Shadow Meta-Controller + Coherence Calibrator + Reward audit"
    " + Agent introspection + Sim-realism tracker"
)
_V33_PATHS = (
    "intelligence_engine/meta_controller/policy/shadow_policy.py",
    "learning_engine/calibration/coherence_calibrator.py",
    "core/contracts/agent.py",
    "intelligence_engine/agents/_base.py",
)
_V33_NOTES = "PR #39 + Phase 6.T1c (PR #47) + wave-2 calibrator (PR #51)."

_V361_PATHS = (
    "system_engine/coupling/hazard_throttle.py",
    "system_engine/coupling/hazard_throttle_adapter.py",
)

DELTA_ROWS: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    ("v3.1", _V31_SUMMARY, _V31_PATHS, "landed", _V31_NOTES),
    ("v3.2", _V32_SUMMARY, _V32_PATHS, "landed", _V32_NOTES),
    ("v3.3", _V33_SUMMARY, _V33_PATHS, "landed", _V33_NOTES),
    (
        "v3.4",
        "Triad Lock (INV-56) — B20/B21/B22 lint rules + canonical pipeline doc",
        ("docs/canonical_pipeline.md", "tools/authority_lint.py"),
        "landed",
        "PR #50.",
    ),
    (
        "v3.5",
        "SCVS — Source & Consumption Validation System",
        ("registry/data_source_registry.yaml", "system_engine/scvs"),
        "landed",
        "PR #56.",
    ),
    (
        "v3.5.1",
        "SCVS Phase 2 — runtime source-liveness FSM (SCVS-03/05/06)",
        (
            "system_engine/scvs/liveness_fsm.py",
            "system_engine/scvs/consumption_tracker.py",
        ),
        "landed",
        "PR #57.",
    ),
    (
        "v3.5.2",
        "SCVS Phase 3 — per-packet schema/staleness guard + AI validator + silent-fallback audit",
        ("system_engine/scvs",),
        "landed",
        "PR #58 closes all 10 SCVS rules from the v1.0 spec.",
    ),
    (
        "v3.5.3",
        "Authority Matrix — single conflict-resolution table",
        ("registry/authority_matrix.yaml",),
        "landed",
        "PR #59.",
    ),
    (
        "v3.5.4",
        "Constraint Engine — single rule-graph oracle",
        ("registry/constraint_rules.yaml", "tools/constraint_lint.py"),
        "landed",
        "PR #60; runtime evaluator under `governance_engine/constraint/`.",
    ),
    (
        "v3.5.5",
        "Wave 5 — Strategic Execution (Almgren-Chriss scheduler)",
        ("execution_engine/strategic",),
        "landed",
        "PR #61.",
    ),
    (
        "v3.6.0",
        "BEHAVIOR-P2 — closed learning loop (Trade Result -> Score -> Adjust Weights)",
        ("learning_engine/loops/closed_loop.py",),
        "landed",
        "PR #62; loop folder is `learning_engine/loops/`, not `learning_engine/closed_loop/`.",
    ),
    (
        "v3.6.1",
        "BEHAVIOR-P3 — hazard-governance hard coupling",
        _V361_PATHS,
        "landed",
        "PR #63; hazard throttle lives under `system_engine/coupling/`.",
    ),
    (
        "v3.6.2",
        "BEHAVIOR-P4 — DecisionTrace per trade",
        ("core/contracts/decision_trace.py",),
        "landed",
        "PR #64.",
    ),
    (
        "v3.6.3",
        "BEHAVIOR-P5 — evolution wiring (5 patch-pipeline stages)",
        ("evolution_engine/patch_pipeline", "core/contracts/patch.py"),
        "landed",
        "PR #65; patch pipeline orchestrator + ledger surface.",
    ),
    (
        "v3.6.4",
        "Dashboard-2026 wave-01 — registry-driven AI providers + cognitive prep",
        ("dashboard2026", "intelligence_engine/cognitive"),
        "landed",
        "PR #69 + Wave-3 (PRs #82-#85).",
    ),
)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_per_package(counts: dict[str, dict[str, int]]) -> str:
    """Render the per-package count table."""

    lines: list[str] = []
    lines.append("| Package | Role | .py | .ts/.tsx | .md | .yaml | .json | other | total |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for pkg in sorted(counts, key=lambda p: (-sum(counts[p].values()), p)):
        c = counts[pkg]
        py = c.get(".py", 0)
        ts = c.get(".ts", 0) + c.get(".tsx", 0)
        md = c.get(".md", 0)
        ya = c.get(".yaml", 0) + c.get(".yml", 0)
        js = c.get(".json", 0)
        tot = sum(c.values())
        other = tot - py - ts - md - ya - js
        lines.append(
            f"| `{pkg}` | {_classify(pkg)} | {py} | {ts} | {md}"
            f" | {ya} | {js} | {other} | **{tot}** |"
        )
    return "\n".join(lines)


def _exists_any(paths: Iterable[str]) -> bool:
    return any((REPO_ROOT / p).exists() for p in paths)


def _render_manifest_delta_table() -> str:
    lines: list[str] = []
    lines.append("| Version | Surface | Status | Notes |")
    lines.append("|---|---|---|---|")
    for version, summary, paths, status, notes in DELTA_ROWS:
        verified = _exists_any(paths)
        marker = status if verified else "pending"
        path_hint = "`" + paths[0] + "`" if paths else ""
        lines.append(f"| `{version}` | {summary} ({path_hint}) | **{marker}** | {notes} |")
    return "\n".join(lines)


HEADER = """# DIX VISION v42.2 — Build Status (regenerated from live tree)

> Regenerated by ``tools/build_status_generator.py`` from a live
> filesystem walk plus a hand-curated manifest delta reconciliation
> (PR-P0-4). The prior v3.3-era doc, which claimed "131 on disk, 361
> missing, 26% coverage", is archived at
> [`docs/archive/build_status_v3.3_stale.md`](archive/build_status_v3.3_stale.md).
>
> **Important:** this file is **not** the CI gate. The strict gate
> lives in ``tools/enforce.py`` (PR-P0-3), driven by
> ``analysis/coverage_summary.json``. This doc is a contributor-facing
> reality view.
"""

INTRO = """## Why the old "74% gap / 361 missing" number was wrong

The previous ``docs/build_status.md`` cross-referenced ``docs/directory_tree.md``
(an architectural intent doc, v3.3) against the on-disk tree as it stood
around PR #39 (mid-v3.3). At that point only ~131 files matched the v3.3
intent paths verbatim, so the doc reported a 74 % gap.

Three things invalidated that number over the following 340+ PRs:

1. **Canonical relocations.** Several v3.3 path names were renamed before
   landing — e.g. `core/coherence/system_intent_engine.py` shipped as
   `core/coherence/system_intent.py` (PR #52); `intelligence_engine/
   meta_controller/evaluation/debate_round.py` shipped under
   `intelligence_engine/agents/debate_round.py`; `system_engine/
   hazard_throttle.py` shipped under `system_engine/coupling/
   hazard_throttle.py` (PR #63); `dashboard/` was renamed to
   `dashboard_backend/` (PR #106) with the React surface moving to
   `dashboard2026/`. The old generator counted each relocation as
   "missing".
2. **Manifest delta growth.** Manifest deltas v3.4–v3.6.4 (PRs #50, #56,
   #58–#65, #69, #82–#85 …) added ~14 new canonical surfaces, none of
   which were in the v3.3 tree the old generator measured against.
3. **No regenerator ran.** The doc was never re-emitted after the v3.3
   snapshot, so it kept reporting v3.3-era numbers forever.

PR-P0-4 replaces the doc with a live filesystem walk + an explicit
reconciliation against the manifest delta chain so contributors can
trust the numbers.
"""


def render(counts: dict[str, dict[str, int]], *, total: int, generated_at_utc: str) -> str:
    """Render the full ``docs/build_status.md`` body as a string."""

    package_count = len(counts)
    parts: list[str] = []
    parts.append(HEADER)
    parts.append(
        f"_Generated at {generated_at_utc} from the live filesystem walk._\n"
        f"_Top-level packages: **{package_count}**.  Tracked files: **{total}**."
        f"  (Walker skips: `.git`, `node_modules`, `__pycache__`, `.pytest_cache`,"
        f" `.ruff_cache`, `.mypy_cache`, `dist`, `build`, `.venv`, `target`,"
        f" `analysis`, `dixvision.egg-info`.)_\n"
    )
    parts.append(INTRO)
    parts.append("## Per-package implementation counts (live walk)\n")
    parts.append(_render_per_package(counts))
    parts.append("\n")
    parts.append(
        "**Role legend.** `runtime` = on the hot path;"
        " `offline` = offline learning/evolution lane;"
        " `ingest` = sensory adapters;"
        " `contracts` = frozen value-object surfaces;"
        " `transport` = HTTP/WebSocket surface;"
        " `frontend` = TS/TSX dashboards;"
        " `registry` = YAML manifests;"
        " `tests` = pytest suite;"
        " `docs` = markdown manifests;"
        " `tooling` = developer scripts;"
        " `compat-shim` = legacy re-export facades;"
        " `other` = ungrouped (root / egg-info).\n"
    )
    parts.append("## Manifest delta reconciliation (v3.1 — v3.6.4)\n")
    parts.append(
        "Each row is verified at render time against the live filesystem. Status\n"
        "is **landed** if at least one canonical path exists on disk; **pending**\n"
        "otherwise. Relocated / renamed paths are called out in the Notes column.\n"
    )
    parts.append(_render_manifest_delta_table())
    parts.append("\n")
    parts.append("## How to update this file\n")
    parts.append("Run ``python tools/build_status_generator.py`` from the repo root.\n")
    parts.append("CI does not regenerate this file automatically (yet); it is the\n")
    parts.append("contributor's responsibility to re-run the generator whenever they\n")
    parts.append("land a manifest delta or move a canonical surface. Pre-commit\n")
    parts.append("integration is tracked under PR-P0-4 follow-up.\n")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-render in memory and exit non-zero if the file differs.",
    )
    args = parser.parse_args(argv)

    counts = _walk_repo(REPO_ROOT)
    total = sum(sum(c.values()) for c in counts.values())
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # noqa: UP017
    body = render(counts, total=total, generated_at_utc=generated_at)

    target = REPO_ROOT / "docs" / "build_status.md"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current == body:
            return 0
        sys.stderr.write(
            "docs/build_status.md is stale. Regenerate with:\n"
            "  python tools/build_status_generator.py\n"
        )
        return 1

    target.write_text(body, encoding="utf-8")
    print(f"wrote {target.relative_to(REPO_ROOT)} ({total} files across {len(counts)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
