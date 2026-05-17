"""TOTAL VALIDATION enforcement — gate on regression-floor thresholds.

Reads ``analysis/coverage_summary.json`` (produced by
``tools/total_validation.py``) and fails the build if any tracked
metric regresses past the floor / ceiling pinned in
:data:`THRESHOLDS` below.

Modes::

    python tools/enforce.py                # advisory: always exit 0
    python tools/enforce.py --strict       # exit 1 on any regression

Threshold philosophy
--------------------

The TOTAL VALIDATION spec aspires to "100% or block". Today's
codebase does not meet that bar — the remediation backlog
includes:

* declared features without an implementing file (feature
  coverage < 100%),
* invariants that are documented but not yet enforced in a
  ``tools/`` lint or a ``tests/`` assertion (invariant coverage <
  100%),
* dead files / unmapped declarations / ambiguous matches that
  still surface in Phase 4 / 7 / 8,
* legacy dependency cycles (Phase 9).

Rather than carry an "advisory" CI step that *never* blocks (the
state PR-P0-3 was filed against — see the issue body in PR #379),
this module pins each metric to today's actual value as a
**regression floor / ceiling**. Future PRs that improve a
metric tighten the floor in the same commit; PRs that regress a
metric trip the gate.

* For percentages we record a *floor*  — a future PR may not
  drop coverage below today's level.
* For counts we record a *ceiling*     — a future PR may not
  raise dead-file / unmapped-declaration / ambiguity / cycle
  counts above today's level.
* For booleans we record an *expected value*. ``ast_validation``
  is required (no parse errors anywhere); ``runtime_validation``
  is permitted to be ``False`` while ``analysis/runtime_logs.txt``
  is not produced by CI.
* ``status`` from Phase 12 is informational — the per-metric
  table below is the authoritative gate.

When tightening a floor, bump it by the actual delta — never
round up. The whole point of pinning to reality is that the
floor only ratchets in the improving direction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = REPO_ROOT / "analysis" / "coverage_summary.json"

# ---------------------------------------------------------------------------
# regression-floor thresholds — pinned to current reality on PR-P0-3
# (see analysis/coverage_summary.json on main at that point).
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, dict[str, Any]] = {
    # Percentages — floor: actual_pct must be >= floor.
    "file_coverage_pct_floor": {"kind": "pct_floor", "value": 100},
    "source_coverage_pct_floor": {"kind": "pct_floor", "value": 100},
    "feature_coverage_pct_floor": {"kind": "pct_floor", "value": 41},
    "invariant_coverage_pct_floor": {"kind": "pct_floor", "value": 96},
    # Counts — ceiling: actual_count must be <= ceiling.
    "dead_files_max": {"kind": "count_ceiling", "value": 29},
    "unmapped_declarations_max": {"kind": "count_ceiling", "value": 117},
    "ambiguity_max": {"kind": "count_ceiling", "value": 58},
    # Booleans — expected value.
    "ast_validation_required": {"kind": "bool", "value": True},
    # runtime_validation is False today because CI does not produce
    # ``analysis/runtime_logs.txt``. When that pipeline is added the
    # value here flips to True.
    "runtime_validation_required": {"kind": "bool", "value": False},
    # dependency_graph_valid is False today because Phase 9 reports
    # 2 legacy cycles (with 0 forbidden cross-domain edges). When
    # the cycles are unwound this flips to True.
    "dependency_graph_valid_required": {"kind": "bool", "value": False},
    # PR-RT-5: every declared runtime node must be statically wired
    # in ``ui/server.py`` or on the allowlist. Phase 12 of
    # ``tools/total_validation.py`` produces this boolean; under the
    # PR-RT-4 wiring every declared node is wired so this floor is
    # ``True`` immediately.
    "topology_drift_valid_required": {"kind": "bool", "value": True},
    # Phase 12 also exports a numeric drift count. Ceiling is 0 — any
    # silent runtime topology drift trips CI in strict mode.
    "topology_drift_count_max": {"kind": "count_ceiling", "value": 0},
}

# Map summary-json keys to the THRESHOLDS entry that gates them.
_GATES: tuple[tuple[str, str], ...] = (
    ("file_coverage", "file_coverage_pct_floor"),
    ("source_coverage", "source_coverage_pct_floor"),
    ("feature_coverage", "feature_coverage_pct_floor"),
    ("invariant_coverage", "invariant_coverage_pct_floor"),
    ("dead_files", "dead_files_max"),
    ("unmapped_declarations", "unmapped_declarations_max"),
    ("ambiguity", "ambiguity_max"),
    ("ast_validation", "ast_validation_required"),
    ("runtime_validation", "runtime_validation_required"),
    ("dependency_graph_valid", "dependency_graph_valid_required"),
    ("topology_drift_valid", "topology_drift_valid_required"),
    ("topology_drift_count", "topology_drift_count_max"),
)

_PCT_RE = re.compile(r"^(\d+)%$")


def _load_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        print(
            f"FAIL: {SUMMARY_PATH.relative_to(REPO_ROOT)} missing "
            "(run tools/total_validation.py first).",
            file=sys.stderr,
        )
        sys.exit(2)
    with SUMMARY_PATH.open(encoding="utf-8") as fp:
        return json.load(fp)


def _parse_pct(raw: Any) -> int | None:
    if isinstance(raw, str):
        match = _PCT_RE.match(raw.strip())
        if match:
            return int(match.group(1))
    return None


def _check_failures(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for summary_key, threshold_key in _GATES:
        spec = THRESHOLDS[threshold_key]
        kind = spec["kind"]
        expected = spec["value"]
        actual = data.get(summary_key)
        if kind == "pct_floor":
            actual_pct = _parse_pct(actual)
            if actual_pct is None:
                failures.append(
                    f"{summary_key} = {actual!r} (could not parse as N%)"
                )
            elif actual_pct < int(expected):
                failures.append(
                    f"{summary_key} = {actual} regressed below floor "
                    f"{expected}% (raise the floor when you improve it)"
                )
        elif kind == "count_ceiling":
            try:
                actual_int = int(actual)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                failures.append(
                    f"{summary_key} = {actual!r} (could not parse as int)"
                )
                continue
            if actual_int > int(expected):
                failures.append(
                    f"{summary_key} = {actual_int} exceeded ceiling "
                    f"{expected} (lower the ceiling when you improve it)"
                )
        elif kind == "bool":
            if bool(actual) != bool(expected):
                failures.append(
                    f"{summary_key} = {actual!r} (expected {expected!r})"
                )
        else:  # pragma: no cover -- defensive
            failures.append(f"{threshold_key}: unknown kind {kind!r}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any regression (CI gate mode).",
    )
    args = parser.parse_args(argv)

    data = _load_summary()
    failures = _check_failures(data)

    if not failures:
        print("ALL CHECKS PASSED -- coverage summary at or above thresholds.")
        return 0

    if args.strict:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("ADVISORY: TOTAL VALIDATION found regressions:")
    for f in failures:
        print(f"  - {f}")
    print()
    print(
        "(advisory mode -- not blocking. Re-run with --strict to gate CI.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
