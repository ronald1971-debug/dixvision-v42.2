"""TOTAL VALIDATION enforcement — read coverage_summary.json, fail-or-pass.

Behaviour::

    python tools/enforce.py                # advisory: always exit 0
    python tools/enforce.py --strict       # exit 1 on any failure

In advisory mode (default during the bring-up window) the script prints
the failure surface so PR reviewers can see what the remediation
backlog looks like, but the build is **not** blocked. Once the backlog
is closed the CI workflow flips to ``--strict`` and the spec's
"100% or block" guarantee takes effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = REPO_ROOT / "analysis" / "coverage_summary.json"


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


def _check_failures(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("file_coverage", "feature_coverage", "source_coverage", "invariant_coverage"):
        if data.get(key) != "100%":
            failures.append(f"{key} = {data.get(key)} (expected 100%)")
    if not data.get("dependency_graph_valid"):
        failures.append("dependency_graph_valid = false")
    if not data.get("ast_validation"):
        failures.append("ast_validation = false")
    if not data.get("runtime_validation"):
        failures.append("runtime_validation = false")
    if int(data.get("dead_files", 0)) > 0:
        failures.append(f"dead_files = {data['dead_files']}")
    if int(data.get("unmapped_declarations", 0)) > 0:
        failures.append(f"unmapped_declarations = {data['unmapped_declarations']}")
    if int(data.get("ambiguity", 0)) > 0:
        failures.append(f"ambiguity = {data['ambiguity']}")
    if data.get("status") != "PASS":
        failures.append(f"status = {data.get('status')}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any failure (CI gate mode).",
    )
    args = parser.parse_args(argv)

    data = _load_summary()
    failures = _check_failures(data)

    if not failures:
        print("ALL CHECKS PASSED")
        return 0

    if args.strict:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    # advisory mode — surface the gaps without blocking
    print("ADVISORY: TOTAL VALIDATION found gaps:")
    for f in failures:
        print(f"  - {f}")
    print()
    print(
        "(advisory mode -- not blocking. flip to --strict once backlog is closed.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
