"""SCVS lint CI entry point (Phase 1).

Validates the bidirectional closure between
``registry/data_source_registry.yaml`` and every ``consumes.yaml``
declaration in the repo. Exits non-zero on any SCVS-01 / SCVS-02
violation so CI fails the build.

Usage::

    python tools/scvs_lint.py [REPO_ROOT]
"""

from __future__ import annotations

import sys
from pathlib import Path

from system_engine.scvs import (
    discover_consumption_declarations,
    find_redundant_sources,
    load_source_registry,
    validate_scvs,
)

_DEFAULT_ROOTS: tuple[str, ...] = (
    "core",
    "execution_engine",
    "evolution_engine",
    "governance_engine",
    "intelligence_engine",
    "learning_engine",
    "sensory",
    "state",
    "system_engine",
    "ui",
)

REGISTRY_PATH = "registry/data_source_registry.yaml"


def main(argv: list[str]) -> int:
    repo_root = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()

    registry_path = repo_root / REGISTRY_PATH
    if not registry_path.exists():
        print(
            f"scvs_lint: source registry not found at {registry_path}",
            file=sys.stderr,
        )
        return 2

    registry = load_source_registry(registry_path)

    discover_roots = [
        repo_root / r for r in _DEFAULT_ROOTS if (repo_root / r).is_dir()
    ]
    declarations = discover_consumption_declarations(discover_roots)

    violations = validate_scvs(registry, declarations)
    warnings = find_redundant_sources(registry)

    if warnings:
        print(
            f"scvs_lint: {len(warnings)} warning(s) (SCVS-08 redundancy, non-fatal)",
            file=sys.stderr,
        )
        for w in warnings:
            print(f"  [{w.rule}] {w.detail}", file=sys.stderr)

    if violations:
        print(
            f"scvs_lint: {len(violations)} violation(s) found",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  [{v.rule}] {v.detail}", file=sys.stderr)
        return 1

    print(
        f"scvs_lint: 0 violations "
        f"({len(registry.sources)} sources, "
        f"{len(declarations)} consumes.yaml files, "
        f"{len(warnings)} warning(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
