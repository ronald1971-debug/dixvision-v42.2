"""CI entry point for the authority matrix lint.

Loads ``registry/authority_matrix.yaml`` through the strict loader.
The loader itself raises on any structural inconsistency (unknown
actor refs, missing precedence coverage, illegal override edges,
etc.). This script wraps the load in a CI-friendly process exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from system_engine.authority import load_authority_matrix

_DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "registry" / "authority_matrix.yaml"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else _DEFAULT_REGISTRY
    try:
        matrix = load_authority_matrix(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"authority_matrix_lint: FAIL — {exc}", file=sys.stderr)
        return 1

    print(
        f"authority_matrix_lint: 0 violations "
        f"({len(matrix.actors)} actors, "
        f"{len(matrix.conflicts)} conflicts, "
        f"{len(matrix.overrides)} overrides)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
