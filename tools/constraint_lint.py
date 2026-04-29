"""CI entry point: compile registry/constraint_rules.yaml and report.

Exits non-zero on any structural inconsistency (unknown owner, dangling
dependency, dependency cycle, unknown enum value, invalid ``when``
expression, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.constraint_engine import compile_rules

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES = REPO_ROOT / "registry" / "constraint_rules.yaml"
DEFAULT_MATRIX = REPO_ROOT / "registry" / "authority_matrix.yaml"


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    rules_path = Path(args[0]) if args else DEFAULT_RULES
    matrix_path = Path(args[1]) if len(args) > 1 else DEFAULT_MATRIX
    try:
        graph = compile_rules(rules_path, matrix_path=matrix_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"constraint_lint: {e}", file=sys.stderr)
        return 1
    n_with_pred = sum(1 for r in graph.rules if r.when_ast is not None)
    print(
        f"constraint_lint: 0 violations "
        f"({len(graph.rules)} rules, {n_with_pred} with 'when' predicates)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
