"""R-4 / Phase-6 audit fix — constraint_engine package import is acyclic.

Pins two contracts:

1.  ``core.constraint_engine.compiler`` must not import the
    ``core.constraint_engine`` package object via the facade
    ``from core.constraint_engine import ...``. The submodule must be
    referenced through its fully-qualified path so the static
    dependency graph records a dependency on the leaf module, not on
    the package. Anything else recreates the cycle
    ``core.constraint_engine ↔ core.constraint_engine.compiler``.

2.  After ``tools/total_validation.py`` regenerates
    ``analysis/dependency_graph.json``, that pair must no longer
    appear in the ``cycles`` list. (The other historical cycle
    ``ui.harness.boot_manager ↔ runtime_registrar ↔ ui.server`` is
    tracked separately by R-1 / C-11 and is not asserted here.)

Both lanes are byte-stable across runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# lane 1 — AST contract on compiler.py
# ---------------------------------------------------------------------------


def _iter_imports(source: str) -> list[ast.AST]:
    tree = ast.parse(source)
    return [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_compiler_does_not_import_via_package_facade() -> None:
    """The forbidden form ``from core.constraint_engine import ...`` must
    not appear in ``compiler.py`` — that's exactly what re-introduces
    the cycle."""
    src = (_REPO_ROOT / "core" / "constraint_engine" / "compiler.py").read_text(encoding="utf-8")
    bad = [
        node
        for node in _iter_imports(src)
        if isinstance(node, ast.ImportFrom) and node.module == "core.constraint_engine"
    ]
    assert bad == [], (
        "core/constraint_engine/compiler.py must not use "
        "`from core.constraint_engine import ...` — import the leaf "
        "submodule directly (e.g. `import core.constraint_engine.expr "
        "as expr_mod`) to avoid the package-facade cycle."
    )


def test_compiler_imports_expr_via_leaf_module() -> None:
    """The replacement form must use the fully-qualified submodule
    path so the dep graph records the leaf as the dependency."""
    src = (_REPO_ROOT / "core" / "constraint_engine" / "compiler.py").read_text(encoding="utf-8")
    leaf_aliases = [
        alias.name
        for node in _iter_imports(src)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "core.constraint_engine.expr"
    ]
    assert leaf_aliases, (
        "core/constraint_engine/compiler.py must import "
        "`core.constraint_engine.expr` via its fully-qualified module "
        "path (this is the form that breaks the dependency cycle)."
    )


# ---------------------------------------------------------------------------
# lane 2 — dependency-graph artefact
# ---------------------------------------------------------------------------


_FORBIDDEN_CYCLE = frozenset({"core.constraint_engine", "core.constraint_engine.compiler"})


def test_dependency_graph_has_no_constraint_engine_cycle() -> None:
    """Phase 9 of total_validation must not report the constraint
    engine cycle anymore.

    If the analysis artefact is missing this test is skipped — the
    CI workflow regenerates it on every run, but local clones may
    not have run the validator yet.
    """
    graph_path = _REPO_ROOT / "analysis" / "dependency_graph.json"
    if not graph_path.exists():
        pytest.skip("analysis/dependency_graph.json not generated yet")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    cycles = graph.get("cycles", [])
    for cycle in cycles:
        nodes = frozenset(cycle)
        assert nodes != _FORBIDDEN_CYCLE, (
            f"R-4 regression: constraint engine cycle reappeared: {cycle}. "
            "Check that core/constraint_engine/compiler.py still imports "
            "core.constraint_engine.expr via its fully-qualified path."
        )


# ---------------------------------------------------------------------------
# lane 3 — runtime equivalence (import works, expr_mod alias intact)
# ---------------------------------------------------------------------------


def test_runtime_expr_mod_alias_preserved() -> None:
    """The ``expr_mod`` alias used throughout compiler.py must still
    resolve to ``core.constraint_engine.expr`` at runtime — this is the
    byte-identical-behaviour guarantee of the fix."""
    import core.constraint_engine.expr as expected
    from core.constraint_engine import compiler

    assert compiler.expr_mod is expected


def test_runtime_compile_path_still_works() -> None:
    """A minimal end-to-end smoke test: ``compile_rules`` must still be
    importable and callable from the package facade."""
    from core.constraint_engine import RuleGraph, compile_rules

    assert callable(compile_rules)
    assert isinstance(RuleGraph, type)
