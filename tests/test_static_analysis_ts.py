"""Tests for C-26 tree-sitter enhanced static analysis."""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

from evolution_engine.patch_pipeline.static_analysis import (
    GOVERNANCE_BOUNDARY_PREFIXES,
    NEW_PIP_DEPENDENCIES,
    TREE_SITTER_ADAPTER_VERSION,
    ASTDiffEntry,
    ASTDiffKind,
    CallEntry,
    CrossFunctionCallAnalyzer,
    FindingSeverity,
    ImportEntry,
    ImportGraphExtractor,
    PatchSafetyAnalyzer,
    PatchSafetyReport,
    SemanticASTDiff,
    StaticAnalysisFinding,
    StaticAnalysisStage,
    tree_sitter_parser_factory,
)


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
def test_pip_deps_advertise_tree_sitter_pair() -> None:
    assert NEW_PIP_DEPENDENCIES == ("tree-sitter", "tree-sitter-python")


def test_adapter_version_is_string() -> None:
    assert TREE_SITTER_ADAPTER_VERSION == "1"


def test_governance_boundary_prefixes_are_sorted_tuple() -> None:
    assert isinstance(GOVERNANCE_BOUNDARY_PREFIXES, tuple)
    assert list(GOVERNANCE_BOUNDARY_PREFIXES) == sorted(GOVERNANCE_BOUNDARY_PREFIXES)
    assert "governance_engine/" in GOVERNANCE_BOUNDARY_PREFIXES


def test_ast_diff_kind_string_values() -> None:
    assert ASTDiffKind.ADDED.value == "ADDED"
    assert ASTDiffKind.REMOVED.value == "REMOVED"
    assert ASTDiffKind.CHANGED.value == "CHANGED"


# ---------------------------------------------------------------------------
# Existing StaticAnalysisStage preserved
# ---------------------------------------------------------------------------
def test_existing_stage_still_aggregates_findings() -> None:
    stage = StaticAnalysisStage(max_severity=FindingSeverity.WARN)
    findings = [
        StaticAnalysisFinding(rule="R1", severity=FindingSeverity.INFO, location="a.py:1"),
        StaticAnalysisFinding(rule="R2", severity=FindingSeverity.WARN, location="a.py:2"),
    ]
    verdict = stage.evaluate(ts_ns=1, findings=findings)
    assert verdict.passed is True
    assert "2 findings" in verdict.detail
    assert verdict.meta["findings"] == "2"


def test_existing_stage_fails_on_error_finding() -> None:
    stage = StaticAnalysisStage(max_severity=FindingSeverity.WARN)
    findings = [
        StaticAnalysisFinding(
            rule="R-FAIL",
            severity=FindingSeverity.ERROR,
            location="x.py:1",
        )
    ]
    verdict = stage.evaluate(ts_ns=2, findings=findings)
    assert verdict.passed is False


def test_existing_stage_passes_on_empty_findings() -> None:
    stage = StaticAnalysisStage()
    verdict = stage.evaluate(ts_ns=3, findings=[])
    assert verdict.passed is True
    assert "NONE" in verdict.detail


# ---------------------------------------------------------------------------
# SemanticASTDiff
# ---------------------------------------------------------------------------
def test_semantic_diff_detects_added_function() -> None:
    diff = SemanticASTDiff()
    before = "def f(): pass\n"
    after = "def f(): pass\ndef g(): pass\n"
    entries = diff.diff(before=before, after=after)
    assert len(entries) == 1
    assert entries[0].kind is ASTDiffKind.ADDED
    assert entries[0].symbol == "g"
    assert entries[0].symbol_kind == "function"


def test_semantic_diff_detects_removed_function() -> None:
    diff = SemanticASTDiff()
    before = "def f(): pass\ndef g(): pass\n"
    after = "def f(): pass\n"
    entries = diff.diff(before=before, after=after)
    assert len(entries) == 1
    assert entries[0].kind is ASTDiffKind.REMOVED
    assert entries[0].symbol == "g"


def test_semantic_diff_detects_changed_function_body() -> None:
    diff = SemanticASTDiff()
    before = "def f():\n    return 1\n"
    after = "def f():\n    return 2\n"
    entries = diff.diff(before=before, after=after)
    assert len(entries) == 1
    assert entries[0].kind is ASTDiffKind.CHANGED
    assert entries[0].symbol == "f"


def test_semantic_diff_detects_added_class() -> None:
    entries = SemanticASTDiff().diff(before="", after="class C:\n    pass\n")
    assert any(
        e.symbol == "C" and e.symbol_kind == "class" and e.kind is ASTDiffKind.ADDED
        for e in entries
    )


def test_semantic_diff_detects_changed_class_method() -> None:
    diff = SemanticASTDiff()
    before = textwrap.dedent("""
    class C:
        def m(self):
            return 1
    """).lstrip()
    after = textwrap.dedent("""
    class C:
        def m(self):
            return 2
    """).lstrip()
    entries = diff.diff(before=before, after=after)
    assert any(e.symbol == "C" and e.kind is ASTDiffKind.CHANGED for e in entries)


def test_semantic_diff_empty_when_identical() -> None:
    diff = SemanticASTDiff()
    src = "def f(): pass\nclass C: pass\n"
    assert diff.diff(before=src, after=src) == ()


def test_semantic_diff_returns_entries_sorted_alphabetically() -> None:
    diff = SemanticASTDiff()
    before = ""
    after = "def z(): pass\ndef a(): pass\ndef m(): pass\n"
    entries = diff.diff(before=before, after=after)
    assert [e.symbol for e in entries] == ["a", "m", "z"]


def test_semantic_diff_rejects_non_str_inputs() -> None:
    diff = SemanticASTDiff()
    with pytest.raises(TypeError):
        diff.diff(before=123, after="")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        diff.diff(before="", after=None)  # type: ignore[arg-type]


def test_semantic_diff_rejects_syntax_errors() -> None:
    diff = SemanticASTDiff()
    with pytest.raises(ValueError, match="before source"):
        diff.diff(before="def f(:", after="")
    with pytest.raises(ValueError, match="after source"):
        diff.diff(before="", after="def f(:")


def test_ast_diff_entry_is_frozen_and_slotted() -> None:
    e = ASTDiffEntry(kind=ASTDiffKind.ADDED, symbol="x", symbol_kind="function")
    with pytest.raises(AttributeError):
        e.symbol = "y"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


# ---------------------------------------------------------------------------
# ImportGraphExtractor
# ---------------------------------------------------------------------------
def test_import_extractor_collects_top_level_import() -> None:
    src = "import os\nimport sys\n"
    entries = ImportGraphExtractor().extract(source=src)
    modules = {e.module for e in entries}
    assert modules == {"os", "sys"}
    assert all(e.is_top_level for e in entries)


def test_import_extractor_collects_from_imports() -> None:
    src = "from os import path\nfrom sys import argv\n"
    entries = ImportGraphExtractor().extract(source=src)
    assert ("os", "path") in {(e.module, e.name) for e in entries}
    assert ("sys", "argv") in {(e.module, e.name) for e in entries}


def test_import_extractor_distinguishes_nested_imports() -> None:
    src = textwrap.dedent("""
    import os
    def f():
        import json
        return json
    """).lstrip()
    entries = ImportGraphExtractor().extract(source=src)
    top = {e.module for e in entries if e.is_top_level}
    nested = {e.module for e in entries if not e.is_top_level}
    assert top == {"os"}
    assert nested == {"json"}


def test_import_extractor_sorts_entries() -> None:
    src = "import sys\nimport os\nimport ast\n"
    entries = ImportGraphExtractor().extract(source=src)
    assert [e.module for e in entries] == ["ast", "os", "sys"]


def test_import_extractor_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        ImportGraphExtractor().extract(source=None)  # type: ignore[arg-type]


def test_import_extractor_rejects_syntax_error() -> None:
    with pytest.raises(ValueError):
        ImportGraphExtractor().extract(source="from x import (")


def test_import_forbidden_hits_match_exact_module() -> None:
    src = "import torch\nimport numpy\nimport os\n"
    extractor = ImportGraphExtractor(forbidden_modules=("torch", "numpy"))
    hits = extractor.forbidden_hits(source=src)
    modules = {h.module for h in hits}
    assert modules == {"torch", "numpy"}


def test_import_forbidden_hits_match_submodules() -> None:
    src = "from torch.nn import Linear\nimport torch.optim\n"
    extractor = ImportGraphExtractor(forbidden_modules=("torch",))
    hits = extractor.forbidden_hits(source=src)
    assert len(hits) == 2
    assert all(h.module.startswith("torch") for h in hits)


def test_import_forbidden_modules_property_is_sorted_tuple() -> None:
    extractor = ImportGraphExtractor(forbidden_modules=("z", "a", "m"))
    assert extractor.forbidden_modules == ("a", "m", "z")


def test_import_entry_is_frozen_and_slotted() -> None:
    e = ImportEntry(module="os", name="", location="1:0", is_top_level=True)
    with pytest.raises(AttributeError):
        e.module = "sys"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


# ---------------------------------------------------------------------------
# CrossFunctionCallAnalyzer
# ---------------------------------------------------------------------------
def test_call_analyzer_extracts_direct_function_calls() -> None:
    src = textwrap.dedent("""
    def a():
        b()
        c()

    def b():
        pass

    def c():
        pass
    """).lstrip()
    entries = CrossFunctionCallAnalyzer().extract(source=src)
    pairs = {(e.caller, e.callee) for e in entries}
    assert ("a", "b") in pairs
    assert ("a", "c") in pairs


def test_call_analyzer_extracts_attribute_call() -> None:
    src = textwrap.dedent("""
    def f():
        obj.method()
        a.b.c()
    """).lstrip()
    entries = CrossFunctionCallAnalyzer().extract(source=src)
    callees = {e.callee for e in entries}
    assert "obj.method" in callees
    assert "a.b.c" in callees


def test_call_analyzer_ignores_calls_outside_top_functions() -> None:
    src = textwrap.dedent("""
    foo()  # module-level call, no caller
    def g():
        bar()
    """).lstrip()
    entries = CrossFunctionCallAnalyzer().extract(source=src)
    callers = {e.caller for e in entries}
    assert callers == {"g"}


def test_call_analyzer_sorts_entries() -> None:
    src = textwrap.dedent("""
    def f():
        z()
        a()
        m()
    """).lstrip()
    entries = CrossFunctionCallAnalyzer().extract(source=src)
    assert [e.callee for e in entries] == ["a", "m", "z"]


def test_call_analyzer_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        CrossFunctionCallAnalyzer().extract(source=42)  # type: ignore[arg-type]


def test_call_analyzer_rejects_syntax_error() -> None:
    with pytest.raises(ValueError):
        CrossFunctionCallAnalyzer().extract(source="def f(:")


def test_tier_violation_runtime_safe_calling_offline_only_flagged() -> None:
    src = textwrap.dedent("""
    def hot():
        cold()

    def cold():
        pass
    """).lstrip()
    tier_map = {"hot": "RUNTIME_SAFE", "cold": "OFFLINE_ONLY"}
    violations = CrossFunctionCallAnalyzer().runtime_tier_violations(source=src, tier_map=tier_map)
    assert len(violations) == 1
    assert violations[0].caller == "hot"
    assert violations[0].callee == "cold"


def test_tier_violation_offline_only_calling_runtime_safe_ok() -> None:
    src = textwrap.dedent("""
    def cold():
        hot()

    def hot():
        pass
    """).lstrip()
    tier_map = {"cold": "OFFLINE_ONLY", "hot": "RUNTIME_SAFE"}
    violations = CrossFunctionCallAnalyzer().runtime_tier_violations(source=src, tier_map=tier_map)
    assert violations == ()


def test_tier_violation_ignores_unmapped_symbols() -> None:
    src = "def a():\n    b()\ndef b():\n    pass\n"
    violations = CrossFunctionCallAnalyzer().runtime_tier_violations(
        source=src, tier_map={"a": "RUNTIME_SAFE"}
    )
    assert violations == ()


def test_tier_violation_research_source_is_highest_tier() -> None:
    src = textwrap.dedent("""
    def a():
        b()

    def b():
        pass
    """).lstrip()
    tier_map = {"a": "OFFLINE_ONLY", "b": "RESEARCH_SOURCE"}
    violations = CrossFunctionCallAnalyzer().runtime_tier_violations(source=src, tier_map=tier_map)
    assert len(violations) == 1


def test_tier_violation_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        CrossFunctionCallAnalyzer().runtime_tier_violations(
            source="",
            tier_map="not-a-mapping",  # type: ignore[arg-type]
        )


def test_call_entry_is_frozen_and_slotted() -> None:
    e = CallEntry(caller="a", callee="b", location="1:0")
    with pytest.raises(AttributeError):
        e.caller = "c"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


# ---------------------------------------------------------------------------
# PatchSafetyAnalyzer
# ---------------------------------------------------------------------------
def test_patch_safety_clean_when_no_violations() -> None:
    analyzer = PatchSafetyAnalyzer()
    before = "def f(): pass\n"
    after = "def f(): return 1\n"
    report = analyzer.analyze(path="utils/helpers.py", before=before, after=after)
    assert report.boundary_touched == ()
    assert report.forbidden_imports == ()
    assert report.tier_violations == ()
    assert report.is_safe is True
    assert len(report.ast_diff) == 1
    assert report.ast_diff[0].kind is ASTDiffKind.CHANGED


def test_patch_safety_flags_governance_boundary_path() -> None:
    analyzer = PatchSafetyAnalyzer()
    report = analyzer.analyze(
        path="governance_engine/policy.py",
        before="def x(): pass\n",
        after="def x(): return 1\n",
    )
    assert report.boundary_touched == ("governance_engine/",)
    assert report.is_safe is False


def test_patch_safety_flags_forbidden_imports() -> None:
    analyzer = PatchSafetyAnalyzer(forbidden_modules=("torch",))
    src = "import torch\ndef f(): pass\n"
    report = analyzer.analyze(path="execution_engine/hot.py", before="", after=src)
    assert len(report.forbidden_imports) == 1
    assert report.forbidden_imports[0].module == "torch"
    assert report.is_safe is False


def test_patch_safety_flags_tier_violations() -> None:
    analyzer = PatchSafetyAnalyzer(
        tier_map={
            "hot": "RUNTIME_SAFE",
            "cold": "OFFLINE_ONLY",
        }
    )
    src = "def hot():\n    cold()\ndef cold(): pass\n"
    report = analyzer.analyze(path="execution_engine/hot.py", before="", after=src)
    assert len(report.tier_violations) == 1
    assert report.tier_violations[0].caller == "hot"
    assert report.is_safe is False


def test_patch_safety_combined_violations() -> None:
    analyzer = PatchSafetyAnalyzer(
        forbidden_modules=("numpy",),
        tier_map={"f": "RUNTIME_SAFE", "g": "OFFLINE_ONLY"},
    )
    src = textwrap.dedent("""
    import numpy
    def f():
        g()
    def g():
        pass
    """).lstrip()
    report = analyzer.analyze(path="governance_engine/core.py", before="", after=src)
    assert report.boundary_touched == ("governance_engine/",)
    assert len(report.forbidden_imports) == 1
    assert len(report.tier_violations) == 1
    assert report.is_safe is False


def test_patch_safety_custom_boundary_prefixes() -> None:
    analyzer = PatchSafetyAnalyzer(boundary_prefixes=("custom/",))
    report = analyzer.analyze(
        path="custom/secret.py",
        before="",
        after="def x(): pass\n",
    )
    assert report.boundary_touched == ("custom/",)


def test_patch_safety_to_dict_is_canonical() -> None:
    analyzer = PatchSafetyAnalyzer(forbidden_modules=("torch",))
    src = "import torch\ndef f(): pass\n"
    report = analyzer.analyze(path="x.py", before="", after=src)
    out = report.to_dict()
    assert isinstance(out["forbidden_imports"], list)
    assert out["forbidden_imports"][0]["module"] == "torch"
    assert isinstance(out["ast_diff"], list)
    assert isinstance(out["boundary_touched"], list)
    assert isinstance(out["tier_violations"], list)


def test_patch_safety_rejects_non_str_path() -> None:
    analyzer = PatchSafetyAnalyzer()
    with pytest.raises(TypeError):
        analyzer.analyze(
            path=None,
            before="",
            after="",  # type: ignore[arg-type]
        )


def test_patch_safety_report_is_frozen_and_slotted() -> None:
    r = PatchSafetyReport(
        boundary_touched=(),
        ast_diff=(),
        forbidden_imports=(),
        tier_violations=(),
    )
    with pytest.raises(AttributeError):
        r.boundary_touched = ("x",)  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_patch_safety_boundary_prefixes_property_sorted() -> None:
    analyzer = PatchSafetyAnalyzer(boundary_prefixes=("z/", "a/", "m/"))
    assert analyzer.boundary_prefixes == ("a/", "m/", "z/")


# ---------------------------------------------------------------------------
# Determinism / INV-15 — same source same output across runs
# ---------------------------------------------------------------------------
def test_three_run_byte_identical_diff() -> None:
    diff = SemanticASTDiff()
    before = textwrap.dedent("""
    def alpha():
        return 1

    class K:
        pass
    """).lstrip()
    after = textwrap.dedent("""
    def alpha():
        return 2

    def beta():
        pass
    """).lstrip()
    runs = [diff.diff(before=before, after=after) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_three_run_byte_identical_imports() -> None:
    extractor = ImportGraphExtractor(forbidden_modules=("torch",))
    src = textwrap.dedent("""
    import os
    import torch
    from sys import argv
    def f():
        import json
        return json
    """).lstrip()
    runs = [extractor.extract(source=src) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_three_run_byte_identical_calls() -> None:
    analyzer = CrossFunctionCallAnalyzer()
    src = textwrap.dedent("""
    def a():
        b()
        x.y()
        z()
    def b(): pass
    def z(): pass
    """).lstrip()
    runs = [analyzer.extract(source=src) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# tree_sitter_parser_factory lazy seam
# ---------------------------------------------------------------------------
def test_tree_sitter_parser_factory_raises_when_dep_missing() -> None:
    import sys

    saved_ts = sys.modules.pop("tree_sitter", None)
    saved_py = sys.modules.pop("tree_sitter_python", None)
    sys.modules["tree_sitter"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="NEW_PIP_DEPENDENCIES"):
            tree_sitter_parser_factory()
    finally:
        if saved_ts is not None:
            sys.modules["tree_sitter"] = saved_ts
        else:
            sys.modules.pop("tree_sitter", None)
        if saved_py is not None:
            sys.modules["tree_sitter_python"] = saved_py
        else:
            sys.modules.pop("tree_sitter_python", None)


# ---------------------------------------------------------------------------
# AST guards — no top-level tree-sitter import
# ---------------------------------------------------------------------------
_THIS = pathlib.Path(__file__).resolve()
_MODULE = _THIS.parents[1] / "evolution_engine" / "patch_pipeline" / "static_analysis.py"


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE.read_text(encoding="utf-8"))


def test_module_has_no_top_level_tree_sitter_import() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("tree_sitter"), (
                    f"tree-sitter must not be top-level imported: {alias.name}"
                )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("tree_sitter"), (
                f"tree-sitter must not be top-level imported: {module}"
            )


def test_tree_sitter_import_confined_to_factory() -> None:
    tree = _module_tree()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            mod = node.module or ""
            names = [mod]
        if not any(n.startswith("tree_sitter") for n in names):
            continue
        # Walk back up to enclosing function — must be tree_sitter_parser_factory.
        # We simulate by checking the parent map.
        enclosing = _enclosing_function_name(tree, node)
        assert enclosing == "tree_sitter_parser_factory", (
            f"tree-sitter import not inside tree_sitter_parser_factory: {names!r}"
        )


def _enclosing_function_name(tree: ast.Module, target: ast.AST) -> str | None:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur = parents.get(id(target))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(id(cur))
    return None


def test_module_has_no_forbidden_runtime_imports() -> None:
    tree = _module_tree()
    forbidden_at_top = {"torch", "numpy", "polars", "tree_sitter"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden_at_top, f"forbidden top-level import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            assert module not in forbidden_at_top, f"forbidden top-level from-import: {node.module}"


def test_module_exports_canonical_symbols() -> None:
    from evolution_engine.patch_pipeline import static_analysis as mod

    for symbol in (
        "FindingSeverity",
        "StaticAnalysisFinding",
        "StaticAnalysisStage",
        "SemanticASTDiff",
        "ImportGraphExtractor",
        "CrossFunctionCallAnalyzer",
        "PatchSafetyAnalyzer",
        "PatchSafetyReport",
        "tree_sitter_parser_factory",
        "NEW_PIP_DEPENDENCIES",
        "TREE_SITTER_ADAPTER_VERSION",
        "GOVERNANCE_BOUNDARY_PREFIXES",
    ):
        assert hasattr(mod, symbol), f"missing public symbol: {symbol}"
