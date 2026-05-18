"""Tests for C-27 sourcegraph-shape codebase intelligence (OFFLINE_ONLY)."""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

from tools.codebase_intelligence import (
    NEW_PIP_DEPENDENCIES,
    SOURCEGRAPH_ADAPTER_VERSION,
    AuthorityViolation,
    CallSite,
    CodebaseIntelligence,
    ImportEdge,
    SymbolKind,
    SymbolRef,
    sg_binary_factory,
)


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
def test_pip_deps_is_empty_tuple() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_adapter_version_is_string() -> None:
    assert SOURCEGRAPH_ADAPTER_VERSION == "1"


def test_symbol_kind_values() -> None:
    assert SymbolKind.FUNCTION.value == "function"
    assert SymbolKind.ASYNC_FUNCTION.value == "async_function"
    assert SymbolKind.CLASS.value == "class"
    assert SymbolKind.METHOD.value == "method"


# ---------------------------------------------------------------------------
# Fixtures — temp source tree
# ---------------------------------------------------------------------------
def _write_tree(root: pathlib.Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@pytest.fixture
def small_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    _write_tree(
        tmp_path,
        {
            "execution_engine/hot.py": textwrap.dedent("""
                def fast_execute(intent):
                    create_execution_intent(intent)
                    return intent

                class HotPath:
                    def step(self):
                        fast_execute(None)
            """).lstrip(),
            "governance_engine/policy.py": textwrap.dedent("""
                import logging

                def create_execution_intent(intent):
                    logging.info("intent created")
                    return intent

                def mark_approved(intent):
                    return intent
            """).lstrip(),
            "tools/devscript.py": textwrap.dedent("""
                from governance_engine.policy import create_execution_intent
                async def main():
                    create_execution_intent(None)
            """).lstrip(),
            "tests/test_helper.py": "def helper(): pass\n",
        },
    )
    return tmp_path


@pytest.fixture
def ci(small_repo: pathlib.Path) -> CodebaseIntelligence:
    return CodebaseIntelligence(root=str(small_repo), exclude=("tests/",))


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
def test_construct_rejects_empty_root() -> None:
    with pytest.raises(ValueError, match="root"):
        CodebaseIntelligence(root="")


def test_construct_rejects_non_str_root() -> None:
    with pytest.raises(ValueError):
        CodebaseIntelligence(root=None)  # type: ignore[arg-type]


def test_construct_rejects_missing_root(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="not a directory"):
        CodebaseIntelligence(root=str(missing))


def test_construct_rejects_file_as_root(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "f.py"
    p.write_text("def f(): pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        CodebaseIntelligence(root=str(p))


def test_root_property_normalized(small_repo: pathlib.Path) -> None:
    ci = CodebaseIntelligence(root=str(small_repo))
    assert ci.root == small_repo.as_posix()


# ---------------------------------------------------------------------------
# Indexing — symbols / calls / imports
# ---------------------------------------------------------------------------
def test_modules_discovered(ci: CodebaseIntelligence) -> None:
    modules = set(ci.modules)
    assert "execution_engine.hot" in modules
    assert "governance_engine.policy" in modules
    assert "tools.devscript" in modules
    assert "tests.test_helper" not in modules


def test_symbols_include_functions_and_classes_and_methods(
    ci: CodebaseIntelligence,
) -> None:
    names = {s.name for s in ci.symbols()}
    assert "fast_execute" in names
    assert "HotPath" in names
    assert "step" in names
    assert "create_execution_intent" in names
    assert "mark_approved" in names


def test_symbol_kind_assigned(ci: CodebaseIntelligence) -> None:
    by_name = {s.name: s.kind for s in ci.symbols()}
    assert by_name["fast_execute"] is SymbolKind.FUNCTION
    assert by_name["HotPath"] is SymbolKind.CLASS
    assert by_name["step"] is SymbolKind.METHOD
    assert by_name["main"] is SymbolKind.ASYNC_FUNCTION


def test_calls_extracted(ci: CodebaseIntelligence) -> None:
    pairs = {(c.caller, c.callee) for c in ci.calls()}
    assert ("fast_execute", "create_execution_intent") in pairs
    assert ("step", "fast_execute") in pairs


def test_imports_extracted(ci: CodebaseIntelligence) -> None:
    pairs = {(i.from_module, i.to_module) for i in ci.imports()}
    assert ("governance_engine.policy", "logging") in pairs
    assert ("tools.devscript", "governance_engine.policy") in pairs


# ---------------------------------------------------------------------------
# find_refs / find_callers / symbol_search
# ---------------------------------------------------------------------------
def test_find_refs_locates_callers(ci: CodebaseIntelligence) -> None:
    refs = ci.find_refs(symbol="create_execution_intent")
    callers = {c.caller for c in refs}
    assert callers == {"fast_execute", "main"}


def test_find_callers_alias_of_find_refs(ci: CodebaseIntelligence) -> None:
    refs = ci.find_refs(symbol="fast_execute")
    callers = ci.find_callers("fast_execute")
    assert refs == callers


def test_find_refs_empty_for_unknown_symbol(
    ci: CodebaseIntelligence,
) -> None:
    assert ci.find_refs(symbol="this_does_not_exist") == ()


def test_find_refs_empty_string_returns_empty(
    ci: CodebaseIntelligence,
) -> None:
    assert ci.find_refs(symbol="") == ()


def test_find_refs_rejects_non_str(ci: CodebaseIntelligence) -> None:
    with pytest.raises(TypeError):
        ci.find_refs(symbol=42)  # type: ignore[arg-type]


def test_find_refs_results_sorted(ci: CodebaseIntelligence) -> None:
    refs = ci.find_refs(symbol="create_execution_intent")
    sort_key = [(r.module, r.caller, r.location) for r in refs]
    assert sort_key == sorted(sort_key)


def test_symbol_search_substring(ci: CodebaseIntelligence) -> None:
    hits = ci.symbol_search(query="execute")
    names = {h.name for h in hits}
    assert "fast_execute" in names


def test_symbol_search_empty_query_returns_empty(
    ci: CodebaseIntelligence,
) -> None:
    assert ci.symbol_search(query="") == ()


def test_symbol_search_rejects_non_str(ci: CodebaseIntelligence) -> None:
    with pytest.raises(TypeError):
        ci.symbol_search(query=None)  # type: ignore[arg-type]


def test_dependency_graph_returns_imports(
    ci: CodebaseIntelligence,
) -> None:
    edges = ci.dependency_graph()
    assert ci.imports() == edges
    assert all(isinstance(e, ImportEdge) for e in edges)


# ---------------------------------------------------------------------------
# include / exclude filters
# ---------------------------------------------------------------------------
def test_exclude_filter_drops_tests(
    small_repo: pathlib.Path,
) -> None:
    ci = CodebaseIntelligence(root=str(small_repo), exclude=("tests/",))
    assert all(not m.startswith("tests") for m in ci.modules)


def test_include_filter_restricts_modules(
    small_repo: pathlib.Path,
) -> None:
    ci = CodebaseIntelligence(root=str(small_repo), include=("execution_engine/",))
    assert all(m.startswith("execution_engine") for m in ci.modules)


def test_filters_sorted_in_tuple_state(
    small_repo: pathlib.Path,
) -> None:
    ci = CodebaseIntelligence(
        root=str(small_repo),
        include=("z/", "a/"),
        exclude=("y/", "b/"),
    )
    # Just exercise the include/exclude path — they must sort internally.
    assert ci.modules == ()


# ---------------------------------------------------------------------------
# Authority violations
# ---------------------------------------------------------------------------
def test_authority_violation_runtime_calling_offline(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "fast_execute": "RUNTIME_SAFE",
            "create_execution_intent": "OFFLINE_ONLY",
        }
    )
    assert len(violations) == 1
    v = violations[0]
    assert v.caller_symbol == "fast_execute"
    assert v.callee_symbol == "create_execution_intent"
    assert v.caller_tier == "RUNTIME_SAFE"
    assert v.callee_tier == "OFFLINE_ONLY"


def test_authority_violation_offline_calling_runtime_ok(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "main": "OFFLINE_ONLY",
            "create_execution_intent": "RUNTIME_SAFE",
        }
    )
    assert violations == ()


def test_authority_violation_ignores_unmapped(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(tier_map={"fast_execute": "RUNTIME_SAFE"})
    assert violations == ()


def test_authority_violation_ignores_unknown_tiers(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "fast_execute": "FOO_TIER",
            "create_execution_intent": "BAR_TIER",
        }
    )
    assert violations == ()


def test_authority_violation_research_source_is_least_strict(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "fast_execute": "OFFLINE_ONLY",
            "create_execution_intent": "RESEARCH_SOURCE",
        }
    )
    assert len(violations) == 1


def test_authority_violation_rejects_non_mapping(
    ci: CodebaseIntelligence,
) -> None:
    with pytest.raises(TypeError):
        ci.authority_violations(tier_map="not-a-mapping")  # type: ignore[arg-type]


def test_authority_violation_results_sorted(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "fast_execute": "RUNTIME_SAFE",
            "create_execution_intent": "OFFLINE_ONLY",
            "main": "RUNTIME_SAFE",
            "step": "RUNTIME_SAFE",
        }
    )
    keys = [(v.caller_module, v.caller_symbol, v.callee_symbol, v.location) for v in violations]
    assert keys == sorted(keys)


def test_authority_violation_to_dict(
    ci: CodebaseIntelligence,
) -> None:
    violations = ci.authority_violations(
        tier_map={
            "fast_execute": "RUNTIME_SAFE",
            "create_execution_intent": "OFFLINE_ONLY",
        }
    )
    d = violations[0].to_dict()
    assert d["caller_symbol"] == "fast_execute"
    assert d["callee_symbol"] == "create_execution_intent"
    assert d["caller_tier"] == "RUNTIME_SAFE"
    assert d["callee_tier"] == "OFFLINE_ONLY"


# ---------------------------------------------------------------------------
# Determinism / INV-15
# ---------------------------------------------------------------------------
def test_three_run_byte_identical_calls(
    small_repo: pathlib.Path,
) -> None:
    runs = [CodebaseIntelligence(root=str(small_repo)).calls() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_three_run_byte_identical_symbols(
    small_repo: pathlib.Path,
) -> None:
    runs = [CodebaseIntelligence(root=str(small_repo)).symbols() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_three_run_byte_identical_imports(
    small_repo: pathlib.Path,
) -> None:
    runs = [CodebaseIntelligence(root=str(small_repo)).imports() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_three_run_byte_identical_authority(
    small_repo: pathlib.Path,
) -> None:
    tier_map = {
        "fast_execute": "RUNTIME_SAFE",
        "create_execution_intent": "OFFLINE_ONLY",
    }
    runs = [
        CodebaseIntelligence(root=str(small_repo)).authority_violations(tier_map=tier_map)
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# Value objects — frozen + slotted
# ---------------------------------------------------------------------------
def test_symbol_ref_is_frozen_and_slotted() -> None:
    s = SymbolRef(name="f", kind=SymbolKind.FUNCTION, module="m", location="1:0")
    with pytest.raises(AttributeError):
        s.name = "g"  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_call_site_is_frozen_and_slotted() -> None:
    c = CallSite(caller="a", callee="b", module="m", location="1:0")
    with pytest.raises(AttributeError):
        c.caller = "c"  # type: ignore[misc]
    assert not hasattr(c, "__dict__")


def test_import_edge_is_frozen_and_slotted() -> None:
    e = ImportEdge(from_module="a", to_module="b", location="1:0")
    with pytest.raises(AttributeError):
        e.from_module = "x"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


def test_authority_violation_is_frozen_and_slotted() -> None:
    v = AuthorityViolation(
        caller_module="m",
        caller_symbol="a",
        callee_module="m",
        callee_symbol="b",
        caller_tier="RUNTIME_SAFE",
        callee_tier="OFFLINE_ONLY",
        location="1:0",
    )
    with pytest.raises(AttributeError):
        v.caller_module = "x"  # type: ignore[misc]
    assert not hasattr(v, "__dict__")


# ---------------------------------------------------------------------------
# sg_binary_factory lazy seam
# ---------------------------------------------------------------------------
def test_sg_binary_factory_rejects_empty_binary() -> None:
    with pytest.raises(ValueError):
        sg_binary_factory(binary="")


def test_sg_binary_factory_rejects_non_str() -> None:
    with pytest.raises(ValueError):
        sg_binary_factory(binary=None)  # type: ignore[arg-type]


def test_sg_binary_factory_raises_when_binary_missing() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        sg_binary_factory(binary="sg-this-does-not-exist-12345")


def test_sg_binary_factory_finds_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    result = sg_binary_factory(binary=pathlib.Path(sys.executable).name)
    # Either it resolves on PATH or it raises — both are acceptable end states.
    # Where it resolves, the result is a dict with binary + path keys.
    assert "binary" in result
    assert "path" in result


# ---------------------------------------------------------------------------
# AST guards — no Sourcegraph top-level imports
# ---------------------------------------------------------------------------
_THIS = pathlib.Path(__file__).resolve()
_MODULE = _THIS.parents[1] / "tools" / "codebase_intelligence.py"


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE.read_text(encoding="utf-8"))


def test_module_has_no_sourcegraph_top_level_imports() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "sourcegraph" not in alias.name.lower()
                assert alias.name not in {"sg", "sg_cli"}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "sourcegraph" not in module.lower()


def test_module_has_no_subprocess_or_network_top_imports() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in {
                    "subprocess",
                    "socket",
                    "http",
                    "urllib",
                    "httpx",
                    "requests",
                }, f"forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            assert module not in {
                "subprocess",
                "socket",
                "http",
                "urllib",
                "httpx",
                "requests",
            }, f"forbidden top-level from-import: {node.module}"


def test_module_exports_canonical_symbols() -> None:
    from tools import codebase_intelligence as mod

    for symbol in (
        "CodebaseIntelligence",
        "SymbolRef",
        "CallSite",
        "ImportEdge",
        "AuthorityViolation",
        "SymbolKind",
        "sg_binary_factory",
        "NEW_PIP_DEPENDENCIES",
        "SOURCEGRAPH_ADAPTER_VERSION",
    ):
        assert hasattr(mod, symbol), f"missing public symbol: {symbol}"
