"""I-26 — tests for the canonical CodeQL-shape dataflow analyzer."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tools.codeql_analyzer import (
    ANALYZER_VERSION,
    MAX_CODE_LEN,
    MAX_PATTERN_LEN,
    MAX_QUERY_NAME_LEN,
    NEW_PIP_DEPENDENCIES,
    AnalysisResult,
    AnalyzerError,
    DataFlowQuery,
    PatternKind,
    Sanitizer,
    TaintSink,
    TaintSource,
    analyze,
    enable_codeql_factory,
)

# ---------------------------------------------------------------------------
# Constants / module identity
# ---------------------------------------------------------------------------


def test_analyzer_version_is_pinned() -> None:
    assert ANALYZER_VERSION == "v1.0-I26"


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("codeql",)


def test_max_lengths_are_pinned() -> None:
    assert MAX_QUERY_NAME_LEN == 128
    assert MAX_PATTERN_LEN == 256
    assert MAX_CODE_LEN == 10_000_000


# ---------------------------------------------------------------------------
# PatternKind enum
# ---------------------------------------------------------------------------


def test_pattern_kind_values() -> None:
    assert PatternKind.CALL.value == "CALL"
    assert PatternKind.NAME.value == "NAME"


def test_pattern_kind_count() -> None:
    assert len(list(PatternKind)) == 2


# ---------------------------------------------------------------------------
# TaintSource validation
# ---------------------------------------------------------------------------


def test_taint_source_constructs_valid() -> None:
    src = TaintSource(
        name="REQUEST.ARGS",
        kind=PatternKind.CALL,
        pattern="request.args.get",
    )
    assert src.name == "REQUEST.ARGS"


def test_taint_source_is_frozen_and_slotted() -> None:
    src = TaintSource(name="X", kind=PatternKind.NAME, pattern="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        src.pattern = "y"  # type: ignore[misc]
    assert not hasattr(src, "__dict__")


def test_taint_source_rejects_empty_name() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(name="", kind=PatternKind.CALL, pattern="x")


def test_taint_source_rejects_overlong_name() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(
            name="x" * (MAX_QUERY_NAME_LEN + 1),
            kind=PatternKind.CALL,
            pattern="x",
        )


def test_taint_source_rejects_invalid_name_chars() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(name="bad name", kind=PatternKind.CALL, pattern="x")


def test_taint_source_rejects_non_kind() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(
            name="X",
            kind="CALL",  # type: ignore[arg-type]
            pattern="x",
        )


def test_taint_source_rejects_empty_pattern() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(name="X", kind=PatternKind.CALL, pattern="")


def test_taint_source_rejects_overlong_pattern() -> None:
    with pytest.raises(AnalyzerError):
        TaintSource(
            name="X",
            kind=PatternKind.CALL,
            pattern="x" * (MAX_PATTERN_LEN + 1),
        )


# ---------------------------------------------------------------------------
# TaintSink validation
# ---------------------------------------------------------------------------


def test_taint_sink_constructs_valid() -> None:
    snk = TaintSink(name="EVAL", kind=PatternKind.CALL, pattern="eval")
    assert snk.name == "EVAL"


def test_taint_sink_is_frozen_and_slotted() -> None:
    snk = TaintSink(name="X", kind=PatternKind.CALL, pattern="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        snk.name = "Y"  # type: ignore[misc]
    assert not hasattr(snk, "__dict__")


def test_taint_sink_rejects_empty_name() -> None:
    with pytest.raises(AnalyzerError):
        TaintSink(name="", kind=PatternKind.CALL, pattern="x")


def test_taint_sink_rejects_name_kind() -> None:
    # Sinks must be CALLs.
    with pytest.raises(AnalyzerError):
        TaintSink(name="X", kind=PatternKind.NAME, pattern="x")


def test_taint_sink_rejects_overlong_pattern() -> None:
    with pytest.raises(AnalyzerError):
        TaintSink(
            name="X",
            kind=PatternKind.CALL,
            pattern="x" * (MAX_PATTERN_LEN + 1),
        )


# ---------------------------------------------------------------------------
# Sanitizer validation
# ---------------------------------------------------------------------------


def test_sanitizer_constructs_valid() -> None:
    san = Sanitizer(name="HTML_ESCAPE", pattern="html.escape")
    assert san.name == "HTML_ESCAPE"


def test_sanitizer_is_frozen_and_slotted() -> None:
    san = Sanitizer(name="X", pattern="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        san.name = "Y"  # type: ignore[misc]
    assert not hasattr(san, "__dict__")


def test_sanitizer_rejects_empty_name() -> None:
    with pytest.raises(AnalyzerError):
        Sanitizer(name="", pattern="x")


def test_sanitizer_rejects_empty_pattern() -> None:
    with pytest.raises(AnalyzerError):
        Sanitizer(name="X", pattern="")


# ---------------------------------------------------------------------------
# DataFlowQuery validation
# ---------------------------------------------------------------------------


_USER_INPUT = TaintSource(name="USER_INPUT", kind=PatternKind.NAME, pattern="user_input")
_REQUEST_ARGS = TaintSource(
    name="REQUEST_ARGS",
    kind=PatternKind.CALL,
    pattern="request.args.get",
)
_EVAL_SINK = TaintSink(name="EVAL", kind=PatternKind.CALL, pattern="eval")
_EXEC_SINK = TaintSink(name="EXEC", kind=PatternKind.CALL, pattern="exec")
_HTML_ESCAPE = Sanitizer(name="HTML_ESCAPE", pattern="html.escape")


def test_data_flow_query_constructs_valid() -> None:
    q = DataFlowQuery(
        name="UNTRUSTED-EVAL",
        sources=(_USER_INPUT,),
        sinks=(_EVAL_SINK,),
    )
    assert q.name == "UNTRUSTED-EVAL"


def test_data_flow_query_is_frozen_and_slotted() -> None:
    q = DataFlowQuery(
        name="x",
        sources=(_USER_INPUT,),
        sinks=(_EVAL_SINK,),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.name = "y"  # type: ignore[misc]
    assert not hasattr(q, "__dict__")


def test_data_flow_query_rejects_empty_sources() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(name="x", sources=(), sinks=(_EVAL_SINK,))


def test_data_flow_query_rejects_empty_sinks() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(name="x", sources=(_USER_INPUT,), sinks=())


def test_data_flow_query_rejects_non_tuple_sources() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(
            name="x",
            sources=[_USER_INPUT],  # type: ignore[arg-type]
            sinks=(_EVAL_SINK,),
        )


def test_data_flow_query_rejects_non_tuple_sinks() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(
            name="x",
            sources=(_USER_INPUT,),
            sinks=[_EVAL_SINK],  # type: ignore[arg-type]
        )


def test_data_flow_query_rejects_non_source_in_sources() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(
            name="x",
            sources=("bad",),  # type: ignore[arg-type]
            sinks=(_EVAL_SINK,),
        )


def test_data_flow_query_rejects_non_sink_in_sinks() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(
            name="x",
            sources=(_USER_INPUT,),
            sinks=("bad",),  # type: ignore[arg-type]
        )


def test_data_flow_query_rejects_non_sanitizer_in_sanitizers() -> None:
    with pytest.raises(AnalyzerError):
        DataFlowQuery(
            name="x",
            sources=(_USER_INPUT,),
            sinks=(_EVAL_SINK,),
            sanitizers=("bad",),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# analyze() — happy paths
# ---------------------------------------------------------------------------


_BASIC_QUERY = DataFlowQuery(
    name="UNTRUSTED-EVAL",
    sources=(_USER_INPUT,),
    sinks=(_EVAL_SINK,),
)


_REQUEST_EVAL_QUERY = DataFlowQuery(
    name="REQUEST-TO-EVAL",
    sources=(_REQUEST_ARGS,),
    sinks=(_EVAL_SINK,),
)


def test_analyze_empty_code_returns_clean_result() -> None:
    result = analyze(_BASIC_QUERY, "")
    assert result.traces == ()
    assert result.is_clean()


def test_analyze_no_taint_returns_clean_result() -> None:
    result = analyze(_BASIC_QUERY, "x = 1\ny = 2\n")
    assert result.traces == ()


def test_analyze_direct_name_flow_emits_trace() -> None:
    code = "user_input = 'x'\neval(user_input)\n"
    result = analyze(_BASIC_QUERY, code)
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.source_name == "USER_INPUT"
    assert trace.sink_name == "EVAL"
    assert trace.tainted_var == "user_input"


def test_analyze_call_source_flow_emits_trace() -> None:
    code = "u = request.args.get('x')\neval(u)\n"
    result = analyze(_REQUEST_EVAL_QUERY, code)
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.source_name == "REQUEST_ARGS"
    assert trace.sink_name == "EVAL"
    assert trace.tainted_var == "u"


def test_analyze_propagates_through_intermediate_assignment() -> None:
    code = "u = request.args.get('x')\nv = u\nw = v\neval(w)\n"
    result = analyze(_REQUEST_EVAL_QUERY, code)
    assert len(result.traces) == 1
    assert result.traces[0].tainted_var == "w"


def test_analyze_sanitizer_blocks_taint() -> None:
    query = DataFlowQuery(
        name="REQUEST-EVAL-SANITIZED",
        sources=(_REQUEST_ARGS,),
        sinks=(_EVAL_SINK,),
        sanitizers=(_HTML_ESCAPE,),
    )
    code = "u = request.args.get('x')\nv = html.escape(u)\neval(v)\n"
    result = analyze(query, code)
    assert result.is_clean()


def test_analyze_emits_multiple_traces_for_multiple_sinks() -> None:
    query = DataFlowQuery(
        name="REQUEST-TO-EVAL-OR-EXEC",
        sources=(_REQUEST_ARGS,),
        sinks=(_EVAL_SINK, _EXEC_SINK),
    )
    code = "u = request.args.get('x')\neval(u)\nexec(u)\n"
    result = analyze(query, code)
    assert len(result.traces) == 2
    sinks = {t.sink_name for t in result.traces}
    assert sinks == {"EVAL", "EXEC"}


def test_analyze_traces_sorted_by_sink_location() -> None:
    code = "u = request.args.get('x')\nexec(u)\neval(u)\n"
    query = DataFlowQuery(
        name="MULTI",
        sources=(_REQUEST_ARGS,),
        sinks=(_EVAL_SINK, _EXEC_SINK),
    )
    result = analyze(query, code)
    # exec is on line 2; eval is on line 3 — sorted ascending
    assert result.traces[0].sink_line == 2
    assert result.traces[1].sink_line == 3


def test_analyze_is_deterministic_across_three_runs() -> None:
    code = "u = request.args.get('a')\nv = request.args.get('b')\neval(u)\nexec(v)\n"
    query = DataFlowQuery(
        name="MULTI",
        sources=(_REQUEST_ARGS,),
        sinks=(_EVAL_SINK, _EXEC_SINK),
    )
    r1 = analyze(query, code)
    r2 = analyze(query, code)
    r3 = analyze(query, code)
    assert r1.digest == r2.digest == r3.digest
    assert r1.traces == r2.traces == r3.traces


def test_analyze_returns_analysis_result_type() -> None:
    result = analyze(_BASIC_QUERY, "")
    assert isinstance(result, AnalysisResult)
    assert result.backend == "stdlib"
    assert result.query_name == "UNTRUSTED-EVAL"


def test_analyze_no_trace_when_source_unrelated() -> None:
    code = "other = 'safe'\neval(other)\n"
    result = analyze(_BASIC_QUERY, code)
    assert result.is_clean()


def test_analyze_source_in_argument_does_not_propagate_to_call() -> None:
    # ``eval(user_input)`` is a sink call — source is the NAME pattern
    # ``user_input``, which matches the inner Name. The argument
    # references the tainted name directly, so the trace fires.
    code = "user_input = 'x'\neval(user_input)\n"
    result = analyze(_BASIC_QUERY, code)
    assert len(result.traces) == 1


def test_analyze_attribute_source_match() -> None:
    src = TaintSource(
        name="OS_GETENV",
        kind=PatternKind.CALL,
        pattern="os.getenv",
    )
    query = DataFlowQuery(
        name="ENV-TO-EVAL",
        sources=(src,),
        sinks=(_EVAL_SINK,),
    )
    code = "secret = os.getenv('KEY')\neval(secret)\n"
    result = analyze(query, code)
    assert len(result.traces) == 1


def test_analyze_attribute_source_suffix_match() -> None:
    # ``request.args.get`` should match an arbitrary prefix like
    # ``flask.request.args.get`` because the pattern contains a dot
    # and the identifier ends with the same dotted suffix.
    src = TaintSource(
        name="REQUEST_ARGS",
        kind=PatternKind.CALL,
        pattern="request.args.get",
    )
    query = DataFlowQuery(
        name="REQUEST-EVAL",
        sources=(src,),
        sinks=(_EVAL_SINK,),
    )
    code = "u = flask.request.args.get('x')\neval(u)\n"
    result = analyze(query, code)
    assert len(result.traces) == 1


def test_analyze_attribute_pattern_does_not_partial_match() -> None:
    src = TaintSource(
        name="OS_GETENV",
        kind=PatternKind.CALL,
        pattern="os.getenv",
    )
    query = DataFlowQuery(
        name="X",
        sources=(src,),
        sinks=(_EVAL_SINK,),
    )
    code = "u = myos.getenv('x')\neval(u)\n"
    result = analyze(query, code)
    assert result.is_clean()


def test_analyze_no_duplicate_trace_for_same_source_sink_pair() -> None:
    # Two name references to the same tainted variable in two
    # arguments collapse to one trace per (source-id, sink-loc).
    code = "u = request.args.get('x')\neval(u + u)\n"
    result = analyze(_REQUEST_EVAL_QUERY, code)
    assert len(result.traces) == 1


# ---------------------------------------------------------------------------
# analyze() — error paths
# ---------------------------------------------------------------------------


def test_analyze_rejects_non_query() -> None:
    with pytest.raises(AnalyzerError):
        analyze("bad-query", "")  # type: ignore[arg-type]


def test_analyze_rejects_non_string_code() -> None:
    with pytest.raises(AnalyzerError):
        analyze(_BASIC_QUERY, 42)  # type: ignore[arg-type]


def test_analyze_rejects_overlong_code() -> None:
    big = "x = 1\n" * (MAX_CODE_LEN // 6 + 1)
    with pytest.raises(AnalyzerError):
        analyze(_BASIC_QUERY, big)


def test_analyze_rejects_syntax_error() -> None:
    with pytest.raises(AnalyzerError):
        analyze(_BASIC_QUERY, "def bad(\n")


def test_analyze_rejects_empty_file_path() -> None:
    with pytest.raises(AnalyzerError):
        analyze(_BASIC_QUERY, "x = 1\n", file_path="")


# ---------------------------------------------------------------------------
# AnalysisResult helpers
# ---------------------------------------------------------------------------


def test_analysis_result_is_clean_on_empty_traces() -> None:
    result = analyze(_BASIC_QUERY, "x = 1\n")
    assert result.is_clean() is True


def test_analysis_result_is_clean_false_with_traces() -> None:
    result = analyze(_BASIC_QUERY, "user_input='x'\neval(user_input)\n")
    assert result.is_clean() is False


def test_analysis_result_by_sink_filters_traces() -> None:
    query = DataFlowQuery(
        name="MULTI",
        sources=(_REQUEST_ARGS,),
        sinks=(_EVAL_SINK, _EXEC_SINK),
    )
    code = "u = request.args.get('x')\neval(u)\nexec(u)\n"
    result = analyze(query, code)
    eval_traces = result.by_sink("EVAL")
    assert len(eval_traces) == 1
    assert eval_traces[0].sink_name == "EVAL"


def test_analysis_result_is_frozen_and_slotted() -> None:
    result = analyze(_BASIC_QUERY, "x = 1\n")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.backend = "codeql"  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_analysis_result_rejects_invalid_backend() -> None:
    with pytest.raises(AnalyzerError):
        AnalysisResult(
            traces=(),
            query_name="X",
            file_path="x.py",
            backend="ROGUE",
        )


def test_analysis_result_rejects_non_tuple_traces() -> None:
    with pytest.raises(AnalyzerError):
        AnalysisResult(
            traces=[],  # type: ignore[arg-type]
            query_name="X",
            file_path="x.py",
        )


def test_analysis_result_rejects_non_trace_member() -> None:
    with pytest.raises(AnalyzerError):
        AnalysisResult(
            traces=("not-a-trace",),  # type: ignore[arg-type]
            query_name="X",
            file_path="x.py",
        )


def test_analysis_result_rejects_empty_query_name() -> None:
    with pytest.raises(AnalyzerError):
        AnalysisResult(traces=(), query_name="", file_path="x.py")


# ---------------------------------------------------------------------------
# Lazy seam — enable_codeql_factory
# ---------------------------------------------------------------------------


def test_enable_codeql_factory_skips_when_uninstalled() -> None:
    try:
        import codeql  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("codeql not installed")
    analyzer = enable_codeql_factory()
    result = analyzer(_BASIC_QUERY, "user_input='x'\neval(user_input)\n", "<test>")
    assert result.backend == "codeql"
    assert len(result.traces) == 1


def test_enable_codeql_factory_rejects_unknown_overrides() -> None:
    try:
        import codeql  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("codeql not installed")
    with pytest.raises(AnalyzerError):
        enable_codeql_factory(overrides={"bogus_key": 1})


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "codeql_analyzer.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_codeql_import() -> None:
    assert all(not name.startswith("codeql") for name in _top_level_imports(_module_ast()))


def test_no_top_level_subprocess_import() -> None:
    assert "subprocess" not in _top_level_imports(_module_ast())


def test_no_top_level_time_or_random_import() -> None:
    banned = {"time", "random", "datetime", "asyncio"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_network_imports() -> None:
    banned = {"socket", "urllib", "requests", "httpx", "aiohttp"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_engine_imports() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "core.contracts.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


def test_codeql_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith("codeql") or name == "subprocess":
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside enable_codeql_factory"
                    )
                    assert parent.name == "enable_codeql_factory", (
                        f"{name} imported in {parent.name!r} — must be inside enable_codeql_factory"
                    )


# ---------------------------------------------------------------------------
# Realistic scan demo
# ---------------------------------------------------------------------------


def test_realistic_request_to_eval_taint() -> None:
    code = """
def handle(request):
    raw = request.args.get('payload')
    decoded = raw.decode()
    pipeline = decoded.strip()
    eval(pipeline)
"""
    query = DataFlowQuery(
        name="UNTRUSTED-EVAL",
        sources=(
            TaintSource(
                name="REQUEST_ARGS",
                kind=PatternKind.CALL,
                pattern="request.args.get",
            ),
        ),
        sinks=(_EVAL_SINK,),
    )
    result = analyze(query, code, file_path="handler.py")
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.source_name == "REQUEST_ARGS"
    assert trace.sink_name == "EVAL"


def test_realistic_sanitized_flow_is_clean() -> None:
    code = """
def render(request):
    raw = request.args.get('name')
    safe = html.escape(raw)
    out = safe.upper()
    print(out)
    eval(safe)
"""
    query = DataFlowQuery(
        name="UNTRUSTED-EVAL-SANITIZED",
        sources=(
            TaintSource(
                name="REQUEST_ARGS",
                kind=PatternKind.CALL,
                pattern="request.args.get",
            ),
        ),
        sinks=(_EVAL_SINK,),
        sanitizers=(_HTML_ESCAPE,),
    )
    result = analyze(query, code, file_path="render.py")
    assert result.is_clean()


# ---------------------------------------------------------------------------
# Reload idempotency (runs last — reload invalidates earlier enum refs)
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import tools.codeql_analyzer as mod1

    importlib.reload(mod1)
    import tools.codeql_analyzer as mod2

    assert mod1.ANALYZER_VERSION == mod2.ANALYZER_VERSION
    assert mod1.MAX_CODE_LEN == mod2.MAX_CODE_LEN
    assert mod1.PatternKind.CALL is mod2.PatternKind.CALL
