"""I-25 — tests for the canonical semgrep-shape pattern scanner."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tools.semgrep_scanner import (
    MAX_CODE_LEN,
    MAX_MESSAGE_LEN,
    MAX_PATTERN_LEN,
    MAX_RULE_ID_LEN,
    NEW_PIP_DEPENDENCIES,
    SCANNER_VERSION,
    NodeKind,
    RuleSet,
    ScannerError,
    ScanResult,
    ScanRule,
    Severity,
    SuiteReport,
    enable_semgrep_factory,
    scan,
    scan_suite,
)

# ---------------------------------------------------------------------------
# Constants / module identity
# ---------------------------------------------------------------------------


def test_scanner_version_is_pinned() -> None:
    assert SCANNER_VERSION == "v1.0-I25"


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("semgrep",)


def test_max_lengths_are_pinned() -> None:
    assert MAX_RULE_ID_LEN == 128
    assert MAX_MESSAGE_LEN == 1024
    assert MAX_PATTERN_LEN == 256
    assert MAX_CODE_LEN == 10_000_000


# ---------------------------------------------------------------------------
# Severity / NodeKind enums
# ---------------------------------------------------------------------------


def test_severity_values() -> None:
    assert Severity.INFO.value == "INFO"
    assert Severity.WARNING.value == "WARNING"
    assert Severity.ERROR.value == "ERROR"


def test_severity_count() -> None:
    assert len(list(Severity)) == 3


def test_node_kind_values() -> None:
    assert NodeKind.CALL.value == "CALL"
    assert NodeKind.ATTRIBUTE.value == "ATTRIBUTE"
    assert NodeKind.IMPORT.value == "IMPORT"
    assert NodeKind.IMPORT_FROM.value == "IMPORT_FROM"
    assert NodeKind.STRING.value == "STRING"
    assert NodeKind.ASSIGN.value == "ASSIGN"
    assert NodeKind.NAME.value == "NAME"


def test_node_kind_count() -> None:
    assert len(list(NodeKind)) == 7


# ---------------------------------------------------------------------------
# ScanRule validation
# ---------------------------------------------------------------------------


def test_scan_rule_constructs_valid() -> None:
    rule = ScanRule(
        rule_id="PY.NO-EVAL",
        kind=NodeKind.CALL,
        pattern="eval",
        severity=Severity.ERROR,
        message="eval() executes arbitrary strings",
    )
    assert rule.rule_id == "PY.NO-EVAL"
    assert rule.kind is NodeKind.CALL


def test_scan_rule_is_frozen_and_slotted() -> None:
    rule = ScanRule(
        rule_id="r1",
        kind=NodeKind.CALL,
        pattern="x",
        severity=Severity.INFO,
        message="m",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.pattern = "y"  # type: ignore[misc]
    assert not hasattr(rule, "__dict__")


def test_scan_rule_rejects_empty_rule_id() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="",
            kind=NodeKind.CALL,
            pattern="x",
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_rule_id_with_bad_chars() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="bad id",
            kind=NodeKind.CALL,
            pattern="x",
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_overlong_rule_id() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="x" * (MAX_RULE_ID_LEN + 1),
            kind=NodeKind.CALL,
            pattern="x",
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_non_node_kind_kind() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="r1",
            kind="CALL",  # type: ignore[arg-type]
            pattern="x",
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_empty_pattern() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="r1",
            kind=NodeKind.CALL,
            pattern="",
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_overlong_pattern() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="r1",
            kind=NodeKind.CALL,
            pattern="x" * (MAX_PATTERN_LEN + 1),
            severity=Severity.INFO,
            message="m",
        )


def test_scan_rule_rejects_empty_message() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="r1",
            kind=NodeKind.CALL,
            pattern="x",
            severity=Severity.INFO,
            message="",
        )


def test_scan_rule_rejects_non_severity() -> None:
    with pytest.raises(ScannerError):
        ScanRule(
            rule_id="r1",
            kind=NodeKind.CALL,
            pattern="x",
            severity="ERROR",  # type: ignore[arg-type]
            message="m",
        )


# ---------------------------------------------------------------------------
# scan() — happy paths
# ---------------------------------------------------------------------------


_EVAL_RULE = ScanRule(
    rule_id="PY.NO-EVAL",
    kind=NodeKind.CALL,
    pattern="eval",
    severity=Severity.ERROR,
    message="eval() is unsafe",
)


_OS_SYSTEM_RULE = ScanRule(
    rule_id="PY.NO-OS-SYSTEM",
    kind=NodeKind.CALL,
    pattern="os.system",
    severity=Severity.ERROR,
    message="os.system shells out",
)


def test_scan_empty_code_returns_clean_result() -> None:
    result = scan([_EVAL_RULE], "")
    assert result.findings == ()
    assert result.rule_count == 1
    assert result.scanned_lines == 0


def test_scan_no_match_returns_empty_findings() -> None:
    result = scan([_EVAL_RULE], "x = 1\ny = 2\n")
    assert result.findings == ()
    assert result.scanned_lines == 2


def test_scan_call_match_emits_finding() -> None:
    result = scan([_EVAL_RULE], "x = eval('1 + 1')\n")
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.rule_id == "PY.NO-EVAL"
    assert finding.severity is Severity.ERROR
    assert finding.line == 1
    assert finding.snippet == "eval"


def test_scan_attribute_call_match() -> None:
    code = "import os\nos.system('ls')\n"
    result = scan([_OS_SYSTEM_RULE], code)
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "os.system"
    assert result.findings[0].line == 2


def test_scan_returns_scan_result_type() -> None:
    result = scan([_EVAL_RULE], "eval('x')\n")
    assert isinstance(result, ScanResult)
    assert result.backend == "stdlib"


def test_scan_is_deterministic_across_three_runs() -> None:
    code = "import os\nos.system('a')\neval('b')\nos.system('c')\n"
    rules = [_EVAL_RULE, _OS_SYSTEM_RULE]
    r1 = scan(rules, code)
    r2 = scan(rules, code)
    r3 = scan(rules, code)
    assert r1.digest == r2.digest == r3.digest
    assert r1.findings == r2.findings == r3.findings


def test_scan_findings_sorted_by_line_col_rule() -> None:
    code = "eval('a')\nos.system('b')\neval('c')\n"
    rules = [_OS_SYSTEM_RULE, _EVAL_RULE]
    result = scan(rules, code)
    assert len(result.findings) == 3
    assert result.findings[0].line == 1
    assert result.findings[1].line == 2
    assert result.findings[2].line == 3


def test_scan_two_rules_same_node_emits_both() -> None:
    # ``eval`` rule on CALL kind fires; an additional NAME-kind rule
    # over ``eval`` also fires on the inner Name node.
    name_rule = ScanRule(
        rule_id="PY.NAME-EVAL",
        kind=NodeKind.NAME,
        pattern="eval",
        severity=Severity.WARNING,
        message="bare eval name reference",
    )
    result = scan([_EVAL_RULE, name_rule], "eval('x')\n")
    rule_ids = {f.rule_id for f in result.findings}
    assert rule_ids == {"PY.NO-EVAL", "PY.NAME-EVAL"}


def test_scan_import_match() -> None:
    rule = ScanRule(
        rule_id="PY.NO-PICKLE",
        kind=NodeKind.IMPORT,
        pattern="pickle",
        severity=Severity.WARNING,
        message="pickle is unsafe",
    )
    result = scan([rule], "import pickle\n")
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "pickle"


def test_scan_import_from_match() -> None:
    rule = ScanRule(
        rule_id="PY.NO-OS-IMPORT",
        kind=NodeKind.IMPORT_FROM,
        pattern="os.path",
        severity=Severity.INFO,
        message="os.path import",
    )
    result = scan([rule], "from os.path import join\n")
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "os.path"


def test_scan_string_match() -> None:
    rule = ScanRule(
        rule_id="PY.SECRET",
        kind=NodeKind.STRING,
        pattern="password",
        severity=Severity.ERROR,
        message="hardcoded password",
    )
    result = scan([rule], "x = 'password'\n")
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "password"


def test_scan_assign_match() -> None:
    rule = ScanRule(
        rule_id="PY.GLOBAL-VAR",
        kind=NodeKind.ASSIGN,
        pattern="GLOBAL_STATE",
        severity=Severity.WARNING,
        message="global state",
    )
    result = scan([rule], "GLOBAL_STATE = {}\n")
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "GLOBAL_STATE"


def test_scan_attribute_chain_match_three_deep() -> None:
    rule = ScanRule(
        rule_id="PY.SUBPROCESS-RUN",
        kind=NodeKind.CALL,
        pattern="subprocess.run",
        severity=Severity.ERROR,
        message="subprocess run",
    )
    code = "import subprocess\nsubprocess.run(['ls'])\n"
    result = scan([rule], code)
    assert len(result.findings) == 1
    assert result.findings[0].snippet == "subprocess.run"


def test_scan_attribute_pattern_does_not_partial_match() -> None:
    # Pattern ``os.system`` must NOT match ``myos.system`` (different
    # head).
    rule = _OS_SYSTEM_RULE
    code = "myos.system('x')\n"
    result = scan([rule], code)
    assert result.findings == ()


def test_scan_rule_count_echoed() -> None:
    result = scan([_EVAL_RULE, _OS_SYSTEM_RULE], "x = 1\n")
    assert result.rule_count == 2


def test_scan_scanned_lines_counts_newlines() -> None:
    code = "x = 1\ny = 2\nz = 3\n"
    result = scan([_EVAL_RULE], code)
    assert result.scanned_lines == 3


# ---------------------------------------------------------------------------
# scan() — error paths
# ---------------------------------------------------------------------------


def test_scan_rejects_non_list_rules() -> None:
    with pytest.raises(ScannerError):
        scan(_EVAL_RULE, "")  # type: ignore[arg-type]


def test_scan_rejects_non_rule_in_list() -> None:
    with pytest.raises(ScannerError):
        scan([_EVAL_RULE, "not-a-rule"], "")  # type: ignore[list-item]


def test_scan_rejects_non_string_code() -> None:
    with pytest.raises(ScannerError):
        scan([_EVAL_RULE], 42)  # type: ignore[arg-type]


def test_scan_rejects_overlong_code() -> None:
    big = "x = 1\n" * (MAX_CODE_LEN // 6 + 1)
    with pytest.raises(ScannerError):
        scan([_EVAL_RULE], big)


def test_scan_rejects_syntax_error() -> None:
    with pytest.raises(ScannerError):
        scan([_EVAL_RULE], "def x(\n")


def test_scan_rejects_empty_file_path() -> None:
    with pytest.raises(ScannerError):
        scan([_EVAL_RULE], "x = 1\n", file_path="")


# ---------------------------------------------------------------------------
# Finding / ScanResult helpers
# ---------------------------------------------------------------------------


def test_scan_result_has_errors_true_on_error_finding() -> None:
    result = scan([_EVAL_RULE], "eval('x')\n")
    assert result.has_errors() is True


def test_scan_result_has_errors_false_on_clean() -> None:
    result = scan([_EVAL_RULE], "x = 1\n")
    assert result.has_errors() is False


def test_scan_result_by_rule_filters_findings() -> None:
    code = "eval('a')\nos.system('b')\n"
    result = scan([_EVAL_RULE, _OS_SYSTEM_RULE], code)
    eval_findings = result.by_rule("PY.NO-EVAL")
    assert len(eval_findings) == 1
    assert eval_findings[0].snippet == "eval"


def test_scan_result_is_frozen_and_slotted() -> None:
    result = scan([_EVAL_RULE], "x = 1\n")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.backend = "semgrep"  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_scan_result_rejects_invalid_backend() -> None:
    with pytest.raises(ScannerError):
        ScanResult(
            findings=(),
            file_path="x.py",
            rule_count=0,
            scanned_lines=0,
            backend="ROGUE",
        )


# ---------------------------------------------------------------------------
# RuleSet validation
# ---------------------------------------------------------------------------


def test_rule_set_constructs_valid() -> None:
    rs = RuleSet(name="security", rules=(_EVAL_RULE, _OS_SYSTEM_RULE))
    assert rs.name == "security"
    assert len(rs.rules) == 2


def test_rule_set_rejects_empty_name() -> None:
    with pytest.raises(ScannerError):
        RuleSet(name="", rules=())


def test_rule_set_rejects_non_rule_member() -> None:
    with pytest.raises(ScannerError):
        RuleSet(name="x", rules=("not-a-rule",))  # type: ignore[arg-type]


def test_rule_set_rejects_duplicate_rule_ids() -> None:
    dup = ScanRule(
        rule_id="PY.NO-EVAL",
        kind=NodeKind.NAME,
        pattern="eval",
        severity=Severity.WARNING,
        message="m",
    )
    with pytest.raises(ScannerError):
        RuleSet(name="x", rules=(_EVAL_RULE, dup))


# ---------------------------------------------------------------------------
# scan_suite()
# ---------------------------------------------------------------------------


def test_scan_suite_returns_suite_report() -> None:
    rs = RuleSet(name="security", rules=(_EVAL_RULE, _OS_SYSTEM_RULE))
    report = scan_suite(rs, "eval('x')\n")
    assert isinstance(report, SuiteReport)
    assert report.suite_name == "security"


def test_scan_suite_per_rule_counts() -> None:
    rs = RuleSet(name="security", rules=(_EVAL_RULE, _OS_SYSTEM_RULE))
    code = "eval('a')\neval('b')\nos.system('c')\n"
    report = scan_suite(rs, code)
    assert report.per_rule_counts["PY.NO-EVAL"] == 2
    assert report.per_rule_counts["PY.NO-OS-SYSTEM"] == 1


def test_scan_suite_total_findings() -> None:
    rs = RuleSet(name="security", rules=(_EVAL_RULE,))
    code = "eval('a')\neval('b')\n"
    report = scan_suite(rs, code)
    assert report.total_findings() == 2


def test_scan_suite_is_clean_when_no_errors() -> None:
    rs = RuleSet(name="x", rules=(_EVAL_RULE,))
    report = scan_suite(rs, "x = 1\n")
    assert report.is_clean() is True


def test_scan_suite_is_clean_false_on_error_finding() -> None:
    rs = RuleSet(name="x", rules=(_EVAL_RULE,))
    report = scan_suite(rs, "eval('x')\n")
    assert report.is_clean() is False


def test_scan_suite_rejects_non_rule_set() -> None:
    with pytest.raises(TypeError):
        scan_suite("not-a-set", "")  # type: ignore[arg-type]


def test_scan_suite_is_deterministic() -> None:
    rs = RuleSet(name="security", rules=(_EVAL_RULE, _OS_SYSTEM_RULE))
    code = "eval('a')\nos.system('b')\n"
    r1 = scan_suite(rs, code)
    r2 = scan_suite(rs, code)
    assert r1.result.digest == r2.result.digest
    assert dict(r1.per_rule_counts) == dict(r2.per_rule_counts)


# ---------------------------------------------------------------------------
# Lazy seam — enable_semgrep_factory
# ---------------------------------------------------------------------------


def test_enable_semgrep_factory_skips_when_uninstalled() -> None:
    try:
        import semgrep  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("semgrep not installed")
    scanner = enable_semgrep_factory()
    result = scanner([_EVAL_RULE], "eval('x')\n", "<test>")
    assert result.backend == "semgrep"
    assert len(result.findings) == 1


def test_enable_semgrep_factory_rejects_unknown_overrides() -> None:
    try:
        import semgrep  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("semgrep not installed")
    with pytest.raises(ScannerError):
        enable_semgrep_factory(overrides={"bogus_key": 1})


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "semgrep_scanner.py"


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


def test_no_top_level_semgrep_import() -> None:
    assert all(not name.startswith("semgrep") for name in _top_level_imports(_module_ast()))


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


def test_semgrep_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith("semgrep") or name == "subprocess":
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside enable_semgrep_factory"
                    )
                    assert parent.name == "enable_semgrep_factory", (
                        f"{name} imported in {parent.name!r} — must be "
                        "inside enable_semgrep_factory"
                    )


# ---------------------------------------------------------------------------
# Realistic scan demo — pin governance-critical patterns
# ---------------------------------------------------------------------------


def test_realistic_security_scan() -> None:
    code = """
import os
import subprocess


def handler(user_input):
    eval(user_input)
    os.system("ls " + user_input)
    subprocess.run(user_input, shell=True)
    return True
"""
    rules = (
        _EVAL_RULE,
        _OS_SYSTEM_RULE,
        ScanRule(
            rule_id="PY.NO-SUBPROCESS-RUN",
            kind=NodeKind.CALL,
            pattern="subprocess.run",
            severity=Severity.ERROR,
            message="subprocess.run is unsafe with shell=True",
        ),
    )
    result = scan(rules, code, file_path="handler.py")
    assert len(result.findings) == 3
    rule_ids = {f.rule_id for f in result.findings}
    assert rule_ids == {
        "PY.NO-EVAL",
        "PY.NO-OS-SYSTEM",
        "PY.NO-SUBPROCESS-RUN",
    }
    assert result.has_errors()


# ---------------------------------------------------------------------------
# Reload idempotency (runs last — reload invalidates earlier enum refs)
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import tools.semgrep_scanner as mod1

    importlib.reload(mod1)
    import tools.semgrep_scanner as mod2

    assert mod1.SCANNER_VERSION == mod2.SCANNER_VERSION
    assert mod1.MAX_CODE_LEN == mod2.MAX_CODE_LEN
    assert mod1.Severity.ERROR is mod2.Severity.ERROR
