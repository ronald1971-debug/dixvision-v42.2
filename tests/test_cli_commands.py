# ADAPTED FROM: https://github.com/tiangolo/typer  (MIT)
# ADAPTED FROM: https://github.com/Textualize/rich  (MIT)
"""Tests for the canonical CLI command surface (I-06 + I-07)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tools import cli
from tools.cli import (
    CANONICAL_APP,
    CLI_VERSION,
    NEW_PIP_DEPENDENCIES,
    CliApp,
    CliCommand,
    CliError,
    CliOption,
    CliResult,
    dispatch,
    enable_rich_formatter_factory,
    enable_typer_factory,
    format_progress,
    format_table,
    main,
)

# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------


def test_version_tag() -> None:
    assert CLI_VERSION == "v1.0-I06-I07"


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("typer", "rich")


def test_all_export_complete() -> None:
    expected = {
        "CANONICAL_APP",
        "CLI_VERSION",
        "CliApp",
        "CliCommand",
        "CliError",
        "CliOption",
        "CliResult",
        "NEW_PIP_DEPENDENCIES",
        "dispatch",
        "enable_rich_formatter_factory",
        "enable_typer_factory",
        "format_progress",
        "format_table",
        "main",
    }
    assert set(cli.__all__) == expected


def test_canonical_app_is_frozen_slotted() -> None:
    assert CANONICAL_APP.__class__.__dataclass_params__.frozen is True
    assert "__slots__" in CANONICAL_APP.__class__.__dict__


def test_canonical_app_command_names() -> None:
    names = tuple(c.name for c in CANONICAL_APP.commands)
    assert names == ("run", "backtest", "governance", "status", "validate", "replay")


# ---------------------------------------------------------------------------
# CliOption validation
# ---------------------------------------------------------------------------


def test_clioption_rejects_empty_name() -> None:
    with pytest.raises(CliError):
        CliOption("", str, "", "h")


def test_clioption_rejects_unsupported_type() -> None:
    with pytest.raises(CliError):
        CliOption("x", list, [], "h")


def test_clioption_rejects_mismatched_default() -> None:
    with pytest.raises(CliError):
        CliOption("x", int, "no", "h")


def test_clioption_allows_none_default() -> None:
    opt = CliOption("x", str, None, "h")
    assert opt.default is None


# ---------------------------------------------------------------------------
# CliCommand / CliApp validation
# ---------------------------------------------------------------------------


def test_clicommand_rejects_empty_name() -> None:
    with pytest.raises(CliError):
        CliCommand("", "h", (), lambda _: CliResult(0, ""))


def test_clicommand_rejects_non_callable_handler() -> None:
    with pytest.raises(CliError):
        CliCommand("x", "h", (), "nope")  # type: ignore[arg-type]


def test_clicommand_rejects_duplicate_option_name() -> None:
    o = CliOption("x", str, "a", "h")
    with pytest.raises(CliError):
        CliCommand("c", "h", (o, o), lambda _: CliResult(0, ""))


def test_cliapp_rejects_duplicate_command_name() -> None:
    cmd = CliCommand("run", "h", (), lambda _: CliResult(0, ""))
    with pytest.raises(CliError):
        CliApp("dix", "h", (cmd, cmd))


# ---------------------------------------------------------------------------
# CliResult validation
# ---------------------------------------------------------------------------


def test_cliresult_rejects_non_int_exit_code() -> None:
    with pytest.raises(CliError):
        CliResult("0", "")  # type: ignore[arg-type]


def test_cliresult_rejects_non_string_stdout() -> None:
    with pytest.raises(CliError):
        CliResult(0, 1)  # type: ignore[arg-type]


def test_cliresult_default_backend_is_stdlib() -> None:
    r = CliResult(0, "ok")
    assert r.backend == "stdlib"


# ---------------------------------------------------------------------------
# dispatch determinism (INV-15)
# ---------------------------------------------------------------------------


def test_dispatch_status_default() -> None:
    r = dispatch(CANONICAL_APP, ["status"])
    assert r.exit_code == 0
    assert "system.status" in r.stdout
    assert "verbose=False" in r.stdout


def test_dispatch_run_with_explicit_port() -> None:
    r = dispatch(CANONICAL_APP, ["run", "--port", "9090"])
    assert r.exit_code == 0
    assert "port=9090" in r.stdout
    assert "host='127.0.0.1'" in r.stdout


def test_dispatch_backtest_seed() -> None:
    r = dispatch(CANONICAL_APP, ["backtest", "--symbol", "ETHUSDT", "--seed", "7"])
    assert r.exit_code == 0
    assert "symbol='ETHUSDT'" in r.stdout
    assert "seed=7" in r.stdout


def test_dispatch_governance_status() -> None:
    r = dispatch(CANONICAL_APP, ["governance"])
    assert r.exit_code == 0
    assert "governance.status" in r.stdout


def test_dispatch_governance_audit() -> None:
    r = dispatch(CANONICAL_APP, ["governance", "--subcommand", "audit"])
    assert r.exit_code == 0
    assert "governance.audit" in r.stdout


def test_dispatch_governance_rejects_unknown_subcommand() -> None:
    r = dispatch(CANONICAL_APP, ["governance", "--subcommand", "nuke"])
    assert r.exit_code == 2
    assert "unknown subcommand" in r.stderr


def test_dispatch_status_verbose_flag() -> None:
    r = dispatch(CANONICAL_APP, ["status", "--verbose"])
    assert "verbose=True" in r.stdout


def test_dispatch_validate_strict_flag() -> None:
    r = dispatch(CANONICAL_APP, ["validate", "--strict"])
    assert r.exit_code == 0
    assert "strict=True" in r.stdout


def test_dispatch_replay_path() -> None:
    r = dispatch(CANONICAL_APP, ["replay", "--path", "/tmp/ledger.db"])
    assert r.exit_code == 0
    assert "path='/tmp/ledger.db'" in r.stdout


def test_dispatch_unknown_command_returns_error_exit_code() -> None:
    r = dispatch(CANONICAL_APP, ["does-not-exist"])
    assert r.exit_code != 0


def test_dispatch_no_args_returns_error_exit_code() -> None:
    r = dispatch(CANONICAL_APP, [])
    assert r.exit_code != 0


def test_dispatch_help_flag_returns_zero() -> None:
    r = dispatch(CANONICAL_APP, ["--help"])
    assert r.exit_code == 0


def test_dispatch_is_pure_three_runs_byte_identical() -> None:
    a = dispatch(CANONICAL_APP, ["run", "--port", "8080"])
    b = dispatch(CANONICAL_APP, ["run", "--port", "8080"])
    c = dispatch(CANONICAL_APP, ["run", "--port", "8080"])
    assert (a, b) == (b, c)


def test_dispatch_status_three_runs_byte_identical() -> None:
    a = dispatch(CANONICAL_APP, ["status"])
    b = dispatch(CANONICAL_APP, ["status"])
    c = dispatch(CANONICAL_APP, ["status"])
    assert a.stdout == b.stdout == c.stdout


def test_main_returns_exit_code() -> None:
    rc = main(["status"])
    assert rc == 0


def test_main_unknown_command_returns_nonzero() -> None:
    rc = main(["nope"])
    assert rc != 0


# ---------------------------------------------------------------------------
# Lazy seam: typer factory
# ---------------------------------------------------------------------------


def test_enable_typer_factory_lazy_import() -> None:
    try:
        d = enable_typer_factory()
    except ImportError:
        pytest.skip("typer not installed")
    r = d(CANONICAL_APP, ["status"])
    assert r.exit_code == 0
    assert r.backend == "typer"


def test_typer_factory_byte_identical_with_stdlib() -> None:
    try:
        d = enable_typer_factory()
    except ImportError:
        pytest.skip("typer not installed")
    stdlib = dispatch(CANONICAL_APP, ["run", "--port", "9000"])
    typer_r = d(CANONICAL_APP, ["run", "--port", "9000"])
    assert stdlib.stdout == typer_r.stdout
    assert stdlib.exit_code == typer_r.exit_code


def test_enable_rich_formatter_factory_lazy() -> None:
    try:
        install = enable_rich_formatter_factory()
    except ImportError:
        pytest.skip("rich not installed")
    # Returned callable is a no-op installer.
    assert callable(install)
    assert install() is None


# ---------------------------------------------------------------------------
# format_table / format_progress
# ---------------------------------------------------------------------------


def test_format_table_simple() -> None:
    out = format_table(["A", "B"], [["1", "2"], ["3", "4"]])
    assert "| A " in out
    assert "| 1 " in out
    assert out.endswith("\n")


def test_format_table_rejects_empty_headers() -> None:
    with pytest.raises(CliError):
        format_table([], [])


def test_format_table_handles_empty_rows() -> None:
    out = format_table(["X"], [])
    assert "| X " in out


def test_format_progress_half() -> None:
    out = format_progress("backtest", 50, 100)
    assert "50%" in out
    assert "50/100" in out
    assert "backtest" in out


def test_format_progress_zero_total_is_complete() -> None:
    out = format_progress("noop", 0, 0)
    assert "100%" in out


def test_format_progress_rejects_negative() -> None:
    with pytest.raises(CliError):
        format_progress("x", -1, 10)


def test_format_progress_clamps_overflow() -> None:
    out = format_progress("x", 200, 100)
    assert "100%" in out


# ---------------------------------------------------------------------------
# AST guardrails (B1 / INV-15 / B27 / B28 / INV-71)
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = (
    "typer",
    "rich",
    "time",
    "datetime",
    "random",
    "asyncio",
    "numpy",
    "torch",
    "polars",
    "requests",
)


def _module_source() -> str:
    return Path(inspect.getfile(cli)).read_text(encoding="utf-8")


def _toplevel_imports() -> set[str]:
    tree = ast.parse(_module_source())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_top_level_imports() -> None:
    found = _toplevel_imports()
    for banned in _FORBIDDEN_TOP_LEVEL_IMPORTS:
        assert banned not in found, f"forbidden top-level import: {banned}"


def test_no_runtime_engine_imports_b1() -> None:
    forbidden_prefixes = (
        "execution_engine",
        "intelligence_engine",
        "governance_engine",
        "system_engine.engine",
    )
    found = _toplevel_imports()
    for prefix in forbidden_prefixes:
        for name in found:
            assert not name.startswith(prefix), f"B1 violation: {name}"


def test_no_typed_event_constructors_b27_b28_inv71() -> None:
    src = _module_source()
    # Typed events use ``*Event(`` ctors; the CLI module only emits plain text.
    for kind in ("SignalEvent(", "ExecutionEvent(", "HazardEvent(", "LearningUpdate("):
        assert kind not in src, f"B27/B28/INV-71 violation: {kind} present"


def test_lazy_seams_have_function_local_imports() -> None:
    src = _module_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enable_typer_factory":
            imports = [n for n in ast.walk(node) if isinstance(n, ast.Import)]
            assert any(alias.name == "typer" for imp in imports for alias in imp.names)
        elif isinstance(node, ast.FunctionDef) and node.name == "enable_rich_formatter_factory":
            imports = [n for n in ast.walk(node) if isinstance(n, ast.Import)]
            assert any(alias.name == "rich" for imp in imports for alias in imp.names)
