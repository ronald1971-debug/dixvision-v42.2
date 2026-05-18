"""Test suite for I-05 :mod:`system_engine.dev_logger`.

Pinned guarantees (canonical doc + AST guardrails):

* Module surface is the documented set (no extras).
* :class:`DevFormatConfig` is frozen + slotted, all-bool fields validated.
* :func:`format_dev_line` is a pure function (INV-15 byte-identical).
* :class:`DevLogger` applies the I-04 redaction + tier + required-field
  matrix unchanged (delegates to :func:`build_log_line`).
* :func:`stdlib_dev_logger_factory` is always available.
* :func:`enable_loguru_dev_logger_factory` is a lazy seam — ``loguru`` is
  only imported inside the function body.
* No top-level imports of ``loguru`` / ``time`` / ``datetime`` /
  ``random`` / ``asyncio`` / ``os`` / ``numpy`` / ``torch`` / ``polars``
  / ``requests`` (AST-pinned).
* No typed-event constructors anywhere in the module (B27/B28/INV-71).
* No runtime-tier cross-imports (B1).
"""

from __future__ import annotations

import ast
import importlib
import io
from pathlib import Path
from typing import Any

import pytest

from system_engine import dev_logger as dev_logger_mod
from system_engine.dev_logger import (
    DEV_LOGGER_VERSION,
    NEW_PIP_DEPENDENCIES,
    DevFormatConfig,
    DevLogger,
    enable_loguru_dev_logger_factory,
    format_dev_line,
    stdlib_dev_logger_factory,
)
from system_engine.logging import (
    LoggingError,
    LogLevel,
    LogLine,
    build_log_line,
)

_MODULE_PATH = Path(dev_logger_mod.__file__)
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_new_pip_dependencies(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ("loguru",)

    def test_version_string(self) -> None:
        assert DEV_LOGGER_VERSION == "v1.0-I05"

    def test_all_exports(self) -> None:
        assert sorted(dev_logger_mod.__all__) == sorted(
            [
                "DEV_LOGGER_VERSION",
                "DevFormatConfig",
                "DevLogger",
                "NEW_PIP_DEPENDENCIES",
                "STRUCT_LOGGER_VERSION",
                "enable_loguru_dev_logger_factory",
                "format_dev_line",
                "stdlib_dev_logger_factory",
            ]
        )


# ---------------------------------------------------------------------------
# DevFormatConfig
# ---------------------------------------------------------------------------


class TestDevFormatConfig:
    def test_defaults(self) -> None:
        cfg = DevFormatConfig()
        assert cfg.use_color is False
        assert cfg.include_engine is True
        assert cfg.include_tier is True
        assert cfg.include_fields is True
        assert cfg.key_sort is True

    def test_frozen(self) -> None:
        cfg = DevFormatConfig()
        with pytest.raises((AttributeError, TypeError, ValueError)):
            cfg.use_color = True  # type: ignore[misc]

    def test_slotted_no_dict(self) -> None:
        cfg = DevFormatConfig()
        assert not hasattr(cfg, "__dict__")

    def test_rejects_non_bool_use_color(self) -> None:
        with pytest.raises(LoggingError):
            DevFormatConfig(use_color="yes")  # type: ignore[arg-type]

    def test_rejects_non_bool_include_engine(self) -> None:
        with pytest.raises(LoggingError):
            DevFormatConfig(include_engine=1)  # type: ignore[arg-type]

    def test_rejects_non_bool_include_tier(self) -> None:
        with pytest.raises(LoggingError):
            DevFormatConfig(include_tier=None)  # type: ignore[arg-type]

    def test_rejects_non_bool_include_fields(self) -> None:
        with pytest.raises(LoggingError):
            DevFormatConfig(include_fields=0)  # type: ignore[arg-type]

    def test_rejects_non_bool_key_sort(self) -> None:
        with pytest.raises(LoggingError):
            DevFormatConfig(key_sort="true")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# format_dev_line
# ---------------------------------------------------------------------------


def _line(
    *,
    ts_ns: int = 1_000,
    engine_id: str = "test-engine",
    tier: str = "system",
    event_kind: str = "hello_world",
    level: LogLevel = LogLevel.INFO,
    fields: dict[str, Any] | None = None,
) -> LogLine:
    return build_log_line(
        ts_ns=ts_ns,
        engine_id=engine_id,
        tier=tier,
        event_kind=event_kind,
        level=level,
        fields=fields or {},
    )


class TestFormatDevLine:
    def test_basic_output(self) -> None:
        out = format_dev_line(_line(fields={"a": 1}), DevFormatConfig())
        assert out == "ts_ns=1000 [system/test-engine] INFO hello_world | a=1"

    def test_byte_stable_same_input(self) -> None:
        line = _line(fields={"b": 2, "a": 1})
        cfg = DevFormatConfig()
        assert format_dev_line(line, cfg) == format_dev_line(line, cfg)

    def test_keys_sorted_by_default(self) -> None:
        out = format_dev_line(_line(fields={"b": 2, "a": 1}), DevFormatConfig())
        assert "a=1 b=2" in out
        assert out.index("a=1") < out.index("b=2")

    def test_key_sort_off_preserves_insertion(self) -> None:
        out = format_dev_line(
            _line(fields={"b": 2, "a": 1}),
            DevFormatConfig(key_sort=False),
        )
        assert "b=2 a=1" in out

    def test_include_fields_off(self) -> None:
        out = format_dev_line(
            _line(fields={"x": 9}),
            DevFormatConfig(include_fields=False),
        )
        assert " | " not in out
        assert "x=9" not in out

    def test_include_engine_off(self) -> None:
        out = format_dev_line(
            _line(),
            DevFormatConfig(include_engine=False),
        )
        assert "test-engine" not in out
        assert "[system]" in out

    def test_include_tier_off(self) -> None:
        out = format_dev_line(
            _line(),
            DevFormatConfig(include_tier=False),
        )
        assert "system" not in out.split("|")[0]
        assert "[test-engine]" in out

    def test_include_both_off(self) -> None:
        out = format_dev_line(
            _line(),
            DevFormatConfig(include_engine=False, include_tier=False),
        )
        assert "[" not in out
        assert "]" not in out

    def test_color_wraps_level(self) -> None:
        out = format_dev_line(
            _line(level=LogLevel.WARNING),
            DevFormatConfig(use_color=True),
        )
        assert "\x1b[33m" in out
        assert "\x1b[0m" in out
        assert "WARNING" in out

    def test_color_off_no_ansi(self) -> None:
        out = format_dev_line(_line(), DevFormatConfig(use_color=False))
        assert "\x1b" not in out

    def test_no_trailing_newline(self) -> None:
        out = format_dev_line(_line(), DevFormatConfig())
        assert not out.endswith("\n")

    def test_string_value_quoted(self) -> None:
        out = format_dev_line(_line(fields={"k": "v"}), DevFormatConfig())
        assert '| k="v"' in out

    def test_string_value_escapes(self) -> None:
        out = format_dev_line(
            _line(fields={"k": 'a"b\\c'}),
            DevFormatConfig(),
        )
        assert 'k="a\\"b\\\\c"' in out

    def test_bool_values(self) -> None:
        out = format_dev_line(
            _line(fields={"t": True, "f": False}),
            DevFormatConfig(),
        )
        assert "f=false" in out and "t=true" in out

    def test_none_value(self) -> None:
        out = format_dev_line(_line(fields={"x": None}), DevFormatConfig())
        assert "x=null" in out

    def test_list_value(self) -> None:
        out = format_dev_line(_line(fields={"xs": [1, 2, 3]}), DevFormatConfig())
        assert "xs=[1,2,3]" in out

    def test_mapping_value_sorted(self) -> None:
        out = format_dev_line(
            _line(fields={"obj": {"z": 1, "a": 2}}),
            DevFormatConfig(),
        )
        assert "obj={a=2,z=1}" in out

    def test_redacted_value_passes_through(self) -> None:
        # build_log_line redacts forbidden keys; the dev formatter must
        # not unredact them.
        out = format_dev_line(
            _line(fields={"api_key": "abc"}),
            DevFormatConfig(),
        )
        assert "<redacted>" in out
        assert "abc" not in out

    def test_rejects_non_logline(self) -> None:
        with pytest.raises(LoggingError):
            format_dev_line("not-a-line", DevFormatConfig())  # type: ignore[arg-type]

    def test_rejects_non_config(self) -> None:
        with pytest.raises(LoggingError):
            format_dev_line(_line(), "not-a-config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DevLogger
# ---------------------------------------------------------------------------


class TestDevLogger:
    def test_basic_log_writes_string(self) -> None:
        buf: list[str] = []
        log = DevLogger(
            engine_id="e",
            tier="system",
            text_sink=buf.append,
        )
        line = log.log(ts_ns=10, event_kind="hi", fields={"a": 1})
        assert line is not None
        assert len(buf) == 1
        assert buf[0].startswith("ts_ns=10 ")
        assert "hi" in buf[0]
        assert "a=1" in buf[0]

    def test_suppress_below_min_level(self) -> None:
        buf: list[str] = []
        log = DevLogger(
            engine_id="e",
            tier="system",
            text_sink=buf.append,
            min_level=LogLevel.WARNING,
        )
        result = log.log(ts_ns=1, event_kind="x", level=LogLevel.INFO)
        assert result is None
        assert buf == []

    def test_emits_at_min_level(self) -> None:
        buf: list[str] = []
        log = DevLogger(
            engine_id="e",
            tier="system",
            text_sink=buf.append,
            min_level=LogLevel.WARNING,
        )
        result = log.log(ts_ns=1, event_kind="x", level=LogLevel.WARNING)
        assert result is not None
        assert len(buf) == 1

    def test_emits_above_min_level(self) -> None:
        buf: list[str] = []
        log = DevLogger(
            engine_id="e",
            tier="system",
            text_sink=buf.append,
            min_level=LogLevel.WARNING,
        )
        result = log.log(ts_ns=1, event_kind="x", level=LogLevel.ERROR)
        assert result is not None
        assert len(buf) == 1

    def test_redaction_applies(self) -> None:
        buf: list[str] = []
        log = DevLogger(engine_id="e", tier="system", text_sink=buf.append)
        log.log(ts_ns=1, event_kind="x", fields={"password": "hunter2"})
        assert len(buf) == 1
        assert "hunter2" not in buf[0]
        assert "<redacted>" in buf[0]

    def test_invalid_tier_rejected_at_log_time(self) -> None:
        buf: list[str] = []
        with pytest.raises(LoggingError):
            DevLogger(engine_id="e", tier="not-a-tier", text_sink=buf.append)

    def test_empty_engine_rejected(self) -> None:
        with pytest.raises(LoggingError):
            DevLogger(engine_id="", tier="system", text_sink=lambda _: None)

    def test_non_callable_sink_rejected(self) -> None:
        with pytest.raises(LoggingError):
            DevLogger(engine_id="e", tier="system", text_sink="not-callable")  # type: ignore[arg-type]

    def test_bad_min_level_rejected(self) -> None:
        with pytest.raises(LoggingError):
            DevLogger(
                engine_id="e",
                tier="system",
                text_sink=lambda _: None,
                min_level="VERBOSE",  # type: ignore[arg-type]
            )

    def test_bad_config_rejected(self) -> None:
        with pytest.raises(LoggingError):
            DevLogger(
                engine_id="e",
                tier="system",
                text_sink=lambda _: None,
                config="not-a-config",  # type: ignore[arg-type]
            )

    def test_frozen(self) -> None:
        log = DevLogger(engine_id="e", tier="system", text_sink=lambda _: None)
        with pytest.raises((AttributeError, TypeError, ValueError)):
            log.engine_id = "f"  # type: ignore[misc]

    def test_3run_byte_identical_replay(self) -> None:
        # INV-15: three runs over the same caller-supplied inputs produce
        # byte-identical text output.
        outputs: list[list[str]] = []
        for _ in range(3):
            buf: list[str] = []
            log = DevLogger(engine_id="e", tier="system", text_sink=buf.append)
            log.log(ts_ns=1, event_kind="a", fields={"b": 2, "a": 1})
            log.log(ts_ns=2, event_kind="b", level=LogLevel.WARNING, fields={})
            outputs.append(buf)
        assert outputs[0] == outputs[1] == outputs[2]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestStdlibFactory:
    def test_returns_dev_logger(self) -> None:
        sink: list[str] = []
        log = stdlib_dev_logger_factory(engine_id="e", tier="system", text_sink=sink.append)
        assert isinstance(log, DevLogger)
        assert log.engine_id == "e"
        assert log.tier == "system"

    def test_min_level_string_parsed(self) -> None:
        log = stdlib_dev_logger_factory(
            engine_id="e",
            tier="system",
            text_sink=lambda _: None,
            min_level="ERROR",
        )
        assert log.min_level == LogLevel.ERROR

    def test_default_config(self) -> None:
        log = stdlib_dev_logger_factory(
            engine_id="e",
            tier="system",
            text_sink=lambda _: None,
        )
        assert log.config == DevFormatConfig()

    def test_custom_config(self) -> None:
        cfg = DevFormatConfig(use_color=True)
        log = stdlib_dev_logger_factory(
            engine_id="e",
            tier="system",
            text_sink=lambda _: None,
            config=cfg,
        )
        assert log.config is cfg


class TestLoguruSeam:
    def test_seam_returns_none_or_shim(self) -> None:
        # The seam may return None if loguru is not installed.  When it
        # returns a shim, that shim must expose a callable ``log`` method.
        result = enable_loguru_dev_logger_factory(engine_id="e", tier="system")
        assert result is None or callable(getattr(result, "log", None))


# ---------------------------------------------------------------------------
# AST guardrails (mandatory per system_guidance)
# ---------------------------------------------------------------------------


_FORBIDDEN_TOPLEVEL_IMPORTS = frozenset(
    {
        "loguru",
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "requests",
    }
)


def _toplevel_imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


class TestAstGuardrails:
    def test_no_forbidden_toplevel_imports(self) -> None:
        toplevel = _toplevel_imports(_MODULE_AST)
        assert toplevel.isdisjoint(_FORBIDDEN_TOPLEVEL_IMPORTS), (
            f"forbidden top-level imports: {toplevel & _FORBIDDEN_TOPLEVEL_IMPORTS}"
        )

    def test_loguru_only_inside_lazy_seam(self) -> None:
        # Every ``import loguru`` / ``from loguru import ...`` must live
        # inside the body of ``enable_loguru_dev_logger_factory``.
        offenders: list[tuple[int, str]] = []
        seam_lineno_range: tuple[int, int] | None = None
        for node in ast.walk(_MODULE_AST):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "enable_loguru_dev_logger_factory"
            ):
                seam_lineno_range = (
                    node.lineno,
                    node.end_lineno or node.lineno,
                )
        assert seam_lineno_range is not None
        lo, hi = seam_lineno_range
        for node in ast.walk(_MODULE_AST):
            if isinstance(node, ast.ImportFrom) and node.module == "loguru":
                if not (lo <= node.lineno <= hi):
                    offenders.append((node.lineno, "from loguru import ..."))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "loguru" and not (lo <= node.lineno <= hi):
                        offenders.append((node.lineno, "import loguru"))
        assert not offenders, f"loguru imported outside lazy seam: {offenders}"

    def test_no_typed_event_constructors(self) -> None:
        forbidden = {
            "PatchProposal",
            "HazardEvent",
            "SignalEvent",
            "ExecutionEvent",
            "SystemEvent",
            "LearningUpdate",
        }
        offenders: list[tuple[int, str]] = []
        for node in ast.walk(_MODULE_AST):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden:
                    offenders.append((node.lineno, node.func.id))
        assert not offenders, f"typed-event constructors: {offenders}"

    def test_no_runtime_tier_imports(self) -> None:
        forbidden_tiers = {
            "intelligence_engine",
            "execution_engine",
            "governance_engine",
            "evolution_engine",
            "learning_engine",
        }
        for node in _MODULE_AST.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_tiers, f"B1 violation: imports {node.module}"

    def test_no_wallclock_reads(self) -> None:
        # No call to time.time / time.monotonic / datetime.now / etc.
        # in module body — INV-15.  Uses AST inspection (not source
        # text grep) so docstring examples are ignored.
        forbidden_attrs = {
            ("time", "time"),
            ("time", "monotonic"),
            ("time", "monotonic_ns"),
            ("time", "time_ns"),
            ("time", "perf_counter"),
            ("datetime", "now"),
            ("datetime", "utcnow"),
        }
        for node in ast.walk(_MODULE_AST):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func
                if isinstance(target.value, ast.Name):
                    pair = (target.value.id, target.attr)
                    assert pair not in forbidden_attrs, (
                        f"wall-clock call {pair} at line {node.lineno}"
                    )

    def test_module_imports_clean(self) -> None:
        # Re-importing the module must not mutate any globals — replay
        # determinism (INV-15).  We deliberately avoid ``importlib.reload``
        # to keep class identity stable across the rest of the suite.
        mod1 = importlib.import_module("system_engine.dev_logger")
        mod2 = importlib.import_module("system_engine.dev_logger")
        assert mod1 is mod2
        assert mod2.NEW_PIP_DEPENDENCIES == NEW_PIP_DEPENDENCIES
        assert mod2.DEV_LOGGER_VERSION == DEV_LOGGER_VERSION


# ---------------------------------------------------------------------------
# Integration: sink contracts
# ---------------------------------------------------------------------------


class TestSinkIntegration:
    def test_io_stringio_sink(self) -> None:
        buf = io.StringIO()
        log = DevLogger(
            engine_id="e",
            tier="system",
            text_sink=lambda s: buf.write(s + "\n"),
        )
        log.log(ts_ns=1, event_kind="hi")
        assert buf.getvalue().startswith("ts_ns=1 ")
        assert buf.getvalue().endswith("\n")

    def test_multiple_log_calls_byte_stable(self) -> None:
        runs: list[list[str]] = []
        for _ in range(3):
            buf: list[str] = []
            log = DevLogger(engine_id="e", tier="system", text_sink=buf.append)
            log.log(ts_ns=1, event_kind="a")
            log.log(ts_ns=2, event_kind="b", fields={"x": 1})
            log.log(ts_ns=3, event_kind="c", level=LogLevel.ERROR)
            runs.append(buf)
        assert runs[0] == runs[1] == runs[2]
