# ADAPTED FROM: https://github.com/hynek/structlog  (Apache-2.0)
"""Tests for the canonical structured-logging facade (I-04)."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from system_engine import logging as logging_mod
from system_engine.logging import (
    LOGGER_VERSION,
    NEW_PIP_DEPENDENCIES,
    LoggingError,
    LogLevel,
    LogLine,
    StructuredLogger,
    build_log_line,
    enable_structlog_factory,
    stdlib_logger_factory,
)

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_version_tag() -> None:
    assert LOGGER_VERSION == "v1.0-I04"


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("structlog",)


def test_log_level_enum_values() -> None:
    assert LogLevel.DEBUG.value == "DEBUG"
    assert LogLevel.INFO.value == "INFO"
    assert LogLevel.WARNING.value == "WARNING"
    assert LogLevel.ERROR.value == "ERROR"
    assert LogLevel.CRITICAL.value == "CRITICAL"


# ---------------------------------------------------------------------------
# build_log_line — happy path
# ---------------------------------------------------------------------------


def _ok_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ts_ns": 1_700_000_000_000_000_000,
        "engine_id": "governance",
        "tier": "governance",
        "event_kind": "decision_signed",
        "level": LogLevel.INFO,
        "fields": {"decision_id": "abc"},
    }
    base.update(overrides)
    return base


def test_build_log_line_returns_frozen_line() -> None:
    line = build_log_line(**_ok_kwargs())  # type: ignore[arg-type]
    assert isinstance(line, LogLine)
    with pytest.raises((AttributeError, TypeError)):
        line.ts_ns = 0  # type: ignore[misc]


def test_build_log_line_accepts_string_level() -> None:
    line = build_log_line(**_ok_kwargs(level="WARNING"))  # type: ignore[arg-type]
    assert line.level is LogLevel.WARNING


def test_build_log_line_defaults_to_info_level() -> None:
    kwargs = _ok_kwargs()
    kwargs.pop("level")
    line = build_log_line(**kwargs)  # type: ignore[arg-type]
    assert line.level is LogLevel.INFO


def test_build_log_line_allows_empty_fields() -> None:
    kwargs = _ok_kwargs(fields=None)
    line = build_log_line(**kwargs)  # type: ignore[arg-type]
    assert line.fields == {}


# ---------------------------------------------------------------------------
# build_log_line — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ts_ns": "string"},
        {"ts_ns": -1},
        {"ts_ns": True},  # bool is NOT a valid int here
        {"engine_id": ""},
        {"engine_id": 123},
        {"tier": "not-a-tier"},
        {"tier": ""},
        {"event_kind": ""},
        {"event_kind": 0},
        {"level": "TRACE"},
    ],
)
def test_build_log_line_rejects_bad_input(kwargs: dict[str, object]) -> None:
    with pytest.raises(LoggingError):
        build_log_line(**_ok_kwargs(**kwargs))  # type: ignore[arg-type]


def test_build_log_line_rejects_non_str_field_key() -> None:
    with pytest.raises(LoggingError):
        build_log_line(**_ok_kwargs(fields={1: "x"}))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scrubber — forbidden keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "credentials",
        "credential",
        "api_key",
        "API_KEY",
        "apikey",
        "secret",
        "private_key",
        "privateKey",
        "password",
        "position_size",
        "positionsize",
        "notional",
        "balance",
        "wallet_balance",
    ],
)
def test_scrubber_redacts_forbidden_keys(key: str) -> None:
    line = build_log_line(**_ok_kwargs(fields={key: "leaked-value"}))  # type: ignore[arg-type]
    assert line.fields[key] == "<redacted>"


def test_scrubber_redacts_nested_forbidden_keys() -> None:
    payload = {"outer": {"api_key": "leak", "ok": 1}}
    line = build_log_line(**_ok_kwargs(fields=payload))  # type: ignore[arg-type]
    rendered = json.loads(line.to_bytes())
    assert rendered["fields"]["outer"]["api_key"] == "<redacted>"
    assert rendered["fields"]["outer"]["ok"] == 1


def test_scrubber_leaves_innocent_keys_alone() -> None:
    payload = {"decision_id": "abc", "regime": "calm"}
    line = build_log_line(**_ok_kwargs(fields=payload))  # type: ignore[arg-type]
    assert line.fields == payload


# ---------------------------------------------------------------------------
# LogLine.to_bytes — canonical JSON shape
# ---------------------------------------------------------------------------


def test_to_bytes_emits_sorted_canonical_json() -> None:
    line = build_log_line(**_ok_kwargs(fields={"b": 1, "a": 2}))  # type: ignore[arg-type]
    blob = line.to_bytes()
    assert isinstance(blob, bytes)
    # alphabetic key order at top level, no whitespace
    text = blob.decode("utf-8")
    assert b", " not in blob and b": " not in blob
    # required top-level fields all present
    parsed = json.loads(text)
    for required in ("ts_ns", "engine_id", "tier", "event_kind", "level", "fields"):
        assert required in parsed


def test_to_bytes_is_pure_function() -> None:
    line = build_log_line(**_ok_kwargs())  # type: ignore[arg-type]
    a = line.to_bytes()
    b = line.to_bytes()
    c = line.to_bytes()
    assert a == b == c


def test_to_bytes_byte_identical_under_field_permutation() -> None:
    base = _ok_kwargs(fields={"a": 1, "b": 2, "c": 3})
    perm = _ok_kwargs(fields={"c": 3, "a": 1, "b": 2})
    assert build_log_line(**base).to_bytes() == build_log_line(**perm).to_bytes()  # type: ignore[arg-type]


def test_to_bytes_rejects_non_canonicalisable_field() -> None:
    class Opaque:
        pass

    line = build_log_line(**_ok_kwargs(fields={"bad": Opaque()}))  # type: ignore[arg-type]
    with pytest.raises(LoggingError):
        line.to_bytes()


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


class _CaptureSink:
    def __init__(self) -> None:
        self.lines: list[bytes] = []

    def __call__(self, blob: bytes) -> None:
        self.lines.append(blob)


def test_logger_writes_above_min_level() -> None:
    sink = _CaptureSink()
    logger = stdlib_logger_factory(
        engine_id="governance",
        tier="governance",
        sink=sink,
        min_level=LogLevel.INFO,
    )
    out = logger.log(ts_ns=42, event_kind="evt", level=LogLevel.WARNING, fields={"k": "v"})
    assert out is not None
    assert len(sink.lines) == 1
    assert json.loads(sink.lines[0])["event_kind"] == "evt"


def test_logger_suppresses_below_min_level() -> None:
    sink = _CaptureSink()
    logger = stdlib_logger_factory(
        engine_id="governance",
        tier="governance",
        sink=sink,
        min_level=LogLevel.WARNING,
    )
    out = logger.log(ts_ns=42, event_kind="evt", level=LogLevel.INFO)
    assert out is None
    assert sink.lines == []


def test_logger_accepts_string_min_level() -> None:
    sink = _CaptureSink()
    logger = stdlib_logger_factory(
        engine_id="governance",
        tier="governance",
        sink=sink,
        min_level="ERROR",
    )
    assert logger.min_level is LogLevel.ERROR


def test_logger_rejects_bad_engine_id() -> None:
    with pytest.raises(LoggingError):
        StructuredLogger(engine_id="", tier="governance", sink=lambda _b: None)


def test_logger_rejects_bad_tier() -> None:
    with pytest.raises(LoggingError):
        StructuredLogger(engine_id="x", tier="bogus", sink=lambda _b: None)


def test_logger_rejects_non_callable_sink() -> None:
    with pytest.raises(LoggingError):
        StructuredLogger(engine_id="x", tier="governance", sink="not-callable")  # type: ignore[arg-type]


def test_logger_propagates_log_line_validation() -> None:
    sink = _CaptureSink()
    logger = stdlib_logger_factory(engine_id="x", tier="governance", sink=sink)
    with pytest.raises(LoggingError):
        logger.log(ts_ns=-1, event_kind="evt")


# ---------------------------------------------------------------------------
# structlog lazy seam
# ---------------------------------------------------------------------------


def test_structlog_is_lazy_seam() -> None:
    """structlog must NOT appear at module-level top of system_engine/logging.py."""

    source = inspect.getsource(logging_mod)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "structlog"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "structlog"


def test_enable_structlog_factory_raises_when_missing() -> None:
    try:
        importlib.import_module("structlog")
    except ImportError:
        with pytest.raises(ImportError):
            enable_structlog_factory(engine_id="x", tier="governance")
    else:
        logger = enable_structlog_factory(engine_id="x", tier="governance")
        assert logger is not None


def test_enable_structlog_factory_rejects_bad_tier() -> None:
    try:
        importlib.import_module("structlog")
    except ImportError:
        pytest.skip("structlog not installed")
    else:
        with pytest.raises(LoggingError):
            enable_structlog_factory(engine_id="x", tier="bogus")


# ---------------------------------------------------------------------------
# INV-15 — three-run byte-identical replay
# ---------------------------------------------------------------------------


def test_inv15_three_run_byte_identical_replay() -> None:
    kwargs_list = [
        _ok_kwargs(fields={"a": 1, "b": [1, 2, 3]}),
        _ok_kwargs(event_kind="other", fields={"nested": {"k": "v"}}),
        _ok_kwargs(level=LogLevel.ERROR, fields={"err": "x"}),
    ]

    def _run() -> list[bytes]:
        return [build_log_line(**kw).to_bytes() for kw in kwargs_list]  # type: ignore[arg-type]

    a = _run()
    b = _run()
    c = _run()
    assert a == b == c


# ---------------------------------------------------------------------------
# Authority constraints — AST guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = (
    "structlog",
    "time",
    "datetime",
    "random",
    "asyncio",
    "numpy",
    "torch",
    "polars",
    "requests",
)


def _module_tree(rel: str) -> ast.AST:
    root = Path(__file__).resolve().parent.parent
    return ast.parse((root / rel).read_text(encoding="utf-8"))


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_tree("system_engine/logging.py")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in _FORBIDDEN_TOP_LEVEL_IMPORTS, root
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in _FORBIDDEN_TOP_LEVEL_IMPORTS, root


def test_b1_no_runtime_engine_imports() -> None:
    forbidden = (
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    )
    tree = _module_tree("system_engine/logging.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in forbidden, root
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden, root


_FORBIDDEN_EVENT_CTORS = (
    "PatchProposal",
    "HazardEvent",
    "SignalEvent",
    "ExecutionEvent",
    "SystemEvent",
    "LearningUpdate",
)


def test_b27_b28_inv71_no_typed_event_constructors() -> None:
    tree = _module_tree("system_engine/logging.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            assert name not in _FORBIDDEN_EVENT_CTORS, name


def test_no_wall_clock_calls() -> None:
    """No call to time.time / time.monotonic / datetime.now in module body."""

    tree = _module_tree("system_engine/logging.py")
    forbidden_attrs = {
        ("time", "time"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("time", "time_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func
            if isinstance(target.value, ast.Name):
                pair = (target.value.id, target.attr)
                assert pair not in forbidden_attrs, pair
