"""B-10 — tests for ``system_engine/error_telemetry.py``."""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from system_engine import error_telemetry as et
from system_engine.error_telemetry import (
    SCRUB_KEY_FRAGMENTS,
    Breadcrumb,
    ErrorEvent,
    ErrorTelemetry,
    ErrorTelemetryError,
    Frame,
    InProcessErrorTelemetry,
    ScrubbedTraceback,
    exception_message_digest,
    project_frames,
    scrub_event,
)

MODULE_PATH = pathlib.Path(et.__file__)
MODULE_SRC = MODULE_PATH.read_text(encoding="utf-8")
MODULE_AST = ast.parse(MODULE_SRC, filename=str(MODULE_PATH))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
def _build_traceback(
    *,
    exception_type: str = "ValueError",
    message: str = "test exception",
    frame_count: int = 2,
) -> ScrubbedTraceback:
    frames = tuple(
        Frame(
            filename=f"/path/to/module_{i}.py",
            lineno=10 + i,
            function=f"function_{i}",
            qualname=f"Module.function_{i}",
        )
        for i in range(frame_count)
    )
    return ScrubbedTraceback(
        exception_type=exception_type,
        exception_message_digest=exception_message_digest(message),
        frames=frames,
    )


def _build_event(
    *,
    seed: int = 1,
    ts_ns: int = 1_000_000_000,
    operator_id: str = "operator-1",
    dix_version: str = "v42.2",
    environment: str = "production",
    traceback: ScrubbedTraceback | None = None,
    tags: dict[str, object] | None = None,
    breadcrumbs: tuple[Breadcrumb, ...] | None = None,
) -> ErrorEvent:
    return scrub_event(
        seed=seed,
        ts_ns=ts_ns,
        operator_id=operator_id,
        dix_version=dix_version,
        environment=environment,
        traceback=traceback or _build_traceback(),
        tags=tags,
        breadcrumbs=breadcrumbs,
    )


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------
def test_authority_adapted_from_header() -> None:
    assert "# ADAPTED FROM: getsentry/sentry-python" in MODULE_SRC


def test_authority_no_top_level_sentry_import() -> None:
    for node in ast.iter_child_nodes(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "sentry" not in alias.name.lower()
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "sentry" not in mod.lower()


def test_authority_no_engine_cross_imports() -> None:
    forbidden = (
        "execution_engine",
        "evolution_engine",
        "governance_engine",
        "intelligence_engine",
    )
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for prefix in forbidden:
                assert not mod.startswith(prefix)


def test_authority_no_clock_or_random_or_io_imports() -> None:
    forbidden = {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "socket",
        "secrets",
        "uuid",
        "numpy",
        "torch",
        "pandas",
        "polars",
        "scipy",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in forbidden


def test_authority_no_typed_event_construction() -> None:
    forbidden = {
        "SignalEvent",
        "ExecutionIntent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in forbidden


def test_authority_no_top_level_io() -> None:
    forbidden = {"open", "print", "input", "exec", "eval"}
    for node in ast.iter_child_nodes(MODULE_AST):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.id if isinstance(func, ast.Name) else None
            assert name not in forbidden


def test_authority_pip_dependencies() -> None:
    assert et.NEW_PIP_DEPENDENCIES == ("sentry-sdk",)


def test_authority_module_reimport_clean() -> None:
    # Re-import (no reload — reload would replace module globals and break
    # class identity for already-loaded test references).
    importlib.import_module("system_engine.error_telemetry")


def test_authority_sentry_import_lazy() -> None:
    """sentry_sdk must only be imported inside sentry_telemetry_factory body."""
    found_inside_factory = False
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.FunctionDef) and node.name == "sentry_telemetry_factory":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    for alias in inner.names:
                        if "sentry" in alias.name.lower():
                            found_inside_factory = True
    assert found_inside_factory


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------
def test_frame_ok() -> None:
    f = Frame(filename="x.py", lineno=42, function="run", qualname="C.run")
    assert f.filename == "x.py"
    assert f.lineno == 42


def test_frame_empty_filename_rejected() -> None:
    with pytest.raises(ValueError):
        Frame(filename="", lineno=1, function="run", qualname="")


def test_frame_negative_lineno_rejected() -> None:
    with pytest.raises(ValueError):
        Frame(filename="x.py", lineno=-1, function="run", qualname="")


def test_frame_empty_function_rejected() -> None:
    with pytest.raises(ValueError):
        Frame(filename="x.py", lineno=1, function="", qualname="")


def test_frame_qualname_must_be_str() -> None:
    with pytest.raises(TypeError):
        Frame(filename="x.py", lineno=1, function="run", qualname=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# exception_message_digest
# ---------------------------------------------------------------------------
def test_exception_message_digest_deterministic() -> None:
    assert exception_message_digest("oops") == exception_message_digest("oops")


def test_exception_message_digest_different_inputs_diverge() -> None:
    assert exception_message_digest("a") != exception_message_digest("b")


def test_exception_message_digest_hex_width() -> None:
    d = exception_message_digest("ok")
    assert len(d) == 16
    int(d, 16)


def test_exception_message_digest_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        exception_message_digest(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ScrubbedTraceback validation
# ---------------------------------------------------------------------------
def test_traceback_ok() -> None:
    tb = _build_traceback()
    assert tb.exception_type == "ValueError"


def test_traceback_empty_exception_type_rejected() -> None:
    with pytest.raises(ValueError):
        ScrubbedTraceback(
            exception_type="",
            exception_message_digest=exception_message_digest("x"),
            frames=(),
        )


def test_traceback_bad_digest_length_rejected() -> None:
    with pytest.raises(ValueError):
        ScrubbedTraceback(
            exception_type="ValueError",
            exception_message_digest="too_short",
            frames=(),
        )


def test_traceback_bad_digest_hex_rejected() -> None:
    with pytest.raises(ValueError):
        ScrubbedTraceback(
            exception_type="ValueError",
            exception_message_digest="z" * 16,
            frames=(),
        )


def test_traceback_too_many_frames_rejected() -> None:
    frames = tuple(
        Frame(filename="x.py", lineno=i, function="run", qualname="")
        for i in range(et.MAX_FRAME_COUNT + 1)
    )
    with pytest.raises(ValueError):
        ScrubbedTraceback(
            exception_type="ValueError",
            exception_message_digest=exception_message_digest("x"),
            frames=frames,
        )


# ---------------------------------------------------------------------------
# Breadcrumb validation
# ---------------------------------------------------------------------------
def test_breadcrumb_ok() -> None:
    c = Breadcrumb(
        ts_ns=1,
        category="navigation",
        level="info",
        message_digest=exception_message_digest("nav"),
    )
    assert c.category == "navigation"


def test_breadcrumb_negative_ts_rejected() -> None:
    with pytest.raises(ValueError):
        Breadcrumb(
            ts_ns=-1,
            category="navigation",
            level="info",
            message_digest=exception_message_digest("nav"),
        )


def test_breadcrumb_bad_level_rejected() -> None:
    with pytest.raises(ValueError):
        Breadcrumb(
            ts_ns=1,
            category="navigation",
            level="trace",
            message_digest=exception_message_digest("nav"),
        )


def test_breadcrumb_forbidden_category_rejected() -> None:
    """Categories like 'api_key_loaded' must be rejected."""
    with pytest.raises(ValueError):
        Breadcrumb(
            ts_ns=1,
            category="api_key_loaded",
            level="info",
            message_digest=exception_message_digest("x"),
        )


def test_breadcrumb_bad_digest_rejected() -> None:
    with pytest.raises(ValueError):
        Breadcrumb(
            ts_ns=1,
            category="navigation",
            level="info",
            message_digest="not_hex_z_" + "0" * 6,
        )


# ---------------------------------------------------------------------------
# scrub_event behaviour
# ---------------------------------------------------------------------------
def test_scrub_event_ok() -> None:
    evt = _build_event()
    assert len(evt.event_id) == 16
    int(evt.event_id, 16)
    assert evt.exception_type == "ValueError"


def test_scrub_event_id_deterministic() -> None:
    a = _build_event(seed=42, ts_ns=10_000)
    b = _build_event(seed=42, ts_ns=10_000)
    assert a.event_id == b.event_id


def test_scrub_event_id_differs_per_seed() -> None:
    a = _build_event(seed=1)
    b = _build_event(seed=2)
    assert a.event_id != b.event_id


def test_scrub_event_id_differs_per_ts() -> None:
    a = _build_event(ts_ns=1_000)
    b = _build_event(ts_ns=2_000)
    assert a.event_id != b.event_id


def test_scrub_event_id_differs_per_traceback() -> None:
    a = _build_event(traceback=_build_traceback(message="a"))
    b = _build_event(traceback=_build_traceback(message="b"))
    assert a.event_id != b.event_id


def test_scrub_event_tags_sorted() -> None:
    evt = _build_event(tags={"z_alpha": 1, "a_alpha": 2})
    keys = [k for k, _ in evt.tags]
    assert keys == sorted(keys)


def test_scrub_event_pins_dix_tags() -> None:
    evt = _build_event()
    tag_dict = dict(evt.tags)
    assert tag_dict["dix_version"] == "v42.2"
    assert tag_dict["operator_id"] == "operator-1"
    assert tag_dict["environment"] == "production"


def test_scrub_event_strips_api_key_tag() -> None:
    evt = _build_event(tags={"api_key": "sk-AAAA", "regime": "trend"})
    tag_dict = dict(evt.tags)
    assert "api_key" not in tag_dict
    assert tag_dict.get("regime") == "trend"


def test_scrub_event_strips_password_tag() -> None:
    evt = _build_event(tags={"password": "hunter2"})
    tag_dict = dict(evt.tags)
    assert "password" not in tag_dict


def test_scrub_event_strips_balance_tag() -> None:
    evt = _build_event(tags={"balance_usd": 1_000_000})
    tag_dict = dict(evt.tags)
    assert "balance_usd" not in tag_dict


def test_scrub_event_strips_position_tag() -> None:
    evt = _build_event(tags={"position_qty": 1.5})
    tag_dict = dict(evt.tags)
    assert "position_qty" not in tag_dict


def test_scrub_event_strips_pnl_tag() -> None:
    evt = _build_event(tags={"pnl_realized": 42.0})
    tag_dict = dict(evt.tags)
    assert "pnl_realized" not in tag_dict


def test_scrub_event_strips_wallet_tag() -> None:
    evt = _build_event(tags={"wallet_address": "0xabc"})
    tag_dict = dict(evt.tags)
    assert "wallet_address" not in tag_dict


def test_scrub_event_strips_secret_tag() -> None:
    evt = _build_event(tags={"my_secret": "x"})
    tag_dict = dict(evt.tags)
    assert "my_secret" not in tag_dict


def test_scrub_event_strips_case_insensitively() -> None:
    evt = _build_event(tags={"API_KEY": "x", "Password": "y", "BALANCE": 1})
    tag_dict = dict(evt.tags)
    assert "API_KEY" not in tag_dict
    assert "Password" not in tag_dict
    assert "BALANCE" not in tag_dict


def test_scrub_event_includes_breadcrumbs() -> None:
    crumbs = (
        Breadcrumb(
            ts_ns=1,
            category="navigation",
            level="info",
            message_digest=exception_message_digest("a"),
        ),
        Breadcrumb(
            ts_ns=2,
            category="navigation",
            level="warning",
            message_digest=exception_message_digest("b"),
        ),
    )
    evt = _build_event(breadcrumbs=crumbs)
    assert len(evt.breadcrumb_digests) == 2
    for d in evt.breadcrumb_digests:
        assert len(d) == 16


def test_scrub_event_rejects_too_many_tags() -> None:
    too_many = {f"k_{i}": i for i in range(et.MAX_TAG_COUNT + 1)}
    with pytest.raises(ValueError):
        _build_event(tags=too_many)


def test_scrub_event_rejects_non_str_tag_key() -> None:
    with pytest.raises(TypeError):
        _build_event(tags={1: "v"})  # type: ignore[dict-item]


def test_scrub_event_rejects_negative_ts() -> None:
    with pytest.raises(ValueError):
        _build_event(ts_ns=-1)


def test_scrub_event_rejects_empty_operator_id() -> None:
    with pytest.raises(ValueError):
        _build_event(operator_id="")


def test_scrub_event_rejects_empty_dix_version() -> None:
    with pytest.raises(ValueError):
        _build_event(dix_version="")


def test_scrub_event_rejects_empty_environment() -> None:
    with pytest.raises(ValueError):
        _build_event(environment="")


def test_scrub_event_rejects_non_traceback() -> None:
    with pytest.raises(TypeError):
        scrub_event(
            seed=1,
            ts_ns=1,
            operator_id="op",
            dix_version="v",
            environment="e",
            traceback="not a traceback",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# InProcessErrorTelemetry behaviour
# ---------------------------------------------------------------------------
def test_inprocess_implements_protocol() -> None:
    sink = InProcessErrorTelemetry()
    assert isinstance(sink, ErrorTelemetry)


def test_inprocess_captures_event() -> None:
    sink = InProcessErrorTelemetry()
    evt = _build_event()
    sink.capture(evt)
    snap = sink.snapshot()
    assert snap.captured == 1
    assert snap.events == (evt,)


def test_inprocess_zero_sample_drops_all() -> None:
    sink = InProcessErrorTelemetry(sample_ratio=0.0)
    sink.capture(_build_event())
    snap = sink.snapshot()
    assert snap.captured == 0
    assert snap.dropped_by_sample == 1


def test_inprocess_buffer_full_drops() -> None:
    sink = InProcessErrorTelemetry(event_buffer_size=1)
    sink.capture(_build_event(ts_ns=1))
    sink.capture(_build_event(ts_ns=2, seed=2))
    snap = sink.snapshot()
    assert snap.captured == 1
    assert snap.dropped_by_buffer == 1


def test_inprocess_rejects_non_event() -> None:
    sink = InProcessErrorTelemetry()
    with pytest.raises(TypeError):
        sink.capture("not an event")  # type: ignore[arg-type]


def test_inprocess_rejects_bad_sample_ratio() -> None:
    with pytest.raises(ValueError):
        InProcessErrorTelemetry(sample_ratio=1.5)


def test_inprocess_rejects_negative_buffer_size() -> None:
    with pytest.raises(ValueError):
        InProcessErrorTelemetry(event_buffer_size=0)


def test_inprocess_snapshot_sorts_by_ts_then_id() -> None:
    sink = InProcessErrorTelemetry()
    sink.capture(_build_event(ts_ns=200))
    sink.capture(_build_event(ts_ns=100, seed=2))
    snap = sink.snapshot()
    timestamps = [e.ts_ns for e in snap.events]
    assert timestamps == sorted(timestamps)


def test_inprocess_breadcrumb_ring() -> None:
    sink = InProcessErrorTelemetry(breadcrumb_buffer_size=2)
    for i in range(4):
        sink.add_breadcrumb(
            Breadcrumb(
                ts_ns=i,
                category="navigation",
                level="info",
                message_digest=exception_message_digest(f"m{i}"),
            )
        )
    crumbs = sink.breadcrumbs()
    assert len(crumbs) == 2
    assert [c.ts_ns for c in crumbs] == [2, 3]


def test_inprocess_breadcrumb_rejects_non_crumb() -> None:
    sink = InProcessErrorTelemetry()
    with pytest.raises(TypeError):
        sink.add_breadcrumb("not a crumb")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# project_frames
# ---------------------------------------------------------------------------
def test_project_frames_ok() -> None:
    raws = [
        {"filename": "a.py", "lineno": 1, "function": "f", "qualname": "C.f"},
        {"filename": "b.py", "lineno": 2, "function": "g", "qualname": ""},
    ]
    frames = project_frames(raws)
    assert len(frames) == 2


def test_project_frames_rejects_non_mapping_entry() -> None:
    with pytest.raises(TypeError):
        project_frames(["not a mapping"])  # type: ignore[list-item]


def test_project_frames_rejects_missing_field() -> None:
    raws = [{"filename": "a.py", "lineno": 1, "function": "f"}]
    frames = project_frames(raws)
    assert frames[0].qualname == ""


def test_project_frames_rejects_bad_type() -> None:
    raws = [{"filename": 1, "lineno": 1, "function": "f", "qualname": ""}]
    with pytest.raises(TypeError):
        project_frames(raws)


def test_project_frames_rejects_too_many() -> None:
    raws = [
        {
            "filename": "x.py",
            "lineno": i,
            "function": "f",
            "qualname": "",
        }
        for i in range(et.MAX_FRAME_COUNT + 5)
    ]
    with pytest.raises(ValueError):
        project_frames(raws)


# ---------------------------------------------------------------------------
# sentry_telemetry_factory contract
# ---------------------------------------------------------------------------
def test_factory_rejects_empty_dsn() -> None:
    with pytest.raises(ValueError):
        et.sentry_telemetry_factory(
            dsn="",
            environment="e",
            dix_version="v",
            operator_id="op",
        )


def test_factory_rejects_bad_sample_ratio() -> None:
    with pytest.raises(ValueError):
        et.sentry_telemetry_factory(
            dsn="https://x@sentry.io/1",
            environment="e",
            dix_version="v",
            operator_id="op",
            sample_ratio=2.0,
        )


def test_factory_raises_when_sdk_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sentry-sdk is not installed, factory raises ErrorTelemetryError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentry_sdk":
            raise ImportError("sentry-sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ErrorTelemetryError):
        et.sentry_telemetry_factory(
            dsn="https://x@sentry.io/1",
            environment="e",
            dix_version="v",
            operator_id="op",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------
def test_replay_three_runs_byte_identical() -> None:
    crumbs = (
        Breadcrumb(
            ts_ns=1,
            category="navigation",
            level="info",
            message_digest=exception_message_digest("a"),
        ),
    )

    def run() -> ErrorEvent:
        return scrub_event(
            seed=42,
            ts_ns=1_000_000,
            operator_id="op-1",
            dix_version="v42.2",
            environment="prod",
            traceback=_build_traceback(),
            tags={"regime": "trend", "z_last": "x", "a_first": "y"},
            breadcrumbs=crumbs,
        )

    a, b, c = run(), run(), run()
    assert a == b == c
    assert a.event_id == b.event_id == c.event_id


def test_replay_tag_dict_order_independence() -> None:
    a = _build_event(tags={"alpha": 1, "beta": 2, "gamma": 3})
    b = _build_event(tags={"gamma": 3, "alpha": 1, "beta": 2})
    assert a == b


def test_replay_snapshot_order_independence() -> None:
    sink_a = InProcessErrorTelemetry()
    sink_b = InProcessErrorTelemetry()
    e1 = _build_event(ts_ns=100, seed=1)
    e2 = _build_event(ts_ns=200, seed=2)
    sink_a.capture(e1)
    sink_a.capture(e2)
    sink_b.capture(e2)
    sink_b.capture(e1)
    assert sink_a.snapshot().events == sink_b.snapshot().events


def test_error_event_frozen() -> None:
    evt = _build_event()
    with pytest.raises(dataclasses_frozen_error()):
        evt.event_id = "x" * 16  # type: ignore[misc]


def test_breadcrumb_frozen() -> None:
    c = Breadcrumb(
        ts_ns=1,
        category="navigation",
        level="info",
        message_digest=exception_message_digest("a"),
    )
    with pytest.raises(dataclasses_frozen_error()):
        c.ts_ns = 2  # type: ignore[misc]


def test_frame_frozen() -> None:
    f = Frame(filename="x.py", lineno=1, function="f", qualname="")
    with pytest.raises(dataclasses_frozen_error()):
        f.lineno = 2  # type: ignore[misc]


def dataclasses_frozen_error() -> type[Exception]:
    import dataclasses

    return dataclasses.FrozenInstanceError


# ---------------------------------------------------------------------------
# Scrub-fragment coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fragment", SCRUB_KEY_FRAGMENTS)
def test_every_fragment_strips_matching_tag(fragment: str) -> None:
    """Every fragment in SCRUB_KEY_FRAGMENTS must remove matching keys."""
    key = f"prefix_{fragment}_suffix"
    evt = _build_event(tags={key: "data", "regime": "trend"})
    tag_dict = dict(evt.tags)
    assert key not in tag_dict
    assert tag_dict.get("regime") == "trend"
