# ADAPTED FROM: https://github.com/ijl/orjson  (Apache-2.0 / MIT)
"""Tests for the canonical orjson-shape JSON codec (I-02)."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from system_engine import codec as codec_pkg
from system_engine.codec import json_codec
from system_engine.codec.json_codec import (
    CODEC_VERSION,
    NEW_PIP_DEPENDENCIES,
    CodecError,
    JsonCodec,
    canonical_dumps,
    canonical_loads,
    default_codec,
    enable_orjson_factory,
    stdlib_codec_factory,
)

# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------


def test_version_tag() -> None:
    assert CODEC_VERSION == "v1.0-I02"


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("orjson",)


def test_package_reexports_match_module() -> None:
    for name in NEW_PIP_DEPENDENCIES, CODEC_VERSION, JsonCodec, CodecError:
        assert name is not None  # smoke
    for sym in (
        "CODEC_VERSION",
        "NEW_PIP_DEPENDENCIES",
        "CodecError",
        "JsonCodec",
        "canonical_dumps",
        "canonical_loads",
        "default_codec",
        "enable_orjson_factory",
        "stdlib_codec_factory",
    ):
        assert hasattr(codec_pkg, sym), sym
        assert getattr(codec_pkg, sym) is getattr(json_codec, sym), sym


def test_all_export_is_complete() -> None:
    assert set(json_codec.__all__) == set(codec_pkg.__all__)
    assert "canonical_dumps" in json_codec.__all__
    assert "canonical_loads" in json_codec.__all__


# ---------------------------------------------------------------------------
# JsonCodec value object
# ---------------------------------------------------------------------------


def test_json_codec_is_frozen() -> None:
    c = stdlib_codec_factory()
    with pytest.raises((AttributeError, TypeError, Exception)):  # noqa: B017 - frozen slots may raise either
        c.backend = "evil"  # type: ignore[misc]


def test_json_codec_backend_must_be_nonempty_string() -> None:
    with pytest.raises(CodecError):
        JsonCodec(dumps=canonical_dumps, loads=canonical_loads, backend="")


def test_default_codec_returns_stdlib_backend() -> None:
    c = default_codec()
    assert c.backend == "stdlib"
    assert c.dumps is canonical_dumps
    assert c.loads is canonical_loads


# ---------------------------------------------------------------------------
# canonical_dumps shape
# ---------------------------------------------------------------------------


def test_dumps_primitives() -> None:
    assert canonical_dumps(None) == b"null"
    assert canonical_dumps(True) == b"true"
    assert canonical_dumps(False) == b"false"
    assert canonical_dumps(0) == b"0"
    assert canonical_dumps(-1) == b"-1"
    assert canonical_dumps(42) == b"42"
    assert canonical_dumps("hello") == b'"hello"'


def test_dumps_empty_containers() -> None:
    assert canonical_dumps([]) == b"[]"
    assert canonical_dumps({}) == b"{}"
    assert canonical_dumps(()) == b"[]"


def test_dumps_sorts_keys() -> None:
    blob = canonical_dumps({"b": 1, "a": 2, "c": 3})
    assert blob == b'{"a":2,"b":1,"c":3}'


def test_dumps_no_whitespace() -> None:
    blob = canonical_dumps({"a": [1, 2, {"b": 3}]})
    # orjson never emits ', ' or ': ' — only ',' / ':'
    assert b", " not in blob
    assert b": " not in blob


def test_dumps_tuple_emits_as_list() -> None:
    assert canonical_dumps((1, 2, 3)) == b"[1,2,3]"


def test_dumps_nested() -> None:
    payload = {"z": [1, {"y": 2, "x": [3, 4]}], "a": "string"}
    blob = canonical_dumps(payload)
    assert blob == b'{"a":"string","z":[1,{"x":[3,4],"y":2}]}'


def test_dumps_unicode_passthrough() -> None:
    blob = canonical_dumps({"k": "héllo"})
    # ensure_ascii=False — orjson never escapes printable unicode
    assert "héllo".encode() in blob


def test_dumps_rejects_nan_inf() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CodecError):
            canonical_dumps(bad)


def test_dumps_rejects_non_str_keys() -> None:
    with pytest.raises(CodecError):
        canonical_dumps({1: "x"})


def test_dumps_rejects_unknown_type() -> None:
    class Opaque:
        pass

    with pytest.raises(CodecError):
        canonical_dumps(Opaque())


def test_dumps_is_pure_function() -> None:
    payload = {"b": 1, "a": [1, 2, 3], "c": {"k": "v"}}
    b1 = canonical_dumps(payload)
    b2 = canonical_dumps(payload)
    b3 = canonical_dumps(payload)
    assert b1 == b2 == b3


def test_dumps_byte_identical_under_key_permutation() -> None:
    p1 = {"a": 1, "b": 2, "c": 3}
    p2 = {"c": 3, "a": 1, "b": 2}
    p3 = {"b": 2, "c": 3, "a": 1}
    assert canonical_dumps(p1) == canonical_dumps(p2) == canonical_dumps(p3)


def test_dumps_returns_bytes() -> None:
    out = canonical_dumps({"a": 1})
    assert isinstance(out, bytes)
    assert not isinstance(out, bytearray)


# ---------------------------------------------------------------------------
# canonical_loads round-trip
# ---------------------------------------------------------------------------


def test_loads_parses_bytes() -> None:
    assert canonical_loads(b"null") is None
    assert canonical_loads(b"true") is True
    assert canonical_loads(b"false") is False
    assert canonical_loads(b"42") == 42
    assert canonical_loads(b'"hi"') == "hi"


def test_loads_accepts_bytearray_and_memoryview() -> None:
    assert canonical_loads(bytearray(b"[1,2,3]")) == [1, 2, 3]
    assert canonical_loads(memoryview(b'{"a":1}')) == {"a": 1}


def test_loads_rejects_non_bytes() -> None:
    for bad in ("a string", 42, None, [1, 2]):
        with pytest.raises(CodecError):
            canonical_loads(bad)  # type: ignore[arg-type]


def test_loads_rejects_invalid_utf8() -> None:
    with pytest.raises(CodecError):
        canonical_loads(b"\xff\xfe")


def test_loads_rejects_malformed_json() -> None:
    with pytest.raises(CodecError):
        canonical_loads(b"{not json")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        True,
        False,
        0,
        -1,
        42,
        3.14,
        "hello",
        "héllo",
        [],
        {},
        [1, 2, 3],
        {"a": 1, "b": 2},
        {"nested": {"a": [1, 2, {"k": "v"}], "b": None}},
        [None, True, False, 1, "x", [1, 2], {"a": 1}],
    ],
)
def test_round_trip_is_byte_stable(payload: object) -> None:
    blob = canonical_dumps(payload)
    assert canonical_loads(blob) == payload
    # double round trip is also byte-stable
    assert canonical_dumps(canonical_loads(blob)) == blob


# ---------------------------------------------------------------------------
# orjson factory
# ---------------------------------------------------------------------------


def test_enable_orjson_factory_is_lazy_seam() -> None:
    """orjson must not be imported at module level (function-local only)."""

    source = inspect.getsource(json_codec)
    tree = ast.parse(source)
    # only check the module-level body, NOT walk into function bodies — the
    # lazy seam is allowed to import orjson from within enable_orjson_factory().
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "orjson", (
                    "orjson must only be imported lazily inside enable_orjson_factory()"
                )
        if isinstance(node, ast.ImportFrom):
            assert node.module != "orjson"


def test_enable_orjson_factory_raises_when_orjson_missing() -> None:
    try:
        importlib.import_module("orjson")
    except ImportError:
        with pytest.raises(ImportError):
            enable_orjson_factory()
    else:
        codec = enable_orjson_factory()
        assert codec.backend == "orjson"


def test_orjson_codec_is_byte_identical_to_stdlib_when_available() -> None:
    try:
        importlib.import_module("orjson")
    except ImportError:
        pytest.skip("orjson not installed in this environment")
    else:
        orjson_codec = enable_orjson_factory()
        std = stdlib_codec_factory()
        for payload in (
            None,
            True,
            False,
            0,
            42,
            -1,
            "hello",
            [1, 2, 3],
            {"a": 1, "b": 2},
            {"z": [1, {"y": 2, "x": [3]}], "a": "s"},
        ):
            assert orjson_codec.dumps(payload) == std.dumps(payload), payload


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def test_inv15_three_run_byte_identical_replay() -> None:
    payloads = [
        {"a": 1, "b": [1, 2, 3]},
        [None, True, False, "x"],
        {"nested": {"k": "v", "n": 0, "arr": [1, {"q": "r"}]}},
        "single string",
        42,
    ]

    def _run() -> list[bytes]:
        return [canonical_dumps(p) for p in payloads]

    a = _run()
    b = _run()
    c = _run()
    assert a == b == c


# ---------------------------------------------------------------------------
# Authority constraints
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = (
    "orjson",
    "time",
    "datetime",
    "random",
    "asyncio",
    "numpy",
    "torch",
    "polars",
    "requests",
)


def _walk_module_source(rel_path: str) -> ast.AST:
    root = Path(__file__).resolve().parent.parent
    return ast.parse((root / rel_path).read_text(encoding="utf-8"))


def test_no_forbidden_top_level_imports() -> None:
    tree = _walk_module_source("system_engine/codec/json_codec.py")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in _FORBIDDEN_TOP_LEVEL_IMPORTS, (
                    f"forbidden top-level import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in _FORBIDDEN_TOP_LEVEL_IMPORTS


def test_b1_no_runtime_engine_imports() -> None:
    """B1 — codec must not import from any runtime engine tier."""

    forbidden_prefixes = (
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    )
    for rel in ("system_engine/codec/__init__.py", "system_engine/codec/json_codec.py"):
        tree = _walk_module_source(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                assert root not in forbidden_prefixes, (
                    f"{rel} imports from forbidden runtime tier {root!r}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root not in forbidden_prefixes, (
                        f"{rel} imports from forbidden runtime tier {root!r}"
                    )


_FORBIDDEN_EVENT_CTORS = (
    "PatchProposal",
    "HazardEvent",
    "SignalEvent",
    "ExecutionEvent",
    "SystemEvent",
    "LearningUpdate",
)


def test_b27_b28_inv71_no_typed_event_constructors() -> None:
    tree = _walk_module_source("system_engine/codec/json_codec.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name: str | None = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name in _FORBIDDEN_EVENT_CTORS:
                raise AssertionError(f"forbidden typed-event constructor call: {name!r}")


def test_codec_matches_stdlib_json_dumps_sort_keys_shape() -> None:
    """Cross-check that canonical_dumps mirrors json.dumps with sort_keys + no-ws."""

    payload = {"b": 1, "a": [1, {"d": 4, "c": 3}], "z": "x"}
    expected = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert canonical_dumps(payload) == expected
