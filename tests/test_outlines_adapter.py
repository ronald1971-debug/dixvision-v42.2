# ADAPTED FROM: https://github.com/dottxt-ai/outlines (Apache-2.0)
#
# Tests for the I-21 outlines adapter — JSON schema → regex compiler +
# matcher.
"""I-21 tests: JSON schema regex compile + match."""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from intelligence_engine.cognitive import outlines_adapter as oa
from intelligence_engine.cognitive.outlines_adapter import (
    NEW_PIP_DEPENDENCIES,
    JsonSchema,
    JsonSchemaField,
    OutlinesError,
    OutlinesMatchError,
    OutlinesSchemaError,
    compile_schema_to_regex,
    enable_outlines_factory,
    match_schema,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("outlines",)


def test_error_hierarchy() -> None:
    assert issubclass(OutlinesError, ValueError)
    assert issubclass(OutlinesSchemaError, OutlinesError)
    assert issubclass(OutlinesMatchError, OutlinesError)


# ---------------------------------------------------------------------------
# JsonSchemaField
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [str, int, float, bool])
def test_json_schema_field_accepts_all_supported_types(t: type) -> None:
    f = JsonSchemaField(name="x", type_=t)
    assert f.type_ is t


def test_json_schema_field_is_frozen() -> None:
    f = JsonSchemaField(name="x", type_=str)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.name = "y"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", 0, None])
def test_json_schema_field_rejects_bad_name(bad: object) -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchemaField(name=bad, type_=str)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["1x", "x-y", "x y", "x.y"])
def test_json_schema_field_rejects_non_identifier_name(bad: str) -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchemaField(name=bad, type_=str)


def test_json_schema_field_rejects_unsupported_type() -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchemaField(name="x", type_=list)


# ---------------------------------------------------------------------------
# JsonSchema
# ---------------------------------------------------------------------------


def test_json_schema_happy_path() -> None:
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="a", type_=int),
            JsonSchemaField(name="b", type_=str),
        )
    )
    assert s.field_names() == ("a", "b")


def test_json_schema_is_frozen() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="a", type_=int),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.fields = ()  # type: ignore[misc]


def test_json_schema_rejects_empty_fields() -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchema(fields=())


def test_json_schema_rejects_non_tuple_fields() -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchema(fields=[JsonSchemaField(name="a", type_=int)])  # type: ignore[arg-type]


def test_json_schema_rejects_non_field_entries() -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchema(fields=("not-a-field",))  # type: ignore[arg-type]


def test_json_schema_rejects_duplicate_names() -> None:
    with pytest.raises(OutlinesSchemaError):
        JsonSchema(
            fields=(
                JsonSchemaField(name="a", type_=int),
                JsonSchemaField(name="a", type_=str),
            )
        )


# ---------------------------------------------------------------------------
# compile_schema_to_regex
# ---------------------------------------------------------------------------


def test_compile_schema_to_regex_returns_pattern() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    p = compile_schema_to_regex(s)
    assert isinstance(p, re.Pattern)


def test_compile_schema_rejects_non_schema() -> None:
    with pytest.raises(OutlinesSchemaError):
        compile_schema_to_regex({"x": int})  # type: ignore[arg-type]


def test_compile_schema_int_matches() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": 42}') is not None
    assert p.search('{"x": -7}') is not None
    assert p.search('{"x": 0}') is not None


def test_compile_schema_int_rejects_float() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": 1.5}') is None


def test_compile_schema_float_matches_int_and_float() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=float),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": 1.5}') is not None
    assert p.search('{"x": 42}') is not None
    assert p.search('{"x": -3.0e2}') is not None


def test_compile_schema_bool_matches_true_false() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=bool),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": true}') is not None
    assert p.search('{"x": false}') is not None
    assert p.search('{"x": 1}') is None


def test_compile_schema_str_matches() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=str),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": "hello"}') is not None
    assert p.search('{"x": ""}') is not None


def test_compile_schema_str_handles_escapes() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=str),))
    p = compile_schema_to_regex(s)
    assert p.search('{"x": "with \\"quote\\""}') is not None


# ---------------------------------------------------------------------------
# match_schema
# ---------------------------------------------------------------------------


def test_match_schema_happy_path() -> None:
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="side", type_=str),
            JsonSchemaField(name="qty", type_=int),
            JsonSchemaField(name="px", type_=float),
            JsonSchemaField(name="dry", type_=bool),
        )
    )
    out = match_schema('{"side": "BUY", "qty": 5, "px": 1.25, "dry": true}', s)
    assert out == {"side": "BUY", "qty": 5, "px": 1.25, "dry": True}


def test_match_schema_returns_fresh_dict() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    a = match_schema('{"x": 1}', s)
    b = match_schema('{"x": 1}', s)
    assert a is not b
    assert a == b


def test_match_schema_preserves_declaration_order() -> None:
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="b", type_=int),
            JsonSchemaField(name="a", type_=int),
        )
    )
    out = match_schema('{"b": 1, "a": 2}', s)
    assert list(out.keys()) == ["b", "a"]


def test_match_schema_searches_inside_surrounding_text() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    out = match_schema('here is data: {"x": 7} ok?', s)
    assert out == {"x": 7}


def test_match_schema_rejects_text_without_match() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    with pytest.raises(OutlinesMatchError):
        match_schema("no data here", s)


def test_match_schema_rejects_non_str_text() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=int),))
    with pytest.raises(OutlinesMatchError):
        match_schema(123, s)  # type: ignore[arg-type]


def test_match_schema_int_promoted_to_float_when_field_is_float() -> None:
    s = JsonSchema(fields=(JsonSchemaField(name="x", type_=float),))
    out = match_schema('{"x": 7}', s)
    assert isinstance(out["x"], float)
    assert out["x"] == 7.0


def test_match_schema_field_order_matters() -> None:
    """The regex enforces declaration order; reordering keys fails."""
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="a", type_=int),
            JsonSchemaField(name="b", type_=int),
        )
    )
    with pytest.raises(OutlinesMatchError):
        match_schema('{"b": 2, "a": 1}', s)


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def test_inv15_compile_three_run_identical_pattern() -> None:
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="side", type_=str),
            JsonSchemaField(name="qty", type_=int),
        )
    )
    a = compile_schema_to_regex(s)
    b = compile_schema_to_regex(s)
    c = compile_schema_to_regex(s)
    assert a.pattern == b.pattern == c.pattern


def test_inv15_match_three_run_byte_identical() -> None:
    s = JsonSchema(
        fields=(
            JsonSchemaField(name="side", type_=str),
            JsonSchemaField(name="qty", type_=int),
        )
    )
    text = '{"side": "BUY", "qty": 5}'
    a = match_schema(text, s)
    b = match_schema(text, s)
    c = match_schema(text, s)
    assert a == b == c
    assert list(a.items()) == list(b.items()) == list(c.items())


# ---------------------------------------------------------------------------
# Outlines lazy seam
# ---------------------------------------------------------------------------


def test_enable_outlines_factory_rejects_none_model() -> None:
    with pytest.raises(OutlinesError):
        enable_outlines_factory(None)


def test_enable_outlines_factory_lazy_seam_when_missing() -> None:
    try:
        import outlines  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            enable_outlines_factory(object())
    else:
        pytest.skip("outlines is installed; lazy-seam path unobservable")


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(oa)


def test_no_top_level_outlines_import() -> None:
    for line in _module_source().splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("import outlines", "from outlines")):
            indent = len(line) - len(stripped)
            if indent == 0:
                raise AssertionError(f"outlines must be a lazy seam — no top-level: {line!r}")


def test_no_top_level_transformers_or_llama_cpp_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(
            (
                "import transformers",
                "from transformers",
                "import llama_cpp",
                "from llama_cpp",
            )
        ):
            raise AssertionError(f"transformers/llama_cpp must be lazy seams: {line!r}")


def test_no_top_level_torch_or_numpy_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(("import torch", "from torch", "import numpy", "from numpy")):
            raise AssertionError(f"torch/numpy must be lazy seams: {line!r}")


def test_no_forbidden_top_level_imports() -> None:
    forbidden = (
        "import time",
        "from time",
        "import datetime",
        "from datetime",
        "import random",
        "from random",
        "import asyncio",
        "from asyncio",
        "import requests",
        "from requests",
    )
    for line in _module_source().splitlines():
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(f"forbidden top-level import: {line!r}")


def test_no_typed_event_constructors() -> None:
    forbidden_ctors = (
        "SignalEvent(",
        "ExecutionEvent(",
        "ExecutionIntent(",
        "HazardEvent(",
        "LearningUpdate(",
        "PatchProposal(",
    )
    src = _module_source()
    for ctor in forbidden_ctors:
        assert ctor not in src, f"adapter must not construct typed events: {ctor!r}"
