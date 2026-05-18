# ADAPTED FROM: https://github.com/567-labs/instructor (MIT)
#
# Tests for the I-19 instructor adapter — structured LLM output validator.
"""I-19 tests: instructor-style structured output validation."""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from intelligence_engine.cognitive import instructor_adapter as ia
from intelligence_engine.cognitive.instructor_adapter import (
    NEW_PIP_DEPENDENCIES,
    InstructorError,
    InstructorSchema,
    InstructorSchemaError,
    SchemaField,
    enable_instructor_factory,
    extract_typed_payload,
    validate_instance,
)


def _basic_schema() -> InstructorSchema:
    return InstructorSchema(
        fields=(
            SchemaField(name="symbol", type_=str),
            SchemaField(name="side", type_=str, choices=("BUY", "SELL")),
            SchemaField(name="confidence", type_=float),
            SchemaField(name="rationale", type_=str, required=False),
        )
    )


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("instructor",)


def test_instructor_schema_error_is_instructor_error() -> None:
    assert issubclass(InstructorSchemaError, InstructorError)


def test_instructor_error_is_value_error() -> None:
    assert issubclass(InstructorError, ValueError)


# ---------------------------------------------------------------------------
# SchemaField validation
# ---------------------------------------------------------------------------


def test_schema_field_happy_path() -> None:
    f = SchemaField(name="x", type_=int)
    assert f.name == "x"
    assert f.type_ is int
    assert f.required is True
    assert f.choices is None


def test_schema_field_is_frozen() -> None:
    f = SchemaField(name="x", type_=int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.name = "y"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", 0, None])
def test_schema_field_rejects_bad_name(bad: object) -> None:
    with pytest.raises(InstructorError):
        SchemaField(name=bad, type_=str)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [list, dict, tuple, bytes, type(None)])
def test_schema_field_rejects_bad_type(bad: type) -> None:
    with pytest.raises(InstructorError):
        SchemaField(name="x", type_=bad)


def test_schema_field_rejects_non_tuple_choices() -> None:
    with pytest.raises(InstructorError):
        SchemaField(name="x", type_=str, choices=["BUY", "SELL"])  # type: ignore[arg-type]


def test_schema_field_rejects_empty_choices() -> None:
    with pytest.raises(InstructorError):
        SchemaField(name="x", type_=str, choices=())


def test_schema_field_rejects_choice_type_mismatch() -> None:
    with pytest.raises(InstructorError):
        SchemaField(name="x", type_=str, choices=("BUY", 1))


# ---------------------------------------------------------------------------
# InstructorSchema validation
# ---------------------------------------------------------------------------


def test_schema_happy_path() -> None:
    s = _basic_schema()
    assert s.field_names() == ("symbol", "side", "confidence", "rationale")


def test_schema_is_frozen() -> None:
    s = _basic_schema()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.fields = ()  # type: ignore[misc]


def test_schema_rejects_empty_fields() -> None:
    with pytest.raises(InstructorError):
        InstructorSchema(fields=())


def test_schema_rejects_duplicate_field_names() -> None:
    with pytest.raises(InstructorError):
        InstructorSchema(
            fields=(
                SchemaField(name="x", type_=str),
                SchemaField(name="x", type_=int),
            )
        )


def test_schema_rejects_non_tuple_fields() -> None:
    with pytest.raises(InstructorError):
        InstructorSchema(fields=[SchemaField(name="x", type_=str)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_instance
# ---------------------------------------------------------------------------


def test_validate_instance_happy_path() -> None:
    payload = {
        "symbol": "EURUSD",
        "side": "BUY",
        "confidence": 0.62,
        "rationale": "trend continuation",
    }
    out = validate_instance(payload, _basic_schema())
    assert out == payload


def test_validate_instance_preserves_schema_field_order() -> None:
    payload = {
        "rationale": "...",
        "confidence": 0.5,
        "side": "BUY",
        "symbol": "EURUSD",
    }
    out = validate_instance(payload, _basic_schema())
    assert list(out.keys()) == ["symbol", "side", "confidence", "rationale"]


def test_validate_instance_coerces_int_to_float() -> None:
    schema = InstructorSchema(fields=(SchemaField(name="x", type_=float),))
    out = validate_instance({"x": 5}, schema)
    assert out == {"x": 5.0}
    assert isinstance(out["x"], float)


def test_validate_instance_rejects_bool_for_int_field() -> None:
    schema = InstructorSchema(fields=(SchemaField(name="x", type_=int),))
    with pytest.raises(InstructorSchemaError):
        validate_instance({"x": True}, schema)


def test_validate_instance_rejects_bool_for_float_field() -> None:
    schema = InstructorSchema(fields=(SchemaField(name="x", type_=float),))
    with pytest.raises(InstructorSchemaError):
        validate_instance({"x": True}, schema)


def test_validate_instance_rejects_int_for_bool_field() -> None:
    schema = InstructorSchema(fields=(SchemaField(name="x", type_=bool),))
    with pytest.raises(InstructorSchemaError):
        validate_instance({"x": 1}, schema)


def test_validate_instance_rejects_wrong_type() -> None:
    with pytest.raises(InstructorSchemaError):
        validate_instance(
            {"symbol": 1, "side": "BUY", "confidence": 0.5},
            _basic_schema(),
        )


def test_validate_instance_rejects_missing_required_field() -> None:
    with pytest.raises(InstructorSchemaError):
        validate_instance(
            {"side": "BUY", "confidence": 0.5},
            _basic_schema(),
        )


def test_validate_instance_accepts_missing_optional_field() -> None:
    out = validate_instance(
        {"symbol": "EURUSD", "side": "BUY", "confidence": 0.5},
        _basic_schema(),
    )
    assert "rationale" not in out


def test_validate_instance_rejects_extra_fields() -> None:
    with pytest.raises(InstructorSchemaError):
        validate_instance(
            {
                "symbol": "EURUSD",
                "side": "BUY",
                "confidence": 0.5,
                "garbage": 1,
            },
            _basic_schema(),
        )


def test_validate_instance_rejects_choice_violation() -> None:
    with pytest.raises(InstructorSchemaError):
        validate_instance(
            {"symbol": "EURUSD", "side": "HOLD", "confidence": 0.5},
            _basic_schema(),
        )


def test_validate_instance_rejects_non_mapping_payload() -> None:
    with pytest.raises(InstructorSchemaError):
        validate_instance("not a mapping", _basic_schema())  # type: ignore[arg-type]


def test_validate_instance_rejects_non_schema() -> None:
    with pytest.raises(InstructorError):
        validate_instance({"x": 1}, "not-a-schema")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_typed_payload
# ---------------------------------------------------------------------------


def test_extract_typed_payload_bare_object() -> None:
    text = json.dumps({"symbol": "EURUSD", "side": "BUY", "confidence": 0.62})
    out = extract_typed_payload(text, _basic_schema())
    assert out["symbol"] == "EURUSD"
    assert out["side"] == "BUY"
    assert out["confidence"] == pytest.approx(0.62)


def test_extract_typed_payload_fenced_json() -> None:
    text = (
        "Here is a proposal:\n\n"
        "```json\n"
        '{"symbol": "EURUSD", "side": "SELL", "confidence": 0.71}\n'
        "```\n"
    )
    out = extract_typed_payload(text, _basic_schema())
    assert out["side"] == "SELL"


def test_extract_typed_payload_uppercase_fence() -> None:
    text = '```JSON\n{"symbol":"X","side":"BUY","confidence":0.5}\n```'
    out = extract_typed_payload(text, _basic_schema())
    assert out["symbol"] == "X"


def test_extract_typed_payload_rejects_non_str_input() -> None:
    with pytest.raises(InstructorSchemaError):
        extract_typed_payload(123, _basic_schema())  # type: ignore[arg-type]


def test_extract_typed_payload_rejects_empty() -> None:
    with pytest.raises(InstructorSchemaError):
        extract_typed_payload("", _basic_schema())


def test_extract_typed_payload_rejects_invalid_json() -> None:
    with pytest.raises(InstructorSchemaError):
        extract_typed_payload("{not json", _basic_schema())


def test_extract_typed_payload_rejects_non_object_root() -> None:
    with pytest.raises(InstructorSchemaError):
        extract_typed_payload("[1, 2, 3]", _basic_schema())


def test_extract_typed_payload_propagates_schema_errors() -> None:
    text = json.dumps({"symbol": "EURUSD", "side": "HOLD", "confidence": 0.5})
    with pytest.raises(InstructorSchemaError):
        extract_typed_payload(text, _basic_schema())


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def test_inv15_extract_three_run_byte_identical() -> None:
    text = '```json\n{"symbol":"EURUSD","side":"BUY","confidence":0.62}\n```'
    schema = _basic_schema()
    a = extract_typed_payload(text, schema)
    b = extract_typed_payload(text, schema)
    c = extract_typed_payload(text, schema)
    a_bytes = json.dumps(a, sort_keys=True).encode("utf-8")
    b_bytes = json.dumps(b, sort_keys=True).encode("utf-8")
    c_bytes = json.dumps(c, sort_keys=True).encode("utf-8")
    assert a_bytes == b_bytes == c_bytes


# ---------------------------------------------------------------------------
# Instructor lazy seam
# ---------------------------------------------------------------------------


def test_enable_instructor_factory_rejects_none_client() -> None:
    with pytest.raises(InstructorError):
        enable_instructor_factory(None)


def test_enable_instructor_factory_lazy_seam_when_missing() -> None:
    """Module imports cleanly without instructor installed; the factory
    must raise :class:`ModuleNotFoundError` only at call time."""
    try:
        import instructor  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            enable_instructor_factory(object())
    else:
        pytest.skip("instructor is installed; lazy-seam path unobservable")


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(ia)


def test_no_top_level_instructor_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(("import instructor", "from instructor")):
            raise AssertionError(f"instructor must be a lazy seam — no top-level: {line!r}")


def test_no_top_level_openai_or_anthropic_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(
            (
                "import openai",
                "from openai",
                "import anthropic",
                "from anthropic",
            )
        ):
            raise AssertionError(f"openai/anthropic must be lazy seams — no top-level: {line!r}")


def test_no_top_level_pydantic_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(("import pydantic", "from pydantic")):
            raise AssertionError(
                f"pydantic must be a lazy seam in this adapter — no top-level: {line!r}"
            )


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
