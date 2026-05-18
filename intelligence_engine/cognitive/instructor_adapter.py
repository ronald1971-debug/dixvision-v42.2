# ADAPTED FROM: https://github.com/567-labs/instructor (MIT)
#
# Tier-I I-19 — structured LLM output validation seam.
#
# ``instructor`` patches the OpenAI / Anthropic SDKs to coerce raw
# completion text into a typed pydantic model. The production default
# here is pure-stdlib: we accept a ``InstructorSchema`` value object
# describing the expected fields + types and run the validation
# ourselves against a JSON payload parsed out of the assistant's reply
# (markdown ``json`` fence or bare object).
#
# ``instructor`` (and its pydantic+openai stack) is the lazy seam — only
# imported inside :func:`enable_instructor_factory` body. Production
# environments without instructor installed still import this module
# cleanly.
#
# NEW_PIP_DEPENDENCIES = ("instructor",)
#
# Authority constraints (pinned by ``tests/test_instructor_adapter.py``):
#
#   * **RUNTIME_SAFE** — pure validator + parser; no clock, no I/O,
#     no PRNG. Three independent calls with identical inputs produce
#     byte-identical output dicts (INV-15).
#   * **B1** — no execution_engine / governance_engine / system_engine
#     cross-imports.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * No top-level imports of :mod:`instructor`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`pydantic`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`requests`.
"""I-19 instructor adapter — structured LLM output validation."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "InstructorError",
    "InstructorSchemaError",
    "SchemaField",
    "InstructorSchema",
    "extract_typed_payload",
    "validate_instance",
    "enable_instructor_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("instructor",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InstructorError(ValueError):
    """Base class for all instructor-adapter errors."""


class InstructorSchemaError(InstructorError):
    """Raised when a payload fails schema validation."""


# ---------------------------------------------------------------------------
# Schema value objects
# ---------------------------------------------------------------------------


_PRIMITIVES: tuple[type, ...] = (str, int, float, bool)


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaField:
    """One field declaration.

    Fields:
        name: JSON key. Must be a non-empty str.
        type_: expected python type. Restricted to ``str``, ``int``,
            ``float``, ``bool``. ``bool`` is rejected when ``int`` is
            expected, and vice versa (Python's standard ``bool``-is-int
            footgun is explicitly closed).
        required: whether the field must be present.
        choices: optional closed enumeration. If set, the parsed value
            must compare equal to one of these.
    """

    name: str
    type_: type
    required: bool = True
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InstructorError(f"SchemaField.name must be a non-empty str, got {self.name!r}")
        if self.type_ not in _PRIMITIVES:
            raise InstructorError(
                f"SchemaField.type_ must be one of {_PRIMITIVES}, got {self.type_!r}"
            )
        if self.choices is not None:
            if not isinstance(self.choices, tuple) or len(self.choices) == 0:
                raise InstructorError(
                    f"SchemaField.choices must be a non-empty tuple or None, got {self.choices!r}"
                )
            for c in self.choices:
                if type(c) is not self.type_:
                    raise InstructorError(
                        f"SchemaField.choices element {c!r} is not of type {self.type_.__name__}"
                    )


@dataclasses.dataclass(frozen=True, slots=True)
class InstructorSchema:
    """An ordered collection of :class:`SchemaField` declarations.

    Field names must be unique. Order is preserved so the validated
    output dict is canonical-sorted on the field declaration order, not
    on the JSON-payload key order.
    """

    fields: tuple[SchemaField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            raise InstructorError(
                f"InstructorSchema.fields must be a tuple, got {type(self.fields).__name__}"
            )
        if len(self.fields) == 0:
            raise InstructorError("InstructorSchema.fields must not be empty")
        seen: set[str] = set()
        for f in self.fields:
            if not isinstance(f, SchemaField):
                raise InstructorError(
                    f"InstructorSchema.fields elements must be SchemaField, got {type(f).__name__}"
                )
            if f.name in seen:
                raise InstructorError(f"InstructorSchema duplicate field name: {f.name!r}")
            seen.add(f.name)

    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _coerce_value(field: SchemaField, raw: Any) -> Any:
    if field.type_ is bool:
        if not isinstance(raw, bool):
            raise InstructorSchemaError(
                f"field {field.name!r}: expected bool, got {type(raw).__name__}"
            )
        return raw
    if field.type_ is int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise InstructorSchemaError(
                f"field {field.name!r}: expected int, got {type(raw).__name__}"
            )
        return raw
    if field.type_ is float:
        if isinstance(raw, bool):
            raise InstructorSchemaError(f"field {field.name!r}: expected float, got bool")
        if isinstance(raw, int):
            return float(raw)
        if isinstance(raw, float):
            return raw
        raise InstructorSchemaError(
            f"field {field.name!r}: expected float, got {type(raw).__name__}"
        )
    if field.type_ is str:
        if not isinstance(raw, str):
            raise InstructorSchemaError(
                f"field {field.name!r}: expected str, got {type(raw).__name__}"
            )
        return raw
    raise InstructorSchemaError(  # pragma: no cover - guarded by __post_init__
        f"field {field.name!r}: unsupported type {field.type_!r}"
    )


def validate_instance(
    payload: Mapping[str, Any],
    schema: InstructorSchema,
) -> dict[str, Any]:
    """Validate ``payload`` against ``schema``.

    Returns a fresh ``dict`` keyed in schema declaration order. Raises
    :class:`InstructorSchemaError` on any violation.
    """

    if not isinstance(payload, Mapping):
        raise InstructorSchemaError(
            f"validate_instance payload must be a mapping, got {type(payload).__name__}"
        )
    if not isinstance(schema, InstructorSchema):
        raise InstructorError(
            f"validate_instance schema must be an InstructorSchema, got {type(schema).__name__}"
        )
    declared = schema.field_names()
    extras = tuple(sorted(k for k in payload.keys() if k not in declared))
    if extras:
        raise InstructorSchemaError(f"payload has extra fields not in schema: {extras!r}")
    out: dict[str, Any] = {}
    for field in schema.fields:
        if field.name not in payload:
            if field.required:
                raise InstructorSchemaError(f"required field missing: {field.name!r}")
            continue
        value = _coerce_value(field, payload[field.name])
        if field.choices is not None and value not in field.choices:
            raise InstructorSchemaError(
                f"field {field.name!r}: value {value!r} not in choices {field.choices!r}"
            )
        out[field.name] = value
    return out


# ---------------------------------------------------------------------------
# Payload extraction
# ---------------------------------------------------------------------------


# Match a ``json``-fenced code block. Anchored, single-block tolerant.
_FENCE_PATTERN: re.Pattern[str] = re.compile(
    r"```(?:json|JSON)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


def extract_typed_payload(
    text: str,
    schema: InstructorSchema,
) -> dict[str, Any]:
    """Parse a JSON object out of ``text`` and validate it.

    The parser is defensive — it accepts:

    1. A markdown-fenced ``json`` code block, or
    2. A bare JSON object that occupies the whole text body.

    Anything else raises :class:`InstructorSchemaError`. The result is
    INV-15-deterministic: identical ``text`` + ``schema`` always
    produce a byte-identical output dict.
    """

    if not isinstance(text, str):
        raise InstructorSchemaError(
            f"extract_typed_payload text must be str, got {type(text).__name__}"
        )
    match = _FENCE_PATTERN.search(text)
    body = match.group("body") if match is not None else text
    body = body.strip()
    if not body:
        raise InstructorSchemaError("extract_typed_payload: empty payload")
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InstructorSchemaError(f"extract_typed_payload: invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise InstructorSchemaError(
            f"extract_typed_payload: top-level must be a JSON object, got {type(raw).__name__}"
        )
    return validate_instance(raw, schema)


# ---------------------------------------------------------------------------
# Lazy ``instructor`` seam
# ---------------------------------------------------------------------------


def enable_instructor_factory(
    client: Any,
) -> Callable[[str, str, InstructorSchema], dict[str, Any]]:
    """Return a callable that wraps an instructor-patched LLM client.

    Importing :mod:`instructor` is deferred to factory-call time. The
    returned callable signature is::

        embed(model: str, prompt: str, schema: InstructorSchema) ->
            dict[str, Any]

    Production environments without instructor installed import this
    module cleanly — the pure-stdlib :func:`extract_typed_payload`
    path remains the default.
    """

    if client is None:
        raise InstructorError("enable_instructor_factory: client is required")
    import instructor  # type: ignore[import-not-found]  # noqa: F401 - lazy seam

    patched = instructor.from_openai(client)

    def _call(
        model: str,
        prompt: str,
        schema: InstructorSchema,
    ) -> dict[str, Any]:
        if not isinstance(model, str) or not model:
            raise InstructorError(f"instructor model must be a non-empty str, got {model!r}")
        if not isinstance(prompt, str):
            raise InstructorError(f"instructor prompt must be str, got {type(prompt).__name__}")
        # Instructor will round-trip through pydantic; we deliberately
        # round-trip through validate_instance so the audit shape stays
        # canonical regardless of backend.
        response = patched.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
        )
        text = response.choices[0].message.content or ""
        return extract_typed_payload(text, schema)

    return _call
