# ADAPTED FROM: https://github.com/dottxt-ai/outlines (Apache-2.0)
#
# Tier-I I-21 — JSON-schema-grammar seam.
#
# Outlines compiles a JSON schema into a finite-state automaton / regex
# and constrains LLM decoding to emit only strings the FSM accepts.
# The production default here is pure-stdlib:
#
#   1. ``compile_schema_to_regex(schema)`` walks a closed-subset JSON
#      schema (object with typed leaf properties) into a single
#      anchored :class:`re.Pattern` whose named capture groups equal
#      the schema field names.
#   2. ``match_schema(text, schema)`` finds the first JSON object
#      substring matching that compiled pattern, then runs the
#      captured groups through stdlib :func:`json.loads` so the
#      returned dict has fully-typed Python values.
#
# ``outlines`` (and its transformers / llama_cpp backends) is the lazy
# seam — only imported inside :func:`enable_outlines_factory` body.
# Production environments without outlines installed still import this
# module cleanly.
#
# NEW_PIP_DEPENDENCIES = ("outlines",)
#
# Authority constraints (pinned by ``tests/test_outlines_adapter.py``):
#
#   * **RUNTIME_SAFE** — pure compiler + matcher. No clock, no I/O,
#     no PRNG. Three independent calls with identical inputs produce
#     byte-identical output dicts (INV-15).
#   * **B1** — no execution_engine / governance_engine / system_engine
#     cross-imports.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * No top-level imports of :mod:`outlines`, :mod:`transformers`,
#     :mod:`llama_cpp`, :mod:`torch`, :mod:`numpy`, :mod:`time`,
#     :mod:`datetime`, :mod:`random`, :mod:`asyncio`, :mod:`requests`.
"""I-21 outlines adapter — JSON schema → regex compiler + matcher."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "OutlinesError",
    "OutlinesSchemaError",
    "OutlinesMatchError",
    "JsonSchemaField",
    "JsonSchema",
    "compile_schema_to_regex",
    "match_schema",
    "enable_outlines_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("outlines",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutlinesError(ValueError):
    """Base class for I-21 outlines-adapter errors."""


class OutlinesSchemaError(OutlinesError):
    """Raised when a JsonSchema is malformed."""


class OutlinesMatchError(OutlinesError):
    """Raised when ``text`` fails to match a compiled schema."""


# ---------------------------------------------------------------------------
# Schema value objects
# ---------------------------------------------------------------------------


_ALLOWED_TYPES: tuple[type, ...] = (str, int, float, bool)


@dataclasses.dataclass(frozen=True, slots=True)
class JsonSchemaField:
    """One typed leaf property in a flat JSON schema.

    Fields:
        name: non-empty identifier matching ``[A-Za-z_][A-Za-z0-9_]*``.
        type_: one of :class:`str`, :class:`int`, :class:`float`,
            :class:`bool`.
    """

    name: str
    type_: type

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise OutlinesSchemaError(
                f"JsonSchemaField.name must be a non-empty str, got {self.name!r}"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise OutlinesSchemaError(
                f"JsonSchemaField.name must match [A-Za-z_][A-Za-z0-9_]*, got {self.name!r}"
            )
        if self.type_ not in _ALLOWED_TYPES:
            raise OutlinesSchemaError(
                f"JsonSchemaField.type_ must be one of {_ALLOWED_TYPES!r}, got {self.type_!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class JsonSchema:
    """Ordered tuple of :class:`JsonSchemaField` declarations.

    All fields are required; the closed subset doesn't support
    ``optional``. Use I-19 (instructor) if optional-field semantics
    are needed.
    """

    fields: tuple[JsonSchemaField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            raise OutlinesSchemaError(
                f"JsonSchema.fields must be a tuple, got {type(self.fields).__name__}"
            )
        if not self.fields:
            raise OutlinesSchemaError("JsonSchema.fields must be non-empty")
        seen: set[str] = set()
        for f in self.fields:
            if not isinstance(f, JsonSchemaField):
                raise OutlinesSchemaError(
                    f"JsonSchema.fields entries must be JsonSchemaField, got {type(f).__name__}"
                )
            if f.name in seen:
                raise OutlinesSchemaError(f"JsonSchema.fields contains duplicate name {f.name!r}")
            seen.add(f.name)

    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


# ---------------------------------------------------------------------------
# Compile JSON schema to regex
# ---------------------------------------------------------------------------


# Match a JSON string literal: opening quote, body of escaped or
# non-quote / non-backslash characters, closing quote.
_JSON_STRING_BODY: str = r'(?:\\.|[^"\\])*'


def _value_regex(t: type) -> str:
    if t is str:
        return f'"{_JSON_STRING_BODY}"'
    if t is bool:
        return r"(?:true|false)"
    if t is int:
        return r"-?(?:0|[1-9]\d*)"
    if t is float:
        return r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    raise OutlinesSchemaError(f"unsupported type {t!r}")


_WS: str = r"\s*"


def compile_schema_to_regex(schema: JsonSchema) -> re.Pattern[str]:
    """Compile a :class:`JsonSchema` into an anchored :class:`re.Pattern`.

    The pattern matches a JSON object literal whose keys are exactly
    the schema field names, in declaration order, with each value
    captured into a named group equal to the field name.

    Pure function — INV-15 byte-identical across runs.
    """

    if not isinstance(schema, JsonSchema):
        raise OutlinesSchemaError(
            f"compile_schema_to_regex requires JsonSchema, got {type(schema).__name__}"
        )
    parts: list[str] = [r"\{", _WS]
    for i, field in enumerate(schema.fields):
        if i > 0:
            parts.append(f"{_WS},{_WS}")
        parts.append(f'"{re.escape(field.name)}"')
        parts.append(f"{_WS}:{_WS}")
        parts.append(f"(?P<{field.name}>{_value_regex(field.type_)})")
    parts.append(f"{_WS}\\}}")
    return re.compile("".join(parts))


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def match_schema(text: str, schema: JsonSchema) -> dict[str, Any]:
    """Match ``text`` against the compiled schema regex, return values.

    Searches for the first JSON object substring matching the schema
    pattern, then feeds the captured slots through :func:`json.loads`
    so the returned dict has fully-typed Python values.

    The output dict is keyed in schema declaration order — INV-15
    byte-identical across runs.
    """

    if not isinstance(text, str):
        raise OutlinesMatchError(f"match_schema text must be str, got {type(text).__name__}")
    pattern = compile_schema_to_regex(schema)
    m = pattern.search(text)
    if m is None:
        raise OutlinesMatchError("match_schema: text did not contain a JSON object matching schema")
    out: dict[str, Any] = {}
    for field in schema.fields:
        raw = m.group(field.name)
        if field.type_ is str:
            out[field.name] = json.loads(f'"{raw[1:-1]}"' if raw.startswith('"') else raw)
        else:
            out[field.name] = json.loads(raw)
        if field.type_ is float and isinstance(out[field.name], int):
            out[field.name] = float(out[field.name])
    return out


# ---------------------------------------------------------------------------
# Lazy ``outlines`` seam
# ---------------------------------------------------------------------------


def enable_outlines_factory(
    model: Any,
) -> Callable[[JsonSchema, str], dict[str, Any]]:
    """Return a callable that drives constrained generation via outlines.

    Importing :mod:`outlines` is deferred to factory-call time. The
    returned callable signature is::

        invoke(schema: JsonSchema, prompt: str) -> dict[str, Any]

    The runtime backend constrains decoding so the generated string is
    guaranteed to match the compiled schema regex — but we still
    re-run :func:`match_schema` to keep the audit shape canonical.
    """

    if model is None:
        raise OutlinesError("enable_outlines_factory: model is required")
    import outlines  # type: ignore[import-not-found]  # noqa: F401 - lazy seam

    def _call(schema: JsonSchema, prompt: str) -> dict[str, Any]:
        if not isinstance(schema, JsonSchema):
            raise OutlinesError(f"outlines schema must be JsonSchema, got {type(schema).__name__}")
        if not isinstance(prompt, str):
            raise OutlinesError(f"outlines prompt must be str, got {type(prompt).__name__}")
        text = str(model(prompt))
        return match_schema(text, schema)

    return _call


# Mapping kept on the public surface so the cognitive runtime can
# round-trip a parsed dict through ``json.dumps`` without re-importing
# typing helpers.
_MappingAlias = Mapping
