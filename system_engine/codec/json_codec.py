# ADAPTED FROM: https://github.com/ijl/orjson  (Apache-2.0 / MIT)
#
# Canonical orjson-shape JSON codec — OFFLINE_ONLY tier.
#
# NEW_PIP_DEPENDENCIES = ("orjson",)
#
# Authority constraints (pinned by tests/test_orjson_codec.py):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :func:`canonical_dumps` is a pure function of its input;
#              three independent calls produce byte-identical bytes.
#   * B27 / B28 / INV-71 — no typed-event constructors here; passive
#              transcoder only.
#   * No top-level imports of :mod:`orjson`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`numpy`, :mod:`torch`,
#     :mod:`polars`, :mod:`requests`.
"""Pure-Python orjson-shape JSON codec (I-02)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

CODEC_VERSION: Final[str] = "v1.0-I02"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("orjson",)

_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
_ALLOWED_PRIMITIVE_TYPES: Final[tuple[type, ...]] = (
    type(None),
    bool,
    int,
    float,
    str,
)


class CodecError(ValueError):
    """Raised when a payload cannot be canonicalised to orjson-shape bytes."""


@dataclass(frozen=True, slots=True)
class JsonCodec:
    """Pure-function pair that emits orjson-shape bytes.

    ``dumps`` produces UTF-8 bytes that match
    ``orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)`` for the closed payload
    alphabet (None / bool / int / float / str / list / tuple / dict).
    ``loads`` parses any UTF-8 bytes produced by ``dumps`` back to a Python
    value tree. The pair is byte-stable under repeated round-trips.
    """

    dumps: Callable[[Any], bytes]
    loads: Callable[[bytes], Any]
    backend: str

    def __post_init__(self) -> None:
        if not callable(self.dumps):  # pragma: no cover - defensive
            raise CodecError("JsonCodec.dumps must be callable")
        if not callable(self.loads):  # pragma: no cover - defensive
            raise CodecError("JsonCodec.loads must be callable")
        if not isinstance(self.backend, str) or not self.backend:
            raise CodecError("JsonCodec.backend must be a non-empty string")


def _canonicalise(value: Any) -> Any:
    """Normalise ``value`` to a representation stdlib :mod:`json` accepts.

    Tuples are collapsed to lists; dict keys are coerced to ``str`` (orjson
    rejects non-string keys; we mirror that behaviour by raising).  Floats are
    validated against the orjson rule that ``nan`` / ``inf`` / ``-inf`` are not
    representable.
    """

    if isinstance(value, bool):  # bool first — subclass of int
        return value
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CodecError("orjson-shape JSON cannot represent nan / inf / -inf")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonicalise(item) for item in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise CodecError(
                    f"orjson-shape JSON keys must be str (got {type(raw_key).__name__!r})"
                )
            out[raw_key] = _canonicalise(raw_value)
        return out
    raise CodecError(f"value of type {type(value).__name__!r} is not orjson-serialisable")


def canonical_dumps(value: Any) -> bytes:
    """Encode ``value`` to orjson-shape UTF-8 bytes.

    Output shape (byte-identical to ``orjson.dumps(obj, OPT_SORT_KEYS)`` for
    the closed alphabet):

    * No insignificant whitespace — separators are ``(",", ":")``.
    * Keys are sorted ascending under Python string ordering.
    * Strings are emitted as UTF-8 (``ensure_ascii=False``).
    * ``True`` / ``False`` / ``None`` → ``true`` / ``false`` / ``null``.
    """

    normalised = _canonicalise(value)
    text = json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=_JSON_SEPARATORS,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_loads(blob: bytes) -> Any:
    """Decode bytes produced by :func:`canonical_dumps` back to a value tree."""

    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise CodecError(f"canonical_loads requires bytes-like input (got {type(blob).__name__!r})")
    try:
        text = bytes(blob).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodecError(f"payload is not valid UTF-8: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CodecError(f"payload is not valid JSON: {exc}") from exc


def stdlib_codec_factory() -> JsonCodec:
    """Return the always-available pure-stdlib canonical codec."""

    return JsonCodec(
        dumps=canonical_dumps,
        loads=canonical_loads,
        backend="stdlib",
    )


def enable_orjson_factory() -> JsonCodec:
    """Return an orjson-backed codec.

    The :mod:`orjson` import happens *inside* this function (never at module
    level) and is only triggered when the operator explicitly opts in.  If
    :mod:`orjson` is not installed, the caller receives an
    :class:`ImportError` (orjson is declared in
    :data:`NEW_PIP_DEPENDENCIES`, not in base :file:`pyproject.toml`).

    The returned codec is byte-identical to :func:`stdlib_codec_factory` for
    the closed payload alphabet — the orjson and stdlib paths produce the same
    bytes for every input the caller is allowed to pass.
    """

    import orjson  # noqa: PLC0415 - intentional lazy seam

    opt_sort_keys = getattr(orjson, "OPT_SORT_KEYS", 0)

    def _dumps(value: Any) -> bytes:
        normalised = _canonicalise(value)
        return orjson.dumps(normalised, option=opt_sort_keys)

    def _loads(blob: bytes) -> Any:
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise CodecError(
                f"canonical_loads requires bytes-like input (got {type(blob).__name__!r})"
            )
        try:
            return orjson.loads(bytes(blob))
        except orjson.JSONDecodeError as exc:  # pragma: no cover - guarded
            raise CodecError(f"payload is not valid JSON: {exc}") from exc

    return JsonCodec(dumps=_dumps, loads=_loads, backend="orjson")


def default_codec() -> JsonCodec:
    """Return the production-default codec (currently stdlib-backed).

    Promotion to the orjson backend is operator-driven via
    :func:`enable_orjson_factory` — the seam stays cold until research
    acceptance, matching every other Tier-I lazy-seam adapter.
    """

    return stdlib_codec_factory()


__all__: tuple[str, ...] = (
    "CODEC_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "CodecError",
    "JsonCodec",
    "canonical_dumps",
    "canonical_loads",
    "default_codec",
    "enable_orjson_factory",
    "stdlib_codec_factory",
)


def _accept_payload(_: Sequence[str]) -> None:
    """Placeholder kept for symmetry with other adapter modules."""
