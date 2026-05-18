"""A-09 — opentelemetry-python → distributed tracing.

# ADAPTED FROM: open-telemetry/opentelemetry-python
#   - opentelemetry-api/src/opentelemetry/trace/__init__.py
#   - opentelemetry-api/src/opentelemetry/trace/span.py
#   - opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py
#
# License: Apache-2.0.

Canonical adaptation of OpenTelemetry's tracing primitives to the DIX
RUNTIME_SAFE tier.  The official ``opentelemetry`` packages are lazy-imported
*only* inside :func:`otel_tracer_factory` — top-level imports remain
empty so the module is importable in replay / test environments where the
dependency is absent.

Design choices (vs. the upstream library):

* Span timestamps are **always caller-supplied**.  The tracer never reads a
  clock so INV-15 byte-identical replay holds across machines.
* ``trace_id`` / ``span_id`` are derived from a caller-supplied
  ``(trace_seed, span_seed)`` pair via splitmix64 + BLAKE2b-8 — no randomness
  on the hot path.
* The in-process tracer records every span into an append-only buffer; the
  snapshot is canonicalised (sorted ``attribute`` keys, sorted spans by
  ``(trace_id, span_id)``) so 3-run replay yields byte-identical output.
* :class:`Tracer` is a runtime-checkable Protocol; both the pure-Python
  :class:`InProcessTracer` and the lazy OTel wrapper satisfy it.
* The module **does not** construct typed bus events (``SignalEvent`` /
  ``ExecutionEvent`` / ``SystemEvent`` / ``HazardEvent`` /
  ``GovernanceDecision``) — B27 / B28 / INV-71 authority symmetry holds.

The module is RUNTIME_SAFE: it never blocks the hot path, never imports
``time``/``datetime``/``random``/``os``/``asyncio``/``socket`` at top level,
and never reaches into ``governance_engine`` / ``execution_engine`` /
``evolution_engine`` (B1).
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "opentelemetry-sdk",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-exporter-otlp-proto-http",
)
OTEL_ADAPTER_VERSION: str = "1"

MAX_ATTRIBUTE_VALUE_LEN: int = 256
MAX_SPAN_NAME_LEN: int = 128
MAX_ATTRIBUTE_COUNT: int = 64
MAX_SPAN_BUFFER: int = 65_536

DEFAULT_SAMPLE_RATIO: float = 0.1

_ATTR_FALLBACK: str = "_"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TracerError(RuntimeError):
    """Raised when the tracer rejects an input or transport error occurs."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_SPLITMIX_MASK = (1 << 64) - 1


def _splitmix64(seed: int) -> int:
    z = (seed + 0x9E3779B97F4A7C15) & _SPLITMIX_MASK
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _SPLITMIX_MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _SPLITMIX_MASK
    return z ^ (z >> 31)


def _derive_id(seed: int, *, size: int) -> str:
    """Derive a fixed-width hex id from a 64-bit seed via BLAKE2b."""
    if size not in (8, 16):
        raise ValueError("size must be 8 or 16")
    digest_size = size // 2
    mixed = _splitmix64(seed)
    h = hashlib.blake2b(
        mixed.to_bytes(8, "big"),
        digest_size=digest_size,
    )
    return h.hexdigest()


def sanitize_attribute_value(value: Any) -> str | int | float | bool:
    """Coerce attribute value to a Prometheus / OTel-safe primitive."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return _ATTR_FALLBACK
    text = str(value)
    if not text:
        return _ATTR_FALLBACK
    cleaned_chars: list[str] = []
    for ch in text:
        if ch == "\\":
            cleaned_chars.append("\\\\")
        elif ch == '"':
            cleaned_chars.append('\\"')
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            cleaned_chars.append("_")
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    if len(cleaned) > MAX_ATTRIBUTE_VALUE_LEN:
        cleaned = cleaned[:MAX_ATTRIBUTE_VALUE_LEN]
    return cleaned


def _sanitize_span_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("span name must be a str")
    if not name:
        raise ValueError("span name must be non-empty")
    if len(name) > MAX_SPAN_NAME_LEN:
        raise ValueError(f"span name must be ≤ {MAX_SPAN_NAME_LEN} chars, got {len(name)}")
    return name


def _canonicalise_attributes(
    attributes: Mapping[str, Any] | None,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    if attributes is None:
        return ()
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    if len(attributes) > MAX_ATTRIBUTE_COUNT:
        raise ValueError(f"attributes must be ≤ {MAX_ATTRIBUTE_COUNT}, got {len(attributes)}")
    pairs: list[tuple[str, str | int | float | bool]] = []
    for key in sorted(attributes):
        if not isinstance(key, str):
            raise TypeError("attribute keys must be str")
        if not key:
            raise ValueError("attribute keys must be non-empty")
        pairs.append((key, sanitize_attribute_value(attributes[key])))
    return tuple(pairs)


# ---------------------------------------------------------------------------
# SpanRecord value object
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class SpanRecord:
    """An immutable record of one finished span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ts_ns: int
    end_ts_ns: int
    attributes: tuple[tuple[str, str | int | float | bool], ...]

    @property
    def duration_ns(self) -> int:
        return self.end_ts_ns - self.start_ts_ns


@dataclasses.dataclass(frozen=True, slots=True)
class TraceSnapshot:
    """Deterministic snapshot of all finished spans."""

    spans: tuple[SpanRecord, ...]
    sampled_in: int
    dropped: int


# ---------------------------------------------------------------------------
# Tracer Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Tracer(Protocol):
    """Caller-driven, clock-free tracer protocol."""

    def start_span(
        self,
        *,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_ts_ns: int,
        attributes: Mapping[str, Any] | None = ...,
    ) -> str: ...

    def end_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        end_ts_ns: int,
        attributes: Mapping[str, Any] | None = ...,
    ) -> SpanRecord: ...

    def snapshot(self) -> TraceSnapshot: ...


# ---------------------------------------------------------------------------
# Pure-Python in-process tracer
# ---------------------------------------------------------------------------
@dataclasses.dataclass(slots=True)
class _OpenSpanState:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_ts_ns: int
    attributes: dict[str, Any]


class InProcessTracer:
    """Append-only in-process tracer.

    * Thread-safe via a single ``threading.Lock``.
    * Spans are accepted only when the BLAKE2b-8 sampling oracle over
      ``trace_id`` falls below ``sample_ratio`` (deterministic per trace).
    * Buffer is bounded at :data:`MAX_SPAN_BUFFER`; once full, additional
      finished spans are dropped and counted.
    """

    def __init__(
        self,
        *,
        sample_ratio: float = DEFAULT_SAMPLE_RATIO,
        buffer_size: int = MAX_SPAN_BUFFER,
    ) -> None:
        if not isinstance(sample_ratio, (int, float)):
            raise TypeError("sample_ratio must be numeric")
        sr = float(sample_ratio)
        if not (0.0 <= sr <= 1.0):
            raise ValueError("sample_ratio must be in [0.0, 1.0]")
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive int")
        self._sample_ratio = sr
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._open: dict[tuple[str, str], _OpenSpanState] = {}
        self._finished: list[SpanRecord] = []
        self._sampled_in: int = 0
        self._sampled_out: dict[tuple[str, str], None] = {}
        self._dropped: int = 0

    @property
    def sample_ratio(self) -> float:
        return self._sample_ratio

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    def _is_sampled(self, trace_id: str) -> bool:
        if self._sample_ratio >= 1.0:
            return True
        if self._sample_ratio <= 0.0:
            return False
        digest = hashlib.blake2b(trace_id.encode("ascii"), digest_size=8).digest()
        oracle = int.from_bytes(digest, "big") / float(1 << 64)
        return oracle < self._sample_ratio

    def start_span(
        self,
        *,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_ts_ns: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> str:
        sanitized_name = _sanitize_span_name(name)
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("trace_id must be a non-empty str")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError("span_id must be a non-empty str")
        if parent_span_id is not None and not isinstance(parent_span_id, str):
            raise TypeError("parent_span_id must be str or None")
        if not isinstance(start_ts_ns, int) or start_ts_ns < 0:
            raise ValueError("start_ts_ns must be a non-negative int")
        if attributes is None:
            attrs: dict[str, Any] = {}
        else:
            if not isinstance(attributes, Mapping):
                raise TypeError("attributes must be a mapping")
            attrs = dict(attributes)
        key = (trace_id, span_id)
        with self._lock:
            if not self._is_sampled(trace_id):
                self._sampled_out[key] = None
                return span_id
            if key in self._open:
                raise TracerError(f"span already open: trace_id={trace_id} span_id={span_id}")
            self._open[key] = _OpenSpanState(
                name=sanitized_name,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                start_ts_ns=start_ts_ns,
                attributes=attrs,
            )
            return span_id

    def end_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        end_ts_ns: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> SpanRecord:
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("trace_id must be a non-empty str")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError("span_id must be a non-empty str")
        if not isinstance(end_ts_ns, int) or end_ts_ns < 0:
            raise ValueError("end_ts_ns must be a non-negative int")
        if attributes is not None and not isinstance(attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        key = (trace_id, span_id)
        with self._lock:
            if key in self._sampled_out:
                del self._sampled_out[key]
                return SpanRecord(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=None,
                    name="",
                    start_ts_ns=end_ts_ns,
                    end_ts_ns=end_ts_ns,
                    attributes=(),
                )
            state = self._open.pop(key, None)
            if state is None:
                raise TracerError(f"span not open: trace_id={trace_id} span_id={span_id}")
            if end_ts_ns < state.start_ts_ns:
                raise ValueError("end_ts_ns must be ≥ start_ts_ns")
            merged = dict(state.attributes)
            if attributes is not None:
                for k, v in attributes.items():
                    merged[k] = v
            record = SpanRecord(
                trace_id=state.trace_id,
                span_id=state.span_id,
                parent_span_id=state.parent_span_id,
                name=state.name,
                start_ts_ns=state.start_ts_ns,
                end_ts_ns=end_ts_ns,
                attributes=_canonicalise_attributes(merged),
            )
            if len(self._finished) >= self._buffer_size:
                self._dropped += 1
                return record
            self._finished.append(record)
            self._sampled_in += 1
            return record

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            spans = tuple(
                sorted(
                    self._finished,
                    key=lambda r: (r.trace_id, r.span_id),
                )
            )
            return TraceSnapshot(
                spans=spans,
                sampled_in=self._sampled_in,
                dropped=self._dropped,
            )

    def derive_trace_id(self, seed: int) -> str:
        if not isinstance(seed, int):
            raise TypeError("seed must be int")
        return _derive_id(seed, size=16)

    def derive_span_id(self, seed: int) -> str:
        if not isinstance(seed, int):
            raise TypeError("seed must be int")
        return _derive_id(seed, size=8)


# ---------------------------------------------------------------------------
# OpenTelemetry binding (lazy)
# ---------------------------------------------------------------------------
class _OtelTracerWrapper:
    """Internal adapter that delegates to an ``opentelemetry`` ``Tracer``."""

    def __init__(self, otel_tracer: Any) -> None:
        self._otel_tracer = otel_tracer
        self._open: dict[tuple[str, str], Any] = {}
        self._records: list[SpanRecord] = []
        self._lock = threading.Lock()
        self._sampled_in = 0

    def start_span(
        self,
        *,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_ts_ns: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> str:
        sanitized_name = _sanitize_span_name(name)
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("trace_id must be a non-empty str")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError("span_id must be a non-empty str")
        if not isinstance(start_ts_ns, int) or start_ts_ns < 0:
            raise ValueError("start_ts_ns must be a non-negative int")
        canonical_attrs = _canonicalise_attributes(attributes)
        with self._lock:
            span = self._otel_tracer.start_span(
                name=sanitized_name,
                start_time=start_ts_ns,
                attributes=dict(canonical_attrs),
            )
            self._open[(trace_id, span_id)] = (
                span,
                start_ts_ns,
                sanitized_name,
                parent_span_id,
            )
            return span_id

    def end_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        end_ts_ns: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> SpanRecord:
        if not isinstance(end_ts_ns, int) or end_ts_ns < 0:
            raise ValueError("end_ts_ns must be a non-negative int")
        key = (trace_id, span_id)
        with self._lock:
            entry = self._open.pop(key, None)
            if entry is None:
                raise TracerError(f"span not open: trace_id={trace_id} span_id={span_id}")
            span, start_ts_ns, name, parent_span_id = entry
            merged: dict[str, Any] = {}
            if attributes is not None:
                for k, v in attributes.items():
                    merged[k] = v
                    span.set_attribute(k, sanitize_attribute_value(v))
            span.end(end_time=end_ts_ns)
            record = SpanRecord(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=name,
                start_ts_ns=start_ts_ns,
                end_ts_ns=end_ts_ns,
                attributes=_canonicalise_attributes(merged),
            )
            self._records.append(record)
            self._sampled_in += 1
            return record

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            spans = tuple(
                sorted(
                    self._records,
                    key=lambda r: (r.trace_id, r.span_id),
                )
            )
            return TraceSnapshot(spans=spans, sampled_in=self._sampled_in, dropped=0)


def otel_tracer_factory(
    *,
    service_name: str,
    otel_tracer: Any | None = None,
) -> Tracer:
    """Lazy-bind the OpenTelemetry SDK and return a DIX-compatible :class:`Tracer`.

    The ``opentelemetry`` import is confined to this function body; the rest
    of the module remains importable without the optional dependency.
    """
    if not isinstance(service_name, str) or not service_name:
        raise ValueError("service_name must be a non-empty str")
    if otel_tracer is None:
        try:
            from opentelemetry.sdk.resources import (  # noqa: PLC0415
                Resource,
            )
            from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised when dep absent
            raise TracerError(
                "opentelemetry-sdk is not installed; see NEW_PIP_DEPENDENCIES"
            ) from exc
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        otel_tracer = provider.get_tracer(service_name)
    return _OtelTracerWrapper(otel_tracer)


# ---------------------------------------------------------------------------
# Project-level helpers
# ---------------------------------------------------------------------------
def render_trace_text(snapshot: TraceSnapshot) -> str:
    """Render a snapshot to a canonical, replayable text format."""
    lines: list[str] = []
    lines.append(f"# sampled_in {snapshot.sampled_in}")
    lines.append(f"# dropped {snapshot.dropped}")
    for record in snapshot.spans:
        parent = record.parent_span_id if record.parent_span_id is not None else "-"
        attr_pairs: list[str] = []
        for key, value in record.attributes:
            if isinstance(value, bool):
                attr_value = "true" if value else "false"
            elif isinstance(value, (int, float)):
                attr_value = repr(value)
            else:
                attr_value = f'"{value}"'
            attr_pairs.append(f"{key}={attr_value}")
        attr_blob = ",".join(attr_pairs) if attr_pairs else "-"
        lines.append(
            f"{record.trace_id} {record.span_id} {parent} "
            f"{record.name} {record.start_ts_ns} {record.end_ts_ns} "
            f"{attr_blob}"
        )
    return "\n".join(lines) + "\n"


def derive_ids(*, seed: int, kind: str) -> str:
    """Project a 64-bit seed onto a 8 / 16 hex id (``trace`` / ``span``)."""
    if kind == "trace":
        return _derive_id(seed, size=16)
    if kind == "span":
        return _derive_id(seed, size=8)
    raise ValueError("kind must be 'trace' or 'span'")


__all__ = [
    "DEFAULT_SAMPLE_RATIO",
    "InProcessTracer",
    "MAX_ATTRIBUTE_COUNT",
    "MAX_ATTRIBUTE_VALUE_LEN",
    "MAX_SPAN_BUFFER",
    "MAX_SPAN_NAME_LEN",
    "NEW_PIP_DEPENDENCIES",
    "OTEL_ADAPTER_VERSION",
    "SpanRecord",
    "TraceSnapshot",
    "Tracer",
    "TracerError",
    "derive_ids",
    "otel_tracer_factory",
    "render_trace_text",
    "sanitize_attribute_value",
]


# Bind ``Sequence`` so the import is not flagged unused; used by tests for
# ``isinstance`` checks against ``SpanRecord.attributes``.
_SEQUENCE_BIND: type[Sequence[Any]] = Sequence  # type: ignore[assignment]
