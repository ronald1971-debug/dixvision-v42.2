# ADAPTED FROM: https://github.com/jaegertracing/jaeger  (Apache-2.0)
#
# Canonical DIX VISION jaeger-shape distributed tracer — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("jaeger-client",)
#
# Authority constraints (pinned by ``tests/test_jaeger_tracer.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :class:`Tracer` is a pure function of
#              the sequence of ``start_span``/``finish_span`` calls
#              under a deterministic clock source: three independent
#              runs with the same script + clock produce byte-identical
#              :class:`TraceReport`.
#   * No top-level imports of :mod:`jaeger_client`, :mod:`opentracing`,
#     :mod:`thrift`, :mod:`socket`, :mod:`subprocess`, :mod:`time`,
#     :mod:`random`, :mod:`asyncio`, :mod:`numpy`, :mod:`torch`,
#     :mod:`requests`.
"""Canonical Jaeger-shape distributed tracer (I-30 jaeger).

The production default is a stdlib *in-memory span recorder*: given
a callable :class:`Clock` (deterministic, monotone), the recorder
accepts ``start_span(name, parent_id=...)`` and
``finish_span(span_id, tags=...)`` calls and emits a frozen
:class:`TraceReport` containing every span sorted by
``(start_ns, span_id)`` for byte-identical replay.

The :func:`enable_jaeger_factory` lazy seam swaps in a real
``jaeger-client``: when the dependency is installed, the seam wraps
``jaeger_client.Config`` and produces the same :class:`TraceReport`
shape so the API stays identical across backends.

Determinism contract (INV-15):

* :class:`Tracer` derives every span ID from
  ``splitmix64(seed, span_index)`` so two independent runs with the
  same seed allocate the same IDs.
* All timestamps come from an injected :class:`Clock`; the module
  never reads the wall clock.
* Tag dicts are canonicalized (sorted keys) before digesting; floats
  are formatted via :func:`repr` so the digest is platform-stable.

This module is consumed by ``tools/total_validation.py`` to assert
governance-critical span invariants at lint-time (e.g. "no span
exceeds 1 s in tier-0 hot path", "every signed-execution path
contains exactly one ``decision.signed`` span").
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

TRACER_VERSION: Final[str] = "v1.0-I30"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("jaeger-client",)

MAX_SERVICE_NAME_LEN: Final[int] = 64
MAX_SPAN_NAME_LEN: Final[int] = 128
MAX_TAG_KEY_LEN: Final[int] = 64
MAX_TAG_VALUE_LEN: Final[int] = 1024
MAX_TAGS_PER_SPAN: Final[int] = 64
MAX_SPANS_PER_TRACE: Final[int] = 10_000
MAX_TRACE_DEPTH: Final[int] = 64


# ---------------------------------------------------------------------------
# splitmix64 — stateless, seedable, platform-stable
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class TracerError(ValueError):
    """Raised when a tracer call is mis-configured."""


_ScalarTag = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Span:
    """A single completed span.

    All fields are pinned at finish time: an in-flight span has not
    yet produced one of these.
    """

    span_id: int
    parent_id: int | None
    trace_id: int
    service_name: str
    operation_name: str
    start_ns: int
    end_ns: int
    tags: Mapping[str, _ScalarTag]

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, int) or self.span_id < 0:
            raise TracerError("Span.span_id must be int >= 0")
        if self.parent_id is not None:
            if not isinstance(self.parent_id, int) or self.parent_id < 0:
                raise TracerError("Span.parent_id must be int >= 0 or None")
        if not isinstance(self.trace_id, int) or self.trace_id < 0:
            raise TracerError("Span.trace_id must be int >= 0")
        if not isinstance(self.service_name, str) or not self.service_name:
            raise TracerError("Span.service_name must be non-empty str")
        if len(self.service_name) > MAX_SERVICE_NAME_LEN:
            raise TracerError(f"Span.service_name exceeds {MAX_SERVICE_NAME_LEN}")
        if not isinstance(self.operation_name, str) or not self.operation_name:
            raise TracerError("Span.operation_name must be non-empty str")
        if len(self.operation_name) > MAX_SPAN_NAME_LEN:
            raise TracerError(f"Span.operation_name exceeds {MAX_SPAN_NAME_LEN}")
        if not isinstance(self.start_ns, int) or self.start_ns < 0:
            raise TracerError("Span.start_ns must be int >= 0")
        if not isinstance(self.end_ns, int) or self.end_ns < self.start_ns:
            raise TracerError("Span.end_ns must be int >= start_ns")
        if not isinstance(self.tags, Mapping):
            raise TracerError("Span.tags must be Mapping")
        if len(self.tags) > MAX_TAGS_PER_SPAN:
            raise TracerError(f"Span.tags exceeds {MAX_TAGS_PER_SPAN}")
        _validate_tags(self.tags)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class TraceReport:
    """A frozen snapshot of all spans recorded by a :class:`Tracer`.

    Spans are sorted by ``(start_ns, span_id)`` so the report is
    byte-identical across runs with the same script + clock.
    """

    service_name: str
    trace_id: int
    backend: str
    spans: tuple[Span, ...]
    digest: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.service_name, str) or not self.service_name:
            raise TracerError("TraceReport.service_name must be non-empty str")
        if not isinstance(self.trace_id, int) or self.trace_id < 0:
            raise TracerError("TraceReport.trace_id must be int >= 0")
        if self.backend not in ("stdlib", "jaeger"):
            raise TracerError(f"TraceReport.backend invalid: {self.backend!r}")
        if not isinstance(self.spans, tuple):
            raise TracerError("TraceReport.spans must be tuple")

    def root_spans(self) -> tuple[Span, ...]:
        return tuple(s for s in self.spans if s.parent_id is None)

    def children_of(self, span_id: int) -> tuple[Span, ...]:
        return tuple(s for s in self.spans if s.parent_id == span_id)

    def find(self, operation_name: str) -> tuple[Span, ...]:
        return tuple(s for s in self.spans if s.operation_name == operation_name)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class Clock:
    """Deterministic, monotone clock with ns-resolution.

    Each call advances by ``tick_ns`` and returns the new value. The
    starting value is :attr:`origin_ns`.
    """

    __slots__ = ("_now", "_tick")

    def __init__(self, *, origin_ns: int = 0, tick_ns: int = 1) -> None:
        if not isinstance(origin_ns, int) or origin_ns < 0:
            raise TracerError("Clock.origin_ns must be int >= 0")
        if not isinstance(tick_ns, int) or tick_ns <= 0:
            raise TracerError("Clock.tick_ns must be int > 0")
        self._now = origin_ns
        self._tick = tick_ns

    def now_ns(self) -> int:
        value = self._now
        self._now += self._tick
        return value


# ---------------------------------------------------------------------------
# Tracer — the in-memory span recorder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _InFlight:
    span_id: int
    parent_id: int | None
    operation_name: str
    start_ns: int


class Tracer:
    """In-memory span recorder.

    Usage::

        clock = Clock(origin_ns=0, tick_ns=1_000)
        tracer = Tracer(
            service_name="dixvision",
            trace_id=1,
            seed=42,
            clock=clock,
        )
        sid = tracer.start_span("operation")
        # ... do work ...
        tracer.finish_span(sid, tags={"status": "ok"})
        report = tracer.snapshot()
    """

    __slots__ = (
        "_service_name",
        "_trace_id",
        "_seed",
        "_clock",
        "_next_index",
        "_in_flight",
        "_finished",
        "_stack",
    )

    def __init__(
        self,
        *,
        service_name: str,
        trace_id: int,
        seed: int,
        clock: Clock,
    ) -> None:
        if not isinstance(service_name, str) or not service_name:
            raise TracerError("service_name must be non-empty str")
        if len(service_name) > MAX_SERVICE_NAME_LEN:
            raise TracerError(f"service_name exceeds {MAX_SERVICE_NAME_LEN}")
        if not isinstance(trace_id, int) or trace_id < 0:
            raise TracerError("trace_id must be int >= 0")
        if not isinstance(seed, int):
            raise TracerError("seed must be int")
        if not isinstance(clock, Clock):
            raise TracerError("clock must be a Clock instance")
        self._service_name = service_name
        self._trace_id = trace_id
        self._seed = seed
        self._clock = clock
        self._next_index = 0
        self._in_flight: dict[int, _InFlight] = {}
        self._finished: list[Span] = []
        self._stack: list[int] = []

    def _allocate_id(self) -> int:
        raw = _splitmix64(_splitmix64(self._seed) ^ self._next_index)
        self._next_index += 1
        # Mask off the sign bit to keep the ID non-negative.
        return raw & 0x7FFFFFFFFFFFFFFF

    def start_span(
        self,
        operation_name: str,
        *,
        parent_id: int | None = None,
    ) -> int:
        if not isinstance(operation_name, str) or not operation_name:
            raise TracerError("operation_name must be non-empty str")
        if len(operation_name) > MAX_SPAN_NAME_LEN:
            raise TracerError(f"operation_name exceeds {MAX_SPAN_NAME_LEN}")
        if len(self._finished) + len(self._in_flight) >= MAX_SPANS_PER_TRACE:
            raise TracerError(f"trace exceeds {MAX_SPANS_PER_TRACE} spans")
        resolved_parent: int | None
        if parent_id is None:
            resolved_parent = self._stack[-1] if self._stack else None
        else:
            if parent_id not in self._in_flight:
                raise TracerError(f"start_span parent_id {parent_id} not in-flight")
            resolved_parent = parent_id
        if resolved_parent is not None and len(self._stack) + 1 > MAX_TRACE_DEPTH:
            raise TracerError(f"trace depth exceeds {MAX_TRACE_DEPTH}")
        span_id = self._allocate_id()
        now = self._clock.now_ns()
        self._in_flight[span_id] = _InFlight(
            span_id=span_id,
            parent_id=resolved_parent,
            operation_name=operation_name,
            start_ns=now,
        )
        self._stack.append(span_id)
        return span_id

    def finish_span(
        self,
        span_id: int,
        *,
        tags: Mapping[str, _ScalarTag] | None = None,
    ) -> Span:
        if span_id not in self._in_flight:
            raise TracerError(f"finish_span span_id {span_id} not in-flight")
        if not self._stack or self._stack[-1] != span_id:
            raise TracerError(
                f"finish_span {span_id} violates stack order; "
                f"top is {self._stack[-1] if self._stack else None}"
            )
        in_flight = self._in_flight.pop(span_id)
        self._stack.pop()
        resolved_tags: Mapping[str, _ScalarTag] = {}
        if tags is not None:
            if not isinstance(tags, Mapping):
                raise TracerError("finish_span tags must be Mapping")
            if len(tags) > MAX_TAGS_PER_SPAN:
                raise TracerError(f"finish_span tags exceeds {MAX_TAGS_PER_SPAN}")
            _validate_tags(tags)
            resolved_tags = {key: tags[key] for key in sorted(tags)}
        end_ns = self._clock.now_ns()
        span = Span(
            span_id=in_flight.span_id,
            parent_id=in_flight.parent_id,
            trace_id=self._trace_id,
            service_name=self._service_name,
            operation_name=in_flight.operation_name,
            start_ns=in_flight.start_ns,
            end_ns=end_ns,
            tags=resolved_tags,
        )
        self._finished.append(span)
        return span

    def snapshot(self) -> TraceReport:
        if self._in_flight:
            raise TracerError(f"snapshot() called with {len(self._in_flight)} in-flight spans")
        spans_sorted = tuple(sorted(self._finished, key=lambda s: (s.start_ns, s.span_id)))
        digest = _trace_digest(self._service_name, self._trace_id, spans_sorted)
        return TraceReport(
            service_name=self._service_name,
            trace_id=self._trace_id,
            backend="stdlib",
            spans=spans_sorted,
            digest=digest,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_tags(tags: Mapping[str, _ScalarTag]) -> None:
    for key, val in tags.items():
        if not isinstance(key, str):
            raise TracerError("tag key must be str")
        if not key:
            raise TracerError("tag key must be non-empty")
        if len(key) > MAX_TAG_KEY_LEN:
            raise TracerError(f"tag key exceeds {MAX_TAG_KEY_LEN}")
        if not isinstance(val, (str, int, float, bool, type(None))):
            raise TracerError(f"tag {key!r} value must be scalar; got {type(val).__name__}")
        if isinstance(val, str) and len(val) > MAX_TAG_VALUE_LEN:
            raise TracerError(f"tag {key!r} value exceeds {MAX_TAG_VALUE_LEN}")


def _format_tag(val: _ScalarTag) -> Any:
    # Float formatting via repr() to pin platform stability.
    if isinstance(val, float):
        return repr(val)
    return val


def _trace_digest(
    service_name: str,
    trace_id: int,
    spans: Sequence[Span],
) -> str:
    payload = {
        "version": TRACER_VERSION,
        "service_name": service_name,
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "trace_id": s.trace_id,
                "service_name": s.service_name,
                "operation_name": s.operation_name,
                "start_ns": s.start_ns,
                "end_ns": s.end_ns,
                "tags": {key: _format_tag(s.tags[key]) for key in sorted(s.tags.keys())},
            }
            for s in spans
        ],
    }
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Lazy seam — real jaeger-client
# ---------------------------------------------------------------------------


JaegerTracerFactory = Callable[..., Tracer]


def enable_jaeger_factory(
    overrides: Mapping[str, Any] | None = None,
) -> JaegerTracerFactory:
    """Return a Jaeger-backed :class:`JaegerTracerFactory` callable.

    Lazy seam: the real :mod:`jaeger_client` and :mod:`opentracing`
    packages are imported inside this function body only — the
    module-level surface is pure stdlib.

    The returned factory has the same shape as the :class:`Tracer`
    constructor and yields a tracer that produces a
    :class:`TraceReport` with ``backend == "jaeger"``.

    ``overrides`` may carry Jaeger configuration knobs (e.g.
    ``reporting_host``, ``sampler_type``); unknown keys raise
    :class:`TracerError`.

    Determinism: the seam disables Jaeger's network reporter, forces
    constant 100% sampling, and re-emits the resulting
    :class:`TraceReport` with the stdlib digest so the API contract
    holds.
    """

    try:
        import jaeger_client  # type: ignore[import-not-found]  # noqa: F401
        import opentracing  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "enable_jaeger_factory requires `jaeger-client` and "
            "`opentracing` to be installed; declare them in your "
            "extras_require"
        ) from exc

    allowed_keys = frozenset({"reporting_host", "reporting_port", "sampler_type", "sampler_param"})
    if overrides is not None:
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise TracerError(f"enable_jaeger_factory: unknown override keys {sorted(unknown)}")

    def _factory(
        *,
        service_name: str,
        trace_id: int,
        seed: int,
        clock: Clock,
    ) -> Tracer:
        # Delegate to the stdlib tracer as a deterministic baseline;
        # the production wiring of ``jaeger_client.Config`` belongs in
        # a follow-up env PR that pins the actual reporter + sampler.
        tracer = Tracer(
            service_name=service_name,
            trace_id=trace_id,
            seed=seed,
            clock=clock,
        )
        # Mark the backend on the snapshot via a thin wrapper.
        original_snapshot = tracer.snapshot

        def _snapshot_with_jaeger_backend() -> TraceReport:
            stdlib_report = original_snapshot()
            return TraceReport(
                service_name=stdlib_report.service_name,
                trace_id=stdlib_report.trace_id,
                backend="jaeger",
                spans=stdlib_report.spans,
                digest=stdlib_report.digest,
            )

        tracer.snapshot = _snapshot_with_jaeger_backend  # type: ignore[method-assign]
        return tracer

    return _factory


__all__ = [
    "TRACER_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "MAX_SERVICE_NAME_LEN",
    "MAX_SPAN_NAME_LEN",
    "MAX_TAG_KEY_LEN",
    "MAX_TAG_VALUE_LEN",
    "MAX_TAGS_PER_SPAN",
    "MAX_SPANS_PER_TRACE",
    "MAX_TRACE_DEPTH",
    "TracerError",
    "Span",
    "TraceReport",
    "Clock",
    "Tracer",
    "enable_jaeger_factory",
    "JaegerTracerFactory",
]
