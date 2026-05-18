"""A-08 — Prometheus metrics export for system_engine.

# ADAPTED FROM: prometheus/client_python
#   * prometheus_client/metrics.py — Counter / Gauge / Histogram semantics
#   * prometheus_client/exposition.py — text/plain ``generate_latest`` format
#
# Classification: RUNTIME_SAFE.
#
# The module is **import-clean**: it never reaches into
# :mod:`prometheus_client` at top-level. The optional Prometheus runtime
# binding is constructed lazily by
# :func:`prometheus_metrics_sink_factory`; the in-process default sink
# is pure-Python and emits the same ``text/plain; version=0.0.4``
# exposition format that Prometheus understands. This keeps the
# hot-path metric updates blocking-free and the test suite fully
# offline-deterministic (INV-15).
#
# Spec line 1086 — six required metric series:
#   * ``executions_total{symbol,side,status}``                 — Counter
#   * ``pnl_usd{symbol}``                                       — Gauge
#   * ``execution_latency_ns``                                  — Histogram
#   * ``hazard_events_total{kind,severity}``                    — Counter
#   * ``governance_approvals_total{verdict}``                   — Counter
#   * ``health_state``                                          — Gauge
#
# Spec line 1087 — metric updates must be async (non-blocking). The
# pure-Python sink uses plain dict writes guarded by a single
# ``threading.Lock`` (microsecond scope); the Prometheus sink delegates
# to the official client which is itself non-blocking per its design.
#
# Spec line 1088 — label values are projected from
# :class:`core.contracts.events.ExecutionEvent` /
# :class:`core.contracts.events.HazardEvent` /
# :class:`core.contracts.governance.GovernanceDecision` fields, never
# hard-coded by the caller. Every label value passes through
# :func:`sanitize_label_value` before it reaches a metric.
#
# Authority symmetry (B27 / B28 / INV-71): this module lives in
# ``system_engine.metrics`` and therefore must **not** construct typed
# bus events. Pinned by an AST guard test.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
)
from core.contracts.governance import GovernanceDecision

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_EXECUTION_LATENCY_BUCKETS_NS",
    "EXECUTIONS_TOTAL",
    "EXECUTION_LATENCY_NS",
    "GOVERNANCE_APPROVALS_TOTAL",
    "HAZARD_EVENTS_TOTAL",
    "HEALTH_STATE",
    "InProcessMetricsSink",
    "MAX_LABEL_VALUE_LEN",
    "MetricsExporter",
    "MetricsExporterError",
    "MetricsSink",
    "MetricsSnapshotError",
    "NEW_PIP_DEPENDENCIES",
    "PNL_USD",
    "PROMETHEUS_ADAPTER_VERSION",
    "prometheus_metrics_sink_factory",
    "render_prometheus_text",
    "sanitize_label_value",
]


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("prometheus-client",)
PROMETHEUS_ADAPTER_VERSION: str = "1"

EXECUTIONS_TOTAL: str = "executions_total"
PNL_USD: str = "pnl_usd"
EXECUTION_LATENCY_NS: str = "execution_latency_ns"
HAZARD_EVENTS_TOTAL: str = "hazard_events_total"
GOVERNANCE_APPROVALS_TOTAL: str = "governance_approvals_total"
HEALTH_STATE: str = "health_state"

MAX_LABEL_VALUE_LEN: int = 128
MAX_METRIC_VALUE: float = 1e18
_LABEL_FALLBACK: str = "_"

# Logarithmic buckets covering 1 µs → 1 s in decade steps. ``+Inf`` is
# always implied per the Prometheus exposition format.
DEFAULT_EXECUTION_LATENCY_BUCKETS_NS: tuple[float, ...] = (
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MetricsExporterError(RuntimeError):
    """Base class for :mod:`system_engine.metrics.exporter` errors."""


class MetricsSnapshotError(MetricsExporterError):
    """Raised when rendering a metrics snapshot fails."""


# ---------------------------------------------------------------------------
# Label sanitisation
# ---------------------------------------------------------------------------


def sanitize_label_value(value: object) -> str:
    """Return a Prometheus-safe label value.

    The Prometheus exposition format requires label values to be valid
    UTF-8 with ``\\``, ``"`` and newline escaped. We additionally:

    * coerce non-string values via ``str``;
    * clamp to :data:`MAX_LABEL_VALUE_LEN` characters;
    * replace control characters with ``_``;
    * fall back to ``_`` for empty values (Prometheus treats the empty
      string as a valid label value but tools like Grafana render it
      poorly, so we substitute the placeholder).
    """

    if value is None:
        return _LABEL_FALLBACK
    text = value if isinstance(value, str) else str(value)
    if not text:
        return _LABEL_FALLBACK
    if len(text) > MAX_LABEL_VALUE_LEN:
        text = text[:MAX_LABEL_VALUE_LEN]
    cleaned_chars: list[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint < 0x20 or codepoint == 0x7F:
            cleaned_chars.append("_")
        elif char == "\\":
            cleaned_chars.append("\\\\")
        elif char == '"':
            cleaned_chars.append('\\"')
        else:
            cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    return cleaned if cleaned else _LABEL_FALLBACK


def _validate_metric_value(value: object, *, field_name: str) -> float:
    """Validate a numeric metric input and return it as ``float``."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value).__name__!r}")
    numeric = float(value)
    if numeric != numeric:  # NaN check
        raise ValueError(f"{field_name} must be finite, got NaN")
    if numeric in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if abs(numeric) > MAX_METRIC_VALUE:
        raise ValueError(f"{field_name} magnitude exceeds {MAX_METRIC_VALUE}: {numeric!r}")
    return numeric


# ---------------------------------------------------------------------------
# Sink Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MetricsSink(Protocol):
    """Lower-level write surface for the six required metric series."""

    def inc_executions(self, *, symbol: str, side: str, status: str) -> None: ...

    def set_pnl_usd(self, *, symbol: str, value: float) -> None: ...

    def observe_execution_latency_ns(self, value: float) -> None: ...

    def inc_hazard(self, *, kind: str, severity: str) -> None: ...

    def inc_governance(self, *, verdict: str) -> None: ...

    def set_health_state(self, value: float) -> None: ...

    def render(self) -> bytes: ...


# ---------------------------------------------------------------------------
# In-process pure-Python sink (test default; never depends on
# prometheus_client; emits the same text/plain exposition format).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _CounterState:
    """Mutable counter state keyed by sorted (label_name, label_value)."""

    samples: dict[tuple[tuple[str, str], ...], float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class _GaugeState:
    """Mutable gauge state keyed by sorted (label_name, label_value)."""

    samples: dict[tuple[tuple[str, str], ...], float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class _HistogramState:
    """Histogram state — bucket counts keyed in sorted order + sum / count."""

    buckets: tuple[float, ...]
    bucket_counts: list[int] = dataclasses.field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.bucket_counts:
            self.bucket_counts = [0 for _ in self.buckets]


class InProcessMetricsSink:
    """Pure-Python :class:`MetricsSink` used by tests and offline tooling."""

    __slots__ = (
        "_counters",
        "_gauges",
        "_histograms",
        "_lock",
        "_metric_help",
        "_metric_labels",
    )

    def __init__(
        self,
        *,
        execution_latency_buckets_ns: Sequence[float] | None = None,
    ) -> None:
        buckets = tuple(
            execution_latency_buckets_ns
            if execution_latency_buckets_ns is not None
            else DEFAULT_EXECUTION_LATENCY_BUCKETS_NS
        )
        if not buckets:
            raise ValueError("execution_latency_buckets_ns must be non-empty")
        if any(not isinstance(b, (int, float)) or isinstance(b, bool) for b in buckets):
            raise TypeError("execution_latency_buckets_ns must be numeric")
        if any(b <= 0 for b in buckets):
            raise ValueError("execution_latency_buckets_ns must be strictly positive")
        # buckets[1:] is intentionally one shorter than buckets.
        pairs = zip(buckets, buckets[1:])  # noqa: B905
        if any(b1 >= b2 for b1, b2 in pairs):
            raise ValueError("execution_latency_buckets_ns must be strictly increasing")

        self._lock = threading.Lock()
        self._counters: dict[str, _CounterState] = {
            EXECUTIONS_TOTAL: _CounterState(),
            HAZARD_EVENTS_TOTAL: _CounterState(),
            GOVERNANCE_APPROVALS_TOTAL: _CounterState(),
        }
        self._gauges: dict[str, _GaugeState] = {
            PNL_USD: _GaugeState(),
            HEALTH_STATE: _GaugeState(),
        }
        self._histograms: dict[str, _HistogramState] = {
            EXECUTION_LATENCY_NS: _HistogramState(buckets=buckets),
        }
        self._metric_labels: Mapping[str, tuple[str, ...]] = {
            EXECUTIONS_TOTAL: ("symbol", "side", "status"),
            PNL_USD: ("symbol",),
            EXECUTION_LATENCY_NS: (),
            HAZARD_EVENTS_TOTAL: ("kind", "severity"),
            GOVERNANCE_APPROVALS_TOTAL: ("verdict",),
            HEALTH_STATE: (),
        }
        self._metric_help: Mapping[str, str] = {
            EXECUTIONS_TOTAL: "Total executions by symbol/side/status.",
            PNL_USD: "Realised PnL in USD by symbol.",
            EXECUTION_LATENCY_NS: "Execution latency distribution (ns).",
            HAZARD_EVENTS_TOTAL: "Total hazard events by kind/severity.",
            GOVERNANCE_APPROVALS_TOTAL: ("Total governance approvals by verdict."),
            HEALTH_STATE: "Aggregated system health state in [0, 1].",
        }

    # ------------------------------------------------------------------
    # Write surface
    # ------------------------------------------------------------------

    def inc_executions(self, *, symbol: str, side: str, status: str) -> None:
        key = (
            ("side", sanitize_label_value(side)),
            ("status", sanitize_label_value(status)),
            ("symbol", sanitize_label_value(symbol)),
        )
        with self._lock:
            samples = self._counters[EXECUTIONS_TOTAL].samples
            samples[key] = samples.get(key, 0.0) + 1.0

    def set_pnl_usd(self, *, symbol: str, value: float) -> None:
        validated = _validate_metric_value(value, field_name="pnl_usd")
        key = (("symbol", sanitize_label_value(symbol)),)
        with self._lock:
            self._gauges[PNL_USD].samples[key] = validated

    def observe_execution_latency_ns(self, value: float) -> None:
        validated = _validate_metric_value(value, field_name="execution_latency_ns")
        if validated < 0:
            raise ValueError(f"execution_latency_ns must be non-negative, got {validated!r}")
        with self._lock:
            histogram = self._histograms[EXECUTION_LATENCY_NS]
            histogram.count += 1
            histogram.sum_value += validated
            for index, upper_bound in enumerate(histogram.buckets):
                if validated <= upper_bound:
                    histogram.bucket_counts[index] += 1

    def inc_hazard(self, *, kind: str, severity: str) -> None:
        key = (
            ("kind", sanitize_label_value(kind)),
            ("severity", sanitize_label_value(severity)),
        )
        with self._lock:
            samples = self._counters[HAZARD_EVENTS_TOTAL].samples
            samples[key] = samples.get(key, 0.0) + 1.0

    def inc_governance(self, *, verdict: str) -> None:
        key = (("verdict", sanitize_label_value(verdict)),)
        with self._lock:
            samples = self._counters[GOVERNANCE_APPROVALS_TOTAL].samples
            samples[key] = samples.get(key, 0.0) + 1.0

    def set_health_state(self, value: float) -> None:
        validated = _validate_metric_value(value, field_name="health_state")
        if not 0.0 <= validated <= 1.0:
            raise ValueError(f"health_state must be in [0, 1], got {validated!r}")
        with self._lock:
            self._gauges[HEALTH_STATE].samples[()] = validated

    # ------------------------------------------------------------------
    # Read / render surface
    # ------------------------------------------------------------------

    def snapshot(self) -> Mapping[str, Any]:
        """Return a deterministic snapshot of the current metric state.

        Used by tests and by the textual renderer. Keys are sorted to
        provide byte-identical output across runs.
        """

        with self._lock:
            counters_snapshot = {
                name: {key: value for key, value in sorted(state.samples.items())}
                for name, state in sorted(self._counters.items())
            }
            gauges_snapshot = {
                name: {key: value for key, value in sorted(state.samples.items())}
                for name, state in sorted(self._gauges.items())
            }
            histograms_snapshot = {
                name: {
                    "buckets": state.buckets,
                    "bucket_counts": tuple(state.bucket_counts),
                    "sum": state.sum_value,
                    "count": state.count,
                }
                for name, state in sorted(self._histograms.items())
            }
        return {
            "counters": counters_snapshot,
            "gauges": gauges_snapshot,
            "histograms": histograms_snapshot,
        }

    def render(self) -> bytes:
        try:
            return render_prometheus_text(self.snapshot(), help_text=self._metric_help)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise MetricsSnapshotError(f"snapshot render failed: {exc!s}") from exc


# ---------------------------------------------------------------------------
# Pure-Python text renderer matching Prometheus exposition format
# ---------------------------------------------------------------------------


_METRIC_TYPE: Mapping[str, str] = {
    EXECUTIONS_TOTAL: "counter",
    HAZARD_EVENTS_TOTAL: "counter",
    GOVERNANCE_APPROVALS_TOTAL: "counter",
    PNL_USD: "gauge",
    HEALTH_STATE: "gauge",
    EXECUTION_LATENCY_NS: "histogram",
}


def _format_value(value: float) -> str:
    """Format a numeric metric value per Prometheus conventions."""

    if value == int(value) and abs(value) < 1e16:
        return f"{int(value)}"
    return repr(float(value))


def _format_labels(pairs: Iterable[tuple[str, str]]) -> str:
    label_text = ",".join(f'{name}="{val}"' for name, val in pairs)
    return f"{{{label_text}}}" if label_text else ""


def render_prometheus_text(
    snapshot: Mapping[str, Any],
    *,
    help_text: Mapping[str, str] | None = None,
) -> bytes:
    """Render a metrics snapshot to ``text/plain; version=0.0.4`` bytes."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a Mapping")

    help_lookup: Mapping[str, str] = help_text or {}
    lines: list[str] = []

    counter_names: Sequence[str] = sorted(snapshot.get("counters", {}).keys())
    gauge_names: Sequence[str] = sorted(snapshot.get("gauges", {}).keys())
    histogram_names: Sequence[str] = sorted(snapshot.get("histograms", {}).keys())

    for name in counter_names:
        lines.extend(
            _render_simple_block(
                name,
                "counter",
                snapshot["counters"][name],
                help_lookup.get(name),
            )
        )

    for name in gauge_names:
        lines.extend(
            _render_simple_block(
                name,
                "gauge",
                snapshot["gauges"][name],
                help_lookup.get(name),
            )
        )

    for name in histogram_names:
        lines.extend(
            _render_histogram_block(
                name,
                snapshot["histograms"][name],
                help_lookup.get(name),
            )
        )

    body = "\n".join(lines)
    if body:
        body += "\n"
    return body.encode("utf-8")


def _render_simple_block(
    name: str,
    metric_type: str,
    samples: Mapping[tuple[tuple[str, str], ...], float],
    help_text: str | None,
) -> list[str]:
    lines: list[str] = []
    if help_text:
        escaped = help_text.replace("\\", "\\\\").replace("\n", "\\n")
        lines.append(f"# HELP {name} {escaped}")
    lines.append(f"# TYPE {name} {metric_type}")
    for label_pairs, value in samples.items():
        label_str = _format_labels(label_pairs)
        lines.append(f"{name}{label_str} {_format_value(value)}")
    return lines


def _render_histogram_block(
    name: str,
    state: Mapping[str, Any],
    help_text: str | None,
) -> list[str]:
    lines: list[str] = []
    if help_text:
        escaped = help_text.replace("\\", "\\\\").replace("\n", "\\n")
        lines.append(f"# HELP {name} {escaped}")
    lines.append(f"# TYPE {name} histogram")
    buckets: Sequence[float] = state["buckets"]
    bucket_counts: Sequence[int] = state["bucket_counts"]
    sum_value: float = state["sum"]
    count: int = state["count"]
    for upper_bound, bucket_count in zip(buckets, bucket_counts, strict=True):
        upper_text = repr(float(upper_bound))
        lines.append(f'{name}_bucket{{le="{upper_text}"}} {_format_value(float(bucket_count))}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {_format_value(float(count))}')
    lines.append(f"{name}_sum {_format_value(sum_value)}")
    lines.append(f"{name}_count {_format_value(float(count))}")
    return lines


# ---------------------------------------------------------------------------
# Optional Prometheus binding (lazy import; production wiring path)
# ---------------------------------------------------------------------------


def prometheus_metrics_sink_factory(
    *,
    registry: object | None = None,
    execution_latency_buckets_ns: Sequence[float] | None = None,
) -> MetricsSink:
    """Construct a :class:`MetricsSink` backed by ``prometheus_client``.

    Lazy-imports :mod:`prometheus_client` only inside this factory so
    the module remains import-clean even when ``prometheus-client`` is
    not installed. The returned sink satisfies :class:`MetricsSink` and
    delegates to the official client classes for thread-safe,
    non-blocking metric writes.
    """

    # Lazy import — confined to this function body so the module-level
    # AST guard tests (no top-level ``prometheus_client`` import) hold.
    import prometheus_client  # noqa: PLC0415

    buckets = tuple(
        execution_latency_buckets_ns
        if execution_latency_buckets_ns is not None
        else DEFAULT_EXECUTION_LATENCY_BUCKETS_NS
    )
    if not buckets:
        raise ValueError("execution_latency_buckets_ns must be non-empty")

    prom_registry = (
        registry
        if registry is not None
        else prometheus_client.CollectorRegistry(auto_describe=False)
    )

    executions_counter = prometheus_client.Counter(
        EXECUTIONS_TOTAL,
        "Total executions by symbol/side/status.",
        labelnames=("symbol", "side", "status"),
        registry=prom_registry,
    )
    pnl_gauge = prometheus_client.Gauge(
        PNL_USD,
        "Realised PnL in USD by symbol.",
        labelnames=("symbol",),
        registry=prom_registry,
    )
    latency_histogram = prometheus_client.Histogram(
        EXECUTION_LATENCY_NS,
        "Execution latency distribution (ns).",
        buckets=buckets,
        registry=prom_registry,
    )
    hazard_counter = prometheus_client.Counter(
        HAZARD_EVENTS_TOTAL,
        "Total hazard events by kind/severity.",
        labelnames=("kind", "severity"),
        registry=prom_registry,
    )
    governance_counter = prometheus_client.Counter(
        GOVERNANCE_APPROVALS_TOTAL,
        "Total governance approvals by verdict.",
        labelnames=("verdict",),
        registry=prom_registry,
    )
    health_gauge = prometheus_client.Gauge(
        HEALTH_STATE,
        "Aggregated system health state in [0, 1].",
        registry=prom_registry,
    )

    return _PrometheusMetricsSink(
        registry=prom_registry,
        executions_counter=executions_counter,
        pnl_gauge=pnl_gauge,
        latency_histogram=latency_histogram,
        hazard_counter=hazard_counter,
        governance_counter=governance_counter,
        health_gauge=health_gauge,
        generate_latest=prometheus_client.generate_latest,
    )


class _PrometheusMetricsSink:
    """Internal :class:`MetricsSink` impl backed by ``prometheus_client``."""

    __slots__ = (
        "_executions_counter",
        "_generate_latest",
        "_governance_counter",
        "_hazard_counter",
        "_health_gauge",
        "_latency_histogram",
        "_pnl_gauge",
        "_registry",
    )

    def __init__(
        self,
        *,
        registry: object,
        executions_counter: object,
        pnl_gauge: object,
        latency_histogram: object,
        hazard_counter: object,
        governance_counter: object,
        health_gauge: object,
        generate_latest: object,
    ) -> None:
        self._registry = registry
        self._executions_counter = executions_counter
        self._pnl_gauge = pnl_gauge
        self._latency_histogram = latency_histogram
        self._hazard_counter = hazard_counter
        self._governance_counter = governance_counter
        self._health_gauge = health_gauge
        self._generate_latest = generate_latest

    def inc_executions(self, *, symbol: str, side: str, status: str) -> None:
        self._executions_counter.labels(
            symbol=sanitize_label_value(symbol),
            side=sanitize_label_value(side),
            status=sanitize_label_value(status),
        ).inc()

    def set_pnl_usd(self, *, symbol: str, value: float) -> None:
        validated = _validate_metric_value(value, field_name="pnl_usd")
        self._pnl_gauge.labels(symbol=sanitize_label_value(symbol)).set(validated)

    def observe_execution_latency_ns(self, value: float) -> None:
        validated = _validate_metric_value(value, field_name="execution_latency_ns")
        if validated < 0:
            raise ValueError(f"execution_latency_ns must be non-negative, got {validated!r}")
        self._latency_histogram.observe(validated)

    def inc_hazard(self, *, kind: str, severity: str) -> None:
        self._hazard_counter.labels(
            kind=sanitize_label_value(kind),
            severity=sanitize_label_value(severity),
        ).inc()

    def inc_governance(self, *, verdict: str) -> None:
        self._governance_counter.labels(verdict=sanitize_label_value(verdict)).inc()

    def set_health_state(self, value: float) -> None:
        validated = _validate_metric_value(value, field_name="health_state")
        if not 0.0 <= validated <= 1.0:
            raise ValueError(f"health_state must be in [0, 1], got {validated!r}")
        self._health_gauge.set(validated)

    def render(self) -> bytes:
        try:
            return bytes(self._generate_latest(self._registry))
        except Exception as exc:  # pragma: no cover - defensive
            raise MetricsSnapshotError(
                f"prometheus_client.generate_latest failed: {exc!s}"
            ) from exc


# ---------------------------------------------------------------------------
# High-level coordinator (event → label projection)
# ---------------------------------------------------------------------------


class MetricsExporter:
    """Typed coordinator that projects events onto a :class:`MetricsSink`.

    All label values are derived from event fields (spec line 1088) and
    pass through :func:`sanitize_label_value`. The coordinator never
    constructs typed bus events itself (B27 / B28 / INV-71).
    """

    __slots__ = ("_sink",)

    def __init__(self, sink: MetricsSink) -> None:
        if not isinstance(sink, MetricsSink):
            raise TypeError(f"sink must implement MetricsSink, got {type(sink).__name__!r}")
        self._sink = sink

    @property
    def sink(self) -> MetricsSink:
        return self._sink

    # ------------------------------------------------------------------
    # Event projections
    # ------------------------------------------------------------------

    def record_execution(
        self,
        event: ExecutionEvent,
        *,
        latency_ns: float | int | None = None,
        pnl_usd: float | int | None = None,
    ) -> None:
        """Project an :class:`ExecutionEvent` onto the metric surface.

        ``latency_ns`` and ``pnl_usd`` are optional out-of-band signals
        because :class:`ExecutionEvent` itself does not carry them. The
        callsite (execution engine post-trade hook) is expected to pass
        them when available; this matches the canonical pattern of
        keeping the event contracts narrow and projecting derived
        observables into metrics rather than the bus.
        """

        if not isinstance(event, ExecutionEvent):
            raise TypeError(f"event must be ExecutionEvent, got {type(event).__name__!r}")

        side_text = event.side.value if isinstance(event.side, Side) else str(event.side)
        status_text = (
            event.status.value if isinstance(event.status, ExecutionStatus) else str(event.status)
        )

        self._sink.inc_executions(
            symbol=event.symbol,
            side=side_text,
            status=status_text,
        )

        if pnl_usd is not None:
            self._sink.set_pnl_usd(symbol=event.symbol, value=float(pnl_usd))

        if latency_ns is not None:
            self._sink.observe_execution_latency_ns(float(latency_ns))

    def record_hazard(self, event: HazardEvent) -> None:
        """Project a :class:`HazardEvent` onto the hazard counter."""

        if not isinstance(event, HazardEvent):
            raise TypeError(f"event must be HazardEvent, got {type(event).__name__!r}")
        severity_text = (
            event.severity.value
            if isinstance(event.severity, HazardSeverity)
            else str(event.severity)
        )
        self._sink.inc_hazard(kind=event.code, severity=severity_text)

    def record_governance_decision(self, decision: GovernanceDecision) -> None:
        """Project a :class:`GovernanceDecision` onto the governance counter.

        ``verdict`` derives from :attr:`GovernanceDecision.approved`:
        ``True`` ⇒ ``APPROVED``; ``False`` ⇒ ``REJECTED``.
        """

        if not isinstance(decision, GovernanceDecision):
            raise TypeError(f"decision must be GovernanceDecision, got {type(decision).__name__!r}")
        verdict = "APPROVED" if decision.approved else "REJECTED"
        self._sink.inc_governance(verdict=verdict)

    def set_health_state(self, value: float | int) -> None:
        """Set the aggregated ``health_state`` gauge (clamped to [0, 1])."""

        self._sink.set_health_state(float(value))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> bytes:
        """Return the ``text/plain; version=0.0.4`` exposition bytes."""

        return self._sink.render()
