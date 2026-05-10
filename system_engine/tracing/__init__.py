"""A-09 — system_engine.tracing package.

Public surface lives in :mod:`system_engine.tracing.tracer`.
"""

from system_engine.tracing.tracer import (
    DEFAULT_SAMPLE_RATIO,
    MAX_ATTRIBUTE_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    OTEL_ADAPTER_VERSION,
    InProcessTracer,
    SpanRecord,
    Tracer,
    TracerError,
    TraceSnapshot,
    otel_tracer_factory,
    sanitize_attribute_value,
)

__all__ = [
    "DEFAULT_SAMPLE_RATIO",
    "InProcessTracer",
    "MAX_ATTRIBUTE_VALUE_LEN",
    "NEW_PIP_DEPENDENCIES",
    "OTEL_ADAPTER_VERSION",
    "SpanRecord",
    "TraceSnapshot",
    "Tracer",
    "TracerError",
    "otel_tracer_factory",
    "sanitize_attribute_value",
]
