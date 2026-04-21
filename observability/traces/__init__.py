"""observability.traces — Lightweight structured tracing."""
from .trace_manager import Span, TraceManager, get_trace_manager

__all__ = ["TraceManager", "get_trace_manager", "Span"]
