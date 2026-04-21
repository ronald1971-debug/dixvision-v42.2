"""
core/runtime/execution_context.py
Thread-local execution context. Carries trace_id + domain for
observability propagation.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecutionContext:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    domain: str = "SYSTEM"  # "MARKET" for Indira, "SYSTEM" for Dyon, "GOV" for governance
    component: str = "unknown"


_local = threading.local()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_context(ctx: ExecutionContext) -> None:
    _local.ctx = ctx


def get_context() -> ExecutionContext:
    ctx: ExecutionContext | None = getattr(_local, "ctx", None)
    if ctx is None:
        ctx = ExecutionContext()
        _local.ctx = ctx
    return ctx
