"""AUDIT-WIRE.5 regression tests — RuntimeContext builder bound to the harness.

P0-4 closure. Production callers had no entry point that produced a
:class:`RuntimeContext`, so the meta-controller never saw non-trivial
perf / risk / drift / latency pressure scalars and INV-48 fallback
could not fire on real elapsed wall-time. After this PR the harness
exposes :func:`build_runtime_context` on ``state.runtime_context_builder``
and a :meth:`build_runtime_context_now` helper that wires the bound
authority surfaces (currently the seeded :class:`RiskSnapshot`) into
the builder.
"""

from __future__ import annotations

import pytest

from intelligence_engine.runtime_context import RuntimeContext
from intelligence_engine.runtime_context_builder import (
    DEFAULT_LATENCY_BUDGET_NS,
    RuntimeMonitorView,
    build_runtime_context,
)


@pytest.fixture()
def state():
    from ui.server import _State

    return _State()


def test_audit_wire_5_state_owns_runtime_context_builder(state):
    """The harness must hold the canonical builder callable so future
    callers (meta-controller hot path, dashboard widgets) have a single
    discoverable entry point."""

    assert state.runtime_context_builder is build_runtime_context
    assert state.runtime_latency_budget_ns == DEFAULT_LATENCY_BUDGET_NS


def test_audit_wire_5_build_runtime_context_now_returns_runtime_context(state):
    ctx = state.build_runtime_context_now()

    assert isinstance(ctx, RuntimeContext)
    # Default boot RiskSnapshot has halted=False so risk pressure is 0.
    assert ctx.risk == 0.0
    assert ctx.drift == 0.0
    assert ctx.latency == 0.0
    assert ctx.perf == 0.0
    assert ctx.vol_spike_z == 0.0
    assert ctx.elapsed_ns == 0


def test_audit_wire_5_build_runtime_context_now_uses_overrides(state):
    """Caller-supplied overrides must propagate through the helper so
    a future per-tick driver can feed real authority-surface scalars
    without bypassing the binding."""

    monitor = RuntimeMonitorView(
        fail_rate=0.25,
        reject_rate=0.10,
        p99_latency_ns=DEFAULT_LATENCY_BUDGET_NS // 2,
    )
    ctx = state.build_runtime_context_now(
        elapsed_ns=1_000_000,
        drift_deviation=0.4,
        vol_spike_z=2.5,
        runtime_monitor=monitor,
    )

    assert isinstance(ctx, RuntimeContext)
    assert ctx.elapsed_ns == 1_000_000
    assert pytest.approx(ctx.drift) == 0.4
    assert pytest.approx(ctx.latency) == 0.5
    # default perf derivation = clamp(fail_rate + reject_rate, 0, 1)
    assert pytest.approx(ctx.perf) == 0.35
    assert ctx.vol_spike_z == 2.5
