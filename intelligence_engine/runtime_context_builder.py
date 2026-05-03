"""Pure builder for the per-tick :class:`RuntimeContext` (P0-4).

Phase 6 wired :class:`MetaControllerHotPath` into ``IntelligenceEngine.run_meta_tick``
but the four runtime scalars (``perf`` / ``risk`` / ``drift`` / ``latency``)
plus ``vol_spike_z`` and ``elapsed_ns`` were caller-supplied raw floats.
In production no caller produced them, so the meta-controller never saw a
non-trivial :class:`RuntimeContext` and INV-48 fallback could not fire on
real elapsed wall-time.

This module closes that gap with a single pure function:

* ``build_runtime_context`` — converts the read-only authority surfaces
  (:class:`RiskSnapshot`, :class:`RuntimeMonitorReport`,
  :class:`DriftReading`) plus a caller-supplied performance scalar and
  ``vol_spike_z`` / timing pair into a fully populated
  :class:`RuntimeContext`.

The builder is deterministic, IO-free, clock-free and PRNG-free
(INV-15 preserved). All four pressure scalars are clamped to ``[0, 1]``
before being handed to :func:`core.coherence.performance_pressure.derive_pressure_vector`.

Authority constraints:

* B1 — depends only on :mod:`core.contracts` and the surrounding
  intelligence-engine package; no cross-engine direct imports.
* The two adapter inputs (``runtime_monitor_report`` and
  ``risk_snapshot``) are both either re-exported in
  :mod:`core.contracts` (RiskSnapshot) or are pure value-types defined
  in their owning engine; we depend on them only via duck-typed
  attribute reads where unavoidable, keeping the builder a pure
  data-flow seam.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.risk import RiskSnapshot
from intelligence_engine.runtime_context import RuntimeContext

__all__ = [
    "DEFAULT_LATENCY_BUDGET_NS",
    "RuntimeMonitorView",
    "build_runtime_context",
]

#: Default latency budget for ``latency`` normalisation. p99 latency
#: above this maps to a saturated ``latency=1.0`` pressure scalar.
#: 100 ms is the manifest-v3.5 hot-path budget (INV-48 fallback band).
DEFAULT_LATENCY_BUDGET_NS: int = 100_000_000


@dataclass(frozen=True, slots=True)
class RuntimeMonitorView:
    """Frozen view of the scalars the builder needs from the runtime monitor.

    ``execution_engine.protections.runtime_monitor.RuntimeMonitorReport``
    satisfies this protocol structurally; we accept either the report
    or any equivalent value-type. Keeping the builder's input as a
    dedicated value-type avoids importing :mod:`execution_engine` from
    :mod:`intelligence_engine` (B1).
    """

    fail_rate: float
    reject_rate: float
    p99_latency_ns: int


def _clamp01(x: float) -> float:
    if x != x:  # NaN guard — pure, no math import needed.
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_runtime_context(
    *,
    risk_snapshot: RiskSnapshot,
    runtime_monitor: RuntimeMonitorView,
    drift_deviation: float = 0.0,
    perf_pressure: float | None = None,
    vol_spike_z: float = 0.0,
    elapsed_ns: int = 0,
    latency_budget_ns: int = DEFAULT_LATENCY_BUDGET_NS,
) -> RuntimeContext:
    """Project read-only authority surfaces into a per-tick :class:`RuntimeContext`.

    Args:
        risk_snapshot: Frozen FastRiskCache snapshot for this tick.
            ``halted=True`` saturates ``risk`` to ``1.0``.
        runtime_monitor: View over the latest ``RuntimeMonitorReport``.
            ``p99_latency_ns`` is normalised against
            ``latency_budget_ns``; ``fail_rate`` + ``reject_rate``
            feed the default ``perf`` derivation when ``perf_pressure``
            is not supplied.
        drift_deviation: Latest deviation from
            :class:`system_engine.state.drift_monitor.DriftReading`.
            Already in ``[0, ∞)`` — clamped to ``[0, 1]``.
        perf_pressure: Optional explicit performance pressure (e.g.
            rolling-drawdown ratio); when ``None`` the builder
            derives ``perf = clamp(fail_rate + reject_rate, 0, 1)``.
        vol_spike_z: Volatility z-score from the signal pipeline.
            Passed through unchanged (sign carries information).
        elapsed_ns: ``TimeAuthority.now_ns() - tick_start_ns`` for the
            current tick; drives INV-48 fallback inside the
            meta-controller.
        latency_budget_ns: Latency-pressure normaliser. Defaults to
            :data:`DEFAULT_LATENCY_BUDGET_NS`.

    Returns:
        A fully populated :class:`RuntimeContext`.

    Raises:
        ValueError: ``elapsed_ns`` or ``latency_budget_ns`` are
            negative / non-positive respectively.
    """
    if elapsed_ns < 0:
        raise ValueError(f"elapsed_ns must be >= 0, got {elapsed_ns}")
    if latency_budget_ns <= 0:
        raise ValueError(
            f"latency_budget_ns must be > 0, got {latency_budget_ns}"
        )

    if risk_snapshot.halted:
        risk = 1.0
    else:
        # When unhalted there is no scalar risk pressure encoded on the
        # snapshot itself; callers that want a richer signal should
        # subclass / extend this builder rather than smuggle logic in
        # here. Default = no pressure.
        risk = 0.0

    if perf_pressure is None:
        perf = _clamp01(
            float(runtime_monitor.fail_rate)
            + float(runtime_monitor.reject_rate)
        )
    else:
        perf = _clamp01(float(perf_pressure))

    drift = _clamp01(float(drift_deviation))

    p99_ns = max(0, int(runtime_monitor.p99_latency_ns))
    latency = _clamp01(p99_ns / float(latency_budget_ns))

    return RuntimeContext(
        perf=perf,
        risk=risk,
        drift=drift,
        latency=latency,
        vol_spike_z=float(vol_spike_z),
        elapsed_ns=int(elapsed_ns),
    )
