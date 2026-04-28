"""Per-tick runtime context for the intelligence engine — Wave 1.

Frozen value-type carrying the runtime monitor scalars the
:class:`MetaControllerHotPath` consumes per tick. Sourced by the
caller from the appropriate read-only authority surfaces:

* ``perf`` — performance pressure (e.g. rolling drawdown).
* ``risk`` — risk pressure from the FastRiskCache / RuntimeMonitor.
* ``drift`` — drift pressure from the DriftOracle.
* ``latency`` — latency pressure from the runtime monitor.
* ``vol_spike_z`` — volatility z-score from the signal pipeline.
* ``elapsed_ns`` — ``TimeAuthority.now_ns() - tick_start_ns`` for the
  current tick. Drives INV-48 fallback.

All fields are pure inputs to deterministic projections; no clocks
or PRNGs are read inside the engine itself (INV-15 preserved).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Frozen per-tick scalars consumed by the meta-controller harness."""

    perf: float
    risk: float
    drift: float
    latency: float
    vol_spike_z: float
    elapsed_ns: int


__all__ = ["RuntimeContext"]
