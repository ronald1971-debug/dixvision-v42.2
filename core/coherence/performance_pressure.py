"""Performance Pressure Vector — 5-D constraint summary (Phase 6.T1a).

Phase 6.T1a — second half. Companion to :mod:`core.coherence.belief_state`.

Design (v3.1 §H2, v3.2 §1.3 / INV-50):

* :class:`PressureVector` is a frozen dataclass; the 5-tuple
  ``(perf, risk, drift, latency, uncertainty)`` is derived from
  inputs only. The ``safety_modifier`` is a **continuous** ``[0, 1]``
  damping factor (v3.1 H2 / INV-31), not a binary kill switch.
* :func:`derive_pressure_vector` is a pure function with no clock,
  no PRNG, and no side effects.
* The ``uncertainty`` derivation incorporates **cross-signal
  entropy** (INV-50): given disagreement among contributing
  signals, the uncertainty rises even when individual signal
  confidences are high. A 5-BUY / 5-SELL window at high individual
  confidence maps to high uncertainty, not low.

Authority constraints:

* Only :mod:`core.contracts` is imported (plus :mod:`math` and
  :mod:`yaml` for config loading).
* No engine imports, no ledger writers, no clocks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.contracts.events import Side, SignalEvent, SystemEvent, SystemEventKind

PRESSURE_VECTOR_VERSION = "v3.3-T1a"

# Maximum entropy across the three sides {BUY, SELL, HOLD}. Used to
# normalise the cross-signal entropy term to ``[0, 1]``.
_MAX_ENTROPY_3 = math.log2(3.0)


# ---------------------------------------------------------------------------
# Config — registry/pressure.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PressureConfig:
    """Coefficients for :func:`derive_pressure_vector`.

    Loaded from ``registry/pressure.yaml`` (single source of truth).
    Frozen so callers cannot mutate the object after load.

    Fields:
        alpha: Weight on the *raw* per-signal uncertainty term in the
            composite ``uncertainty``.
        beta: Weight on the *cross-signal entropy* term (INV-50).
            Must satisfy ``alpha + beta <= 1`` so the result stays in
            ``[0, 1]`` without clamping in the typical path.
        entropy_high_water: Threshold above which the continuous
            ``safety_modifier`` starts compressing (v3.1 H2 / INV-31).
        entropy_high_water_modifier: Compression slope applied above
            the high water mark. ``safety_modifier`` is computed as
            ``1 - max(0, (uncertainty - entropy_high_water) *
            entropy_high_water_modifier)`` and clamped to ``[0, 1]``.
        version: Config schema version; recorded into every snapshot.
    """

    alpha: float
    beta: float
    entropy_high_water: float
    entropy_high_water_modifier: float
    version: str = "v3.3-T1a"

    def __post_init__(self) -> None:
        # Hard constraints — invalid configs fail fast at boot
        # (SAFE-43 envelope: pressure derivation must be well-defined).
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"PressureConfig.alpha out of [0,1]: {self.alpha}")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"PressureConfig.beta out of [0,1]: {self.beta}")
        if self.alpha + self.beta > 1.0 + 1e-9:
            raise ValueError(
                "PressureConfig requires alpha + beta <= 1 to keep "
                f"uncertainty in [0,1]: alpha={self.alpha}, beta={self.beta}"
            )
        if not (0.0 <= self.entropy_high_water <= 1.0):
            raise ValueError(
                "PressureConfig.entropy_high_water must be in [0,1]: "
                f"{self.entropy_high_water}"
            )
        if self.entropy_high_water_modifier < 0.0:
            raise ValueError(
                "PressureConfig.entropy_high_water_modifier must be >= 0: "
                f"{self.entropy_high_water_modifier}"
            )


def load_pressure_config(path: str | Path) -> PressureConfig:
    """Load :class:`PressureConfig` from a YAML file.

    The schema is::

        alpha: 0.5
        beta:  0.5
        entropy_high_water: 0.6
        entropy_high_water_modifier: 1.5
        version: v3.3-T1a
    """
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"pressure config at {path} is not a mapping")
    return PressureConfig(
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        entropy_high_water=float(raw["entropy_high_water"]),
        entropy_high_water_modifier=float(raw["entropy_high_water_modifier"]),
        version=str(raw.get("version", "v3.3-T1a")),
    )


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PressureVector:
    """5-D constraint summary projected from runtime telemetry.

    All five dimensions live in ``[0, 1]`` where 0 = no pressure and
    1 = maximum pressure.

    Fields:
        ts_ns: Snapshot timestamp.
        perf: Performance pressure (recent execution quality).
        risk: Risk pressure (drawdown / VaR / exposure).
        drift: Distributional drift / model staleness pressure.
        latency: End-to-end latency pressure on the hot path.
        uncertainty: Composite uncertainty including cross-signal
            entropy (INV-50).
        safety_modifier: Continuous damping factor in ``[0, 1]``
            (v3.1 H2 / INV-31). Governance retains hard override =
            forced 0 outside this projection.
        cross_signal_entropy: Normalised Shannon entropy of the
            contributing signal-side distribution (the entropy term
            already folded into ``uncertainty`` via ``beta``).
            Exposed for audit and for the calibration loop.
        signal_count: Window size that produced this snapshot.
        version: Derivation version (``PRESSURE_VECTOR_VERSION``).
    """

    ts_ns: int
    perf: float
    risk: float
    drift: float
    latency: float
    uncertainty: float
    safety_modifier: float
    cross_signal_entropy: float
    signal_count: int
    version: str = PRESSURE_VECTOR_VERSION

    def to_event(
        self,
        source: str = "core.coherence.performance_pressure",
    ) -> SystemEvent:
        """Project the snapshot into a ledgerable :class:`SystemEvent`.

        The ledger row is the only export path. Calibrator
        (``learning_engine.calibration.coherence_calibrator``,
        INV-53) consumes these against realised constraints.
        """
        payload: Mapping[str, str] = {
            "perf": f"{self.perf:.6f}",
            "risk": f"{self.risk:.6f}",
            "drift": f"{self.drift:.6f}",
            "latency": f"{self.latency:.6f}",
            "uncertainty": f"{self.uncertainty:.6f}",
            "safety_modifier": f"{self.safety_modifier:.6f}",
            "cross_signal_entropy": f"{self.cross_signal_entropy:.6f}",
            "signal_count": str(self.signal_count),
            "version": self.version,
        }
        return SystemEvent(
            ts_ns=self.ts_ns,
            sub_kind=SystemEventKind.PRESSURE_VECTOR_SNAPSHOT,
            source=source,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    """Clamp a float to ``[0, 1]``."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _cross_signal_entropy(signals: Sequence[SignalEvent]) -> float:
    """Normalised Shannon entropy of the side distribution (INV-50).

    Returns a value in ``[0, 1]`` where 0 = all signals on one side
    (perfect agreement) and 1 = uniform split across {BUY, SELL,
    HOLD} (maximum disagreement).

    Sub-cases:
    * Empty input → 0.0 (no disagreement to measure).
    * All signals on one side → 0.0.
    * Uniform 50/50 BUY/SELL → ``log2(2) / log2(3) ≈ 0.6309``.
    """
    n = len(signals)
    if n == 0:
        return 0.0

    counts: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0, Side.HOLD: 0}
    for s in signals:
        counts[s.side] = counts.get(s.side, 0) + 1

    total = float(n)
    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)

    # Normalise against the 3-side maximum so the result stays in [0,1]
    # regardless of how many sides are populated.
    return entropy / _MAX_ENTROPY_3


def _raw_uncertainty(signals: Sequence[SignalEvent]) -> float:
    """Average of ``(1 - confidence)`` across the input signals.

    Empty input maps to 0.0 — there is no per-signal uncertainty when
    there are no signals (the cross-signal entropy term is also 0 in
    that case, so the composite stays well-defined).
    """
    if not signals:
        return 0.0
    return sum(max(0.0, 1.0 - s.confidence) for s in signals) / len(signals)


def derive_pressure_vector(
    *,
    ts_ns: int,
    signals: Sequence[SignalEvent],
    perf: float,
    risk: float,
    drift: float,
    latency: float,
    config: PressureConfig,
) -> PressureVector:
    """Pure derivation of :class:`PressureVector`.

    The four scalar inputs (``perf``, ``risk``, ``drift``,
    ``latency``) are sourced by the caller (Phase 6.T1a wiring layer
    in ``intelligence_engine`` runtime) from the appropriate
    monitors. They are clamped to ``[0, 1]`` here so that misbehaving
    upstream sensors cannot push the projection out of domain.

    The ``uncertainty`` field is the **only** dimension this function
    actually composes:

        uncertainty = clamp01(alpha * raw_uncertainty
                              + beta * cross_signal_entropy)

    The ``safety_modifier`` (v3.1 H2 / INV-31) is a continuous damper
    monotone-decreasing in ``uncertainty``:

        safety_modifier = clamp01(
            1 - max(0, (uncertainty - entropy_high_water) *
                       entropy_high_water_modifier)
        )

    Both expressions are pure functions of inputs + config; replay
    determinism (INV-15) is preserved.
    """
    cs_entropy = _cross_signal_entropy(signals)
    raw_unc = _raw_uncertainty(signals)
    uncertainty = _clamp01(config.alpha * raw_unc + config.beta * cs_entropy)

    over = max(0.0, uncertainty - config.entropy_high_water)
    safety_modifier = _clamp01(1.0 - over * config.entropy_high_water_modifier)

    return PressureVector(
        ts_ns=ts_ns,
        perf=_clamp01(perf),
        risk=_clamp01(risk),
        drift=_clamp01(drift),
        latency=_clamp01(latency),
        uncertainty=uncertainty,
        safety_modifier=safety_modifier,
        cross_signal_entropy=cs_entropy,
        signal_count=len(signals),
    )


__all__ = [
    "PRESSURE_VECTOR_VERSION",
    "PressureConfig",
    "PressureVector",
    "derive_pressure_vector",
    "load_pressure_config",
]
