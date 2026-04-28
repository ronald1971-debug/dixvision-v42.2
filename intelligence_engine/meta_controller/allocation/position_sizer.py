"""Position Sizer — composite sizing with J3-aligned audit breakdown.

Phase 6.T1b. Pure deterministic function over
``(confidence, regime, pressure, config)`` producing the
``proposed_size`` value the policy layer consumes.

Composite formula (audit-friendly per J3):

    confidence_factor = clamp01(confidence)              if confidence ≥ floor
                       = 0                                otherwise
    regime_factor     = config.multiplier_for(regime)    in ``[0, ∞)``
    risk_factor       = clamp01(1 - risk_damping · pressure.risk)
    pre_cap_size      = clamp01(base_fraction
                                · confidence_factor
                                · regime_factor
                                · risk_factor)
    final_size        = min(pre_cap_size, kelly_cap)

Each component is exposed in :class:`SizingComponents` so the J3
calibrator (T1c) can attribute size drift to the right cause.

Authority constraints:

* Imports only :mod:`core.contracts`,
  :mod:`core.coherence`, and the standard library
  (plus :mod:`yaml` for the registry loader).
* No clock, no PRNG; replay-deterministic per INV-15.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import PressureVector

POSITION_SIZER_VERSION = "v3.3-T1b"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SizingComponents:
    """Audit breakdown of a single position-size computation.

    ``final_size`` is the value the policy layer consumes; the other
    fields exist so the offline calibrator can attribute drift.
    """

    confidence_factor: float
    regime_factor: float
    risk_factor: float
    pre_cap_size: float
    final_size: float
    rationale: str
    version: str = POSITION_SIZER_VERSION

    def __post_init__(self) -> None:
        for name in ("confidence_factor", "risk_factor", "pre_cap_size", "final_size"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"SizingComponents.{name} must be in [0, 1]: {v}"
                )
        if self.regime_factor < 0.0:
            raise ValueError(
                f"SizingComponents.regime_factor must be >= 0: "
                f"{self.regime_factor}"
            )


@dataclass(frozen=True, slots=True)
class PositionSizerConfig:
    """Sizing-formula coefficients (registry-controlled)."""

    base_fraction: float
    kelly_cap: float
    trend_multiplier: float
    range_multiplier: float
    vol_spike_multiplier: float
    confidence_floor: float
    risk_damping: float
    version: str = POSITION_SIZER_VERSION

    def __post_init__(self) -> None:
        for name in ("base_fraction", "kelly_cap"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"PositionSizerConfig.{name} must be in [0, 1]: {v}"
                )
        for name in (
            "trend_multiplier",
            "range_multiplier",
            "vol_spike_multiplier",
        ):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(
                    f"PositionSizerConfig.{name} must be >= 0: {v}"
                )
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise ValueError(
                f"PositionSizerConfig.confidence_floor must be in [0, 1]: "
                f"{self.confidence_floor}"
            )
        if not (0.0 <= self.risk_damping <= 1.0):
            raise ValueError(
                f"PositionSizerConfig.risk_damping must be in [0, 1]: "
                f"{self.risk_damping}"
            )

    def multiplier_for(self, regime: Regime) -> float:
        """Resolve the regime → multiplier mapping.

        UNKNOWN regimes always size to zero — without a thesis, no
        position. VOL_SPIKE is configurable but defaults to zero.
        """
        if regime is Regime.TREND_UP or regime is Regime.TREND_DOWN:
            return self.trend_multiplier
        if regime is Regime.RANGE:
            return self.range_multiplier
        if regime is Regime.VOL_SPIKE:
            return self.vol_spike_multiplier
        return 0.0  # UNKNOWN


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def load_position_sizer_config(path: str | Path) -> PositionSizerConfig:
    """Load the position-sizer config from a YAML file.

    Required keys (top-level): ``base_fraction``, ``kelly_cap``,
    ``trend_multiplier``, ``range_multiplier``,
    ``vol_spike_multiplier``, ``confidence_floor``, ``risk_damping``.
    Optional: ``version``. Fail-fast on any missing or extra key.
    """
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"position sizer config must be a YAML mapping, got "
            f"{type(raw).__name__}"
        )
    required = {
        "base_fraction",
        "kelly_cap",
        "trend_multiplier",
        "range_multiplier",
        "vol_spike_multiplier",
        "confidence_floor",
        "risk_damping",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"position sizer config missing keys: {sorted(missing)}"
        )
    extra = raw.keys() - (required | {"version"})
    if extra:
        raise ValueError(
            f"position sizer config has unknown keys: {sorted(extra)}"
        )
    kwargs: dict[str, Any] = {
        "base_fraction": float(raw["base_fraction"]),
        "kelly_cap": float(raw["kelly_cap"]),
        "trend_multiplier": float(raw["trend_multiplier"]),
        "range_multiplier": float(raw["range_multiplier"]),
        "vol_spike_multiplier": float(raw["vol_spike_multiplier"]),
        "confidence_floor": float(raw["confidence_floor"]),
        "risk_damping": float(raw["risk_damping"]),
    }
    if "version" in raw:
        kwargs["version"] = str(raw["version"])
    return PositionSizerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Composite sizing
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


_RATIONALE_PRIMARY = "primary"
_RATIONALE_BELOW_FLOOR = "confidence_below_floor"
_RATIONALE_REGIME_ZERO = "regime_zero_multiplier"
_RATIONALE_KELLY_CAPPED = "kelly_capped"


def compute_position_size(
    *,
    confidence: float,
    regime: Regime,
    pressure: PressureVector,
    config: PositionSizerConfig,
) -> SizingComponents:
    """Compute composite position size + per-component breakdown.

    Order of evaluation (each step is constant-time):

    1. **Confidence floor** — confidence below ``confidence_floor``
       sizes to zero with rationale ``"confidence_below_floor"``.
    2. **Regime zero-multiplier guard** — UNKNOWN regime (or any
       configured-zero regime) sizes to zero with rationale
       ``"regime_zero_multiplier"``.
    3. **Composite path** — multiply confidence_factor · regime_factor
       · risk_factor · base_fraction, clamp to ``[0, 1]``.
    4. **Kelly cap** — cap at ``kelly_cap``; rationale becomes
       ``"kelly_capped"`` when the cap actually fires (``pre_cap >
       kelly_cap``).
    """
    confidence_clamped = _clamp01(confidence)
    if confidence_clamped < config.confidence_floor:
        return SizingComponents(
            confidence_factor=0.0,
            regime_factor=config.multiplier_for(regime),
            risk_factor=_clamp01(
                1.0 - config.risk_damping * _clamp01(pressure.risk)
            ),
            pre_cap_size=0.0,
            final_size=0.0,
            rationale=_RATIONALE_BELOW_FLOOR,
        )

    regime_factor = config.multiplier_for(regime)
    risk_factor = _clamp01(
        1.0 - config.risk_damping * _clamp01(pressure.risk)
    )

    if regime_factor == 0.0:
        return SizingComponents(
            confidence_factor=confidence_clamped,
            regime_factor=0.0,
            risk_factor=risk_factor,
            pre_cap_size=0.0,
            final_size=0.0,
            rationale=_RATIONALE_REGIME_ZERO,
        )

    pre_cap = _clamp01(
        config.base_fraction * confidence_clamped * regime_factor * risk_factor
    )
    if pre_cap > config.kelly_cap:
        return SizingComponents(
            confidence_factor=confidence_clamped,
            regime_factor=regime_factor,
            risk_factor=risk_factor,
            pre_cap_size=pre_cap,
            final_size=config.kelly_cap,
            rationale=_RATIONALE_KELLY_CAPPED,
        )

    return SizingComponents(
        confidence_factor=confidence_clamped,
        regime_factor=regime_factor,
        risk_factor=risk_factor,
        pre_cap_size=pre_cap,
        final_size=pre_cap,
        rationale=_RATIONALE_PRIMARY,
    )


__all__ = [
    "POSITION_SIZER_VERSION",
    "PositionSizerConfig",
    "SizingComponents",
    "compute_position_size",
    "load_position_sizer_config",
]
