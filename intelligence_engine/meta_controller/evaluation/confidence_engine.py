"""Confidence Engine — composite confidence + side resolution.

Phase 6.T1b. Pure deterministic function over a sequence of
:class:`SignalEvent` records, producing both:

* ``proposed_side`` — the consensus direction the policy layer will
  consume (BUY / SELL / HOLD).
* ``proposed_confidence`` — composite ``[0, 1]`` score that the
  policy layer further dampens via ``pressure.safety_modifier``.

Composite formula (audit-friendly per J3):

    consensus  = |buy_count - sell_count| / total_signals
    strength   = mean(signal.confidence)  over consensus side
    coverage   = min(1, total_signals / saturation_count)

    composite  = (w_c · consensus + w_s · strength + w_v · coverage)
                  / (w_c + w_s + w_v)

The split into named components is intentional: when J3 reward
shaping lands in T1c, each component is ledgered separately so the
calibrator can attribute confidence drift to the right cause.

Authority constraints:

* Imports only :mod:`core.contracts` and the standard library
  (plus :mod:`yaml` for the registry loader).
* No clock, no PRNG; replay-deterministic per INV-15.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.contracts.events import Side, SignalEvent

CONFIDENCE_ENGINE_VERSION = "v3.3-T1b"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceComponents:
    """J3-aligned audit breakdown of a single confidence computation.

    ``composite`` is the value the policy layer consumes; the other
    fields exist so the offline calibrator can attribute drift.
    """

    consensus: float
    strength: float
    coverage: float
    composite: float
    signal_count: int
    version: str = CONFIDENCE_ENGINE_VERSION

    def __post_init__(self) -> None:
        for name in ("consensus", "strength", "coverage", "composite"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"ConfidenceComponents.{name} must be in [0, 1]: {v}"
                )
        if self.signal_count < 0:
            raise ValueError(
                f"ConfidenceComponents.signal_count must be >= 0: "
                f"{self.signal_count}"
            )


@dataclass(frozen=True, slots=True)
class ConfidenceEngineConfig:
    """Composite-formula coefficients (registry-controlled).

    Weights are normalised in :func:`compute_confidence`; the absolute
    values matter only as a ratio.
    """

    consensus_weight: float
    strength_weight: float
    coverage_weight: float
    saturation_count: int
    version: str = CONFIDENCE_ENGINE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "consensus_weight",
            "strength_weight",
            "coverage_weight",
        ):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(
                    f"ConfidenceEngineConfig.{name} must be >= 0: {v}"
                )
        total = (
            self.consensus_weight
            + self.strength_weight
            + self.coverage_weight
        )
        if total <= 0.0:
            raise ValueError(
                "ConfidenceEngineConfig: at least one weight must be > 0"
            )
        if self.saturation_count <= 0:
            raise ValueError(
                f"ConfidenceEngineConfig.saturation_count must be > 0: "
                f"{self.saturation_count}"
            )


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def load_confidence_engine_config(path: str | Path) -> ConfidenceEngineConfig:
    """Load the confidence-engine config from a YAML file.

    Required keys (all top-level): ``consensus_weight``,
    ``strength_weight``, ``coverage_weight``, ``saturation_count``.
    Optional: ``version``. Fail-fast on any missing or extra key.
    """
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"confidence engine config must be a YAML mapping, got "
            f"{type(raw).__name__}"
        )
    required = {
        "consensus_weight",
        "strength_weight",
        "coverage_weight",
        "saturation_count",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"confidence engine config missing keys: {sorted(missing)}"
        )
    extra = raw.keys() - (required | {"version"})
    if extra:
        raise ValueError(
            f"confidence engine config has unknown keys: {sorted(extra)}"
        )
    kwargs: dict[str, Any] = {
        "consensus_weight": float(raw["consensus_weight"]),
        "strength_weight": float(raw["strength_weight"]),
        "coverage_weight": float(raw["coverage_weight"]),
        "saturation_count": int(raw["saturation_count"]),
    }
    if "version" in raw:
        kwargs["version"] = str(raw["version"])
    return ConfidenceEngineConfig(**kwargs)


# ---------------------------------------------------------------------------
# Side resolution
# ---------------------------------------------------------------------------


def resolve_proposed_side(signals: Sequence[SignalEvent]) -> Side:
    """Pick the consensus direction.

    Tie-breaking rules:

    * No signals → HOLD.
    * BUY count == SELL count → HOLD (no consensus).
    * Strict majority of BUY → BUY; strict majority of SELL → SELL.
    * HOLD signals abstain (do not contribute either way).
    """
    if not signals:
        return Side.HOLD
    buy = sum(1 for s in signals if s.side is Side.BUY)
    sell = sum(1 for s in signals if s.side is Side.SELL)
    if buy > sell:
        return Side.BUY
    if sell > buy:
        return Side.SELL
    return Side.HOLD


# ---------------------------------------------------------------------------
# Composite confidence
# ---------------------------------------------------------------------------


_ZERO_COMPONENTS: ConfidenceComponents = ConfidenceComponents(
    consensus=0.0,
    strength=0.0,
    coverage=0.0,
    composite=0.0,
    signal_count=0,
)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def compute_confidence(
    signals: Sequence[SignalEvent],
    config: ConfidenceEngineConfig,
) -> ConfidenceComponents:
    """Compute composite confidence + per-component breakdown.

    Returns :data:`_ZERO_COMPONENTS` when ``signals`` is empty (no
    information → no confidence), and when there is no strict
    majority direction (tie or all-HOLD).
    """
    total = len(signals)
    if total == 0:
        return _ZERO_COMPONENTS

    buy = sum(1 for s in signals if s.side is Side.BUY)
    sell = sum(1 for s in signals if s.side is Side.SELL)

    if buy == sell:
        # No consensus; coverage and strength are not meaningful.
        return ConfidenceComponents(
            consensus=0.0,
            strength=0.0,
            coverage=0.0,
            composite=0.0,
            signal_count=total,
        )

    consensus_side = Side.BUY if buy > sell else Side.SELL
    consensus = abs(buy - sell) / total

    relevant = [s for s in signals if s.side is consensus_side]
    # consensus_side has > 0 entries because buy != sell here.
    strength = _clamp01(
        sum(_clamp01(s.confidence) for s in relevant) / len(relevant)
    )

    coverage = _clamp01(total / config.saturation_count)

    weight_sum = (
        config.consensus_weight
        + config.strength_weight
        + config.coverage_weight
    )
    composite = _clamp01(
        (
            config.consensus_weight * consensus
            + config.strength_weight * strength
            + config.coverage_weight * coverage
        )
        / weight_sum
    )
    return ConfidenceComponents(
        consensus=consensus,
        strength=strength,
        coverage=coverage,
        composite=composite,
        signal_count=total,
    )


__all__ = [
    "CONFIDENCE_ENGINE_VERSION",
    "ConfidenceComponents",
    "ConfidenceEngineConfig",
    "compute_confidence",
    "load_confidence_engine_config",
    "resolve_proposed_side",
]
