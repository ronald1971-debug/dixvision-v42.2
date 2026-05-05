"""Macro regime contracts — system-wide market context classification.

The macro regime layer answers a different question than the per-asset
classifier in :mod:`intelligence_engine.strategy_runtime.regime_detector`
(which produces TRENDING/RANGING/VOLATILE on a single asset's tick window)
and the hysteresis transition gate in
:mod:`intelligence_engine.meta_controller.perception.regime_router`
(which decides when to *commit* to a coherence regime). This layer asks:

    "What is the macro environment right now across the whole portfolio?"

Output is a :class:`MacroRegime` (RISK_ON / RISK_OFF / NEUTRAL / CRISIS)
that downstream consumers (trader-archetype regime_performance dict,
allocator caps, hazard escalation policy, cognitive-chat context) read
to gate their behaviour.

The input :class:`MacroSnapshot` is a normalised aggregate of cross-asset
signals + macro indicators. The engine itself is a pure function on
``(snapshot, config) → reading`` (INV-15).

Refs:
- manifest_v3.3_delta.md §1.1 (J1 macro regime)
- full_feature_spec.md §"Macro regime engine"
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum


class MacroRegime(StrEnum):
    """Coarse macro environment label.

    UNKNOWN is the boot/uninitialised value; engines must not act on
    UNKNOWN as if it were NEUTRAL — it means "no classification yet".
    """

    UNKNOWN = "UNKNOWN"
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"


@dataclasses.dataclass(frozen=True, slots=True)
class MacroSnapshot:
    """Read-only normalised aggregate input to the macro regime engine.

    All values are expected to be normalised by the calling adapter so
    the engine is pure and the same snapshot always classifies the same
    way (INV-15). Adapters are out of scope for this contract; the
    snapshot is whatever the caller can compute from BeliefState +
    MacroObservation history + cross-asset return matrix.

    Attributes:
        ts_ns: Monotonic ingestion timestamp (TimeAuthority). Strictly
            positive.
        vol_index: Implied / realised vol proxy in the 0..100 range
            (e.g. VIX-like). NaN / negative is rejected.
        breadth: Advance-decline ratio in [-1, 1] where +1 = uniformly
            up, -1 = uniformly down.
        credit_spread_bps: HY-vs-treasury spread in basis points;
            non-negative.
        dollar_strength: Normalised z-score of a broad-dollar index in
            [-3, 3].
        return_correlation: Realised cross-asset return correlation in
            [0, 1] (1.0 = total contagion).
        meta: Free-form structural metadata (no PII, no secrets) so
            tracing / replay can carry adapter context without coupling
            this contract to a specific adapter shape.
    """

    ts_ns: int
    vol_index: float
    breadth: float
    credit_spread_bps: float
    dollar_strength: float
    return_correlation: float
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("MacroSnapshot.ts_ns must be positive")
        if not (0.0 <= self.vol_index <= 100.0):
            raise ValueError(
                f"MacroSnapshot.vol_index must be in [0, 100], "
                f"got {self.vol_index!r}"
            )
        if not (-1.0 <= self.breadth <= 1.0):
            raise ValueError(
                f"MacroSnapshot.breadth must be in [-1, 1], got {self.breadth!r}"
            )
        if self.credit_spread_bps < 0:
            raise ValueError(
                "MacroSnapshot.credit_spread_bps must be non-negative, "
                f"got {self.credit_spread_bps!r}"
            )
        if not (-3.0 <= self.dollar_strength <= 3.0):
            raise ValueError(
                "MacroSnapshot.dollar_strength must be in [-3, 3], "
                f"got {self.dollar_strength!r}"
            )
        if not (0.0 <= self.return_correlation <= 1.0):
            raise ValueError(
                "MacroSnapshot.return_correlation must be in [0, 1], "
                f"got {self.return_correlation!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MacroRegimeReading:
    """Immutable reading produced by the macro regime engine.

    Attributes:
        regime: The classified macro regime.
        confidence: Engine confidence in [0, 1].
        rule_fired: Stable name of the rule branch that fired
            (e.g. ``"crisis_vol"``); used for audit / why-trace.
        snapshot_ts_ns: ``ts_ns`` of the originating
            :class:`MacroSnapshot`. Carried so the reading is
            self-describing in the audit ledger.
    """

    regime: MacroRegime
    confidence: float
    rule_fired: str
    snapshot_ts_ns: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"MacroRegimeReading.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )
        if not self.rule_fired:
            raise ValueError("MacroRegimeReading.rule_fired must be non-empty")
        if self.snapshot_ts_ns <= 0:
            raise ValueError(
                "MacroRegimeReading.snapshot_ts_ns must be positive"
            )


__all__ = [
    "MacroRegime",
    "MacroSnapshot",
    "MacroRegimeReading",
]
