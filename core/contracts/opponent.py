"""Opponent-model contracts — typed inputs/outputs for OPP-XX modules.

The opponent layer answers a different question than the per-asset
regime detector or the macro regime engine. It asks:

    "Given the current microstructure, who is the *other side* of the
    book right now and what are they likely to do next?"

Output is an :class:`OpponentClassification` (HFT_MAKER /
MOMENTUM_TAKER / SWEEPER / SLOW_RESTING_LIQUIDITY / NOISE) plus a
:class:`BehaviorPrediction` (CONTINUE_AGGRESSION / FADE / WITHDRAW /
HOLD) that downstream consumers (execution scheduler, slippage
estimator, hazard layer) read to gate their behaviour.

The input :class:`OpponentObservation` is a normalised aggregate of
order-flow / L2 features. The predictor itself is a pure function on
``(observation, config) → (classification, prediction)`` (INV-15) —
same observation, same config → byte-identical output.

Refs:
- manifest_v3.1_delta.md §"Opponent model"
- full_feature_spec.md §"Behavior predictor"
- build_plan.md Phase 10.10 OPPONENT MODEL
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from enum import StrEnum


class OpponentArchetype(StrEnum):
    """Coarse opponent label — *who* is dominating the other side.

    UNKNOWN is the boot/uninitialised value; consumers must not treat
    UNKNOWN as ``NOISE`` — it means "no classification yet".
    """

    UNKNOWN = "UNKNOWN"
    HFT_MAKER = "HFT_MAKER"
    MOMENTUM_TAKER = "MOMENTUM_TAKER"
    SWEEPER = "SWEEPER"
    SLOW_RESTING_LIQUIDITY = "SLOW_RESTING_LIQUIDITY"
    NOISE = "NOISE"


class PredictedAction(StrEnum):
    """Coarse predicted next action of the dominant opponent class."""

    HOLD = "HOLD"
    CONTINUE_AGGRESSION = "CONTINUE_AGGRESSION"
    FADE = "FADE"
    WITHDRAW = "WITHDRAW"


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


@dataclasses.dataclass(frozen=True, slots=True)
class OpponentObservation:
    """Read-only normalised microstructure aggregate.

    All values are expected to be normalised by the calling adapter so
    the predictor is pure and the same observation always classifies
    the same way (INV-15). Adapters are out of scope for this contract.

    Attributes:
        ts_ns: Monotonic ingestion timestamp (TimeAuthority). Strictly
            positive.
        symbol: Asset symbol the observation is about. Non-empty.
        aggressor_imbalance: Buy-vs-sell aggressor flow in [-1, 1]
            where +1 = all buy aggression, -1 = all sell aggression.
        avg_taker_size_usd: Mean aggressor trade size in USD,
            non-negative.
        avg_resting_size_usd: Mean top-of-book resting size in USD on
            the dominant side, non-negative.
        cancel_to_fill_ratio: Cancellations per fill on the dominant
            side, non-negative. HFT makers run very high here.
        top_of_book_refresh_rate_hz: Best-bid/ask refresh frequency in
            Hz, non-negative. HFT makers run very high here too.
        spread_bps: Current top-of-book spread in basis points,
            non-negative.
        meta: Free-form structural metadata (no PII, no secrets) so
            tracing / replay can carry adapter context without coupling
            this contract to a specific adapter shape.
    """

    ts_ns: int
    symbol: str
    aggressor_imbalance: float
    avg_taker_size_usd: float
    avg_resting_size_usd: float
    cancel_to_fill_ratio: float
    top_of_book_refresh_rate_hz: float
    spread_bps: float
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError(
                f"OpponentObservation.ts_ns must be positive, got {self.ts_ns!r}"
            )
        if not self.symbol:
            raise ValueError("OpponentObservation.symbol must be non-empty")
        # Phrased as ``not (lo <= x <= hi)`` so NaN — which compares False
        # against every numeric — is rejected here instead of silently
        # passing through into rule comparisons below.
        if not (-1.0 <= self.aggressor_imbalance <= 1.0):
            raise ValueError(
                "OpponentObservation.aggressor_imbalance must be in [-1, 1], "
                f"got {self.aggressor_imbalance!r}"
            )
        for name, value in (
            ("avg_taker_size_usd", self.avg_taker_size_usd),
            ("avg_resting_size_usd", self.avg_resting_size_usd),
            ("cancel_to_fill_ratio", self.cancel_to_fill_ratio),
            ("top_of_book_refresh_rate_hz", self.top_of_book_refresh_rate_hz),
            ("spread_bps", self.spread_bps),
        ):
            if not (_is_finite(value) and value >= 0.0):
                raise ValueError(
                    f"OpponentObservation.{name} must be a finite "
                    f"non-negative number, got {value!r}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class OpponentClassification:
    """Immutable archetype classification of one observation.

    Attributes:
        archetype: The classified :class:`OpponentArchetype`.
        confidence: Classifier confidence in [0, 1].
        rule_fired: Stable name of the rule branch that fired
            (e.g. ``"hft_maker_high_cancel"``); used for audit /
            why-trace.
        observation_ts_ns: ``ts_ns`` of the originating
            :class:`OpponentObservation`. Carried so the classification
            is self-describing in the audit ledger.
    """

    archetype: OpponentArchetype
    confidence: float
    rule_fired: str
    observation_ts_ns: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"OpponentClassification.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )
        if not self.rule_fired:
            raise ValueError(
                "OpponentClassification.rule_fired must be non-empty"
            )
        if self.observation_ts_ns <= 0:
            raise ValueError(
                "OpponentClassification.observation_ts_ns must be positive, "
                f"got {self.observation_ts_ns!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class BehaviorPrediction:
    """Immutable prediction of the dominant opponent's next action.

    Attributes:
        symbol: Asset symbol the prediction is about. Non-empty.
        predicted_action: The forecast :class:`PredictedAction`.
        confidence: Predictor confidence in [0, 1].
        classification: The :class:`OpponentClassification` the
            prediction was derived from. Carried so the prediction is
            self-describing in the audit ledger.
        observation_ts_ns: ``ts_ns`` of the originating
            :class:`OpponentObservation`.
    """

    symbol: str
    predicted_action: PredictedAction
    confidence: float
    classification: OpponentClassification
    observation_ts_ns: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("BehaviorPrediction.symbol must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"BehaviorPrediction.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )
        if self.observation_ts_ns <= 0:
            raise ValueError(
                "BehaviorPrediction.observation_ts_ns must be positive, "
                f"got {self.observation_ts_ns!r}"
            )
        if self.observation_ts_ns != self.classification.observation_ts_ns:
            raise ValueError(
                "BehaviorPrediction.observation_ts_ns must match "
                "classification.observation_ts_ns"
            )
