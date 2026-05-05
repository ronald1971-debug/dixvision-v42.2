"""OPP-01 BehaviorPredictor — rule-based opponent classifier.

Phase 10.10 opponent-model layer. Consumes an
:class:`~core.contracts.opponent.OpponentObservation` (read-only) and
produces an :class:`~core.contracts.opponent.OpponentClassification`
plus :class:`~core.contracts.opponent.BehaviorPrediction`. The
predictor is a pure function: same observation, same config → same
output (INV-15).

Authority constraints (manifest §H1):

* This module imports only from :mod:`core.contracts` and the standard
  library plus PyYAML. No engine cross-imports.
* No clock, no PRNG, no IO outside config load.
* Replay-deterministic.

Rule order (first-match wins so the audit trail is unambiguous):

1. **HFT_MAKER** — high cancel/fill AND fast refresh AND small resting size
2. **SWEEPER** — very large taker print on a narrow book
3. **MOMENTUM_TAKER** — sustained one-sided aggressor flow with mid-size prints
4. **SLOW_RESTING_LIQUIDITY** — low cancel/fill, slow refresh, large resting size
5. **NOISE** — fallback when no other rule fires

Confidence is a deterministic function of how far the observation's
worst-violating dimension exceeds (or undershoots) the rule threshold,
clamped to ``[confidence_floor, confidence_ceiling]``. The predictor
never returns ``UNKNOWN`` once it has classified at least one
observation — UNKNOWN is reserved for the boot state held outside this
module.

Refs:
- manifest_v3.1_delta.md §"Opponent model"
- full_feature_spec.md §"Behavior predictor"
- build_plan.md Phase 10.10 OPPONENT MODEL
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from core.contracts.opponent import (
    BehaviorPrediction,
    OpponentArchetype,
    OpponentClassification,
    OpponentObservation,
    PredictedAction,
)


@dataclasses.dataclass(frozen=True, slots=True)
class BehaviorPredictorConfig:
    """Versioned thresholds for the rule-based classifier.

    Loaded from ``registry/opponent_behavior.yaml`` so the patch
    pipeline is the only mutator (no runtime mutation,
    INV-08 / INV-15).
    """

    # HFT_MAKER ---------------------------------------------------------
    hft_cancel_to_fill_min: float
    hft_refresh_rate_hz_min: float
    hft_resting_size_usd_max: float

    # SWEEPER -----------------------------------------------------------
    sweeper_taker_size_usd_min: float
    sweeper_spread_bps_max: float

    # MOMENTUM_TAKER ----------------------------------------------------
    momentum_imbalance_min: float
    momentum_taker_size_usd_min: float

    # SLOW_RESTING_LIQUIDITY -------------------------------------------
    slow_cancel_to_fill_max: float
    slow_refresh_rate_hz_max: float
    slow_resting_size_usd_min: float

    # Confidence shaping -----------------------------------------------
    confidence_floor: float
    confidence_ceiling: float

    # Predicted-action mapping -----------------------------------------
    prediction_confidence_scale: float
    noise_action_confidence: float

    def __post_init__(self) -> None:
        # Phrased as ``not (x > 0.0)`` so NaN — which compares False
        # against every numeric — is rejected at construction time.
        positives = {
            "hft_cancel_to_fill_min": self.hft_cancel_to_fill_min,
            "hft_refresh_rate_hz_min": self.hft_refresh_rate_hz_min,
            "hft_resting_size_usd_max": self.hft_resting_size_usd_max,
            "sweeper_taker_size_usd_min": self.sweeper_taker_size_usd_min,
            "sweeper_spread_bps_max": self.sweeper_spread_bps_max,
            "momentum_taker_size_usd_min": self.momentum_taker_size_usd_min,
            "slow_resting_size_usd_min": self.slow_resting_size_usd_min,
        }
        for name, value in positives.items():
            if not (value > 0.0):
                raise ValueError(
                    f"BehaviorPredictorConfig.{name} must be positive, "
                    f"got {value!r}"
                )

        non_negatives = {
            "slow_cancel_to_fill_max": self.slow_cancel_to_fill_max,
            "slow_refresh_rate_hz_max": self.slow_refresh_rate_hz_max,
        }
        for name, value in non_negatives.items():
            if not (value >= 0.0):
                raise ValueError(
                    f"BehaviorPredictorConfig.{name} must be non-negative, "
                    f"got {value!r}"
                )

        if not (0.0 <= self.momentum_imbalance_min <= 1.0):
            raise ValueError(
                "BehaviorPredictorConfig.momentum_imbalance_min must be in "
                f"[0, 1], got {self.momentum_imbalance_min!r}"
            )

        if not (0.0 < self.confidence_floor <= self.confidence_ceiling <= 1.0):
            raise ValueError(
                "BehaviorPredictorConfig requires "
                "0 < confidence_floor <= confidence_ceiling <= 1, "
                f"got floor={self.confidence_floor!r}, "
                f"ceiling={self.confidence_ceiling!r}"
            )
        if not (0.0 < self.prediction_confidence_scale <= 1.0):
            raise ValueError(
                "BehaviorPredictorConfig.prediction_confidence_scale must "
                f"be in (0, 1], got {self.prediction_confidence_scale!r}"
            )
        if not (0.0 <= self.noise_action_confidence <= 1.0):
            raise ValueError(
                "BehaviorPredictorConfig.noise_action_confidence must be in "
                f"[0, 1], got {self.noise_action_confidence!r}"
            )


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "registry"
        / "opponent_behavior.yaml"
    )


def load_behavior_predictor_config(
    path: Path | None = None,
) -> BehaviorPredictorConfig:
    """Load and validate :class:`BehaviorPredictorConfig` from YAML."""

    p = path or _default_config_path()
    with p.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"opponent_behavior config at {p} must be a YAML mapping"
        )
    fields = {f.name for f in dataclasses.fields(BehaviorPredictorConfig)}
    missing = fields - raw.keys()
    if missing:
        raise ValueError(
            f"opponent_behavior config at {p} missing fields: "
            f"{sorted(missing)}"
        )
    extra = raw.keys() - fields
    if extra:
        raise ValueError(
            f"opponent_behavior config at {p} has unknown fields: "
            f"{sorted(extra)}"
        )
    return BehaviorPredictorConfig(**{name: float(raw[name]) for name in fields})


def _shape_confidence(
    excess: float, span: float, floor: float, ceiling: float
) -> float:
    """Map a non-negative excess into ``[floor, ceiling]``.

    ``excess`` is the unclamped distance the observation cleared its
    threshold by. ``span`` is the dimension's full saturation distance
    above threshold. The shaped value rises linearly from ``floor`` at
    excess=0 to ``ceiling`` at excess>=span.
    """

    if span <= 0.0:
        raise ValueError(f"_shape_confidence span must be positive, got {span!r}")
    frac = max(0.0, excess) / span
    if frac >= 1.0:
        return ceiling
    return floor + (ceiling - floor) * frac


_PREDICTED_ACTION: dict[OpponentArchetype, PredictedAction] = {
    OpponentArchetype.HFT_MAKER: PredictedAction.WITHDRAW,
    OpponentArchetype.MOMENTUM_TAKER: PredictedAction.CONTINUE_AGGRESSION,
    OpponentArchetype.SWEEPER: PredictedAction.CONTINUE_AGGRESSION,
    OpponentArchetype.SLOW_RESTING_LIQUIDITY: PredictedAction.FADE,
    OpponentArchetype.NOISE: PredictedAction.HOLD,
}


class BehaviorPredictor:
    """Rule-based opponent classifier (OPP-01).

    Stateless — every call to :meth:`predict` reads only its arguments
    and the bound config, so two predictors with the same config produce
    identical outputs on identical inputs (INV-15).
    """

    def __init__(self, config: BehaviorPredictorConfig | None = None) -> None:
        self._config = config or load_behavior_predictor_config()

    @property
    def config(self) -> BehaviorPredictorConfig:
        return self._config

    def classify(
        self, observation: OpponentObservation
    ) -> OpponentClassification:
        """Return the dominant :class:`OpponentArchetype` classification."""

        cfg = self._config
        floor = cfg.confidence_floor
        ceiling = cfg.confidence_ceiling

        # 1. HFT_MAKER — first-match. ALL three signals must trip.
        if (
            observation.cancel_to_fill_ratio >= cfg.hft_cancel_to_fill_min
            and observation.top_of_book_refresh_rate_hz
            >= cfg.hft_refresh_rate_hz_min
            and observation.avg_resting_size_usd
            <= cfg.hft_resting_size_usd_max
        ):
            # Confidence keyed off whichever dimension cleared by the
            # widest fraction. ``span`` is the threshold itself for the
            # >= signals and the threshold for the <= signal too — so
            # all three normalise to the same [0, 1] frac scale.
            cancel_frac = (
                observation.cancel_to_fill_ratio - cfg.hft_cancel_to_fill_min
            ) / cfg.hft_cancel_to_fill_min
            refresh_frac = (
                observation.top_of_book_refresh_rate_hz
                - cfg.hft_refresh_rate_hz_min
            ) / cfg.hft_refresh_rate_hz_min
            # Smaller resting size = stronger HFT_MAKER signal.
            resting_frac = (
                cfg.hft_resting_size_usd_max - observation.avg_resting_size_usd
            ) / cfg.hft_resting_size_usd_max
            best_frac = max(cancel_frac, refresh_frac, resting_frac)
            confidence = _shape_confidence(best_frac, 1.0, floor, ceiling)
            # Pick the rule label off the dimension that cleared by the
            # widest *normalised* fraction so the audit row always
            # names a dimension that actually fired.
            if cancel_frac >= refresh_frac and cancel_frac >= resting_frac:
                rule = "hft_maker_high_cancel"
            elif refresh_frac >= resting_frac:
                rule = "hft_maker_fast_refresh"
            else:
                rule = "hft_maker_small_resting"
            return OpponentClassification(
                archetype=OpponentArchetype.HFT_MAKER,
                confidence=confidence,
                rule_fired=rule,
                observation_ts_ns=observation.ts_ns,
            )

        # 2. SWEEPER — large taker print on a narrow book.
        if (
            observation.avg_taker_size_usd >= cfg.sweeper_taker_size_usd_min
            and observation.spread_bps <= cfg.sweeper_spread_bps_max
        ):
            taker_frac = (
                observation.avg_taker_size_usd - cfg.sweeper_taker_size_usd_min
            ) / cfg.sweeper_taker_size_usd_min
            # Tighter spread = stronger SWEEPER signal. Clamp at >=0
            # since spread is non-negative; spread > threshold can't
            # reach this branch anyway.
            spread_frac = (
                cfg.sweeper_spread_bps_max - observation.spread_bps
            ) / cfg.sweeper_spread_bps_max
            best_frac = max(taker_frac, spread_frac)
            confidence = _shape_confidence(best_frac, 1.0, floor, ceiling)
            rule = (
                "sweeper_size_print"
                if taker_frac >= spread_frac
                else "sweeper_narrow_book"
            )
            return OpponentClassification(
                archetype=OpponentArchetype.SWEEPER,
                confidence=confidence,
                rule_fired=rule,
                observation_ts_ns=observation.ts_ns,
            )

        # 3. MOMENTUM_TAKER — sustained one-sided aggressor flow with
        # mid-size prints. SWEEPER above already grabbed the very-large
        # prints, so this branch is the "smaller but persistent" tier.
        abs_imbalance = abs(observation.aggressor_imbalance)
        if (
            abs_imbalance >= cfg.momentum_imbalance_min
            and observation.avg_taker_size_usd
            >= cfg.momentum_taker_size_usd_min
        ):
            imb_frac = (
                abs_imbalance - cfg.momentum_imbalance_min
            ) / max(1.0 - cfg.momentum_imbalance_min, 1e-9)
            size_frac = (
                observation.avg_taker_size_usd
                - cfg.momentum_taker_size_usd_min
            ) / cfg.momentum_taker_size_usd_min
            best_frac = max(imb_frac, size_frac)
            confidence = _shape_confidence(best_frac, 1.0, floor, ceiling)
            rule = (
                "momentum_imbalance"
                if imb_frac >= size_frac
                else "momentum_size"
            )
            return OpponentClassification(
                archetype=OpponentArchetype.MOMENTUM_TAKER,
                confidence=confidence,
                rule_fired=rule,
                observation_ts_ns=observation.ts_ns,
            )

        # 4. SLOW_RESTING_LIQUIDITY — patient size, low cancel/fill,
        # slow refresh.
        if (
            observation.cancel_to_fill_ratio <= cfg.slow_cancel_to_fill_max
            and observation.top_of_book_refresh_rate_hz
            <= cfg.slow_refresh_rate_hz_max
            and observation.avg_resting_size_usd
            >= cfg.slow_resting_size_usd_min
        ):
            # Bigger resting size = stronger signal; lower cancel/fill
            # and refresh = stronger signal too.
            resting_frac = (
                observation.avg_resting_size_usd
                - cfg.slow_resting_size_usd_min
            ) / cfg.slow_resting_size_usd_min
            cancel_span = max(cfg.slow_cancel_to_fill_max, 1e-9)
            cancel_frac = (
                cfg.slow_cancel_to_fill_max - observation.cancel_to_fill_ratio
            ) / cancel_span
            refresh_span = max(cfg.slow_refresh_rate_hz_max, 1e-9)
            refresh_frac = (
                cfg.slow_refresh_rate_hz_max
                - observation.top_of_book_refresh_rate_hz
            ) / refresh_span
            best_frac = max(resting_frac, cancel_frac, refresh_frac)
            confidence = _shape_confidence(best_frac, 1.0, floor, ceiling)
            if resting_frac >= cancel_frac and resting_frac >= refresh_frac:
                rule = "slow_large_resting"
            elif cancel_frac >= refresh_frac:
                rule = "slow_low_cancel"
            else:
                rule = "slow_steady_refresh"
            return OpponentClassification(
                archetype=OpponentArchetype.SLOW_RESTING_LIQUIDITY,
                confidence=confidence,
                rule_fired=rule,
                observation_ts_ns=observation.ts_ns,
            )

        # 5. NOISE — fallback. Confidence pinned at floor since no rule
        # cleared its bar.
        return OpponentClassification(
            archetype=OpponentArchetype.NOISE,
            confidence=floor,
            rule_fired="noise_fallback",
            observation_ts_ns=observation.ts_ns,
        )

    def predict(self, observation: OpponentObservation) -> BehaviorPrediction:
        """Classify the observation and project the next-action forecast."""

        classification = self.classify(observation)
        action = _PREDICTED_ACTION[classification.archetype]

        if classification.archetype is OpponentArchetype.NOISE:
            confidence = self._config.noise_action_confidence
        else:
            scaled = (
                classification.confidence
                * self._config.prediction_confidence_scale
            )
            # ``classification.confidence`` is already in [0, 1] and
            # ``prediction_confidence_scale`` is in (0, 1] so the
            # product is in [0, 1] without explicit clamping. Clamp
            # defensively so a future config-validator change can't
            # leak an out-of-range value into the contract.
            confidence = max(0.0, min(1.0, scaled))

        return BehaviorPrediction(
            symbol=observation.symbol,
            predicted_action=action,
            confidence=confidence,
            classification=classification,
            observation_ts_ns=observation.ts_ns,
        )
