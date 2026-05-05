"""Macro regime engine — rule-based system-wide environment classifier.

Phase 10 macro layer. Consumes a :class:`MacroSnapshot` (read-only)
and produces a :class:`MacroRegimeReading`. The engine is a pure
function: same input, same config → same output (INV-15).

Authority constraints (manifest §H1):

* This module imports only from :mod:`core.contracts` and the standard
  library plus PyYAML. No engine cross-imports.
* No clock, no PRNG, no IO outside config load.
* Replay-deterministic.

Rule order (first-match wins so the audit trail is unambiguous):

1. **CRISIS** — extreme vol AND extreme cross-asset correlation
   (vol_index ≥ vol_crisis OR return_correlation ≥ corr_crisis)
2. **RISK_OFF** — elevated vol OR negative breadth OR wide credit
3. **RISK_ON** — low vol AND positive breadth AND tight credit
4. **NEUTRAL** — fallback when no other rule fires

Confidence is a deterministic function of how far the snapshot's
worst-violating dimension exceeds (or undershoots) the rule threshold,
clamped to [0, 1]. The engine never returns ``UNKNOWN`` once it has
classified at least one snapshot — UNKNOWN is reserved for the boot
state held outside this engine.

Refs:
- manifest_v3.3_delta.md §1.1 (J1 macro regime)
- full_feature_spec.md §"Macro regime engine"
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from core.contracts.macro_regime import (
    MacroRegime,
    MacroRegimeReading,
    MacroSnapshot,
)


@dataclasses.dataclass(frozen=True, slots=True)
class MacroRegimeEngineConfig:
    """Versioned thresholds for the rule-based classifier.

    Loaded from ``registry/macro_regime.yaml`` so the patch pipeline is
    the only mutator (no runtime mutation, INV-08 / INV-15).
    """

    # CRISIS branch ------------------------------------------------------
    vol_crisis: float
    correlation_crisis: float

    # RISK_OFF branch ----------------------------------------------------
    vol_risk_off: float
    breadth_risk_off: float
    credit_risk_off_bps: float

    # RISK_ON branch -----------------------------------------------------
    vol_risk_on: float
    breadth_risk_on: float
    credit_risk_on_bps: float

    # Confidence shaping -------------------------------------------------
    confidence_floor: float
    confidence_ceiling: float

    def __post_init__(self) -> None:
        if not (0.0 < self.vol_crisis <= 100.0):
            raise ValueError("vol_crisis must be in (0, 100]")
        if not (0.0 < self.correlation_crisis <= 1.0):
            raise ValueError("correlation_crisis must be in (0, 1]")
        if not (0.0 < self.vol_risk_off <= self.vol_crisis):
            raise ValueError(
                "vol_risk_off must be in (0, vol_crisis]"
            )
        if not (-1.0 <= self.breadth_risk_off < 0.0):
            raise ValueError("breadth_risk_off must be in [-1, 0)")
        if self.credit_risk_off_bps < 0:
            raise ValueError("credit_risk_off_bps must be non-negative")
        if not (0.0 < self.vol_risk_on <= self.vol_risk_off):
            raise ValueError(
                "vol_risk_on must be in (0, vol_risk_off]"
            )
        if not (0.0 < self.breadth_risk_on <= 1.0):
            raise ValueError("breadth_risk_on must be in (0, 1]")
        if self.credit_risk_on_bps < 0:
            raise ValueError("credit_risk_on_bps must be non-negative")
        if self.credit_risk_on_bps > self.credit_risk_off_bps:
            raise ValueError(
                "credit_risk_on_bps must not exceed credit_risk_off_bps"
            )
        if not (0.0 <= self.confidence_floor <= self.confidence_ceiling <= 1.0):
            raise ValueError(
                "confidence floor/ceiling must satisfy 0 <= floor <= ceiling <= 1"
            )


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "registry" / "macro_regime.yaml"


def load_macro_regime_config(
    path: Path | None = None,
) -> MacroRegimeEngineConfig:
    """Load thresholds from ``registry/macro_regime.yaml``."""

    p = path or _default_config_path()
    raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected mapping at top level, got {type(raw)!r}")
    try:
        return MacroRegimeEngineConfig(
            vol_crisis=float(raw["vol_crisis"]),
            correlation_crisis=float(raw["correlation_crisis"]),
            vol_risk_off=float(raw["vol_risk_off"]),
            breadth_risk_off=float(raw["breadth_risk_off"]),
            credit_risk_off_bps=float(raw["credit_risk_off_bps"]),
            vol_risk_on=float(raw["vol_risk_on"]),
            breadth_risk_on=float(raw["breadth_risk_on"]),
            credit_risk_on_bps=float(raw["credit_risk_on_bps"]),
            confidence_floor=float(raw["confidence_floor"]),
            confidence_ceiling=float(raw["confidence_ceiling"]),
        )
    except KeyError as e:
        raise ValueError(f"{p}: missing required key {e.args[0]!r}") from None


def _shape_confidence(
    excess: float,
    span: float,
    floor: float,
    ceiling: float,
) -> float:
    """Map a non-negative ``excess`` over a rule threshold to [floor, ceiling].

    ``span`` is the maximum credible excess we expect; everything beyond
    is clamped to ``ceiling``. The mapping is linear so the function is
    deterministic and trivially auditable.
    """

    if span <= 0.0:
        return ceiling
    e = max(0.0, excess)
    fraction = min(1.0, e / span)
    return floor + (ceiling - floor) * fraction


class MacroRegimeEngine:
    """Pure rule-based classifier producing :class:`MacroRegimeReading`.

    Stateless: every call to :meth:`classify` reads only the snapshot
    and the bound config, so two engines bound to the same config will
    produce the same readings on the same input sequence (INV-15).
    """

    def __init__(self, config: MacroRegimeEngineConfig) -> None:
        self._config = config

    @property
    def config(self) -> MacroRegimeEngineConfig:
        return self._config

    def classify(self, snapshot: MacroSnapshot) -> MacroRegimeReading:
        cfg = self._config

        # 1. CRISIS — first-match. Either dimension alone qualifies.
        if (
            snapshot.vol_index >= cfg.vol_crisis
            or snapshot.return_correlation >= cfg.correlation_crisis
        ):
            vol_excess = snapshot.vol_index - cfg.vol_crisis
            corr_excess = snapshot.return_correlation - cfg.correlation_crisis
            # span: vol can blow out by ~30 above crisis line; correlation
            # by ~0.15 (already saturated near 1).
            confidence = max(
                _shape_confidence(
                    vol_excess, 30.0, cfg.confidence_floor, cfg.confidence_ceiling
                ),
                _shape_confidence(
                    corr_excess, 0.15, cfg.confidence_floor, cfg.confidence_ceiling
                ),
            )
            rule = "crisis_vol" if vol_excess >= corr_excess else "crisis_correlation"
            return MacroRegimeReading(
                regime=MacroRegime.CRISIS,
                confidence=confidence,
                rule_fired=rule,
                snapshot_ts_ns=snapshot.ts_ns,
            )

        # 2. RISK_OFF — any one of: elevated vol, negative breadth, wide credit.
        risk_off_score = 0
        if snapshot.vol_index >= cfg.vol_risk_off:
            risk_off_score += 1
        if snapshot.breadth <= cfg.breadth_risk_off:
            risk_off_score += 1
        if snapshot.credit_spread_bps >= cfg.credit_risk_off_bps:
            risk_off_score += 1
        if risk_off_score >= 1:
            # confidence scales with how many dimensions agree.
            confidence = _shape_confidence(
                float(risk_off_score - 1),  # 0..2 over the floor
                2.0,
                cfg.confidence_floor,
                cfg.confidence_ceiling,
            )
            return MacroRegimeReading(
                regime=MacroRegime.RISK_OFF,
                confidence=confidence,
                rule_fired=f"risk_off_{risk_off_score}_of_3",
                snapshot_ts_ns=snapshot.ts_ns,
            )

        # 3. RISK_ON — all three pro-risk dimensions must agree.
        if (
            snapshot.vol_index <= cfg.vol_risk_on
            and snapshot.breadth >= cfg.breadth_risk_on
            and snapshot.credit_spread_bps <= cfg.credit_risk_on_bps
        ):
            # confidence: distance below vol_risk_on AND above breadth_risk_on.
            vol_room = cfg.vol_risk_on - snapshot.vol_index
            breadth_room = snapshot.breadth - cfg.breadth_risk_on
            confidence = min(
                _shape_confidence(
                    vol_room, 10.0, cfg.confidence_floor, cfg.confidence_ceiling
                ),
                _shape_confidence(
                    breadth_room,
                    0.5,
                    cfg.confidence_floor,
                    cfg.confidence_ceiling,
                ),
            )
            return MacroRegimeReading(
                regime=MacroRegime.RISK_ON,
                confidence=confidence,
                rule_fired="risk_on_all_dimensions",
                snapshot_ts_ns=snapshot.ts_ns,
            )

        # 4. NEUTRAL fallback — confidence sits at the floor (we know what
        # it isn't, not what it is).
        return MacroRegimeReading(
            regime=MacroRegime.NEUTRAL,
            confidence=cfg.confidence_floor,
            rule_fired="neutral_fallback",
            snapshot_ts_ns=snapshot.ts_ns,
        )


__all__ = [
    "MacroRegimeEngine",
    "MacroRegimeEngineConfig",
    "load_macro_regime_config",
]
