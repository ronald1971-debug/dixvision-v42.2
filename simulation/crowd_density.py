"""SIM-12 crowd_density — crowded-position squeeze step function.

Models the failure mode where a strategy sits on the same side of
the market as the wider crowd; once aggregate same-side exposure
crosses a critical density, the inevitable unwind dispatches an
adverse price move that disproportionately hits the most crowded
positions. The simulator does NOT model microstructure; it maps a
frozen :class:`RealityScenario` plus a seed to a deterministic
:class:`RealityOutcome` describing the P&L impact a single position
would suffer through a squeeze of given intensity.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at position open.
* ``position_size_usd`` (float, > 0): notional held.
* ``side`` (str, one of ``"long"`` / ``"short"``).
* ``crowd_share`` (float, in [0, 1]): fraction of aggregate
  open interest on the same side as this position. 0.5 is
  neutral; 0.9 means 90% of the cohort is positioned the same
  direction.
* ``squeeze_intensity`` (float, in [0, 1]): how violent the
  expected unwind is. Used together with crowd_share to determine
  whether a squeeze fires this scenario.
* ``unwind_pct`` (float, in [0, 1]): maximum adverse move (as a
  fraction of entry_price) when a squeeze does fire.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. ``base_pressure = crowd_share * squeeze_intensity``.
3. Apply seeded jitter:
   ``pressure = clamp(base_pressure + jitter * pressure_jitter,
   [0, 1])``.
4. If ``pressure >= squeeze_threshold``: a squeeze fires.

   * ``move_factor = 1 + jitter * unwind_jitter``.
   * ``adverse = clamp(unwind_pct * move_factor, [0, 1])``.
   * Long: ``terminal = entry * (1 - adverse)``;
     pnl = ``size * (terminal - entry) / entry``.
   * Short: ``terminal = entry * (1 + adverse)``;
     pnl = ``size * (entry - terminal) / entry``.
   * ``rule_fired`` = ``"long_squeeze"`` or ``"short_squeeze"``.
5. Otherwise: no squeeze, ``terminal = entry``, pnl = 0,
   ``rule_fired`` = ``"no_squeeze"``.
6. ``terminal_drawdown_usd`` = absolute pnl when negative, else 0.
7. ``fills_count`` = 1 (entry) + 1 if squeeze fires.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.5 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class CrowdDensityConfig:
    """Versioned configuration for SIM-12 crowd_density.

    Attributes:
        squeeze_threshold: Pressure level above which a squeeze
            fires. Default 0.5.
        pressure_jitter: Symmetric jitter applied to the
            ``crowd_share * squeeze_intensity`` product. Default
            0.15.
        unwind_jitter: Symmetric jitter applied to the realised
            adverse move when a squeeze does fire. Default 0.2.
    """

    squeeze_threshold: float = 0.5
    pressure_jitter: float = 0.15
    unwind_jitter: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.squeeze_threshold <= 1.0:
            raise ValueError(
                "CrowdDensityConfig.squeeze_threshold must be in (0, 1], "
                f"got {self.squeeze_threshold!r}"
            )
        if not 0.0 <= self.pressure_jitter <= 1.0:
            raise ValueError(
                "CrowdDensityConfig.pressure_jitter must be in [0, 1], "
                f"got {self.pressure_jitter!r}"
            )
        if not 0.0 <= self.unwind_jitter <= 1.0:
            raise ValueError(
                "CrowdDensityConfig.unwind_jitter must be in [0, 1], "
                f"got {self.unwind_jitter!r}"
            )


_LONG = "long"
_SHORT = "short"


def _require_positive_float(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not v > 0.0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {v!r}")
    return v


def _require_unit_interval(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"meta[{key!r}] must be in [0, 1], got {v!r}")
    return v


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_LONG, _SHORT):
        raise ValueError(
            f"meta['side'] must be 'long' or 'short', got {side!r}"
        )
    return side


class CrowdDensity:
    """SIM-12 deterministic crowded-position squeeze step function."""

    def __init__(self, config: CrowdDensityConfig | None = None) -> None:
        self._config = config or CrowdDensityConfig()

    @property
    def config(self) -> CrowdDensityConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "position_size_usd")
        crowd_share = _require_unit_interval(meta, "crowd_share")
        squeeze_intensity = _require_unit_interval(meta, "squeeze_intensity")
        unwind_pct = _require_unit_interval(meta, "unwind_pct")
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        pressure_jit = (rng.random() - 0.5) * 2.0 * cfg.pressure_jitter
        unwind_jit = (rng.random() - 0.5) * 2.0 * cfg.unwind_jitter

        base_pressure = crowd_share * squeeze_intensity
        pressure = max(0.0, min(1.0, base_pressure + pressure_jit))

        if pressure < cfg.squeeze_threshold:
            return RealityOutcome(
                scenario_id=scenario.scenario_id,
                seed=seed,
                pnl_usd=0.0,
                terminal_drawdown_usd=0.0,
                fills_count=1,
                rule_fired="no_squeeze",
            )

        adverse = max(0.0, min(1.0, unwind_pct * (1.0 + unwind_jit)))

        if side == _LONG:
            terminal = entry * (1.0 - adverse)
            pnl = size_usd * (terminal - entry) / entry
            rule_fired = "long_squeeze"
        else:
            terminal = entry * (1.0 + adverse)
            pnl = size_usd * (entry - terminal) / entry
            rule_fired = "short_squeeze"

        drawdown = -pnl if pnl < 0.0 else 0.0
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=2,
            rule_fired=rule_fired,
        )


__all__ = ["CrowdDensity", "CrowdDensityConfig"]
