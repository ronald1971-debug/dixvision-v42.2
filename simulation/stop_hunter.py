"""SIM-08 stop_hunter — adversarial stop-cluster hunting step function.

Models the hostile-market-maker pattern where a counterparty pushes
price through a known stop-loss cluster to harvest forced liquidations,
then mean-reverts. The simulator does NOT model microstructure — it
maps a frozen :class:`RealityScenario` plus a seed to a deterministic
:class:`RealityOutcome` describing the P&L impact a long position
would have suffered through the hunt.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): caller's average entry price.
* ``position_size_usd`` (float, > 0): notional of caller's open long.
* ``stop_price`` (float, > 0): stop-loss level (must be < entry_price).
* ``cluster_thickness_usd`` (float, >= 0): aggregate notional camped
  *at* ``stop_price`` (the larger this is, the more rewarding it is
  for the hunter to wick through it).
* ``hunt_intensity`` (float, in ``[0, 1]``): how hard the hunter is
  willing to overshoot. ``0.0`` = no hunt (price respects stop and
  bounces above it); ``1.0`` = full hunt (max overshoot below stop).

The deterministic step:

1. Derive a stable ``rng`` from ``(seed, scenario.scenario_id)`` via
   stdlib :class:`random.Random` (NOT used for production logic — only
   to perturb cluster_thickness sensitivity per reality).
2. Compute ``triggered = hunt_intensity * (1 + cluster_pull)`` where
   ``cluster_pull`` is the seed-perturbed cluster sensitivity.
3. If ``triggered`` ≥ ``trigger_threshold`` (config), a hunt occurs:
   the position is force-closed at
   ``stop_price - overshoot_factor * (entry_price - stop_price)``
   so P&L is the (more-negative) overshot loss.
4. Otherwise the stop is respected: position closes at ``stop_price``
   and P&L is ``-position_size_usd * (entry_price - stop_price) /
   entry_price`` (linear approximation; the contract surface is
   USD-denominated P&L per INV-29).

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
``random.Random`` itself is fully deterministic on its seed (PCG-XSH
Mersenne Twister), so two runs with the same scenario + seed produce
byte-identical outcomes.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.1, the first of the 15 missing SIM
  modules in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class StopHunterConfig:
    """Versioned configuration for the SIM-08 stop_hunter module.

    Attributes:
        trigger_threshold: ``triggered`` value above which a hunt
            occurs. Default 0.5 (median hunt intensity + non-empty
            cluster fires).
        overshoot_factor: How far below the stop price the hunt
            wicks, expressed as a fraction of the (entry - stop)
            distance. ``0.0`` = no overshoot; ``1.0`` = wick doubles
            the loss. Default 0.5.
        cluster_jitter: Magnitude of seed-driven jitter applied to
            ``cluster_thickness_usd`` sensitivity. Default 0.2.
    """

    trigger_threshold: float = 0.5
    overshoot_factor: float = 0.5
    cluster_jitter: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_threshold <= 2.0:
            raise ValueError(
                "StopHunterConfig.trigger_threshold must be in [0, 2], "
                f"got {self.trigger_threshold!r}"
            )
        if not 0.0 <= self.overshoot_factor <= 1.0:
            raise ValueError(
                "StopHunterConfig.overshoot_factor must be in [0, 1], "
                f"got {self.overshoot_factor!r}"
            )
        if not 0.0 <= self.cluster_jitter <= 1.0:
            raise ValueError(
                "StopHunterConfig.cluster_jitter must be in [0, 1], "
                f"got {self.cluster_jitter!r}"
            )


def _require_positive(meta: dict[str, Any], key: str) -> float:
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


def _require_nonneg(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not v >= 0.0:
        raise ValueError(f"meta[{key!r}] must be >= 0, got {v!r}")
    return v


def _require_unit(meta: dict[str, Any], key: str) -> float:
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


class StopHunter:
    """SIM-08 deterministic stop-cluster hunting step function."""

    def __init__(self, config: StopHunterConfig | None = None) -> None:
        self._config = config or StopHunterConfig()

    @property
    def config(self) -> StopHunterConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        entry = _require_positive(meta, "entry_price")
        size_usd = _require_positive(meta, "position_size_usd")
        stop = _require_positive(meta, "stop_price")
        thickness = _require_nonneg(meta, "cluster_thickness_usd")
        intensity = _require_unit(meta, "hunt_intensity")

        if not stop < entry:
            raise ValueError(
                f"stop_price ({stop!r}) must be < entry_price ({entry!r}) "
                "for a long-position stop hunt"
            )

        cfg = self._config
        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        # Map cluster thickness to a [0, 1] pull saturating at $1M
        # (arbitrary cap; calibration is upstream's responsibility).
        normalised_thickness = min(1.0, thickness / 1_000_000.0)
        jitter = (rng.random() - 0.5) * 2.0 * cfg.cluster_jitter
        cluster_pull = max(0.0, min(1.0, normalised_thickness + jitter))
        triggered = intensity * (1.0 + cluster_pull)

        loss_at_stop = size_usd * (entry - stop) / entry

        if triggered >= cfg.trigger_threshold:
            overshoot = cfg.overshoot_factor * (entry - stop)
            wick_price = stop - overshoot
            pnl = -size_usd * (entry - wick_price) / entry
            drawdown = -pnl
            rule_fired = "stop_hunt_triggered"
            fills = 2
        else:
            pnl = -loss_at_stop
            drawdown = loss_at_stop
            rule_fired = "stop_respected"
            fills = 1

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=fills,
            rule_fired=rule_fired,
        )


__all__ = ["StopHunter", "StopHunterConfig"]
