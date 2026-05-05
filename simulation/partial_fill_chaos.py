"""SIM-14 partial_fill_chaos — chaotic partial-fill chase step fn.

Models the failure mode where a venue fills only a chaotic fraction
of each submission and the strategy must repeatedly re-submit the
remaining size; between attempts the price drifts adversely (we are
chasing a queue), so each successive partial fill is filled at a
worse price. After ``num_attempts`` failed re-submissions any
remaining size is cancelled, which is itself a soft failure
(opportunity cost not captured here).

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): the intended fill price.
* ``order_size_usd`` (float, > 0): notional we want to fill.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_attempts`` (int, > 0): max number of fill attempts; each
  attempt fills a chaotic fraction of the remaining size.
* ``fill_ratio_mean`` (float, in [0, 1]): expected fraction of the
  remaining size filled per attempt; 1.0 means each attempt is
  fully filled (no chaos), values < 1 leak across attempts.
* ``fill_ratio_std`` (float, in [0, 0.5]): standard deviation of
  the per-attempt fill ratio.
* ``adverse_drift_per_attempt`` (float, in [0, 1]): adverse price
  drift per attempt as a fraction of entry_price; this is always
  paid against the side direction (buy chases up, sell chases down).

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Set ``remaining_usd = order_size_usd``, ``cost_usd = 0``,
   ``fills = 0``.
3. For ``attempt_idx`` in 0..num_attempts:
   - Draw ``ratio = clamp(rng.gauss(mean, std), [0, 1])``.
   - ``fill_usd = remaining_usd * ratio``.
   - If ``fill_usd > 0``:
       - ``adverse = adverse_drift_per_attempt * attempt_idx`` —
         later attempts pay deeper drift than earlier ones.
       - ``cost_usd += fill_usd * adverse``.
       - ``remaining_usd -= fill_usd``;  ``fills += 1``.
   - If ``remaining_usd <= 0``: break.
4. ``pnl_usd = -cost_usd`` (always non-positive — the chase only
   ever costs).
5. ``terminal_drawdown_usd = cost_usd``.
6. ``rule_fired``:
   - ``"fully_filled"`` if remaining_usd ≈ 0.
   - ``"incomplete_fill"`` otherwise.
7. ``fills_count`` is the number of attempts that produced a
   non-zero fill (>=1 when at least one attempt landed,
   0 if every attempt drew ratio == 0).

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.7 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class PartialFillChaosConfig:
    """Versioned configuration for SIM-14 partial_fill_chaos.

    Attributes:
        max_attempts: Hard cap on ``num_attempts`` to keep loop
            bounded. Default 1_000.
        residual_epsilon_usd: Numerical floor for "fully filled" —
            remaining size below this is treated as zero so we
            don't classify a 1e-9 dust residue as incomplete.
            Default 1e-6.
    """

    max_attempts: int = 1_000
    residual_epsilon_usd: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.max_attempts <= 10_000:
            raise ValueError(
                "PartialFillChaosConfig.max_attempts must be in (0, 10_000], "
                f"got {self.max_attempts!r}"
            )
        if not 0.0 < self.residual_epsilon_usd <= 1.0:
            raise ValueError(
                "residual_epsilon_usd must be in (0, 1] USD, "
                f"got {self.residual_epsilon_usd!r}"
            )


_BUY = "buy"
_SELL = "sell"


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


def _require_positive_int(meta: dict[str, Any], key: str) -> int:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"meta[{key!r}] must be int, got {type(raw).__name__}")
    if raw <= 0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {raw!r}")
    return raw


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


def _require_bounded_float(
    meta: dict[str, Any], key: str, low: float, high: float
) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not low <= v <= high:
        raise ValueError(
            f"meta[{key!r}] must be in [{low}, {high}], got {v!r}"
        )
    return v


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_BUY, _SELL):
        raise ValueError(
            f"meta['side'] must be 'buy' or 'sell', got {side!r}"
        )
    return side


class PartialFillChaos:
    """SIM-14 deterministic chaotic-partial-fill step fn."""

    def __init__(self, config: PartialFillChaosConfig | None = None) -> None:
        self._config = config or PartialFillChaosConfig()

    @property
    def config(self) -> PartialFillChaosConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        num_attempts = _require_positive_int(meta, "num_attempts")
        if num_attempts > cfg.max_attempts:
            raise ValueError(
                f"meta['num_attempts'] {num_attempts} exceeds "
                f"max_attempts {cfg.max_attempts}"
            )
        fill_mean = _require_unit_interval(meta, "fill_ratio_mean")
        fill_std = _require_bounded_float(meta, "fill_ratio_std", 0.0, 0.5)
        drift = _require_unit_interval(meta, "adverse_drift_per_attempt")
        side = _require_side(meta)

        # entry is exposed for downstream consumers; pin it.
        assert entry > 0.0  # noqa: S101

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        remaining = size_usd
        cost = 0.0
        fills = 0
        for attempt_idx in range(num_attempts):
            ratio_raw = rng.gauss(fill_mean, fill_std)
            ratio = max(0.0, min(1.0, ratio_raw))
            fill_usd = remaining * ratio
            if fill_usd > 0.0:
                adverse = drift * attempt_idx
                cost += fill_usd * adverse
                remaining -= fill_usd
                fills += 1
            if remaining <= cfg.residual_epsilon_usd:
                remaining = 0.0
                break

        if remaining <= cfg.residual_epsilon_usd:
            rule_fired = "fully_filled"
        else:
            rule_fired = "incomplete_fill"

        # rule_fired distinguishes "we got out cleanly" from "we
        # left size on the table"; the cost component is the same
        # in both cases (we only paid for what we filled).
        # side is exposed for downstream consumers; pin via assert.
        assert side in (_BUY, _SELL)  # noqa: S101

        pnl = -cost
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=cost,
            fills_count=fills,
            rule_fired=rule_fired,
        )


__all__ = ["PartialFillChaos", "PartialFillChaosConfig"]
