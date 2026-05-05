"""SIM-13 latency_jitter — round-trip-latency fill-slippage step fn.

Models the failure mode where the round-trip latency from intent
arrival to broker acknowledgement is variable, and during that
window the price has moved away from the intended fill. The
strategy still fills, but at a price drifted from ``entry_price``
by a Gaussian function of realised latency.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): the price at intent submission.
* ``order_size_usd`` (float, > 0): notional being filled.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``expected_latency_ms`` (float, > 0): mean round-trip latency in
  milliseconds.
* ``jitter_std_ms`` (float, >= 0): standard deviation of latency
  jitter in milliseconds.
* ``price_drift_per_ms`` (float, in [0, 1]): expected price drift
  rate per millisecond, expressed as a fraction of entry_price.
  This is the deterministic component of fill slippage and can be
  zero in calm markets.
* ``price_volatility`` (float, in [0, 1]): standard deviation of
  per-millisecond price noise as a fraction of entry_price; the
  realised noise scales with sqrt(latency).

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Draw two N(0, 1) variates ``z_lat`` and ``z_px``.
3. ``realised_latency_ms = max(0, expected + z_lat * jitter_std)``.
4. ``base_drift = price_drift_per_ms * realised_latency_ms``.
5. ``noise = price_volatility * sqrt(realised_latency_ms) * z_px``.
6. ``signed_drift = clamp(base_drift + noise, [-1, 1])``.
7. Buy fill: ``pnl = -size_usd * signed_drift``;
   Sell fill: ``pnl = size_usd * signed_drift``.
   (Positive drift is adverse to a buyer and favourable to a
   seller; the formulas have opposite signs.)
8. ``terminal_drawdown_usd = max(0, -pnl)``.
9. ``fills_count = 1``.
10. ``rule_fired = "buy_jitter" | "sell_jitter"``.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.6 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class LatencyJitterConfig:
    """Versioned configuration for SIM-13 latency_jitter.

    Attributes:
        max_latency_ms: Hard cap on realised latency to keep price
            drift bounded. Default 60_000 ms (one minute).
    """

    max_latency_ms: float = 60_000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_latency_ms <= 24 * 3_600_000.0:
            raise ValueError(
                "LatencyJitterConfig.max_latency_ms must be in (0, 24h], "
                f"got {self.max_latency_ms!r}"
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


def _require_non_negative_float(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if v < 0.0:
        raise ValueError(f"meta[{key!r}] must be >= 0, got {v!r}")
    return v


def _require_unit_interval(meta: dict[str, Any], key: str) -> float:
    v = _require_non_negative_float(meta, key)
    if v > 1.0:
        raise ValueError(f"meta[{key!r}] must be in [0, 1], got {v!r}")
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


class LatencyJitter:
    """SIM-13 deterministic latency-jitter fill-slippage step fn."""

    def __init__(self, config: LatencyJitterConfig | None = None) -> None:
        self._config = config or LatencyJitterConfig()

    @property
    def config(self) -> LatencyJitterConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        expected_latency = _require_positive_float(meta, "expected_latency_ms")
        jitter_std = _require_non_negative_float(meta, "jitter_std_ms")
        drift_per_ms = _require_unit_interval(meta, "price_drift_per_ms")
        volatility = _require_unit_interval(meta, "price_volatility")
        side = _require_side(meta)

        # entry is exposed for downstream consumers; pin it via an
        # invariant check.
        assert entry > 0.0  # noqa: S101

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        z_lat = rng.gauss(0.0, 1.0)
        z_px = rng.gauss(0.0, 1.0)

        realised_latency = max(
            0.0,
            min(cfg.max_latency_ms, expected_latency + z_lat * jitter_std),
        )
        base_drift = drift_per_ms * realised_latency
        noise = volatility * math.sqrt(realised_latency) * z_px
        signed_drift = max(-1.0, min(1.0, base_drift + noise))

        if side == _BUY:
            pnl = -size_usd * signed_drift
            rule_fired = "buy_jitter"
        else:
            pnl = size_usd * signed_drift
            rule_fired = "sell_jitter"

        drawdown = -pnl if pnl < 0.0 else 0.0
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=1,
            rule_fired=rule_fired,
        )


__all__ = ["LatencyJitter", "LatencyJitterConfig"]
