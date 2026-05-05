"""SIM-10 impact_feedback — own-trade price-impact step function.

Models the closed feedback loop where a large order's own market
impact moves the price against itself. The impact follows the
Almgren-Chriss square-root law: ``slippage = impact_coef *
sqrt(size / depth)``, which is the canonical model used by the
strategic execution scheduler at ``execution_engine/strategic/
almgren_chriss.py`` (PR #61).

Inputs (read from ``RealityScenario.meta``):

* ``reference_price`` (float, > 0): mid-market price at order arrival.
* ``order_size_usd`` (float, > 0): notional being traded by the
  caller.
* ``liquidity_depth_usd`` (float, > 0): aggregate notional sitting
  on the touch (typically L1 + a few levels).
* ``side`` (str, one of ``"buy"`` / ``"sell"``).

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. ``ratio = order_size_usd / liquidity_depth_usd`` (clamped to
   ``[0, max_ratio]`` to bound slippage at extreme inputs).
3. ``base_slippage = impact_coef * sqrt(ratio)``.
4. Apply seeded jitter: ``slippage = base_slippage * (1 + jitter *
   impact_jitter)`` (clamped to ``[0, 1]``).
5. ``avg_fill = reference_price * (1 + slippage)`` for buy;
   ``reference_price * (1 - slippage)`` for sell.
6. ``cost_usd = order_size_usd * slippage`` (always non-negative).
7. P&L = ``-cost_usd`` (impact is pure cost from the order's
   perspective; downstream meta-controller can offset with alpha).
8. ``terminal_drawdown_usd = cost_usd``.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.3 in the canonical-rebuild walk).
* execution_engine/strategic/almgren_chriss.py — production
  square-root impact scheduler this simulation mirrors.
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
class ImpactFeedbackConfig:
    """Versioned configuration for SIM-10 impact_feedback.

    Attributes:
        impact_coef: Square-root-law coefficient. Production
            calibrations of the Almgren-Chriss model put this in the
            range [0.05, 0.5]; default 0.1 corresponds to a 10%
            slippage at parity (size == depth).
        impact_jitter: Symmetric jitter applied to the realised
            slippage to model intra-day variance. Default 0.2.
        max_ratio: Hard cap on size/depth ratio used in the
            sqrt-law. Prevents pathological inputs (``depth = 1$``)
            from producing nonsensical slippage. Default 4.0
            (i.e. size up to 4x depth still maps to a finite
            slippage of ``impact_coef * 2``).
    """

    impact_coef: float = 0.1
    impact_jitter: float = 0.2
    max_ratio: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.impact_coef <= 1.0:
            raise ValueError(
                "ImpactFeedbackConfig.impact_coef must be in (0, 1], "
                f"got {self.impact_coef!r}"
            )
        if not 0.0 <= self.impact_jitter <= 1.0:
            raise ValueError(
                "ImpactFeedbackConfig.impact_jitter must be in [0, 1], "
                f"got {self.impact_jitter!r}"
            )
        if not 0.0 < self.max_ratio <= 100.0:
            raise ValueError(
                "ImpactFeedbackConfig.max_ratio must be in (0, 100], "
                f"got {self.max_ratio!r}"
            )


_BUY = "buy"
_SELL = "sell"


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


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_BUY, _SELL):
        raise ValueError(
            f"meta['side'] must be 'buy' or 'sell', got {side!r}"
        )
    return side


class ImpactFeedback:
    """SIM-10 deterministic own-trade impact-feedback step function."""

    def __init__(self, config: ImpactFeedbackConfig | None = None) -> None:
        self._config = config or ImpactFeedbackConfig()

    @property
    def config(self) -> ImpactFeedbackConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        reference = _require_positive(meta, "reference_price")
        size_usd = _require_positive(meta, "order_size_usd")
        depth = _require_positive(meta, "liquidity_depth_usd")
        side = _require_side(meta)

        cfg = self._config
        rng = random.Random(f"{seed}:{scenario.scenario_id}")

        ratio = min(size_usd / depth, cfg.max_ratio)
        base_slippage = cfg.impact_coef * math.sqrt(ratio)
        jitter = (rng.random() - 0.5) * 2.0 * cfg.impact_jitter
        slippage = max(0.0, min(1.0, base_slippage * (1.0 + jitter)))

        if side == _BUY:
            avg_fill = reference * (1.0 + slippage)
            rule_fired = "buy_impact"
        else:
            avg_fill = reference * (1.0 - slippage)
            rule_fired = "sell_impact"

        cost_usd = size_usd * slippage
        pnl = -cost_usd
        drawdown = cost_usd

        # avg_fill is computed for downstream consumers but doesn't
        # change the outcome contract; pin it via the rule_fired
        # tag for audit and assert non-negative.
        assert avg_fill > 0.0  # noqa: S101 - INV-29 invariant

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=1,
            rule_fired=rule_fired,
        )


__all__ = ["ImpactFeedback", "ImpactFeedbackConfig"]
