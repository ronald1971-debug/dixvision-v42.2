"""SIM-22 oracle_lag — stale-oracle settlement step function.

Models the failure mode where the price oracle used to mark a
position is delayed relative to the live market. The strategy
believes its mark is fresh; in reality it is reading
``true_price[t - lag]`` (with optional bps noise). The realised
pnl is computed against the *true* terminal price, while the
ledger's "perceived" pnl is computed against the *oracle* terminal
price. The simulator does NOT model microstructure; it maps a
frozen :class:`RealityScenario` plus a seed to a deterministic
:class:`RealityOutcome` describing the size of the oracle blindspot
and the realised pnl.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at position open. Both
  trader's perceived entry and the true market are entered at this
  price (oracle is fresh at t=0 by convention).
* ``order_size_usd`` (float, > 0): notional held.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, > 0): walk length.
* ``oracle_lag_steps`` (int, in [0, num_steps]): how many steps
  behind the oracle is. ``0`` means fresh oracle; ``num_steps``
  means the oracle never updates past entry.
* ``per_step_drift`` (float, in [-0.01, 0.01]): drift applied to
  each step of the true price walk.
* ``per_step_std`` (float, in [0, 0.1]): std applied to each step
  of the true price walk.
* ``oracle_noise_bps`` (float, in [0, 100]): symmetric uniform
  noise added to every oracle read, in bps of the oracle level.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk ``true_price`` for ``num_steps`` Gaussian increments.
   ``true_price[0] = entry``.
3. At each step ``t``, compute
   ``oracle_price[t] = true_price[max(0, t - oracle_lag_steps)] *
   (1 + uniform(-noise, +noise))``.
4. Compute ``actual_pnl`` = side-signed
   ``size_usd * (true_price[num_steps] - entry) / entry``.
5. Compute ``perceived_pnl`` = side-signed
   ``size_usd * (oracle_price[num_steps] - entry) / entry``.
6. ``terminal_drawdown_usd`` = absolute oracle blindspot,
   ``abs(actual_pnl - perceived_pnl)``. (Distinct from earlier SIM
   modules that report ``max(0, -actual_pnl)``.)
7. ``fills_count`` = ``oracle_lag_steps`` (encodes lag depth in
   the ledger projection).
8. ``rule_fired`` ∈ ``{"fresh", "slight_lag", "moderate_lag",
   "severe_lag"}`` per the ratio
   ``oracle_lag_steps / num_steps``:

   * ``< 0.05`` → ``"fresh"``
   * ``< 0.20`` → ``"slight_lag"``
   * ``< 0.50`` → ``"moderate_lag"``
   * otherwise  → ``"severe_lag"``

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe + +inf-safe (PR #263 review pattern); inf-overflow guard
on the price walk (PR #268 review pattern).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.15 in the canonical-rebuild walk —
  the 15th and final SIM module).
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
class OracleLagConfig:
    """Versioned configuration for SIM-22 oracle_lag.

    Attributes:
        max_steps: Hard upper bound on ``num_steps`` accepted from
            ``RealityScenario.meta``. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "OracleLagConfig.max_steps must be in (0, 1_000_000], "
                f"got {self.max_steps!r}"
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
    # NaN- and +inf-safe positive-float guard (PR #263 pattern).
    if not v > 0.0 or not math.isfinite(v):
        raise ValueError(f"meta[{key!r}] must be a finite positive float, got {v!r}")
    return v


def _require_bounded_float(
    meta: dict[str, Any], key: str, lo: float, hi: float
) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"meta[{key!r}] must be finite, got {v!r}")
    if not lo <= v <= hi:
        raise ValueError(f"meta[{key!r}] must be in [{lo}, {hi}], got {v!r}")
    return v


def _require_non_negative_int(meta: dict[str, Any], key: str) -> int:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"meta[{key!r}] must be int, got {raw!r}")
    if raw < 0:
        raise ValueError(f"meta[{key!r}] must be >= 0, got {raw!r}")
    return raw


def _require_positive_int(meta: dict[str, Any], key: str) -> int:
    raw = _require_non_negative_int(meta, key)
    if raw <= 0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {raw!r}")
    return raw


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_BUY, _SELL):
        raise ValueError(f"meta['side'] must be 'buy' or 'sell', got {side!r}")
    return side


def _classify_lag(lag: int, total: int) -> str:
    ratio = lag / total
    if ratio < 0.05:
        return "fresh"
    if ratio < 0.20:
        return "slight_lag"
    if ratio < 0.50:
        return "moderate_lag"
    return "severe_lag"


class OracleLag:
    """SIM-22 deterministic stale-oracle settlement step function."""

    def __init__(self, config: OracleLagConfig | None = None) -> None:
        self._config = config or OracleLagConfig()

    @property
    def config(self) -> OracleLagConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        side = _require_side(meta)
        num_steps = _require_positive_int(meta, "num_steps")
        if num_steps > cfg.max_steps:
            raise ValueError(
                f"num_steps={num_steps} exceeds max_steps={cfg.max_steps}"
            )
        oracle_lag_steps = _require_non_negative_int(meta, "oracle_lag_steps")
        if oracle_lag_steps > num_steps:
            raise ValueError(
                f"oracle_lag_steps={oracle_lag_steps} cannot exceed "
                f"num_steps={num_steps}"
            )
        drift = _require_bounded_float(meta, "per_step_drift", -0.01, 0.01)
        std = _require_bounded_float(meta, "per_step_std", 0.0, 0.1)
        noise_bps = _require_bounded_float(meta, "oracle_noise_bps", 0.0, 100.0)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        # Walk true price; record full path so we can lookup
        # oracle reads at arbitrary lag.
        true_path: list[float] = [entry]
        for _ in range(num_steps):
            prev = true_path[-1]
            nxt = prev * (1.0 + rng.gauss(drift, std))
            true_path.append(nxt)

        # Reject overflow / non-finite prices (PR #268 pattern).
        if not math.isfinite(true_path[-1]):
            raise ValueError(
                "oracle_lag walk overflowed to non-finite true price; "
                "tighten num_steps or per_step_drift / per_step_std"
            )

        noise_frac = noise_bps / 10_000.0
        # Single oracle read at terminal — what the trader's mark
        # ledger sees at settlement.
        oracle_idx = max(0, num_steps - oracle_lag_steps)
        oracle_terminal_clean = true_path[oracle_idx]
        # Per-call noise (one rng draw, replay-deterministic).
        noise_jit = (rng.random() - 0.5) * 2.0 * noise_frac
        oracle_terminal = oracle_terminal_clean * (1.0 + noise_jit)

        true_terminal = true_path[-1]
        sign = 1.0 if side == _BUY else -1.0
        actual_pnl = sign * size_usd * (true_terminal - entry) / entry
        perceived_pnl = sign * size_usd * (oracle_terminal - entry) / entry
        blindspot = abs(actual_pnl - perceived_pnl)

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=actual_pnl,
            terminal_drawdown_usd=blindspot,
            fills_count=oracle_lag_steps,
            rule_fired=_classify_lag(oracle_lag_steps, num_steps),
        )


__all__ = ["OracleLag", "OracleLagConfig"]
