"""Simulation contract surface — frozen dataclasses for the SIM-XX modules.

Phase 10 simulation. The intelligence engine and the meta-controller read
:class:`RealitySummary` projections (slow cadence, off-hot-path) so the T1
≤1ms hot-path budget is never compromised.

Authority constraints (manifest §H1):

* Pure data only — no engine cross-imports, no clock, no PRNG, no IO.
* Every dataclass is frozen + slotted so simulation outputs are immutable
  audit records.
* Every field is range-checked in ``__post_init__``.

Refs:
- manifest §549 (simulation/ tree)
- full_feature_spec §624 (SIM-00..SIM-09 module list)
- features_list §264 (SIM-00 description)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class RealityScenario:
    """Frozen input state shared by every reality in a parallel batch.

    The same ``RealityScenario`` is fed to N seeded realities so the only
    thing that varies between realities is the seed. Replay determinism
    (INV-15) requires that the scenario itself is content-addressable, so
    callers must populate ``initial_state_hash`` with a stable hash of
    whatever upstream state they reduced into ``meta``.
    """

    scenario_id: str
    ts_ns: int
    initial_state_hash: str
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("RealityScenario.scenario_id must be non-empty")
        if self.ts_ns <= 0:
            raise ValueError(
                f"RealityScenario.ts_ns must be positive, got {self.ts_ns!r}"
            )
        if not self.initial_state_hash:
            raise ValueError(
                "RealityScenario.initial_state_hash must be non-empty"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RealityOutcome:
    """One reality's terminal state in a parallel-realities batch.

    ``pnl_usd`` is the realised P&L of the reality (positive = gain),
    ``terminal_drawdown_usd`` is the worst peak-to-trough drawdown the
    reality experienced expressed as a non-negative number, and
    ``fills_count`` is the number of synthetic fills the step function
    produced. Together they are enough to compute the standard
    distributional summary the strategy arena needs (mean / median /
    quantile / win-rate / max drawdown).
    """

    scenario_id: str
    seed: int
    pnl_usd: float
    terminal_drawdown_usd: float
    fills_count: int
    rule_fired: str

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("RealityOutcome.scenario_id must be non-empty")
        if self.seed < 0:
            raise ValueError(
                f"RealityOutcome.seed must be non-negative, got {self.seed!r}"
            )
        if self.terminal_drawdown_usd < 0.0:
            raise ValueError(
                "RealityOutcome.terminal_drawdown_usd must be non-negative, "
                f"got {self.terminal_drawdown_usd!r}"
            )
        if self.fills_count < 0:
            raise ValueError(
                "RealityOutcome.fills_count must be non-negative, "
                f"got {self.fills_count!r}"
            )
        if not self.rule_fired:
            raise ValueError("RealityOutcome.rule_fired must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class RealitySummary:
    """Distributional summary over a batch of :class:`RealityOutcome`.

    Stable, deterministic fields the meta-controller's scoring layer can
    read without re-iterating the per-reality sequence. Every field is
    derived from a sorted-by-seed input so the summary is replay-stable
    (INV-15).
    """

    scenario_id: str
    n_realities: int
    pnl_mean_usd: float
    pnl_median_usd: float
    pnl_p05_usd: float
    pnl_p95_usd: float
    win_rate: float
    max_drawdown_usd: float

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("RealitySummary.scenario_id must be non-empty")
        if self.n_realities <= 0:
            raise ValueError(
                "RealitySummary.n_realities must be positive, "
                f"got {self.n_realities!r}"
            )
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(
                f"RealitySummary.win_rate must be in [0, 1], "
                f"got {self.win_rate!r}"
            )
        if self.max_drawdown_usd < 0.0:
            raise ValueError(
                "RealitySummary.max_drawdown_usd must be non-negative, "
                f"got {self.max_drawdown_usd!r}"
            )
        if not (
            self.pnl_p05_usd
            <= self.pnl_median_usd
            <= self.pnl_p95_usd
        ):
            raise ValueError(
                "RealitySummary requires pnl_p05 <= pnl_median <= pnl_p95, "
                f"got p05={self.pnl_p05_usd!r}, "
                f"median={self.pnl_median_usd!r}, "
                f"p95={self.pnl_p95_usd!r}"
            )
