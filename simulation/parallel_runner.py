"""SIM-07 — deterministic N-reality runner.

Runs ``n_realities`` independent realities of one
:class:`RealityScenario`, given a caller-supplied step function and a
caller-supplied seed sequence, and returns:

* a sorted-by-seed tuple of :class:`RealityOutcome`, and
* an aggregated :class:`RealitySummary` distribution.

The step function is the only place actual market dynamics live — this
module is the deterministic harness that wraps it. By forcing the caller
to supply the seeds (instead of generating them internally), we keep
INV-15 replay determinism: the same scenario + same seed list + same
step function → byte-identical outcomes on any host.

Authority constraints (manifest §H1):

* Imports only from :mod:`core.contracts` and stdlib. No engine
  cross-imports (INV-08).
* No clock, no PRNG, no IO inside this module — the seed list is the
  only source of pseudo-randomness, and the step function is the only
  thing allowed to read it.
* Replay-deterministic: identical inputs always produce identical
  outputs (INV-15).

Refs:
- full_feature_spec §624–633 (SIM-XX module list)
- features_list §264 (SIM-00 description)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from core.contracts.simulation import (
    RealityOutcome,
    RealityScenario,
    RealitySummary,
)

StepFn = Callable[[int, RealityScenario], RealityOutcome]
"""Caller-supplied step function.

Given a non-negative seed and the frozen :class:`RealityScenario`,
returns a :class:`RealityOutcome`. The function MUST be deterministic
on its inputs and MUST set ``RealityOutcome.scenario_id`` equal to the
scenario's ``scenario_id`` and ``RealityOutcome.seed`` equal to the
input seed. Both invariants are enforced by the runner.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelRunnerConfig:
    """Versioned configuration for the deterministic N-reality runner.

    ``min_realities`` and ``max_realities`` bound the batch size so a
    misconfigured caller cannot DoS the simulation tier; ``win_threshold_usd``
    is the cutoff above which a reality counts as a "win" in the summary's
    ``win_rate`` field.
    """

    min_realities: int = 1
    max_realities: int = 10_000
    win_threshold_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.min_realities < 1:
            raise ValueError(
                "ParallelRunnerConfig.min_realities must be >= 1, "
                f"got {self.min_realities!r}"
            )
        if self.max_realities < self.min_realities:
            raise ValueError(
                "ParallelRunnerConfig.max_realities must be >= min_realities, "
                f"got max={self.max_realities!r}, min={self.min_realities!r}"
            )


def _percentile_sorted(sorted_pnls: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    ``q`` is in [0, 1]. Equivalent to numpy's default linear method but
    implemented in stdlib so the simulation tier stays leaf-pure.
    """

    n = len(sorted_pnls)
    if n == 1:
        return sorted_pnls[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_pnls[lo] * (1.0 - frac) + sorted_pnls[hi] * frac


class ParallelRunner:
    """Deterministic N-reality runner (SIM-07).

    Stateless — every call to :meth:`run` reads only its arguments and
    the bound config, so two runners with the same config produce
    identical outputs on identical inputs (INV-15).
    """

    def __init__(self, config: ParallelRunnerConfig | None = None) -> None:
        self._config = config or ParallelRunnerConfig()

    @property
    def config(self) -> ParallelRunnerConfig:
        return self._config

    def run(
        self,
        scenario: RealityScenario,
        seeds: Sequence[int],
        step: StepFn,
    ) -> tuple[tuple[RealityOutcome, ...], RealitySummary]:
        """Run the scenario under each seed and aggregate the outcomes.

        Returns ``(outcomes, summary)`` where ``outcomes`` is sorted by
        ``seed`` ascending so downstream consumers see a stable ledger
        ordering. Duplicate seeds are rejected up-front so the same
        seed-list always produces the same number of outcomes.
        """

        cfg = self._config

        if not seeds:
            raise ValueError("ParallelRunner.run requires at least one seed")
        if len(seeds) < cfg.min_realities:
            raise ValueError(
                "ParallelRunner.run: too few seeds "
                f"({len(seeds)} < min_realities={cfg.min_realities})"
            )
        if len(seeds) > cfg.max_realities:
            raise ValueError(
                "ParallelRunner.run: too many seeds "
                f"({len(seeds)} > max_realities={cfg.max_realities})"
            )

        seen: set[int] = set()
        for s in seeds:
            if s < 0:
                raise ValueError(
                    f"ParallelRunner.run: seeds must be non-negative, got {s!r}"
                )
            if s in seen:
                raise ValueError(
                    f"ParallelRunner.run: duplicate seed {s!r} would "
                    "produce duplicate realities"
                )
            seen.add(s)

        ordered_seeds = sorted(seeds)
        outcomes: list[RealityOutcome] = []
        for seed in ordered_seeds:
            outcome = step(seed, scenario)
            if outcome.scenario_id != scenario.scenario_id:
                raise ValueError(
                    "ParallelRunner.run: step function returned outcome "
                    f"for scenario_id {outcome.scenario_id!r}, expected "
                    f"{scenario.scenario_id!r}"
                )
            if outcome.seed != seed:
                raise ValueError(
                    "ParallelRunner.run: step function returned outcome "
                    f"with seed {outcome.seed!r}, expected {seed!r}"
                )
            outcomes.append(outcome)

        summary = self._summarise(scenario, outcomes)
        return tuple(outcomes), summary

    def _summarise(
        self,
        scenario: RealityScenario,
        outcomes: Sequence[RealityOutcome],
    ) -> RealitySummary:
        n = len(outcomes)
        pnls = sorted(o.pnl_usd for o in outcomes)
        mean = sum(pnls) / n
        median = _percentile_sorted(pnls, 0.5)
        p05 = _percentile_sorted(pnls, 0.05)
        p95 = _percentile_sorted(pnls, 0.95)
        wins = sum(1 for o in outcomes if o.pnl_usd > self._config.win_threshold_usd)
        max_dd = max(o.terminal_drawdown_usd for o in outcomes)
        return RealitySummary(
            scenario_id=scenario.scenario_id,
            n_realities=n,
            pnl_mean_usd=mean,
            pnl_median_usd=median,
            pnl_p05_usd=p05,
            pnl_p95_usd=p95,
            win_rate=wins / n,
            max_drawdown_usd=max_dd,
        )
