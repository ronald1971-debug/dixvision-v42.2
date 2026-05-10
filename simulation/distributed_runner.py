"""Distributed N-reality runner — B-01.1 / SIM (extension of SIM-07).

Drop-in counterpart to :class:`simulation.parallel_runner.ParallelRunner`
that dispatches each seed to an isolated worker. Worker isolation is
expressed through a small :class:`WorkerExecutor` Protocol so the
runtime tier never imports Ray; production callers wire in the
``ray_worker_executor_factory`` factory (which lazy-imports Ray inside
its body), tests use the default :class:`InProcessWorkerExecutor`.

# ADAPTED FROM: ray-project/ray
# (python/ray/remote_function.py @ray.remote decorator pattern;
# ray/_private/worker.py ray.get() result gathering;
# ray/actor.py isolated-actor-per-task pattern)

Tier discipline
---------------

* OFFLINE_ONLY — never runs inside the hot path. The runner reads no
  clock, opens no socket, mutates no ledger; it returns
  :class:`RealityOutcome` tuples and a :class:`RealitySummary` exactly
  like :class:`ParallelRunner`.
* INV-15 byte-identical replay: identical inputs (scenario, seed list,
  step function) MUST produce identical outputs. The runner enforces
  this by sorting seeds before dispatch and by re-sorting outcomes by
  ``seed`` after gather, so executor scheduling order cannot leak into
  the result.
* B27 / B28 / INV-71 authority symmetry: no typed bus event
  construction. Pinned by AST tests.
* B1 isolation: no ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``evolution_engine`` / ``learning_engine``
  imports. Pinned by AST tests.
* No top-level ``ray`` import. The Ray executor factory lazy-imports
  ``ray`` inside the factory body only.
* No ``random`` / ``time`` / ``datetime`` / ``asyncio`` / ``os``
  imports at module top-level (INV-15).

``NEW_PIP_DEPENDENCIES = ("ray[default]",)`` — only required when the
Ray executor is actually wired in. The in-process fallback executor
needs nothing beyond stdlib.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Final, Protocol, runtime_checkable

from core.contracts.simulation import (
    RealityOutcome,
    RealityScenario,
    RealitySummary,
)
from simulation.parallel_runner import (
    ParallelRunnerConfig,
    StepFn,
    _percentile_sorted,
)

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("ray[default]",)


# ---------------------------------------------------------------------------
# Worker executor Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkerExecutor(Protocol):
    """Abstract worker pool.

    Implementations MUST:

    * dispatch each ``(seed, scenario)`` pair to a logically isolated
      worker (no shared mutable state between workers);
    * gather the resulting :class:`RealityOutcome` for every seed; and
    * preserve INV-15 determinism — the executor is allowed to
      schedule in any order, but the runner re-sorts results by
      ``seed`` so the final outcome tuple is order-stable.
    """

    def map(
        self,
        step: StepFn,
        scenario: RealityScenario,
        seeds: Sequence[int],
    ) -> Iterable[RealityOutcome]: ...


class InProcessWorkerExecutor:
    """Default executor — runs every step in the caller's thread.

    Acts as the canonical drop-in replacement when Ray is not installed
    or not wired in (e.g. tests, CI). Producing the same answer as
    :class:`ParallelRunner` is the contract; see test suite.
    """

    def map(
        self,
        step: StepFn,
        scenario: RealityScenario,
        seeds: Sequence[int],
    ) -> Iterable[RealityOutcome]:
        for seed in seeds:
            yield step(seed, scenario)


# ---------------------------------------------------------------------------
# DistributedRunner
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedRunnerConfig:
    """Versioned configuration for :class:`DistributedRunner`.

    Mirrors :class:`ParallelRunnerConfig` so a caller can swap one for
    the other without changing the surrounding code. ``num_workers``
    is an executor hint — it does not affect the result, only the
    expected concurrency.
    """

    min_realities: int = 1
    max_realities: int = 10_000
    win_threshold_usd: float = 0.0
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.min_realities < 1:
            raise ValueError(
                "DistributedRunnerConfig.min_realities must be >= 1, "
                f"got {self.min_realities!r}"
            )
        if self.max_realities < self.min_realities:
            raise ValueError(
                "DistributedRunnerConfig.max_realities must be >= min_realities, "
                f"got max={self.max_realities!r}, min={self.min_realities!r}"
            )
        if self.num_workers < 1:
            raise ValueError(
                "DistributedRunnerConfig.num_workers must be >= 1, "
                f"got {self.num_workers!r}"
            )

    def as_parallel(self) -> ParallelRunnerConfig:
        """Project to a :class:`ParallelRunnerConfig` for drop-in fallback."""
        return ParallelRunnerConfig(
            min_realities=self.min_realities,
            max_realities=self.max_realities,
            win_threshold_usd=self.win_threshold_usd,
        )


class DistributedRunner:
    """Distributed drop-in for :class:`ParallelRunner`.

    Same ``run(scenario, seeds, step)`` signature, same
    ``(outcomes, summary)`` return shape, same INV-15 determinism
    contract. The only difference is *where* each step executes — the
    bound :class:`WorkerExecutor` decides.
    """

    def __init__(
        self,
        config: DistributedRunnerConfig | None = None,
        executor: WorkerExecutor | None = None,
    ) -> None:
        self._config = config or DistributedRunnerConfig()
        self._executor: WorkerExecutor = executor or InProcessWorkerExecutor()

    @property
    def config(self) -> DistributedRunnerConfig:
        return self._config

    @property
    def executor(self) -> WorkerExecutor:
        return self._executor

    def run(
        self,
        scenario: RealityScenario,
        seeds: Sequence[int],
        step: StepFn,
    ) -> tuple[tuple[RealityOutcome, ...], RealitySummary]:
        """Run the scenario under each seed and aggregate.

        Returns ``(outcomes, summary)`` where ``outcomes`` is sorted by
        ``seed`` ascending — independent of executor scheduling order.
        """
        cfg = self._config

        if not seeds:
            raise ValueError("DistributedRunner.run requires at least one seed")
        if len(seeds) < cfg.min_realities:
            raise ValueError(
                "DistributedRunner.run: too few seeds "
                f"({len(seeds)} < min_realities={cfg.min_realities})"
            )
        if len(seeds) > cfg.max_realities:
            raise ValueError(
                "DistributedRunner.run: too many seeds "
                f"({len(seeds)} > max_realities={cfg.max_realities})"
            )

        seen: set[int] = set()
        for s in seeds:
            if s < 0:
                raise ValueError(
                    f"DistributedRunner.run: seeds must be non-negative, got {s!r}"
                )
            if s in seen:
                raise ValueError(
                    f"DistributedRunner.run: duplicate seed {s!r} would "
                    "produce duplicate realities"
                )
            seen.add(s)

        ordered_seeds = sorted(seeds)
        gathered = list(self._executor.map(step, scenario, ordered_seeds))

        # Validate and re-sort by seed — executor scheduling order MUST
        # NOT leak into the outcome tuple.
        if len(gathered) != len(ordered_seeds):
            raise ValueError(
                "DistributedRunner.run: executor returned "
                f"{len(gathered)} outcomes, expected {len(ordered_seeds)}"
            )
        for outcome in gathered:
            if outcome.scenario_id != scenario.scenario_id:
                raise ValueError(
                    "DistributedRunner.run: step function returned outcome "
                    f"for scenario_id {outcome.scenario_id!r}, expected "
                    f"{scenario.scenario_id!r}"
                )
        seed_to_outcome: dict[int, RealityOutcome] = {}
        for outcome in gathered:
            if outcome.seed in seed_to_outcome:
                raise ValueError(
                    "DistributedRunner.run: duplicate outcome for seed "
                    f"{outcome.seed!r}"
                )
            seed_to_outcome[outcome.seed] = outcome
        if set(seed_to_outcome) != set(ordered_seeds):
            missing = set(ordered_seeds) - set(seed_to_outcome)
            extra = set(seed_to_outcome) - set(ordered_seeds)
            raise ValueError(
                "DistributedRunner.run: outcome seed set mismatch "
                f"(missing={sorted(missing)!r}, extra={sorted(extra)!r})"
            )
        outcomes = tuple(seed_to_outcome[seed] for seed in ordered_seeds)

        summary = self._summarise(scenario, outcomes)
        return outcomes, summary

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


# ---------------------------------------------------------------------------
# Ray executor factory (lazy)
# ---------------------------------------------------------------------------


def ray_worker_executor_factory(
    *,
    num_cpus: int | None = None,
    address: str | None = None,
    init_options: dict[str, object] | None = None,
) -> WorkerExecutor:
    """Build a Ray-backed :class:`WorkerExecutor`.

    Lazy-imports ``ray`` inside this function so the simulation tier
    can be imported without Ray installed (INV-08 isolation +
    NEW_PIP_DEPENDENCIES dispensation). Each step is dispatched to a
    fresh ``@ray.remote`` task (no shared state between workers per the
    canonical spec).

    The returned executor implements :class:`WorkerExecutor` but is
    deliberately a thin wrapper — keep this function free of any
    business logic so the determinism contract lives in
    :class:`DistributedRunner`.
    """
    # Lazy import — see module docstring.
    import ray  # type: ignore[import-not-found]

    init_kwargs: dict[str, object] = dict(init_options or {})
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus
    if address is not None:
        init_kwargs["address"] = address
    if not ray.is_initialized():
        ray.init(**init_kwargs)

    remote_step: Callable[..., object] = ray.remote(_ray_step_entry)

    class _RayExecutor:
        def map(
            self,
            step: StepFn,
            scenario: RealityScenario,
            seeds: Sequence[int],
        ) -> Iterable[RealityOutcome]:
            futures = [remote_step.remote(step, seed, scenario) for seed in seeds]
            results = ray.get(futures)
            return list(results)

    return _RayExecutor()


def _ray_step_entry(
    step: StepFn,
    seed: int,
    scenario: RealityScenario,
) -> RealityOutcome:
    """Top-level entry that Ray serialises and ships to each worker.

    Kept at module level so Ray's cloudpickle path can pickle it; the
    closure-free shape also helps INV-15 determinism by removing
    any captured-state surface.
    """
    return step(seed, scenario)


__all__ = [
    "DistributedRunner",
    "DistributedRunnerConfig",
    "InProcessWorkerExecutor",
    "NEW_PIP_DEPENDENCIES",
    "WorkerExecutor",
    "ray_worker_executor_factory",
]
