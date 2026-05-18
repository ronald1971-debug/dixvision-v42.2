"""Concrete builders for :class:`ClosedLearningLoop` (PR-Z2 / P0).

The closed learning loop is constructed with two caller-supplied
builders:

* ``sample_builder`` — maps drained :class:`TradeOutcome` rows into
  :class:`FeedbackSample` rows fed to the bounded
  :class:`SlowLoopLearner`.
* ``update_builder`` — diffs the previous and current
  :class:`ParameterSnapshot` and emits one :class:`LearningUpdate`
  per changed parameter value.

PR-Z2 replaces the no-op defaults in
``learning_engine.loops.closed_loop`` with these concrete builders so
the loop actually drives parameter learning under HARDEN-04 unfrozen
+ ``LIVE`` + ``operator_override`` (instead of submitting zero
samples and emitting zero updates even when unfrozen).

The builders are pure / deterministic — same inputs produce byte-
identical outputs (INV-15) — and they never read clocks, PRNGs, or
issue IO. Strategy identifiers, parameter values, and timestamps all
arrive through their function arguments. Only ``learning_engine.*``
may construct :class:`LearningUpdate` rows (B27 / HARDEN-06 / INV-71);
this module honors that authority by living under ``learning_engine``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from core.contracts.learning import LearningUpdate, TradeOutcome

#: Canonical strategy identifier on every :class:`LearningUpdate` the
#: default diff builder emits. The closed learning loop owns
#: *parameter* updates (non-structural) so we anchor them to a single
#: stable ID rather than projecting outcome-level strategy IDs through
#: the diff. Downstream governance / audit code keys on this so a
#: future refactor can split per-strategy lanes if needed.
DEFAULT_LEARNING_STRATEGY_ID = "closed_learning_loop"

#: Reason stamped on every diff-derived :class:`LearningUpdate`. Pinned
#: by tests so the canonical ledger projection is stable.
DEFAULT_UPDATE_REASON = "slow_loop_parameter_diff"


def make_pnl_sample_builder(
    parameters: tuple[str, ...],
    sample_factory: Callable[..., Any],
    *,
    weight: float = 1.0,
) -> Callable[[tuple[TradeOutcome, ...]], tuple[Any, ...]]:
    """Build a :class:`SampleBuilder` over ``parameters``.

    Maps each drained :class:`TradeOutcome` to one
    :class:`FeedbackSample` per tracked parameter, using the outcome's
    PnL as the reward signal. The unix-seconds timestamp is derived
    from ``outcome.ts_ns // 1_000_000_000`` (pure arithmetic — no
    wall-clock read).

    The concrete sample dataclass is injected via ``sample_factory``
    so this module stays within the offline-learning tier under B1 /
    L2 (it never imports ``intelligence_engine.*`` itself; the harness
    in ``ui/server.py`` is the only legal place to bridge the two
    tiers and threads the factory through here).

    Args:
        parameters: Tuple of parameter names tracked by the bounded
            :class:`SlowLoopLearner`. Each outcome produces one sample
            per parameter so the EMA gradient observes the trade
            outcome consistently across all parameters.
        sample_factory: Keyword-only constructor of the
            :class:`FeedbackSample` value object. Must accept
            ``ts_unix_s``, ``parameter``, ``reward``, and ``weight``.
        weight: Optional positive weight applied to every sample
            (default ``1.0``). Tests can dial it up to pin the EMA
            response.

    Returns:
        A pure :class:`SampleBuilder` callable.

    Raises:
        ValueError: if ``parameters`` is empty or contains duplicates.
    """

    if not parameters:
        raise ValueError(
            "make_pnl_sample_builder requires at least one parameter name; got an empty tuple"
        )
    if len(set(parameters)) != len(parameters):
        raise ValueError(
            f"make_pnl_sample_builder requires unique parameter names; got {parameters!r}"
        )
    if not callable(sample_factory):
        raise TypeError(
            f"make_pnl_sample_builder requires a callable sample_factory; got {sample_factory!r}"
        )
    if not (weight > 0.0):
        raise ValueError(f"make_pnl_sample_builder weight must be > 0, got {weight!r}")

    params_tuple = tuple(parameters)
    sample_weight = float(weight)

    def _build_samples(
        outcomes: tuple[TradeOutcome, ...],
    ) -> tuple[Any, ...]:
        samples: list[Any] = []
        for outcome in outcomes:
            ts_unix_s = outcome.ts_ns // 1_000_000_000
            reward = float(outcome.pnl)
            for parameter in params_tuple:
                samples.append(
                    sample_factory(
                        ts_unix_s=ts_unix_s,
                        parameter=parameter,
                        reward=reward,
                        weight=sample_weight,
                    )
                )
        return tuple(samples)

    return _build_samples


def make_diff_update_builder(
    *,
    strategy_id: str = DEFAULT_LEARNING_STRATEGY_ID,
    reason: str = DEFAULT_UPDATE_REASON,
) -> Callable[
    [Any, Any, int],
    tuple[LearningUpdate, ...],
]:
    """Build an :class:`UpdateBuilder` that diffs parameter snapshots.

    Compares the previous and current :class:`ParameterSnapshot`
    ``values`` mappings; for every parameter whose post-tick value
    differs from the previous tick, emits one :class:`LearningUpdate`
    capturing the old/new value pair. The previous snapshot is
    ``None`` on the very first tick — in that case the builder returns
    an empty tuple (there is no "previous" to diff against).

    Updates are emitted in canonical parameter-name sort order so the
    closed-loop digest stays byte-identical across runs (INV-15).

    Args:
        strategy_id: Stamped on every emitted update so governance /
            audit projections key on a stable, canonical identifier.
            Defaults to :data:`DEFAULT_LEARNING_STRATEGY_ID`.
        reason: Stamped on every emitted update.
            Defaults to :data:`DEFAULT_UPDATE_REASON`.

    Returns:
        A pure :class:`UpdateBuilder` callable.

    Raises:
        ValueError: if ``strategy_id`` or ``reason`` is empty.
    """

    if not strategy_id:
        raise ValueError("make_diff_update_builder requires a non-empty strategy_id")
    if not reason:
        raise ValueError("make_diff_update_builder requires a non-empty reason")

    bound_strategy_id = str(strategy_id)
    bound_reason = str(reason)

    def _build_updates(
        previous: Any,
        current: Any,
        ts_ns: int,
    ) -> tuple[LearningUpdate, ...]:
        if previous is None:
            return ()
        previous_values: Mapping[str, float] = previous.values
        current_values: Mapping[str, float] = current.values
        updates: list[LearningUpdate] = []
        for parameter in sorted(current_values.keys()):
            new_value = current_values[parameter]
            old_value = previous_values.get(parameter)
            if old_value is None or old_value == new_value:
                continue
            updates.append(
                LearningUpdate(
                    ts_ns=ts_ns,
                    strategy_id=bound_strategy_id,
                    parameter=parameter,
                    old_value=f"{old_value:.12g}",
                    new_value=f"{new_value:.12g}",
                    reason=bound_reason,
                    meta={},
                )
            )
        return tuple(updates)

    return _build_updates


__all__ = [
    "DEFAULT_LEARNING_STRATEGY_ID",
    "DEFAULT_UPDATE_REASON",
    "make_pnl_sample_builder",
    "make_diff_update_builder",
]
