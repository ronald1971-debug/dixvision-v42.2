"""Strategy approval registry contract — Wave-04.6 PR-D.

The :class:`StrategyRegistry` is the **governance-side** view of every
trading strategy ever proposed by Indira's composition engine. It is
orthogonal to the intelligence-side
:class:`intelligence_engine.strategy_runtime.StrategyStateMachine`
which tracks deployment-tier promotions
(``PROPOSED → SHADOW → CANARY → LIVE``). The registry asks a different
question: *"is this strategy approved for use at all?"*

Lifecycle::

    DRAFT  ─┐
            ├──► VALIDATING ─┬─► APPROVED ─► RETIRED
            └────────────────┴────────────► RETIRED  (validation failed
                                                       or operator
                                                       withdrew)

Both terminal states (``APPROVED``→``RETIRED``,
``VALIDATING``→``RETIRED``) are reachable; only ``RETIRED`` is final.

Determinism (INV-15 / TEST-01): every transition is materialised as a
``STRATEGY_LIFECYCLE`` row in the authority ledger so the registry can
be replayed bit-identically by walking the chain.

The contract module is **pure** — no IO, no state, no clocks. The
governance-engine adapter
(``governance_engine.strategy_registry``) owns the in-memory dictionary,
the ledger writer, and the replay loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class StrategyLifecycle(StrEnum):
    """Governance-side approval lifecycle.

    Members:
        DRAFT: Strategy has been proposed by the composition engine but
            no validator has run yet.
        VALIDATING: A validator is actively evaluating the strategy
            against the approval criteria (paper / shadow score
            thresholds, rule-graph oracle, etc.).
        APPROVED: The validator passed; the strategy is permitted to
            run live (deployment-tier promotion is then governed by the
            intelligence-side ``StrategyStateMachine``).
        RETIRED: Terminal. The strategy is permanently removed from
            consideration. Reachable from any non-terminal state.
    """

    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


# Forward-only edges; ``RETIRED`` is the single terminal state.
LEGAL_LIFECYCLE_TRANSITIONS: dict[StrategyLifecycle, frozenset[StrategyLifecycle]] = {
    StrategyLifecycle.DRAFT: frozenset(
        {StrategyLifecycle.VALIDATING, StrategyLifecycle.RETIRED}
    ),
    StrategyLifecycle.VALIDATING: frozenset(
        {StrategyLifecycle.APPROVED, StrategyLifecycle.RETIRED}
    ),
    StrategyLifecycle.APPROVED: frozenset({StrategyLifecycle.RETIRED}),
    StrategyLifecycle.RETIRED: frozenset(),
}


class StrategyLifecycleError(ValueError):
    """Raised when a transition is not in :data:`LEGAL_LIFECYCLE_TRANSITIONS`."""


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    """One immutable governance-side strategy snapshot.

    The registry holds the *current* :class:`StrategyRecord` for each
    ``strategy_id``; older versions are reachable by replaying the
    ledger.

    Attributes:
        strategy_id: Stable, unique identifier for the strategy
            (typically a content hash of the composition).
        version: Monotonically increasing per ``strategy_id``. ``1``
            for the initial DRAFT, incremented on each transition.
        lifecycle: Current :class:`StrategyLifecycle` state.
        parameters: Frozen mapping of the strategy's parameter set
            (``str → str`` for ledger-friendly serialisation; numeric
            values are stringified by the producer).
        composed_from: Tuple of component IDs the strategy was
            decomposed into (Wave-04 PR-3). Ordering is the canonical
            order returned by the composition engine.
        why: Tuple of structured DecisionTrace.why references
            (Wave-04 PR-5) explaining the strategy's intent. Empty
            tuple is permitted for strategies older than Wave-04.
        created_ts_ns: Caller-supplied nanosecond timestamp at which
            the DRAFT record was first registered.
        last_transition_ts_ns: Caller-supplied nanosecond timestamp
            of the most recent lifecycle transition. Equals
            ``created_ts_ns`` while the record is still in DRAFT and
            no transition has occurred.
        mutable_parameters: Tuple of parameter keys that the
            ``UpdateValidator`` (Wave-04.6 PR-E) will permit the
            learning engine to mutate at runtime. Empty tuple means
            **immutable** — every learning-driven update is rejected
            for this strategy.
        parameter_bounds: Optional inclusive ``[lo, hi]`` numeric
            range per mutable parameter. ``UpdateValidator`` rejects
            any proposed ``new_value`` whose ``float`` cast falls
            outside the declared range. Parameters absent from this
            mapping are accepted on whitelist alone (no numeric
            range check).
    """

    strategy_id: str
    version: int
    lifecycle: StrategyLifecycle
    parameters: Mapping[str, str] = field(default_factory=dict)
    composed_from: tuple[str, ...] = ()
    why: tuple[str, ...] = ()
    created_ts_ns: int = 0
    last_transition_ts_ns: int = 0
    mutable_parameters: tuple[str, ...] = ()
    parameter_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )


def is_legal_transition(
    *, prev: StrategyLifecycle, new: StrategyLifecycle
) -> bool:
    """Return whether ``prev → new`` is in :data:`LEGAL_LIFECYCLE_TRANSITIONS`.

    Pure — used by the registry adapter as a single, testable
    pre-flight check before mutating any state.
    """

    return new in LEGAL_LIFECYCLE_TRANSITIONS[prev]


__all__ = [
    "LEGAL_LIFECYCLE_TRANSITIONS",
    "StrategyLifecycle",
    "StrategyLifecycleError",
    "StrategyRecord",
    "is_legal_transition",
]
