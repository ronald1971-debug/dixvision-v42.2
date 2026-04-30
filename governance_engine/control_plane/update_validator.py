"""UpdateValidator — Wave-04.6 PR-E.

Pure decision layer that turns a :class:`ProposedUpdate` into a
ratify-or-reject verdict. Called from
:class:`governance_engine.engine.GovernanceEngine` whenever a
``SystemEvent(sub_kind=UPDATE_PROPOSED)`` arrives. The validator is
the **first time** in the platform's history that a non-operator
actor (the learning engine) is permitted to mutate a strategy
parameter at runtime; everything below is therefore phrased as
"reject by default unless every condition is met".

``ProposedUpdate`` is a governance-owned mirror of
:class:`core.contracts.learning.LearningUpdate`. The two carry the
same fields but the ownership boundary is enforced by lint rule
B27 / INV-71: only ``learning_engine.*`` may construct a
``LearningUpdate``; governance reads the typed bus and rebuilds the
payload as a ``ProposedUpdate``.

Validation rules (deterministic, ordered):

1. ``MODE_EFFECTS[mode].learning_apply`` must be ``True``. PAPER /
   SHADOW / SAFE / LOCKED reject *everything* — the learning loop
   is closed only in CANARY / LIVE / AUTO.
2. ``strategy_id`` must exist in :class:`StrategyRegistry`.
3. ``StrategyRecord.lifecycle`` must be ``APPROVED``.
4. ``parameter`` must appear in ``StrategyRecord.mutable_parameters``.
5. If the strategy declares ``parameter_bounds[parameter]``,
   ``float(new_value)`` must be in ``[lo, hi]``. (Parameters with no
   declared bound pass on whitelist alone.)

Pure — no IO, no clocks, no randomness. INV-15 / TEST-01.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.contracts.governance import SystemMode
from core.contracts.mode_effects import effect_for
from core.contracts.strategy_registry import StrategyLifecycle
from governance_engine.strategy_registry import StrategyRegistry


@dataclass(frozen=True, slots=True)
class ProposedUpdate:
    """Governance-owned mirror of a parameter-update proposal.

    The engine deserializes ``SystemEvent(sub_kind=UPDATE_PROPOSED)``
    payload directly into this DTO rather than constructing a
    :class:`core.contracts.learning.LearningUpdate`, which lint rule
    B27 / INV-71 reserves to ``learning_engine.*``. Authority
    symmetry: governance observes parameter mutations on the typed
    bus, it does not synthesise them.
    """

    ts_ns: int
    strategy_id: str
    parameter: str
    old_value: str
    new_value: str
    reason: str
    meta: Mapping[str, str] = field(default_factory=dict)


class UpdateVerdict(StrEnum):
    """Two-valued verdict — every input is either ratified or rejected."""

    RATIFY = "RATIFY"
    REJECT = "REJECT"


class UpdateRejectCode(StrEnum):
    """Stable reason codes for rejected updates.

    Used by the audit ledger row to keep replay-tractable
    classifications independent of human-readable reasons (which may
    drift across releases).
    """

    MODE_LEARNING_DISABLED = "MODE_LEARNING_DISABLED"
    UNKNOWN_STRATEGY = "UNKNOWN_STRATEGY"
    LIFECYCLE_NOT_APPROVED = "LIFECYCLE_NOT_APPROVED"
    PARAMETER_NOT_MUTABLE = "PARAMETER_NOT_MUTABLE"
    NEW_VALUE_NOT_NUMERIC = "NEW_VALUE_NOT_NUMERIC"
    NEW_VALUE_OUT_OF_BOUNDS = "NEW_VALUE_OUT_OF_BOUNDS"


@dataclass(frozen=True, slots=True)
class UpdateDecision:
    """Result of :meth:`UpdateValidator.validate`.

    Attributes:
        verdict: Either ``RATIFY`` or ``REJECT``.
        code: ``None`` for ``RATIFY``; the canonical
            :class:`UpdateRejectCode` for ``REJECT``.
        detail: Human-readable supplement, suitable for a ledger
            row's ``reason`` field.
    """

    verdict: UpdateVerdict
    code: UpdateRejectCode | None
    detail: str


class UpdateValidator:
    """Stateless decider over a :class:`StrategyRegistry`.

    The validator does not mutate the registry; it only reads
    :class:`StrategyRecord` shape. Wiring (mode source, registry
    handle) is injected at construction time so the validator stays
    deterministic and re-entrant.
    """

    name: str = "update_validator"
    spec_id: str = "GOV-CP-08-VALIDATOR"

    def __init__(self, *, registry: StrategyRegistry) -> None:
        self._registry = registry

    def validate(
        self, *, update: ProposedUpdate, mode: SystemMode
    ) -> UpdateDecision:
        if not effect_for(mode).learning_apply:
            return UpdateDecision(
                verdict=UpdateVerdict.REJECT,
                code=UpdateRejectCode.MODE_LEARNING_DISABLED,
                detail=f"mode {mode.value} forbids learning_apply",
            )

        record = self._registry.get(update.strategy_id)
        if record is None:
            return UpdateDecision(
                verdict=UpdateVerdict.REJECT,
                code=UpdateRejectCode.UNKNOWN_STRATEGY,
                detail=f"strategy {update.strategy_id!r} not registered",
            )

        if record.lifecycle is not StrategyLifecycle.APPROVED:
            return UpdateDecision(
                verdict=UpdateVerdict.REJECT,
                code=UpdateRejectCode.LIFECYCLE_NOT_APPROVED,
                detail=(
                    f"strategy {update.strategy_id!r} is "
                    f"{record.lifecycle.value}, not APPROVED"
                ),
            )

        if update.parameter not in record.mutable_parameters:
            return UpdateDecision(
                verdict=UpdateVerdict.REJECT,
                code=UpdateRejectCode.PARAMETER_NOT_MUTABLE,
                detail=(
                    f"parameter {update.parameter!r} is not in the "
                    f"mutable whitelist for {update.strategy_id!r}"
                ),
            )

        bound = record.parameter_bounds.get(update.parameter)
        if bound is not None:
            try:
                value = float(update.new_value)
            except (TypeError, ValueError):
                return UpdateDecision(
                    verdict=UpdateVerdict.REJECT,
                    code=UpdateRejectCode.NEW_VALUE_NOT_NUMERIC,
                    detail=(
                        f"new_value {update.new_value!r} is not numeric "
                        f"and parameter {update.parameter!r} declares bounds"
                    ),
                )
            lo, hi = bound
            if not (lo <= value <= hi):
                return UpdateDecision(
                    verdict=UpdateVerdict.REJECT,
                    code=UpdateRejectCode.NEW_VALUE_OUT_OF_BOUNDS,
                    detail=(
                        f"new_value {value} is outside [{lo}, {hi}] for "
                        f"{update.strategy_id!r}.{update.parameter}"
                    ),
                )

        return UpdateDecision(
            verdict=UpdateVerdict.RATIFY,
            code=None,
            detail=(
                f"{update.strategy_id!r}.{update.parameter} "
                f"{update.old_value!r} → {update.new_value!r}"
            ),
        )


__all__ = [
    "UpdateDecision",
    "UpdateRejectCode",
    "UpdateValidator",
    "UpdateVerdict",
]
