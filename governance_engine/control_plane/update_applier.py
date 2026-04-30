"""UpdateApplier — Wave-04.6 PR-E.

Bridges a ratified :class:`UpdateDecision` to the canonical mutation
on :class:`StrategyRegistry`. Separated from :class:`UpdateValidator`
so the decision-vs-effect split can be lint-enforced (B33 in a
follow-on PR): only this module is permitted to call
:meth:`StrategyRegistry.apply_parameter_update`.

Pure with respect to its inputs — every call is a deterministic
function of ``(decision, update, ts_ns)`` plus the current registry
state. INV-15 / TEST-01.
"""

from __future__ import annotations

from core.contracts.strategy_registry import StrategyRecord
from governance_engine.control_plane.update_validator import (
    ProposedUpdate,
    UpdateDecision,
    UpdateVerdict,
)
from governance_engine.strategy_registry import StrategyRegistry


class UpdateApplier:
    """Applies ratified :class:`ProposedUpdate` rows.

    The applier is a thin shell around
    :meth:`StrategyRegistry.apply_parameter_update`. It exists as a
    separate class so the decision (validator) is testable
    independently of the effect (registry mutation), and so future
    lint rules can pin "only :class:`UpdateApplier` may call
    ``apply_parameter_update``" without grepping the entire
    governance package.
    """

    name: str = "update_applier"
    spec_id: str = "GOV-CP-08-APPLIER"

    def __init__(self, *, registry: StrategyRegistry) -> None:
        self._registry = registry

    def apply(
        self,
        *,
        decision: UpdateDecision,
        update: ProposedUpdate,
    ) -> StrategyRecord:
        """Apply a ratified ``decision`` to the registry.

        Raises:
            ValueError: ``decision.verdict`` is not ``RATIFY``.
            KeyError / StrategyLifecycleError: propagated from
                :meth:`StrategyRegistry.apply_parameter_update`. (The
                validator should have caught these already; if one
                escapes the validator we *want* the apply to fail
                loudly rather than silently mutate state.)
        """

        if decision.verdict is not UpdateVerdict.RATIFY:
            raise ValueError(
                f"UpdateApplier.apply requires RATIFY, got "
                f"{decision.verdict.value}"
            )
        return self._registry.apply_parameter_update(
            strategy_id=update.strategy_id,
            parameter=update.parameter,
            new_value=update.new_value,
            ts_ns=update.ts_ns,
            reason=update.reason,
        )


__all__ = ["UpdateApplier"]
