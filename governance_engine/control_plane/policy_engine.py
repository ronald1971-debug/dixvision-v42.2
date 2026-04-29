"""GOV-CP-01 — Policy Engine.

Owns the canonical constraint store and answers two yes/no questions:

* ``permit_mode_transition(req)``  — is the proposed Mode FSM edge
  legal under current policy?
* ``permit_operator_action(req)``  — is this dashboard-originated
  operator action allowed under the active mode?

Per Build Compiler Spec §1: the policy engine never **changes** state.
It only judges whether a proposal is acceptable. Mode writes go
through :class:`StateTransitionManager` (GOV-CP-03).

Constraints are loaded as :class:`Constraint` records; each carries a
``scope`` (GLOBAL / MODE / SYMBOL / DOMAIN) and a ``kind``. The kinds
relevant to mode transitions are:

* ``REQUIRE_OPERATOR`` — gate on ``request.operator_authorized``
* ``DOMAIN_ISOLATION`` — declarative; enforced elsewhere

Determinism contract: same constraint set + same request → same
verdict (INV-15).
"""

from __future__ import annotations

from collections.abc import Iterable

from core.contracts.governance import (
    Constraint,
    ConstraintKind,
    ConstraintScope,
    ModeTransitionRequest,
    OperatorAction,
    OperatorRequest,
    SystemMode,
)


class PolicyEngine:
    name: str = "policy_engine"
    spec_id: str = "GOV-CP-01"

    def __init__(self, constraints: Iterable[Constraint] | None = None) -> None:
        self._constraints: tuple[Constraint, ...] = tuple(constraints or ())

    # ------------------------------------------------------------------
    # Constraint store
    # ------------------------------------------------------------------

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        return self._constraints

    def load(self, constraints: Iterable[Constraint]) -> None:
        """Replace the constraint set (atomic)."""

        self._constraints = tuple(constraints)

    def for_kind(
        self, kind: ConstraintKind, *, scope: ConstraintScope | None = None
    ) -> tuple[Constraint, ...]:
        return tuple(
            c
            for c in self._constraints
            if c.kind is kind and (scope is None or c.scope is scope)
        )

    # ------------------------------------------------------------------
    # Mode transition gate
    # ------------------------------------------------------------------

    def permit_mode_transition(
        self, request: ModeTransitionRequest
    ) -> tuple[bool, str]:
        """Apply the policy half of the mode-transition gate.

        Returns ``(approved, rejection_code)``. The Mode FSM legality
        (legal edge set) is owned by ``StateTransitionManager``;
        ``PolicyEngine`` enforces *additional* policy gates layered on
        top:

        * AUTO and LIVE require ``operator_authorized`` when the
          transition is a *forward* ratchet (current rank below
          target rank). De-escalation toward LIVE/AUTO is always
          permitted by policy because the Mode FSM treats backward
          edges as safety operations (Build Compiler Spec §7).
        * REQUIRE_OPERATOR scoped to a target mode forces an explicit
          operator authorisation regardless of the requestor.
        """

        target = request.target_mode
        current = request.current_mode

        if target in (SystemMode.LIVE, SystemMode.AUTO):
            forward = (
                current is not SystemMode.LOCKED
                and int(target) > int(current)
            )
            if forward and not request.operator_authorized:
                return False, "POLICY_OPERATOR_REQUIRED"

        for c in self.for_kind(ConstraintKind.REQUIRE_OPERATOR):
            scoped_mode = c.params.get("mode")
            if scoped_mode and scoped_mode == target.name:
                if not request.operator_authorized:
                    return False, f"POLICY_OPERATOR_REQUIRED:{c.id}"

        return True, ""

    # ------------------------------------------------------------------
    # Operator action gate
    # ------------------------------------------------------------------

    def permit_operator_action(
        self, request: OperatorRequest, current_mode: SystemMode
    ) -> tuple[bool, str]:
        """Decide whether ``request`` is permitted in ``current_mode``."""

        if current_mode is SystemMode.LOCKED:
            # In LOCKED only an unlock request is acceptable.
            if request.action is OperatorAction.REQUEST_UNLOCK:
                return True, ""
            return False, "POLICY_LOCKED"

        if request.action is OperatorAction.REQUEST_KILL:
            return True, ""  # always allowed (emergency)

        if request.action is OperatorAction.REQUEST_PLUGIN_LIFECYCLE:
            target = request.payload.get("target_status", "")
            if target == "ACTIVE" and current_mode is SystemMode.SAFE:
                return False, "POLICY_LIFECYCLE_REQUIRES_NON_SAFE"
            return True, ""

        if request.action is OperatorAction.REQUEST_MODE:
            return True, ""

        if request.action is OperatorAction.REQUEST_INTENT:
            # Intent is operator strategy, not a runtime trade. It is
            # always policy-permitted while the system is not LOCKED;
            # the validation of the requested objective / risk_mode /
            # horizon enums lives in
            # ``StateTransitionManager.propose_intent``.
            return True, ""

        if request.action is OperatorAction.REQUEST_UNLOCK:
            return False, "POLICY_NOT_LOCKED"

        return False, "POLICY_UNKNOWN_ACTION"


__all__ = ["PolicyEngine"]
