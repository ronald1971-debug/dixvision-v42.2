"""HARDEN-04 — LearningEvolutionFreeze policy (INV-70).

The fourth and final piece of the runtime defence of the Triad Lock.
Pairs with HARDEN-01 (``ExecutionIntent`` + B25), HARDEN-02
(``ExecutionEngine.execute(intent)`` chokepoint + ``AuthorityGuard``),
and HARDEN-03 (per-event ``produced_by_engine`` + receiver assertions).

The user's framing:

    Wave-03 introduces non-deterministic LLM advice; freezing adaptive
    paths during cognitive bring-up isolates the variable. Today
    shadow-only by convention, not contract.

This module turns the convention into a contract:

* The :class:`LearningEvolutionFreezePolicy` is a frozen, immutable
  dataclass keyed on the current :class:`~core.contracts.governance.SystemMode`
  and an explicit ``operator_override``.
* Adaptive engines (``learning_engine``, ``evolution_engine``) construct
  the policy and call :func:`assert_unfrozen` at every mutation
  emission point.
* By default — in any mode that is *not* ``LIVE``, or in ``LIVE``
  without an explicit operator override — the policy is **frozen** and
  the call raises :class:`LearningEvolutionFrozenError`.

This keeps adaptive mutation a deliberate, mode-gated act. Patch
proposals, weight nudges, and structural mutations all flow through
the same contract; nothing in the offline engines can silently
re-arm itself outside ``LIVE`` mode.

Backwards compatibility: receivers that take a
``LearningEvolutionFreezePolicy | None`` field tolerate ``None`` as
"freeze not yet wired here", which preserves every existing
deterministic offline test that constructs an emitter without a
``SystemMode`` in scope. Production wiring (governance runtime →
adaptive engines) must pass a real policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.contracts.governance import SystemMode

if TYPE_CHECKING:  # pragma: no cover — type-only seam
    from core.contracts.events import SystemEvent

# P0 refinement — canonical version tag projected into every
# ``POLICY_STATE`` row this policy emits. Bumping this constant is the
# audit-anchor for any future change to the freeze-policy contract
# (frozen-condition predicate, new fields, etc.); the offline
# replay-validator can pin the version it expects to see and refuse
# rows from a future contract version it cannot reason about.
POLICY_VERSION: str = "v42.2-P0-RELAX"

__all__ = [
    "POLICY_VERSION",
    "LearningEvolutionFreezePolicy",
    "LearningEvolutionFrozenError",
    "assert_unfrozen",
    "is_unfrozen",
]


class LearningEvolutionFrozenError(RuntimeError):
    """Raised when an adaptive mutation is attempted under a frozen
    :class:`LearningEvolutionFreezePolicy`."""


@dataclass(frozen=True, slots=True)
class LearningEvolutionFreezePolicy:
    """Operator-gated freeze policy for adaptive engines.

    Attributes:
        mode: Current :class:`SystemMode`. Carried for audit and
            POLICY_STATE projection; the mode is no longer a freeze
            predicate input under ``v42.2-P0-RELAX``.
        operator_override: The single freeze gate. When ``True`` the
            policy is unfrozen and adaptive mutations are permitted in
            every :class:`SystemMode`; when ``False`` the policy is
            frozen regardless of mode. Defaults to ``False`` so a
            zero-arg construction stays fail-closed.

    Contract version history:

    * ``v42.2-P0`` — dual gate: ``mode is LIVE AND operator_override``.
      Required the FSM to be promoted to LIVE before adaptive
      mutations could run.
    * ``v42.2-P0-RELAX`` — single gate: ``operator_override``. The
      ``mode is LIVE`` predicate was dropped per direct operator
      directive so the closed learning + structural evolution loops
      unfreeze on the very first ``/api/tick`` after boot when the
      override flag is ``True`` (the post-PR #376 boot seed). Mode is
      still projected into every POLICY_STATE row for audit and
      offline replay. Re-freeze paths remain:

      * ``POST /api/operator/learning-override {enabled: false}``
        (audited via ``OPERATOR_LEARNING_OVERRIDE_CHANGED`` +
        ``POLICY_STATE`` row pair under ``STATE.lock``).
      * ``DIXVISION_LEARNING_OVERRIDE=false`` env at boot (existing
        operator pin, exercised by
        ``tests/test_pr_z1_harden04_conditional_relax.py``).

    The kill-switch, ``RiskSnapshot.halted``, the hazard-throttle
    chain, and the typed consent envelopes on the FSM mode-transition
    edges all remain in force — the relaxation governs *adaptive
    mutation only*, not order dispatch.
    """

    mode: SystemMode
    operator_override: bool = False

    def is_frozen(self) -> bool:
        """Return ``True`` if adaptive mutations must be refused."""
        return not self.is_unfrozen()

    def is_unfrozen(self) -> bool:
        """Return ``True`` iff adaptive mutations are permitted.

        Single operator-gated predicate under ``v42.2-P0-RELAX``: the
        loop is unfrozen iff ``operator_override is True``. The mode
        is intentionally NOT consulted (compare ``v42.2-P0`` which
        also required ``mode is SystemMode.LIVE``).
        """
        return self.operator_override is True

    def to_system_event(
        self,
        ts_ns: int,
        *,
        source: str = "governance.policy.freeze",
    ) -> SystemEvent:
        """Project the policy's current state into a canonical SystemEvent.

        P0 refinement — the policy itself owns the audit payload shape so
        every emitter (operator override flip, periodic tick projection,
        offline replay validator) renders the same five keys in the same
        order. The :class:`SystemEvent` payload is
        ``Mapping[str, str]`` by contract; bools are projected as
        lowercase ``"true"`` / ``"false"`` and the mode as ``mode.name``
        so the row is plain JSON-safe.

        Args:
            ts_ns: Wall-time ns for the audit row. Caller-supplied so
                INV-15 byte-identical replay is preserved (this method
                MUST NOT read any clock itself).
            source: Free-form producer label. Defaults to
                ``"governance.policy.freeze"``; operator-route writers
                pass ``"operator.api"`` so the offline replay can
                distinguish a periodic projection from an operator
                request.

        Returns:
            :class:`~core.contracts.events.SystemEvent` with
            :attr:`SystemEventKind.POLICY_STATE`, the supplied source,
            and a five-key string payload:
            ``{"policy", "frozen", "mode", "operator_override", "version"}``.
        """

        # Function-local import keeps the contracts/ package free of a
        # static cycle (events imports nothing from this module today,
        # but a top-level import here would close the seam to a future
        # ``events`` extension that wants ``POLICY_VERSION``).
        from core.contracts.events import SystemEvent, SystemEventKind

        return SystemEvent(
            ts_ns=ts_ns,
            sub_kind=SystemEventKind.POLICY_STATE,
            source=source,
            payload={
                "policy": "LearningEvolutionFreezePolicy",
                "frozen": "true" if self.is_frozen() else "false",
                "mode": self.mode.name,
                "operator_override": ("true" if self.operator_override else "false"),
                "version": POLICY_VERSION,
            },
            produced_by_engine="governance",
            proposed=False,
        )


def assert_unfrozen(
    policy: LearningEvolutionFreezePolicy | None,
    *,
    action: str,
) -> None:
    """Raise :class:`LearningEvolutionFrozenError` if ``policy`` is frozen.

    Args:
        policy: The freeze policy in scope. ``None`` is treated as a
            migration-window sentinel meaning *no policy wired yet*,
            and the call is permitted; production call sites must
            pass a real policy.
        action: Short label of the action being attempted (for
            diagnostics — e.g. ``"emit_update"``, ``"propose_patch"``,
            ``"transition_patch"``). Included in the error message.

    Raises:
        LearningEvolutionFrozenError: when ``policy`` is non-``None``
            and :meth:`LearningEvolutionFreezePolicy.is_frozen` is
            ``True``.
    """

    if policy is None:
        return
    if policy.is_frozen():
        raise LearningEvolutionFrozenError(
            f"learning/evolution action {action!r} refused: "
            f"freeze policy is active "
            f"(mode={policy.mode.name}, "
            f"operator_override={policy.operator_override})"
        )


def is_unfrozen(policy: LearningEvolutionFreezePolicy | None) -> bool:
    """Advisory probe — return ``True`` iff ``policy`` permits mutations.

    ``None`` is treated as unfrozen (migration-window default), matching
    :func:`assert_unfrozen`.
    """

    if policy is None:
        return True
    return policy.is_unfrozen()
