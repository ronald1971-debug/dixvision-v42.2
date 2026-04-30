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

from core.contracts.governance import SystemMode

__all__ = [
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
    """Mode-gated freeze policy for adaptive engines.

    Attributes:
        mode: Current :class:`SystemMode`. The policy is frozen for
            every mode except ``LIVE``.
        operator_override: When ``True`` *and* ``mode`` is ``LIVE``,
            the policy is unfrozen and adaptive mutations are
            permitted. Defaults to ``False`` so that unfreezing is
            always an explicit operator act, never a side-effect of a
            mode change.

    The policy is frozen (``is_frozen() is True``) when:

    * ``mode`` is anything other than ``LIVE`` (``SAFE``, ``PAPER``,
      ``SHADOW``, ``CANARY``, ``AUTO``, ``LOCKED``), or
    * ``mode`` is ``LIVE`` but ``operator_override`` is ``False``.

    The policy is unfrozen iff both conditions hold simultaneously:
    ``mode is SystemMode.LIVE and operator_override is True``.
    """

    mode: SystemMode
    operator_override: bool = False

    def is_frozen(self) -> bool:
        """Return ``True`` if adaptive mutations must be refused."""
        return not self.is_unfrozen()

    def is_unfrozen(self) -> bool:
        """Return ``True`` iff adaptive mutations are permitted."""
        return self.mode is SystemMode.LIVE and self.operator_override is True


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
