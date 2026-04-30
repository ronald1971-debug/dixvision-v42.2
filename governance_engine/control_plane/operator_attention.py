"""Wave-04.6 PR-F — operator attention seam (AUTO mode oversight relaxation).

Reviewer #3 (audit v3, §"three things that need honest scrutiny", item 3)
identified that the Mode FSM was missing AUTO as a distinct runtime
state. Wave-04.6 PR-A made AUTO declarative in the canonical mode-effect
table (``core/contracts/mode_effects.py``); PR-B / PR-C / PR-E wired the
``signals_emit`` / ``executions_dispatch`` / ``size_cap_pct`` /
``learning_*`` columns into the runtime engines.

This PR closes the last column: ``oversight_kind``. AUTO is the *target
full-deploy state* — its operational distinction from LIVE is that
operator approval is **exception-based**, not per-trade. The cognitive
approval queue currently gates every cognitive proposal on an explicit
operator click (Wave-03 PR-5, INV-72). In AUTO mode that gate must
relax to "approve unless a hazard is active".

This module is the canonical decision point. It owns *one* question:

    Given the current mode and the hazard state, does this cognitive
    proposal require per-trade operator attention, or may it auto-emit?

The answer is computed deterministically from the mode-effect table and
an injected hazard-active provider:

* ``oversight_kind == "per_trade"`` → always True (LIVE / CANARY /
  SHADOW / PAPER / SAFE).
* ``oversight_kind == "exception_only"`` → True iff a hazard is active
  (AUTO).
* ``oversight_kind == "none"`` → always False (LOCKED — but the
  PolicyEngine and execution gate already block proposals upstream;
  this branch is a defensive fall-through).

Isolation contract (B1): this module imports from ``core.contracts.*``
only. The mode and hazard state are passed via callable seams so the
caller decides which subsystem provides them.

Determinism contract (INV-15): the function is stateless and pure once
the seams are bound; identical seam outputs → identical decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.contracts.governance import SystemMode
from core.contracts.mode_effects import MODE_EFFECTS

__all__ = [
    "AUTO_DECIDED_BY_TAG",
    "OperatorAttention",
]


AUTO_DECIDED_BY_TAG: str = "auto:AUTO_MODE_EXCEPTION_ONLY"
"""Stamp written to ``decided_by`` when AUTO mode auto-approves.

Surfaces in the ``OPERATOR_APPROVED_SIGNAL`` ledger row so a replay can
distinguish operator clicks from mode-driven auto-decisions. The string
is canonical — any divergence breaks ledger replay equivalence."""


ModeProvider = Callable[[], SystemMode]
"""Returns the current SystemMode (typically ``StateTransitionManager.current_mode``)."""


HazardActiveProvider = Callable[[], bool]
"""Returns True iff at least one hazard is active right now.

Production binding wraps :class:`system_engine.coupling.HazardObserver`'s
``current_throttle(now_ns).block`` call. A throttle that blocks reflects
an active hazard window; a non-blocking throttle reflects a quiet
system. Tests pass a recording lambda."""


@dataclass(frozen=True, slots=True)
class OperatorAttention:
    """Decision: does the next cognitive proposal need per-trade approval?

    The runtime calls :meth:`per_trade_required` once per queued
    proposal. When it returns ``False`` the caller may invoke the
    approval edge with ``decided_by=AUTO_DECIDED_BY_TAG`` instead of
    waiting for an operator click. When it returns ``True`` the caller
    must leave the proposal PENDING and surface it to the dashboard.

    Construction takes two seams (mode + hazard); the dataclass is
    frozen so the binding is immutable for the lifetime of the runtime.
    """

    mode_provider: ModeProvider
    hazard_active_provider: HazardActiveProvider

    def per_trade_required(self) -> bool:
        """Return True iff the proposal needs an explicit operator decision."""

        mode = self.mode_provider()
        oversight = MODE_EFFECTS[mode].oversight_kind
        if oversight == "per_trade":
            return True
        if oversight == "none":
            # LOCKED — proposals should not reach this seam at all
            # (PolicyEngine blocks operator-origin actions and the
            # execution gate suppresses dispatch). Fail-closed:
            # require per-trade so a misrouted proposal is never
            # auto-emitted from a locked system.
            return True
        # exception_only — AUTO. Per-trade only when a hazard fires.
        return self.hazard_active_provider()
