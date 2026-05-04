"""Hardening-S1 item 4-ext (wiring) -- policy drift -> Governance routing.

The pure detector :class:`PolicyHashAnchor` produces a
:class:`HazardEvent` with code ``HAZ-POLICY-DRIFT`` and severity
``CRITICAL`` whenever any anchored policy file is missing, unreadable,
or mutated mid-session. Detection alone does not protect the system --
something has to take that hazard, route it through
:meth:`GovernanceEngine.process`, and let the single FSM mutator
(B32 / GOV-CP-03) downgrade the mode through the audited chain every
other CRITICAL hazard takes (the classifier sets
``emergency_lock=True`` on CRITICAL/HIGH, which transitions to LOCKED
through :meth:`StateTransitions.propose`).

:class:`PolicyDriftSentry` is that thin coupling. It owns:

  * a :class:`PolicyHashAnchor` (pure detector);
  * a callable that mirrors :meth:`GovernanceEngine.process`
    (i.e. ``Callable[[HazardEvent], Sequence[Event]]``);
  * an optional ``on_hazard`` callback that fires *before* the hazard
    is handed to Governance, so the harness can persist the hazard on
    the audit ring even though :meth:`GovernanceEngine.process` itself
    returns ``()`` for ``HAZARD`` events (the FSM transition happens
    inside ``_handle_hazard`` and no event is emitted downstream).

Its single public method :meth:`check` calls
:meth:`~PolicyHashAnchor.verify_no_drift`, returns ``()`` on no drift,
and on drift fires ``on_hazard`` (if set), hands the hazard to the
governance callable, and returns whatever Governance emits downstream
(which is the empty tuple in production but kept polymorphic for tests).

Pure / deterministic / no I/O beyond the file-read inside
:meth:`PolicyHashAnchor.verify_no_drift`. The sentry never reads a
clock; callers must pass ``now_ns`` explicitly so replay determinism
(INV-15) is preserved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from core.contracts.events import Event, HazardEvent
from governance_engine.control_plane.policy_hash_anchor import PolicyHashAnchor

__all__ = ["PolicyDriftSentry", "GovernanceProcessor"]


class GovernanceProcessor(Protocol):
    """Structural type for the subset of :class:`GovernanceEngine` we use."""

    def process(self, event: Event) -> Sequence[Event]:  # pragma: no cover
        ...


@dataclass(slots=True)
class PolicyDriftSentry:
    """Couples :class:`PolicyHashAnchor` to :class:`GovernanceEngine`.

    Args:
        anchor: The session-bound multi-file SHA-256 anchor.
        governance_process: Callable that consumes a :class:`HazardEvent`
            and returns the events Governance emits in response. Either
            :meth:`GovernanceEngine.process` directly or a test double.
        on_hazard: Optional callback invoked with the detected hazard
            *before* it is forwarded to ``governance_process``. The
            harness uses this to persist the hazard on the audit ring;
            without it, operators would have no record of the drift
            because :meth:`GovernanceEngine.process` returns ``()`` for
            ``HAZARD`` events (the FSM is mutated inside
            ``_handle_hazard`` rather than via downstream emission).
            Exceptions raised by the callback propagate to the caller
            -- it must be cheap and not raise on the happy path.

    Example::

        sentry = PolicyDriftSentry(
            anchor=policy_hash_anchor,
            governance_process=governance_engine.process,
            on_hazard=lambda h: state.record("governance.policy_drift", h),
        )
        # On the hot path, before doing decision-relevant work:
        sentry.check(now_ns=wall_ns())

    The check is idempotent: repeated calls when there is no drift
    return ``()`` and have no side effects. When drift is detected,
    each call routes a fresh hazard through Governance -- the audit row
    accumulates one entry per detection, which is the desired behaviour
    (it lets the operator see *when* the drift was first detected and
    *for how long* it persisted before remediation).
    """

    anchor: PolicyHashAnchor
    governance_process: Callable[[HazardEvent], Sequence[Event]]
    on_hazard: Callable[[HazardEvent], None] | None = None

    def check(self, *, now_ns: int) -> Sequence[Event]:
        """Run the drift detector; route any hazard through Governance.

        On no drift returns ``()`` immediately -- no governance call,
        no callback, no ledger row.

        On drift, fires ``on_hazard`` (if set) with the detected hazard
        *before* forwarding to ``governance_process``, then returns the
        (possibly empty) sequence of events Governance emits.
        """

        hazard = self.anchor.verify_no_drift(ts_ns=now_ns)
        if hazard is None:
            return ()
        if self.on_hazard is not None:
            self.on_hazard(hazard)
        return tuple(self.governance_process(hazard))
