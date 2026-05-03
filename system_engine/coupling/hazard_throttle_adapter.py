"""Hazard throttle adapter — runtime closure for the BEHAVIOR-P3 chain.

P0-2 (PHASE6_action_plan.md) — until this module landed, the chain
that the BEHAVIOR-P3 / INV-64 docs described existed only in pieces:

  * :class:`HazardObserver` could collect :class:`HazardEvent`
    observations into a bounded ring buffer.
  * :func:`compute_throttle` could fold an observation window into
    a deterministic :class:`ThrottleDecision`.
  * :func:`apply_throttle` could project that decision onto a frozen
    :class:`RiskSnapshot`.

But nothing wired them together: ``apply_throttle()`` was never
called on the live hot path, so hazards observed by Dyon could not
gradually degrade execution -- only the CRITICAL/HIGH emergency-LOCK
path owned by Governance had any runtime effect. The action plan
flagged this as the second-highest P0 risk after the missing safety
primitives.

:class:`HazardThrottleAdapter` is the chain closure. It owns one
:class:`HazardObserver` and exposes:

  * :meth:`observe` -- accept a :class:`HazardEvent` (or an already-
    converted :class:`HazardObservation`) and remember it;
  * :meth:`current_decision` -- replay the active observations
    through :func:`compute_throttle` for a given ``now_ns``;
  * :meth:`project` -- the canonical hot-path call: take a frozen
    baseline :class:`RiskSnapshot` plus ``now_ns`` and return a
    *new* throttled snapshot with :func:`apply_throttle` applied.

Pure / deterministic / no I/O (INV-64). The adapter never reads a
clock; callers must pass ``now_ns`` explicitly so replay determinism
(INV-15) is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts.events import HazardEvent
from core.contracts.risk import RiskSnapshot
from system_engine.coupling.hazard_throttle import (
    HazardObservation,
    HazardObserver,
    HazardThrottleConfig,
    ThrottleDecision,
)
from system_engine.coupling.risk_snapshot_throttle import apply_throttle

__all__ = ["HazardThrottleAdapter"]


@dataclass(slots=True)
class HazardThrottleAdapter:
    """Wires :class:`HazardObserver` to :func:`apply_throttle`.

    Args:
        config: Throttle policy table. Defaults to
            :meth:`HazardThrottleConfig.default`.
        capacity: Bound on the underlying observer ring buffer.
            Default ``1024``; observations decay out of the active
            window long before that fills under steady state.
    """

    config: HazardThrottleConfig = field(
        default_factory=HazardThrottleConfig.default
    )
    capacity: int = 1024
    _observer: HazardObserver = field(init=False)

    def __post_init__(self) -> None:
        self._observer = HazardObserver(
            config=self.config, capacity=self.capacity
        )

    # ------------------------------------------------------------------
    # Observation intake
    # ------------------------------------------------------------------

    def observe(self, hazard: HazardEvent | HazardObservation) -> None:
        """Record a hazard observation.

        Accepts either a raw :class:`HazardEvent` from the bus or a
        pre-converted :class:`HazardObservation`. Idempotent in the
        ring-buffer sense -- the observer accepts duplicates and
        each contributes to the active window until it decays.
        """

        self._observer.observe(hazard)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def current_decision(self, *, now_ns: int) -> ThrottleDecision:
        """Replay the active observations through :func:`compute_throttle`.

        Useful for telemetry and decision-trace builders that want
        the raw throttle decision without applying it.
        """

        return self._observer.current_throttle(now_ns=now_ns)

    def project(
        self,
        *,
        snapshot: RiskSnapshot,
        now_ns: int,
    ) -> RiskSnapshot:
        """Project the current throttle decision onto ``snapshot``.

        The canonical hot-path call. The return value is a *new*
        :class:`RiskSnapshot` with the throttle applied; the input is
        unchanged.
        """

        decision = self._observer.current_throttle(now_ns=now_ns)
        return apply_throttle(snapshot=snapshot, decision=decision)

    # ------------------------------------------------------------------
    # Telemetry passthroughs
    # ------------------------------------------------------------------

    def active_observations(
        self, *, now_ns: int
    ) -> tuple[HazardObservation, ...]:
        """Return the observations still inside their active window."""

        return self._observer.active_observations(now_ns=now_ns)
