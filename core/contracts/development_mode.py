"""PR-DEV-A — Operator Master Development Mode policy.

Operator vision (PR-DEV-A): the system is a research + evolution
platform first. Indira (full trader discovery + profile modeling +
strategy mutation) and Dyon (heavy learning + structural evolution +
self-reflection + slow-loop critique) must run at full potential
**before** any real trading occurs. The operator is the only authority
that decides when the system transitions from "build phase" to "trading
phase".

This module turns that vision into a contract:

* :class:`DevelopmentModePolicy` is a frozen, slotted dataclass keyed
  on two independent flags:

  * ``development_enabled`` — gates the learning + research +
    self-evolution surface. Defaults to ``True`` so that Indira and
    Dyon are unblocked from boot regardless of :class:`SystemMode`.
  * ``trading_allowed`` — gates the execution surface. Defaults to
    ``False`` so that the Execution Gate (HARDEN-02) refuses to
    dispatch to any broker until the operator explicitly flips this
    flag. This is the **only** switch that opens trading.

* The two predicates :meth:`is_learning_unblocked` and
  :meth:`is_trading_unblocked` are pure projections of the flags.

* Re-freeze paths (operator-driven):

  * ``POST /api/operator/development-mode {enabled: false}`` — pauses
    the research + learning surface (audited under
    ``OPERATOR_DEVELOPMENT_MODE_CHANGED`` + ``POLICY_STATE`` row pair).
  * ``POST /api/operator/trading-allowed {enabled: bool}`` — the
    single switch that opens or closes the Execution Gate (audited
    under ``OPERATOR_TRADING_ALLOWED_CHANGED`` + ``POLICY_STATE`` row
    pair).
  * ``DIXVISION_DEVELOPMENT_MODE=false`` — boot-time pin for the
    development flag.
  * ``DIXVISION_TRADING_ALLOWED=true`` — boot-time pin for the trading
    flag (default ``false``).

Defense-in-depth contract (this policy is an **outer** gate, never a
replacement):

* The kill-switch, ``RiskSnapshot.halted``, the hazard-throttle chain,
  the ``AuthorityGuard``, the typed consent envelopes on the FSM
  mode-transition edges, the HARDEN-04 freeze policy, the INV-15
  replay anchor, and the HARDEN-03 ``produced_by_engine`` receiver
  assertions **all** remain in force under this policy. PR-DEV-A only
  adds an outer gate that defaults *closed for trading* and *open for
  learning*; nothing inside the runtime can flip those defaults
  without operator audit.

The :func:`to_system_event` projection emits a canonical
``POLICY_STATE`` system event with the policy version anchor so an
offline replay validator can pin the contract version and refuse rows
from a future version it cannot reason about. The policy version is
intentionally separate from :data:`POLICY_VERSION` on
:mod:`core.contracts.learning_evolution_freeze` — the two policies are
independent contracts and can be re-versioned independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.contracts.governance import SystemMode

if TYPE_CHECKING:  # pragma: no cover — type-only seam
    from core.contracts.events import SystemEvent

# PR-DEV-A — canonical version tag projected into every POLICY_STATE
# row this policy emits. Independent of
# ``learning_evolution_freeze.POLICY_VERSION``; bumping this constant
# is the audit anchor for any future change to the development-mode
# contract (new fields, new predicates, etc.).
POLICY_VERSION: str = "v42.2-DEV-A"

# Re-export the canonical reasons used by the Execution Gate so the
# audit projection (and the receivers that look for these strings)
# share a single source of truth.
DEVELOPMENT_MODE_TRADING_BLOCKED: str = "development_mode_trading_blocked"

__all__ = [
    "DEVELOPMENT_MODE_TRADING_BLOCKED",
    "POLICY_VERSION",
    "DevelopmentModePolicy",
    "DevelopmentModeTradingBlockedError",
    "assert_trading_unblocked",
    "is_learning_unblocked",
    "is_trading_unblocked",
]


class DevelopmentModeTradingBlockedError(RuntimeError):
    """Raised when a trading-side caller attempts to dispatch while
    :attr:`DevelopmentModePolicy.trading_allowed` is ``False``.

    The Execution Gate does **not** raise this — it returns a
    synthetic ``REJECTED`` :class:`ExecutionEvent` with
    ``meta.reason == DEVELOPMENT_MODE_TRADING_BLOCKED`` so the learning
    loop still observes the refusal. This exception is the explicit
    failure mode for callers that want a hard fail (e.g. CLI tools,
    integration tests).
    """


@dataclass(frozen=True, slots=True)
class DevelopmentModePolicy:
    """Operator-gated dual-flag development policy.

    Attributes:
        development_enabled: ``True`` (default) when the learning +
            research surface (Indira trader discovery, Dyon evolution,
            slow-loop critique, patch pipeline) is unblocked. ``False``
            pauses the surface; the underlying HARDEN-04 freeze policy
            and FSM mode gates are still in force regardless.
        trading_allowed: ``False`` (default) when the Execution Gate
            refuses to dispatch any intent to a broker. ``True`` opens
            the gate; the underlying ``AuthorityGuard``, hazard
            throttle, kill-switch, mode-effect table, and HARDEN-05
            chokepoint are still in force regardless.
        mode: Current :class:`SystemMode`. Carried for audit and
            ``POLICY_STATE`` projection; the mode is **not** a
            predicate input under this contract — the operator-driven
            flags are the single authority. ``None`` is permitted for
            offline tests that construct the policy without a
            governance runtime in scope.

    Default construction yields ``DevelopmentModePolicy()`` →
    ``(development_enabled=True, trading_allowed=False, mode=None)`` —
    Indira + Dyon run full-bore, trading refused. The operator opens
    trading with a single audited flip.
    """

    development_enabled: bool = True
    trading_allowed: bool = False
    mode: SystemMode | None = None

    def is_learning_unblocked(self) -> bool:
        """Return ``True`` iff the learning + research surface is
        permitted to emit adaptive mutations (subject to defense-in-
        depth from HARDEN-04 / hazard throttle / kill-switch)."""

        return self.development_enabled

    def is_trading_unblocked(self) -> bool:
        """Return ``True`` iff the Execution Gate is permitted to
        dispatch to a broker (subject to defense-in-depth from the
        AuthorityGuard / hazard throttle / kill-switch / mode-effect
        table / FSM consent envelopes)."""

        return self.trading_allowed

    def to_system_event(
        self,
        *,
        ts_ns: int,
        source: str = "governance.policy.development_mode",
    ) -> SystemEvent:
        """Project the current policy state into a typed
        :class:`SystemEvent` so the audit ledger records the canonical
        contract state on every operator flip."""

        from core.contracts.events import SystemEvent, SystemEventKind

        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError("DevelopmentModePolicy.to_system_event requires non-bool int ts_ns")
        if ts_ns < 0:
            raise ValueError("DevelopmentModePolicy.to_system_event requires non-negative ts_ns")
        if not isinstance(source, str) or not source:
            raise ValueError("DevelopmentModePolicy.to_system_event requires non-empty source")

        return SystemEvent(
            ts_ns=ts_ns,
            sub_kind=SystemEventKind.POLICY_STATE,
            source=source,
            payload={
                "policy": "DevelopmentModePolicy",
                "version": POLICY_VERSION,
                "development_enabled": ("true" if self.development_enabled else "false"),
                "trading_allowed": ("true" if self.trading_allowed else "false"),
                "mode": self.mode.name if self.mode is not None else "",
                "learning_unblocked": ("true" if self.is_learning_unblocked() else "false"),
                "trading_unblocked": ("true" if self.is_trading_unblocked() else "false"),
            },
            produced_by_engine="governance",
            proposed=False,
        )


# ---------------------------------------------------------------------------
# Convenience predicates over an optional policy reference.
#
# Adaptive engines and the Execution Gate carry the policy as an
# optional field so existing offline tests (which construct emitters
# without a governance runtime in scope) keep working. ``None`` is the
# *migration sentinel* meaning "no policy wired here" — it resolves to
# fail-open at both gates so pre-PR-DEV-A offline harness flows are
# unchanged. Production wiring (the cockpit / bootstrap_kernel)
# explicitly injects a real :class:`DevelopmentModePolicy` with the
# operator's defaults (``development_enabled=True``,
# ``trading_allowed=False``); this is the only place the
# fail-closed-for-trading default actually closes the gate. The
# pattern matches :class:`LearningEvolutionFreezePolicy` (HARDEN-04)
# which uses the same ``None``-means-migration-sentinel convention.
# ---------------------------------------------------------------------------


def is_learning_unblocked(policy: DevelopmentModePolicy | None) -> bool:
    """Return ``True`` iff the learning surface is permitted.

    ``None`` is the migration sentinel and resolves to ``True`` so
    pre-PR-DEV-A offline tests that do not construct a policy retain
    their previous behaviour. Production wires a real policy with
    ``development_enabled=True`` at boot.
    """

    if policy is None:
        return True
    return policy.is_learning_unblocked()


def is_trading_unblocked(policy: DevelopmentModePolicy | None) -> bool:
    """Return ``True`` iff the Execution Gate is permitted to dispatch.

    ``None`` is the migration sentinel and resolves to ``True`` so
    pre-PR-DEV-A offline tests that do not construct a policy retain
    their previous behaviour. Production wires a real policy with
    ``trading_allowed=False`` at boot so the runtime gate is closed
    until the operator explicitly flips it via
    ``POST /api/operator/trading-allowed`` (or boot-time
    ``DIXVISION_TRADING_ALLOWED=true``). The cockpit boot is the
    *single* place that enforces the fail-closed-for-trading default.
    """

    if policy is None:
        return True
    return policy.is_trading_unblocked()


def assert_trading_unblocked(
    policy: DevelopmentModePolicy | None,
) -> None:
    """Raise :class:`DevelopmentModeTradingBlockedError` if trading is
    refused under the supplied policy.

    ``None`` is the migration sentinel and is treated as permitted
    (matches :func:`is_trading_unblocked`). The runtime Execution Gate
    does **not** use this helper — it emits a typed ``ExecutionEvent``
    so the learning loop still observes the refusal; this helper is
    for CLI tools and integration tests that want a hard fail.
    """

    if not is_trading_unblocked(policy):
        raise DevelopmentModeTradingBlockedError(
            "DevelopmentModePolicy.trading_allowed is False; the Execution "
            "Gate is closed. Flip via "
            "`POST /api/operator/trading-allowed {enabled: true}` or boot "
            "with `DIXVISION_TRADING_ALLOWED=true`."
        )
