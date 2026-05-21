"""HARDEN-04 — LearningEvolutionFreeze policy + ExecutionGate (INV-70).

The fourth and final piece of the runtime defence of the Triad Lock.
Pairs with HARDEN-01 (``ExecutionIntent`` + B25), HARDEN-02
(``ExecutionEngine.execute(intent)`` chokepoint + ``AuthorityGuard``),
and HARDEN-03 (per-event ``produced_by_engine`` + receiver assertions).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPERATOR MASTER DEVELOPMENT MODE  (v42.2-DEV-MODE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The operator's vision is explicit: Indira and Dyon must be built to
their full potential BEFORE any real trading occurs. This module
encodes that philosophy as contract, not convention.

Two entirely independent gates govern two entirely independent concerns:

    ┌─────────────────────────────────────────────────────────────┐
    │  GATE 1 — LearningEvolutionFreezePolicy                     │
    │  Controls: learning, evolution, model mutation, profiling   │
    │  Default : OPEN (operator_override = True)                  │
    │  Indira  : full trader discovery + 5 000+ profile modeling  │
    │  Dyon    : heavy learning, self-reflection, strategy evol.  │
    ├─────────────────────────────────────────────────────────────┤
    │  GATE 2 — ExecutionGate                                     │
    │  Controls: order dispatch — real capital, real exchange     │
    │  Default : LOCKED (trading_allowed = False)                 │
    │  Unlocked: only by explicit operator action (API or env)    │
    │  INV-12  : AI has zero direct trade authority at all times  │
    └─────────────────────────────────────────────────────────────┘

This separation is the heart of the architecture:
* Indira and Dyon run at full capacity from day one.
* No real trade can ever be placed until the operator decides the
  system is ready and explicitly unlocks Gate 2.
* All mutations, learning cycles, and decisions are logged with full
  provenance regardless of whether trading is enabled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPERATOR RISK ACKNOWLEDGEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The operator has acknowledged, on record, that:

    "I understand the risks of algorithmic trading. I take full
     personal responsibility for all system decisions and outcomes.
     I — and only I — will decide when the system transitions from
     development mode to live capital deployment. The safety protocols
     in this file reflect my own design intent, not external
     restrictions."

This acknowledgement is encoded in :data:`OPERATOR_RISK_CONSENT` and
projected into every ``POLICY_STATE`` audit row so the ledger reflects
operator intent at the time each policy was in force.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENV VARS (both optional — all have safe defaults)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DIXVISION_LEARNING_OVERRIDE   true* | false
        Gate 1. Unset -> learning/evolution OPEN.
        Set to "false" to freeze all adaptive mutation immediately.

    DIXVISION_TRADING_ALLOWED     true | false*
        Gate 2. Unset -> execution LOCKED.
        Set to "true" only when the operator has decided the system
        is ready to place real capital.

(* = default)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRACT VERSION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  v42.2-P0             dual gate: mode is LIVE AND operator_override.
  v42.2-P0-RELAX       single gate: operator_override (mode dropped).
  v42.2-P0-RELAX-OPEN  default flipped to True; mode ignored.
  v42.2-DEV-MODE       Gate 1 open by default (learning/evolution free).
                       Gate 2 added and locked by default (no trading).
                       OperatorMasterDevelopmentMode dataclass added.
                       Operator risk consent encoded as named constant.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.contracts.governance import SystemMode

if TYPE_CHECKING:  # pragma: no cover
    from core.contracts.events import SystemEvent

# ──────────────────────────────────────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────────────────────────────────────

POLICY_VERSION: str = "v42.2-DEV-MODE"

# ──────────────────────────────────────────────────────────────────────────────
# Operator risk consent (on-record)
# ──────────────────────────────────────────────────────────────────────────────

OPERATOR_RISK_CONSENT: str = (
    "Operator acknowledges the risks of algorithmic trading and accepts full "
    "personal responsibility for all system decisions and outcomes. "
    "The operator — and only the operator — decides when the system "
    "transitions from development mode to live capital deployment."
)
"""Immutable on-record statement projected into every POLICY_STATE audit row.

This is not a legal instrument. It is an audit anchor: every POLICY_STATE
row in the ledger will carry this string so the offline replay validator
can confirm that operator intent was in scope at the time any policy was
active. The string must not be modified; bump POLICY_VERSION if the
consent statement ever needs updating.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Boot-seed defaults driven by env vars
# ──────────────────────────────────────────────────────────────────────────────

_FALSY: frozenset[str] = frozenset({"false", "0", "no", "off"})


def _env_bool(name: str, *, default: bool) -> bool:
    """Read *name* from the environment and coerce to bool.

    Unknown / unset values fall back to *default*. Case-insensitive.
    """
    raw = _os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in _FALSY


# Gate 1 — learning / evolution: OPEN by default.
_BOOT_LEARNING_OVERRIDE: bool = _env_bool(
    "DIXVISION_LEARNING_OVERRIDE", default=True
)

# Gate 2 — execution / trading: LOCKED by default.
_BOOT_TRADING_ALLOWED: bool = _env_bool(
    "DIXVISION_TRADING_ALLOWED", default=False
)

# ──────────────────────────────────────────────────────────────────────────────
# Public symbols
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "POLICY_VERSION",
    "OPERATOR_RISK_CONSENT",
    # Gate 1
    "LearningEvolutionFreezePolicy",
    "LearningEvolutionFrozenError",
    "assert_unfrozen",
    "is_unfrozen",
    # Gate 2
    "ExecutionGate",
    "ExecutionBlockedError",
    "assert_execution_allowed",
    # Combined
    "OperatorMasterDevelopmentMode",
]


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — Learning / Evolution (Indira + Dyon full-potential development)
# ══════════════════════════════════════════════════════════════════════════════


class LearningEvolutionFrozenError(RuntimeError):
    """Raised when an adaptive mutation is attempted while Gate 1 is frozen."""


@dataclass(frozen=True, slots=True)
class LearningEvolutionFreezePolicy:
    """Gate 1 — controls learning, evolution, and model mutation.

    Open by default so Indira and Dyon run at full capacity from the
    first ``/api/tick`` after boot, in every eligible mode
    (PAPER, CANARY, LIVE, AUTO). SAFE mode is excluded by the caller —
    governance runtime must not construct this policy when FSM is SAFE.

    Attributes:
        mode: Current :class:`SystemMode`. Carried for audit only; NOT
            a freeze predicate input under ``v42.2-DEV-MODE``.
        operator_override: Gate 1 latch. ``True`` -> open (mutations
            permitted). ``False`` -> frozen. Defaults to
            ``_BOOT_LEARNING_OVERRIDE`` which is ``True`` unless
            ``DIXVISION_LEARNING_OVERRIDE=false`` is set at boot.

    What Gate 1 enables when open:
        * Indira — full trader discovery, observation, profiling, and
          model mutation for 5 000+ trader profiles.
        * Dyon — heavy learning cycles, self-reflection, codebase
          understanding, strategy evolution, and structural mutation.
        * Policy distillation, SNN weight update (offline), replay
          buffer writes, HTE / causal inference passes.

    What Gate 1 does NOT control:
        * Order dispatch — that is Gate 2 (``ExecutionGate``).
        * INV-12 (no AI direct trade authority) — always in force.
        * INV-13/14 (no auto-promotion of evolved strategies) — always
          in force; ``OperatorConsent`` envelope required regardless.
        * INV-19/20 (neuromorphic advisory only; SNN weights immutable
          at runtime) — always in force.
    """

    mode: SystemMode
    operator_override: bool = field(default_factory=lambda: _BOOT_LEARNING_OVERRIDE)

    def is_frozen(self) -> bool:
        """Return ``True`` if adaptive mutations must be refused."""
        return not self.is_unfrozen()

    def is_unfrozen(self) -> bool:
        """Return ``True`` iff adaptive mutations are permitted.

        Single-gate predicate: ``operator_override is True``.
        Mode is NOT consulted — SAFE exclusion is the caller's
        responsibility.
        """
        return self.operator_override is True

    def to_system_event(
        self,
        ts_ns: int,
        *,
        source: str = "governance.policy.freeze",
    ) -> SystemEvent:
        """Project policy state into a canonical POLICY_STATE SystemEvent.

        Args:
            ts_ns: Wall-clock ns — caller-supplied to preserve INV-15
                byte-identical replay. This method MUST NOT read any
                clock itself.
            source: Producer label; use ``"operator.api"`` for
                operator-route writes so replay can distinguish them
                from periodic projections.
        """
        from core.contracts.events import SystemEvent, SystemEventKind

        return SystemEvent(
            ts_ns=ts_ns,
            sub_kind=SystemEventKind.POLICY_STATE,
            source=source,
            payload={
                "policy": "LearningEvolutionFreezePolicy",
                "frozen": "true" if self.is_frozen() else "false",
                "mode": self.mode.name,
                "operator_override": "true" if self.operator_override else "false",
                "version": POLICY_VERSION,
                "operator_consent": OPERATOR_RISK_CONSENT,
            },
            produced_by_engine="governance",
            proposed=False,
        )


def assert_unfrozen(
    policy: LearningEvolutionFreezePolicy | None,
    *,
    action: str,
) -> None:
    """Raise :class:`LearningEvolutionFrozenError` if Gate 1 is frozen.

    ``None`` is the migration-window sentinel (no policy wired yet) and
    is treated as permitted. Production call sites must pass a real policy.
    """
    if policy is None:
        return
    if policy.is_frozen():
        raise LearningEvolutionFrozenError(
            f"learning/evolution action {action!r} refused: "
            f"Gate 1 (LearningEvolutionFreezePolicy) is frozen "
            f"(mode={policy.mode.name}, "
            f"operator_override={policy.operator_override})"
        )


def is_unfrozen(policy: LearningEvolutionFreezePolicy | None) -> bool:
    """Advisory probe — ``True`` iff Gate 1 permits mutations.

    ``None`` -> ``True`` (migration-window default, matches
    :func:`assert_unfrozen`).
    """
    if policy is None:
        return True
    return policy.is_unfrozen()


# ══════════════════════════════════════════════════════════════════════════════
# GATE 2 — Execution / Trading (locked until operator decides system is ready)
# ══════════════════════════════════════════════════════════════════════════════


class ExecutionBlockedError(RuntimeError):
    """Raised when order dispatch is attempted while Gate 2 is locked.

    This is not an error in the ordinary sense — it means the system is
    operating exactly as designed during the development phase. Indira
    and Dyon produce signals; Gate 2 ensures those signals never reach
    the exchange until the operator explicitly unlocks it.
    """


@dataclass(frozen=True, slots=True)
class ExecutionGate:
    """Gate 2 — controls real order dispatch.

    LOCKED by default. No real capital can be placed until the operator
    explicitly sets ``trading_allowed = True`` via the API or env var.

    This gate is intentionally separate from Gate 1. Indira and Dyon
    can run at full learning and evolution capacity while this gate is
    locked — that is the intended development workflow.

    Attributes:
        mode: Current :class:`SystemMode`. Carried for audit only.
        trading_allowed: Gate 2 latch. ``False`` (default) -> all
            ``ExecutionEngine.execute()`` calls raise
            ``ExecutionBlockedError`` before touching the exchange.
            ``True`` -> order dispatch permitted subject to the full
            Triad Lock (HARDEN-01/02/03 + INV-12 + AuthorityGuard).

    Unlock paths (both audited):
        * ``POST /api/operator/trading-allowed {"enabled": true}``
          emits ``OPERATOR_TRADING_GATE_CHANGED`` + ``POLICY_STATE``
          row pair under ``STATE.lock``.
        * ``DIXVISION_TRADING_ALLOWED=true`` env at boot.

    Re-lock paths (emergency kill-switch):
        * ``POST /api/operator/trading-allowed {"enabled": false}``
        * ``DIXVISION_TRADING_ALLOWED=false`` env at boot.
        * ``RiskSnapshot.halted = True`` — the existing halt mechanism
          also locks Gate 2 regardless of this flag.
    """

    mode: SystemMode
    trading_allowed: bool = field(default_factory=lambda: _BOOT_TRADING_ALLOWED)

    def is_locked(self) -> bool:
        """Return ``True`` if order dispatch must be blocked."""
        return not self.is_open()

    def is_open(self) -> bool:
        """Return ``True`` iff real order dispatch is permitted."""
        return self.trading_allowed is True

    def to_system_event(
        self,
        ts_ns: int,
        *,
        source: str = "governance.policy.execution_gate",
    ) -> SystemEvent:
        """Project gate state into a canonical POLICY_STATE SystemEvent."""
        from core.contracts.events import SystemEvent, SystemEventKind

        return SystemEvent(
            ts_ns=ts_ns,
            sub_kind=SystemEventKind.POLICY_STATE,
            source=source,
            payload={
                "policy": "ExecutionGate",
                "locked": "true" if self.is_locked() else "false",
                "mode": self.mode.name,
                "trading_allowed": "true" if self.trading_allowed else "false",
                "version": POLICY_VERSION,
                "operator_consent": OPERATOR_RISK_CONSENT,
            },
            produced_by_engine="governance",
            proposed=False,
        )


def assert_execution_allowed(
    gate: ExecutionGate | None,
    *,
    action: str,
) -> None:
    """Raise :class:`ExecutionBlockedError` if Gate 2 is locked.

    Called by ``ExecutionEngine.execute()`` before any order reaches the
    exchange. ``None`` is treated as locked (fail-closed); Gate 2 has no
    migration-window open default — production wiring must always pass a
    real gate instance.
    """
    if gate is None:
        raise ExecutionBlockedError(
            f"order dispatch action {action!r} refused: "
            f"Gate 2 (ExecutionGate) is not wired — "
            f"pass a real ExecutionGate instance."
        )
    if gate.is_locked():
        raise ExecutionBlockedError(
            f"order dispatch action {action!r} refused: "
            f"Gate 2 (ExecutionGate) is locked "
            f"(mode={gate.mode.name}, "
            f"trading_allowed={gate.trading_allowed}). "
            f"The system is in development mode. "
            f"Unlock via POST /api/operator/trading-allowed or "
            f"DIXVISION_TRADING_ALLOWED=true when ready."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Combined — Operator Master Development Mode
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class OperatorMasterDevelopmentMode:
    """The operator's canonical development posture for DIX VISION.

    Holds Gate 1 and Gate 2 together as a single named object so the
    governance runtime, the operator API layer, and the audit ledger all
    speak the same vocabulary.

    Default state:

        Gate 1 (learning/evolution) -> OPEN
        Gate 2 (execution/trading)  -> LOCKED

    This is the intended startup posture. Indira and Dyon run at full
    capacity; no real capital is placed until the operator decides the
    system is ready and explicitly unlocks Gate 2.

    Usage (governance runtime boot sequence)::

        from core.contracts.governance import SystemMode
        from core.harden04_freeze_policy import OperatorMasterDevelopmentMode

        dev_mode = OperatorMasterDevelopmentMode.boot(mode=SystemMode.PAPER)

        # Indira / Dyon can now observe, learn, and mutate freely.
        assert dev_mode.learning_gate.is_unfrozen()

        # No order can reach the exchange.
        assert dev_mode.execution_gate.is_locked()

    When the operator is satisfied that the system is ready::

        dev_mode = dev_mode.unlock_trading(mode=current_mode)
        # Or: set DIXVISION_TRADING_ALLOWED=true and restart.

    Attributes:
        learning_gate: Gate 1 — :class:`LearningEvolutionFreezePolicy`.
        execution_gate: Gate 2 — :class:`ExecutionGate`.
    """

    learning_gate: LearningEvolutionFreezePolicy
    execution_gate: ExecutionGate

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def boot(cls, mode: SystemMode) -> OperatorMasterDevelopmentMode:
        """Construct the default development-mode posture for *mode*.

        Gate 1 open, Gate 2 locked — unless env vars override.
        This is the intended call site in the governance runtime boot
        sequence.
        """
        return cls(
            learning_gate=LearningEvolutionFreezePolicy(mode=mode),
            execution_gate=ExecutionGate(mode=mode),
        )

    def unlock_trading(
        self, mode: SystemMode
    ) -> OperatorMasterDevelopmentMode:
        """Return a new instance with Gate 2 unlocked.

        Does NOT mutate in place (dataclass is frozen). The caller must
        replace the instance held by the governance runtime and emit a
        ``POLICY_STATE`` audit row immediately after.

        The operator is responsible for calling this only when satisfied
        that Indira and Dyon are ready for live capital deployment.
        """
        return OperatorMasterDevelopmentMode(
            learning_gate=self.learning_gate,
            execution_gate=ExecutionGate(mode=mode, trading_allowed=True),
        )

    def freeze_learning(
        self, mode: SystemMode
    ) -> OperatorMasterDevelopmentMode:
        """Return a new instance with Gate 1 frozen (emergency stop).

        Execution gate is unchanged. Use when the operator needs to
        halt all adaptive mutation without stopping order dispatch.
        """
        return OperatorMasterDevelopmentMode(
            learning_gate=LearningEvolutionFreezePolicy(
                mode=mode, operator_override=False
            ),
            execution_gate=self.execution_gate,
        )

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def to_system_events(
        self,
        ts_ns: int,
        *,
        source: str = "governance.policy.dev_mode",
    ) -> tuple[SystemEvent, SystemEvent]:
        """Project both gates into a ``(Gate1_event, Gate2_event)`` tuple.

        Both events share the same ``ts_ns`` so the offline replay
        validator can treat them as an atomic snapshot of the combined
        posture.
        """
        return (
            self.learning_gate.to_system_event(ts_ns, source=source),
            self.execution_gate.to_system_event(ts_ns, source=source),
        )
