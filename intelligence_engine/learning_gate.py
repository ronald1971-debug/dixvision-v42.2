"""PR-DEV-B — Indira learning gate (Operator Master Development Mode).

This module is the **single point of consultation** for Indira-side
runtime code that needs to ask "is the operator currently permitting
the learning + research surface to emit?". It wraps
:class:`core.contracts.development_mode.DevelopmentModePolicy` so the
intelligence engine never reaches into the policy directly — that
keeps Indira free of any direct :class:`SystemMode` dependence (B31
already enforces this at lint time) and free of any direct
governance-tier import (L1/L2/L3 already enforce that).

Architectural contract
~~~~~~~~~~~~~~~~~~~~~~

PR-DEV-A inverted the default safety stance:

* ``development_enabled`` defaults to ``True`` — Indira (and Dyon)
  run full-bore at boot regardless of :class:`SystemMode`.
* ``trading_allowed`` defaults to ``False`` — the Execution Gate
  refuses to dispatch to any broker until the operator explicitly
  flips the flag.

PR-DEV-B pins this for Indira's signal-emission surface specifically:
the :class:`IntelligenceEngine` consults a single
:class:`LearningGate` reference. The gate is **open** by default
(``policy_supplier`` resolves to ``None`` — the same migration
sentinel pattern :mod:`core.contracts.development_mode` uses); when
the operator flips ``DevelopmentModePolicy.development_enabled`` to
``False`` (via ``POST /api/operator/development-mode {enabled: false}``
or the boot env var ``DIXVISION_DEVELOPMENT_MODE=false``) the gate
closes and the engine returns an empty signal tuple.

The gate is **not** a replacement for HARDEN-04, the hazard throttle,
or the kill-switch — those remain active as defense-in-depth on the
adaptive-mutation surface (closed learning loop, structural evolution
loop, slow-loop critique, patch pipeline). PR-DEV-B specifically
targets the *signal-emission* surface so the operator can pause new
signals without tearing down the whole engine.

Audit
~~~~~

The gate exposes :meth:`audit_payload` so callers can emit a typed
``LEARNING_GATE_CLOSED`` audit row whenever a tick is short-circuited.
The payload mirrors the canonical ``POLICY_STATE`` projection so
offline replay validators can correlate ``LEARNING_GATE_CLOSED`` rows
with the ``OPERATOR_DEVELOPMENT_MODE_CHANGED`` row that produced the
state change.

Why a supplier callable
~~~~~~~~~~~~~~~~~~~~~~~

The gate accepts a ``Callable[[], DevelopmentModePolicy | None]``
rather than a static policy reference. This matches the
:class:`LearningEvolutionFreezePolicy` supplier pattern used by the
HARDEN-04 freeze loops and lets the operator's
``/api/operator/development-mode`` route mutate the underlying
:attr:`_State.development_mode_policy` without re-wiring the engine.
The gate re-reads on every consultation; there is no caching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.contracts.development_mode import (
    POLICY_VERSION,
    DevelopmentModePolicy,
    is_learning_unblocked,
)

if TYPE_CHECKING:  # pragma: no cover — type-only seam
    from collections.abc import Mapping

__all__ = [
    "LEARNING_GATE_CLOSED_REASON",
    "LearningGate",
]


# PR-DEV-B — canonical reason string the engine attaches to a
# short-circuit audit row when the gate is closed. Receivers and
# replay validators look for this exact value; do not change without
# bumping :data:`core.contracts.development_mode.POLICY_VERSION`.
LEARNING_GATE_CLOSED_REASON: str = "learning_gate_closed_by_operator"


@dataclass(frozen=True, slots=True)
class LearningGate:
    """Indira-side wrapper over :class:`DevelopmentModePolicy`.

    Attributes:
        policy_supplier: A zero-argument callable that returns the
            current :class:`DevelopmentModePolicy` (or ``None`` for
            the fail-open migration sentinel). Re-read on every
            consultation; there is no caching. ``None`` resolves to
            "no policy wired" which means the gate is open — this
            preserves backward compatibility with pre-PR-DEV-B
            offline tests that construct :class:`IntelligenceEngine`
            without a governance runtime in scope.

    Default construction yields ``LearningGate()`` →
    ``policy_supplier`` returning ``None`` → :meth:`is_open` returns
    ``True``. Production wiring (the cockpit / bootstrap_kernel)
    explicitly injects a supplier that reads
    ``STATE.get_development_mode_policy()`` so the operator's flip
    via the audited route propagates without re-wiring the engine.
    """

    policy_supplier: Callable[[], DevelopmentModePolicy | None] = lambda: None

    def current_policy(self) -> DevelopmentModePolicy | None:
        """Re-read the policy from the supplier. Pure pass-through."""

        return self.policy_supplier()

    def is_open(self) -> bool:
        """Return ``True`` iff Indira is permitted to emit signals.

        The migration sentinel (``policy_supplier`` returning
        ``None``) resolves to **open** so pre-PR-DEV-B offline tests
        that do not inject a real policy retain their previous
        unconditional-emit behaviour.
        """

        return is_learning_unblocked(self.current_policy())

    def is_closed(self) -> bool:
        """Convenience inverse of :meth:`is_open`."""

        return not self.is_open()

    def audit_payload(self) -> Mapping[str, str]:
        """Return a typed payload for a ``LEARNING_GATE_CLOSED`` row.

        The payload mirrors the
        :meth:`DevelopmentModePolicy.to_system_event` projection so
        an offline replay validator can correlate a gate-closure
        row with the ``OPERATOR_DEVELOPMENT_MODE_CHANGED`` row that
        produced the state change. Returned values are always
        strings (the canonical ``SystemEvent.payload`` shape).
        """

        policy = self.current_policy()
        if policy is None:
            return {
                "policy": "DevelopmentModePolicy",
                "version": POLICY_VERSION,
                "reason": LEARNING_GATE_CLOSED_REASON,
                "development_enabled": "true",
                "trading_allowed": "true",
                "mode": "",
                "supplier": "sentinel",
            }
        return {
            "policy": "DevelopmentModePolicy",
            "version": POLICY_VERSION,
            "reason": LEARNING_GATE_CLOSED_REASON,
            "development_enabled": ("true" if policy.development_enabled else "false"),
            "trading_allowed": ("true" if policy.trading_allowed else "false"),
            "mode": policy.mode.name if policy.mode is not None else "",
            "supplier": "live",
        }
