"""GOV-CP-03 — State Transition Manager (Mode FSM).

The **only writer** of system mode (Build Compiler Spec §6 / §7;
``manifest.md`` §0.5 GOV-CP-03). All other modules — including the
dashboard and the operator bridge — *propose* transitions; only this
manager flips the mode bit, and only after the policy engine has
approved and the authority ledger has accepted the row.

Legal edges (Build Compiler Spec §7):

* Forward ratchet: ``SAFE → PAPER → SHADOW → CANARY → LIVE → AUTO``
  (one step at a time; LIVE and AUTO additionally require operator
  authorisation, gated upstream by :class:`PolicyEngine`).
* De-escalation: any backward step in the chain is permitted.
* Emergency: any state may transition to ``LOCKED`` (kill).
* Recovery: ``LOCKED → SAFE`` is the **only** way out of LOCKED.

Determinism contract: given the same sequence of ``propose`` calls,
``current_mode`` evolves identically and the ledger contains the
same rows in the same order (INV-15).
"""

from __future__ import annotations

from threading import Lock

from core.contracts.governance import (
    ModeTransitionDecision,
    ModeTransitionRequest,
    SystemMode,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_engine import PolicyEngine

# The forward ratchet, in order. Index gaps mean: SHADOW → LIVE is
# illegal (must go SHADOW → CANARY → LIVE).
_FORWARD_CHAIN: tuple[SystemMode, ...] = (
    SystemMode.SAFE,
    SystemMode.PAPER,
    SystemMode.SHADOW,
    SystemMode.CANARY,
    SystemMode.LIVE,
    SystemMode.AUTO,
)


def _chain_index(mode: SystemMode) -> int:
    """Return the index of ``mode`` in the forward chain, or ``-1``."""

    for idx, m in enumerate(_FORWARD_CHAIN):
        if m is mode:
            return idx
    return -1


def _is_legal_edge(prev: SystemMode, target: SystemMode) -> tuple[bool, str]:
    """Pure FSM check. Returns ``(legal, rejection_code)``."""

    if prev is target:
        return False, "FSM_NO_OP"

    if target is SystemMode.LOCKED:
        return True, ""  # any state can be locked

    if prev is SystemMode.LOCKED:
        if target is SystemMode.SAFE:
            return True, ""
        return False, "FSM_LOCKED_ONLY_TO_SAFE"

    p = _chain_index(prev)
    t = _chain_index(target)
    if p == -1 or t == -1:
        return False, "FSM_UNKNOWN_MODE"

    if t == p + 1:
        return True, ""  # one-step forward
    if t < p:
        return True, ""  # any-step backward / de-escalation
    return False, "FSM_FORWARD_SKIP"


class StateTransitionManager:
    name: str = "state_transition_manager"
    spec_id: str = "GOV-CP-03"

    def __init__(
        self,
        *,
        policy: PolicyEngine,
        ledger: LedgerAuthorityWriter,
        initial_mode: SystemMode = SystemMode.SAFE,
    ) -> None:
        self._policy = policy
        self._ledger = ledger
        self._mode = initial_mode
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_mode(self) -> SystemMode:
        with self._lock:
            return self._mode

    def propose(
        self, request: ModeTransitionRequest
    ) -> ModeTransitionDecision:
        """Apply the full transition pipeline atomically.

        Order (deterministic):

        1. Resolve current mode under the manager's lock.
        2. FSM legality check.
        3. Policy gate (operator authorisation, etc.).
        4. Append to authority ledger (only if approved).
        5. Flip the mode bit.
        """

        with self._lock:
            prev = self._mode

            normalised = ModeTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                current_mode=prev,
                target_mode=request.target_mode,
                reason=request.reason,
                operator_authorized=request.operator_authorized,
            )

            legal, fsm_code = _is_legal_edge(prev, normalised.target_mode)
            if not legal:
                rejection_payload = {
                    "requestor": normalised.requestor,
                    "prev_mode": prev.name,
                    "target_mode": normalised.target_mode.name,
                    "reason": normalised.reason,
                    "rejection_code": fsm_code,
                }
                entry = self._ledger.append(
                    ts_ns=normalised.ts_ns,
                    kind="MODE_TRANSITION_REJECTED",
                    payload=rejection_payload,
                )
                return ModeTransitionDecision(
                    ts_ns=normalised.ts_ns,
                    approved=False,
                    prev_mode=prev,
                    new_mode=prev,
                    reason=normalised.reason,
                    rejection_code=fsm_code,
                    ledger_seq=entry.seq,
                )

            policy_ok, policy_code = self._policy.permit_mode_transition(
                normalised
            )
            if not policy_ok:
                rejection_payload = {
                    "requestor": normalised.requestor,
                    "prev_mode": prev.name,
                    "target_mode": normalised.target_mode.name,
                    "reason": normalised.reason,
                    "rejection_code": policy_code,
                }
                entry = self._ledger.append(
                    ts_ns=normalised.ts_ns,
                    kind="MODE_TRANSITION_REJECTED",
                    payload=rejection_payload,
                )
                return ModeTransitionDecision(
                    ts_ns=normalised.ts_ns,
                    approved=False,
                    prev_mode=prev,
                    new_mode=prev,
                    reason=normalised.reason,
                    rejection_code=policy_code,
                    ledger_seq=entry.seq,
                )

            approval_payload = {
                "requestor": normalised.requestor,
                "prev_mode": prev.name,
                "new_mode": normalised.target_mode.name,
                "reason": normalised.reason,
                "operator_authorized": (
                    "true" if normalised.operator_authorized else "false"
                ),
            }
            entry = self._ledger.append(
                ts_ns=normalised.ts_ns,
                kind="MODE_TRANSITION",
                payload=approval_payload,
            )
            self._mode = normalised.target_mode
            return ModeTransitionDecision(
                ts_ns=normalised.ts_ns,
                approved=True,
                prev_mode=prev,
                new_mode=normalised.target_mode,
                reason=normalised.reason,
                rejection_code="",
                ledger_seq=entry.seq,
            )


__all__ = ["StateTransitionManager"]
