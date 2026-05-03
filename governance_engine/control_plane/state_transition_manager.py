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

from core.coherence.system_intent import (
    INTENT_KEY_FOCUS,
    INTENT_KEY_HORIZON,
    INTENT_KEY_OBJECTIVE,
    INTENT_KEY_REASON,
    INTENT_KEY_REQUESTOR,
    INTENT_KEY_RISK_MODE,
    INTENT_KEY_VERSION,
    INTENT_TRANSITION_KIND,
    SYSTEM_INTENT_VERSION,
    encode_focus,
)
from core.contracts.governance import (
    IntentTransitionDecision,
    IntentTransitionRequest,
    ModeTransitionDecision,
    ModeTransitionRequest,
    SystemMode,
)
from core.contracts.operator_consent import (
    OperatorConsent,
    OperatorConsentValidator,
    edge_requires_consent,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_engine import PolicyEngine
from governance_engine.control_plane.promotion_gates import PromotionGates

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
        promotion_gates: PromotionGates | None = None,
        consent_validator: OperatorConsentValidator | None = None,
    ) -> None:
        self._policy = policy
        self._ledger = ledger
        self._mode = initial_mode
        self._lock = Lock()
        # Reviewer #4 finding 4 -- hash-anchored promotion-gate check.
        # Optional for backwards compatibility with existing call sites
        # (tests, harnesses) that don't yet pass a PromotionGates. When
        # ``None`` the manager skips the gate check entirely; production
        # call sites must pass an instance so the gate is enforced.
        self._promotion_gates = promotion_gates
        # Hardening-S1 item 8 — typed OperatorConsent envelope check.
        # When ``None`` the manager auto-constructs a fresh validator;
        # production wiring should pass a singleton so the nonce ring
        # is shared across the whole process. Backward compatibility:
        # call sites that don't pass consent on edges that require it
        # are rejected with ``CONSENT_MISSING`` — the manager never
        # silently downgrades to the legacy bool gate for those edges.
        self._consent_validator = (
            consent_validator
            if consent_validator is not None
            else OperatorConsentValidator()
        )

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
                consent=request.consent,
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

            # Hardening-S1 item 8 — typed OperatorConsent envelope.
            # Edges in CONSENT_REQUIRED_EDGES (today: SAFE → PAPER and
            # LIVE → AUTO) refuse to advance without a fresh,
            # policy-bound, replay-protected OperatorConsent envelope.
            # The check runs *after* FSM legality (so we never validate
            # an illegal edge) and *before* the promotion-gates check
            # (so the consent rejection surfaces ahead of any
            # downstream policy code).
            if edge_requires_consent(prev, normalised.target_mode):
                consent_obj = normalised.consent
                if consent_obj is not None and not isinstance(
                    consent_obj, OperatorConsent
                ):
                    rejection_payload = {
                        "requestor": normalised.requestor,
                        "prev_mode": prev.name,
                        "target_mode": normalised.target_mode.name,
                        "reason": normalised.reason,
                        "rejection_code": "CONSENT_TYPE_INVALID",
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
                        rejection_code="CONSENT_TYPE_INVALID",
                        ledger_seq=entry.seq,
                    )
                consent_envelope: OperatorConsent | None = consent_obj  # type: ignore[assignment]
                consent_result = self._consent_validator.validate(
                    consent=consent_envelope,
                    request_ts_ns=normalised.ts_ns,
                    prev_mode=prev,
                    target_mode=normalised.target_mode,
                    live_policy_hash=self._policy.table_hash,
                )
                if not consent_result.ok:
                    rejection_payload = {
                        "requestor": normalised.requestor,
                        "prev_mode": prev.name,
                        "target_mode": normalised.target_mode.name,
                        "reason": normalised.reason,
                        "rejection_code": consent_result.code,
                        "detail": consent_result.detail,
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
                        rejection_code=consent_result.code,
                        ledger_seq=entry.seq,
                    )
                # Consent accepted — write a paired audit row before
                # the MODE_TRANSITION row so replay tools can see the
                # consent → transition pairing without parsing the
                # transition row's payload.
                self._ledger.append(
                    ts_ns=normalised.ts_ns,
                    kind="OPERATOR_CONSENT_ACCEPTED",
                    payload={
                        "operator_id": consent_envelope.operator_id,  # type: ignore[union-attr]
                        "mode_from": prev.name,
                        "mode_to": normalised.target_mode.name,
                        "policy_hash": consent_envelope.policy_hash,  # type: ignore[union-attr]
                        "nonce": consent_envelope.nonce,  # type: ignore[union-attr]
                        "consent_ts_ns": str(consent_envelope.ts_ns),  # type: ignore[union-attr]
                    },
                )
                # A valid OperatorConsent envelope is strictly
                # stronger than the legacy ``operator_authorized``
                # bool — refresh the normalised request so the
                # downstream policy gate sees authorised=True without
                # the call site having to set both fields.
                normalised = ModeTransitionRequest(
                    ts_ns=normalised.ts_ns,
                    requestor=normalised.requestor,
                    current_mode=prev,
                    target_mode=normalised.target_mode,
                    reason=normalised.reason,
                    operator_authorized=True,
                    consent=normalised.consent,
                )

            # Reviewer #4 finding 4 -- hash-anchored promotion-gate
            # enforcement. CANARY / LIVE / AUTO refuse to enter unless the
            # live ``docs/promotion_gates.yaml`` hash matches the hash that
            # was bound at SHADOW entry. Pre-FSM-state-flip and pre-policy
            # so the rejection_code surfaces the gate violation, not a
            # downstream policy code that would obscure the real cause.
            if self._promotion_gates is not None:
                gate_ok, gate_code = self._promotion_gates.check(
                    normalised.target_mode.name
                )
                if not gate_ok:
                    rejection_payload = {
                        "requestor": normalised.requestor,
                        "prev_mode": prev.name,
                        "target_mode": normalised.target_mode.name,
                        "reason": normalised.reason,
                        "rejection_code": gate_code,
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
                        rejection_code=gate_code,
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

            # Reviewer #4 finding 4 -- bind the live promotion-gates
            # file hash on every SHADOW entry. ``bind`` writes a
            # ``PROMOTION_GATES_BOUND`` ledger row immediately after
            # the ``MODE_TRANSITION`` row so replay sees them as a
            # paired commit. Re-entry into SHADOW (after a
            # de-escalation) overwrites the bound hash, which is the
            # documented "edit the file, restart the clock" path.
            if (
                self._promotion_gates is not None
                and normalised.target_mode is SystemMode.SHADOW
            ):
                self._promotion_gates.bind(
                    ts_ns=normalised.ts_ns,
                    requestor=normalised.requestor,
                )

            return ModeTransitionDecision(
                ts_ns=normalised.ts_ns,
                approved=True,
                prev_mode=prev,
                new_mode=normalised.target_mode,
                reason=normalised.reason,
                rejection_code="",
                ledger_seq=entry.seq,
            )


    # ------------------------------------------------------------------
    # Intent transitions (Phase 6.T1d, INV-38)
    # ------------------------------------------------------------------

    def propose_intent(
        self, request: IntentTransitionRequest
    ) -> IntentTransitionDecision:
        """Commit (or reject) an operator-set System Intent.

        ``StateTransitionManager.propose_intent`` is the **only** writer
        of ``INTENT_TRANSITION`` ledger rows (INV-38). Validation is
        intentionally narrow:

        * ``objective`` / ``risk_mode`` / ``horizon`` must be valid enum
          values — these are typed at the contracts boundary, so the
          arrival of an invalid value here means the operator bridge
          built a malformed request and we ledger a rejection.
        * ``focus`` is preserved in order; empty is permitted (the
          operator may unset focus).

        Mode is unaffected — intent is the strategic axis above the Mode
        FSM. The ledger row is appended under the manager's lock so
        intent and mode rows never interleave non-deterministically.
        """

        with self._lock:
            try:
                objective = request.objective
                risk_mode = request.risk_mode
                horizon = request.horizon
                # Touching the .name property forces an enum validity
                # check without changing the value, so a malformed enum
                # arriving here (e.g. from ``object.__setattr__``) is
                # caught and rejected rather than silently committed.
                _ = (objective.name, risk_mode.name, horizon.name)
            except (AttributeError, ValueError):
                rejection_payload = {
                    INTENT_KEY_REQUESTOR: request.requestor,
                    "rejection_code": "INTENT_INVALID_ENUM",
                    INTENT_KEY_VERSION: SYSTEM_INTENT_VERSION,
                }
                entry = self._ledger.append(
                    ts_ns=request.ts_ns,
                    kind="INTENT_TRANSITION_REJECTED",
                    payload=rejection_payload,
                )
                return IntentTransitionDecision(
                    ts_ns=request.ts_ns,
                    approved=False,
                    objective=request.objective,
                    risk_mode=request.risk_mode,
                    horizon=request.horizon,
                    focus=tuple(request.focus),
                    reason=request.reason,
                    rejection_code="INTENT_INVALID_ENUM",
                    ledger_seq=entry.seq,
                )

            focus = tuple(request.focus)
            approval_payload = {
                INTENT_KEY_REQUESTOR: request.requestor,
                INTENT_KEY_OBJECTIVE: objective.value,
                INTENT_KEY_RISK_MODE: risk_mode.value,
                INTENT_KEY_HORIZON: horizon.value,
                INTENT_KEY_FOCUS: encode_focus(focus),
                INTENT_KEY_REASON: request.reason,
                INTENT_KEY_VERSION: SYSTEM_INTENT_VERSION,
            }
            entry = self._ledger.append(
                ts_ns=request.ts_ns,
                kind=INTENT_TRANSITION_KIND,
                payload=approval_payload,
            )
            return IntentTransitionDecision(
                ts_ns=request.ts_ns,
                approved=True,
                objective=objective,
                risk_mode=risk_mode,
                horizon=horizon,
                focus=focus,
                reason=request.reason,
                rejection_code="",
                ledger_seq=entry.seq,
            )


__all__ = ["StateTransitionManager"]
