"""Hardening-S1 item 8 — typed Operator Consent envelope.

Today the Mode FSM gates forward transitions on a free-floating
``operator_authorized: bool`` flag carried on
:class:`ModeTransitionRequest`. That bool says nothing about *which*
operator authorised, *which* edge they consented to, *under which
policy version*, and provides no replay protection — anything that
flips it to ``True`` at the right moment passes the gate.

This module introduces a **typed consent envelope** that binds the
operator's approval to a specific edge under a specific policy
version, with replay protection via a per-operator nonce window:

* :class:`OperatorConsent` — frozen dataclass carrying
  ``mode_from`` / ``mode_to`` / ``policy_hash`` / ``operator_id`` /
  ``ts_ns`` / ``nonce``.
* :class:`OperatorConsentValidator` — pure validator that
  cross-checks an envelope against the actual transition request,
  the live policy hash, and a bounded nonce window.

The two **required-consent** edges (per the architecture critique
item 8) are:

* ``SAFE → PAPER``   — first execution surface; real loss exposure
  begins (paper, but operator authorises trading at all).
* ``LIVE → AUTO``    — operator-removed autonomy; AI now acts
  without per-trade ratification.

Other forward edges (``PAPER → SHADOW``, ``SHADOW → CANARY``,
``CANARY → LIVE``) keep their existing operator-authorised gate
plus the hash-anchored promotion-gates check (PR #124). They are
candidates to upgrade to typed consent in a follow-up PR.

This module is contract-only — it has no governance-engine,
state-transition-manager, or policy-engine dependencies, so it can
be imported anywhere on the read side (dashboard, CLI, tests)
without dragging the entire control plane along.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

from core.contracts.governance import SystemMode

# ---------------------------------------------------------------------------
# Validation parameters
# ---------------------------------------------------------------------------

# Maximum age, in nanoseconds, of an :class:`OperatorConsent` envelope
# relative to the request's ``ts_ns``. 60 seconds is intentionally
# loose — operators construct consent through the dashboard, then
# submit a transition request; we want network jitter + UI lag to
# fit comfortably, but not so loose that a leaked envelope lives
# forever. 60 s × 10⁹ ns/s.
CONSENT_FRESHNESS_WINDOW_NS: Final[int] = 60 * 1_000_000_000

# Maximum number of recently-seen nonces retained per validator. A
# nonce that lands inside this window is rejected as a replay; older
# nonces fall out and could in principle be reused, but the
# freshness window above already rejects them on staleness grounds.
NONCE_RING_CAPACITY: Final[int] = 1024

# The two consent-required edges per Hardening-S1 item 8. ``frozenset``
# of ``(SystemMode, SystemMode)`` tuples; expanding the set is an
# authority-matrix change that should land alongside test coverage.
CONSENT_REQUIRED_EDGES: Final[frozenset[tuple[SystemMode, SystemMode]]] = (
    frozenset(
        {
            (SystemMode.SAFE, SystemMode.PAPER),
            (SystemMode.LIVE, SystemMode.AUTO),
        }
    )
)


def edge_requires_consent(prev: SystemMode, target: SystemMode) -> bool:
    """Return ``True`` if the ``prev → target`` edge requires a typed
    :class:`OperatorConsent` envelope per Hardening-S1 item 8."""

    return (prev, target) in CONSENT_REQUIRED_EDGES


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorConsent:
    """Typed envelope carrying an operator's explicit approval of a
    specific Mode FSM edge under a specific policy version.

    Fields:

    * ``ts_ns``        — operator's signing timestamp (wall-clock at
      the dashboard / CLI). Used for freshness check against the
      request's ``ts_ns``.
    * ``operator_id``  — stable identifier for the operator (e.g.
      ``"ronald"``, ``"operator:dashboard"``). Bound to the
      authority ledger row when consent is accepted.
    * ``mode_from``    — the Mode FSM state the operator believes
      the system is in. Validator rejects if it does not match the
      live ``current_mode``.
    * ``mode_to``      — the target state the operator is consenting
      to. Validator rejects if it does not match the request's
      ``target_mode``.
    * ``policy_hash``  — SHA-256 hex of the policy version that was
      active when the operator constructed the consent. Validator
      rejects if it does not match the live policy hash — this
      catches the "policy was loosened mid-session" foot-gun.
    * ``nonce``        — opaque unique token (UUID4 hex is the
      canonical shape). Replay-rejected by the validator after
      first use within the freshness window.
    """

    ts_ns: int
    operator_id: str
    mode_from: SystemMode
    mode_to: SystemMode
    policy_hash: str
    nonce: str


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsentValidationResult:
    """Outcome of :meth:`OperatorConsentValidator.validate`.

    ``ok=True`` iff every check passed. ``code`` carries a
    machine-stable rejection identifier for the audit ledger.
    """

    ok: bool
    code: str = ""
    detail: str = ""


# Stable rejection codes — these surface in ``MODE_TRANSITION_REJECTED``
# ledger rows so replay tools can group failures without parsing
# free-form ``detail`` text.
CODE_OK: Final[str] = ""
CODE_MISSING: Final[str] = "CONSENT_MISSING"
CODE_MODE_FROM_MISMATCH: Final[str] = "CONSENT_MODE_FROM_MISMATCH"
CODE_MODE_TO_MISMATCH: Final[str] = "CONSENT_MODE_TO_MISMATCH"
CODE_STALE: Final[str] = "CONSENT_STALE"
CODE_FUTURE: Final[str] = "CONSENT_FUTURE"
CODE_POLICY_HASH_MISMATCH: Final[str] = "CONSENT_POLICY_HASH_MISMATCH"
CODE_REPLAY: Final[str] = "CONSENT_REPLAY"
CODE_OPERATOR_EMPTY: Final[str] = "CONSENT_OPERATOR_EMPTY"
CODE_NONCE_EMPTY: Final[str] = "CONSENT_NONCE_EMPTY"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class OperatorConsentValidator:
    """Pure validator for :class:`OperatorConsent` envelopes.

    The validator owns a bounded ring of recently-seen nonces (per
    operator) so a replayed envelope inside the freshness window is
    rejected with ``CONSENT_REPLAY``. The ring is per-instance, so
    the production wiring constructs **one** validator and shares
    it across every transition path.

    Determinism contract: given the same sequence of ``validate``
    calls (same envelopes, same request ts/mode/hash), the
    validator returns the same results in the same order — there is
    no clock read inside the validator.
    """

    name: str = "operator_consent_validator"
    spec_id: str = "HARDENING-S1-08"

    def __init__(
        self,
        *,
        freshness_window_ns: int = CONSENT_FRESHNESS_WINDOW_NS,
        nonce_capacity: int = NONCE_RING_CAPACITY,
    ) -> None:
        if freshness_window_ns <= 0:
            raise ValueError(
                "freshness_window_ns must be positive nanoseconds"
            )
        if nonce_capacity <= 0:
            raise ValueError("nonce_capacity must be positive")
        self._freshness_window_ns = freshness_window_ns
        self._seen_nonces: deque[tuple[str, str]] = deque(
            maxlen=nonce_capacity
        )

    def validate(
        self,
        *,
        consent: OperatorConsent | None,
        request_ts_ns: int,
        prev_mode: SystemMode,
        target_mode: SystemMode,
        live_policy_hash: str,
    ) -> ConsentValidationResult:
        """Cross-check ``consent`` against the live request context.

        Order of checks is deterministic — the *first* failure wins,
        so the rejection code surfaces the most actionable cause.

        Successful validation returns ``ok=True`` but does **not**
        register the nonce in the seen-ring. Registration is deferred
        to :meth:`commit`, which the caller must invoke once *all*
        downstream checks (promotion gates, policy decision table,
        etc.) have also accepted the transition. This two-phase
        commit prevents a downstream rejection from burning a
        semantically-valid nonce. Future calls inside the freshness
        window with the same ``(operator_id, nonce)`` pair will fail
        with ``CONSENT_REPLAY`` only after :meth:`commit` has run.
        """

        if consent is None:
            return ConsentValidationResult(
                ok=False,
                code=CODE_MISSING,
                detail=(
                    "OperatorConsent envelope required for edge "
                    f"{prev_mode.name} → {target_mode.name} "
                    "(Hardening-S1 item 8)."
                ),
            )

        if not consent.operator_id:
            return ConsentValidationResult(
                ok=False,
                code=CODE_OPERATOR_EMPTY,
                detail="OperatorConsent.operator_id must be non-empty.",
            )

        if not consent.nonce:
            return ConsentValidationResult(
                ok=False,
                code=CODE_NONCE_EMPTY,
                detail="OperatorConsent.nonce must be non-empty.",
            )

        if consent.mode_from is not prev_mode:
            return ConsentValidationResult(
                ok=False,
                code=CODE_MODE_FROM_MISMATCH,
                detail=(
                    f"consent.mode_from={consent.mode_from.name} but "
                    f"live current_mode={prev_mode.name}"
                ),
            )

        if consent.mode_to is not target_mode:
            return ConsentValidationResult(
                ok=False,
                code=CODE_MODE_TO_MISMATCH,
                detail=(
                    f"consent.mode_to={consent.mode_to.name} but "
                    f"request target_mode={target_mode.name}"
                ),
            )

        # Future-stamped consent (clock skew or fabrication) — reject.
        if consent.ts_ns > request_ts_ns:
            return ConsentValidationResult(
                ok=False,
                code=CODE_FUTURE,
                detail=(
                    f"consent.ts_ns={consent.ts_ns} > "
                    f"request.ts_ns={request_ts_ns}"
                ),
            )

        # Stale consent — beyond the freshness window.
        if request_ts_ns - consent.ts_ns > self._freshness_window_ns:
            return ConsentValidationResult(
                ok=False,
                code=CODE_STALE,
                detail=(
                    f"consent.ts_ns is {request_ts_ns - consent.ts_ns} ns "
                    f"older than request.ts_ns; freshness window is "
                    f"{self._freshness_window_ns} ns"
                ),
            )

        if consent.policy_hash != live_policy_hash:
            return ConsentValidationResult(
                ok=False,
                code=CODE_POLICY_HASH_MISMATCH,
                detail=(
                    "consent.policy_hash does not match live policy "
                    "hash — policy was loosened or rotated after "
                    "operator constructed the consent."
                ),
            )

        key = (consent.operator_id, consent.nonce)
        if key in self._seen_nonces:
            return ConsentValidationResult(
                ok=False,
                code=CODE_REPLAY,
                detail=(
                    "OperatorConsent nonce already used by this operator "
                    "within the freshness window."
                ),
            )

        # Accept — but do NOT register the nonce here. Nonce
        # registration is a side effect that is only safe to commit
        # AFTER all downstream checks (promotion gates, policy)
        # have also passed. The caller (StateTransitionManager) must
        # invoke ``commit(consent)`` once it is ready to write the
        # ``OPERATOR_CONSENT_ACCEPTED`` ledger row, otherwise a
        # downstream rejection would burn a semantically-valid
        # nonce and leave an orphan audit row in the ledger.
        return ConsentValidationResult(ok=True, code=CODE_OK)

    def commit(self, consent: OperatorConsent) -> None:
        """Register the consent's nonce in the replay ring.

        Must only be called after :meth:`validate` returned ``ok=True``
        AND every downstream check (promotion gates, policy decision
        table, etc.) the caller intends to run has also passed.
        Calling this twice for the same nonce is a no-op as far as
        replay protection is concerned, but the caller should not
        rely on idempotency — :class:`StateTransitionManager`
        invokes :meth:`commit` exactly once per accepted transition.
        """

        self._seen_nonces.append((consent.operator_id, consent.nonce))


__all__ = [
    "CODE_FUTURE",
    "CODE_MISSING",
    "CODE_MODE_FROM_MISMATCH",
    "CODE_MODE_TO_MISMATCH",
    "CODE_NONCE_EMPTY",
    "CODE_OK",
    "CODE_OPERATOR_EMPTY",
    "CODE_POLICY_HASH_MISMATCH",
    "CODE_REPLAY",
    "CODE_STALE",
    "CONSENT_FRESHNESS_WINDOW_NS",
    "CONSENT_REQUIRED_EDGES",
    "ConsentValidationResult",
    "NONCE_RING_CAPACITY",
    "OperatorConsent",
    "OperatorConsentValidator",
    "edge_requires_consent",
]
