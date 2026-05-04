"""HARDEN-05 — harness governance approval shim (INV-68).

Production code paths construct an :class:`ExecutionIntent` inside an
intelligence subsystem and approve it through the live governance
pipeline; only then does the intent reach
:meth:`ExecutionEngine.execute`. The harness (``ui.server``) and a
handful of replay tests need a deterministic, single-call equivalent
that does the same thing without spinning up the full async control
loop.

This module exposes exactly one helper,
:func:`approve_signal_for_execution`, that:

1. Re-stamps the input :class:`SignalEvent` so it carries the canonical
   ``produced_by_engine`` for its origin (HARDEN-03 wire contract).
2. Constructs a frozen :class:`ExecutionIntent` via the B25-allowed
   :func:`create_execution_intent` factory.
3. Marks the intent approved through :func:`mark_approved`, recording
   a deterministic ``governance_decision_id`` derived from the signal
   timestamp so the result is reproducible across replays (INV-15).

The helper does **not** call ``execute`` — the caller does, so the
chokepoint stays exactly where HARDEN-02 placed it
(:meth:`ExecutionEngine.execute`). All this module does is collapse
"build intent + governance approves" into one deterministic call.

**Hardening-S1 item 1 — explicit opt-in gate (no implicit approvals).**

The original architecture review (operator-control critique, item 1)
flagged this module as a hidden authority surface: any caller that
imports :func:`approve_signal_for_execution` silently bypasses the
live governance loop and gets a fully-approved intent back. The fix
is to make the gate explicit at call time:

* The shim now refuses to run unless ``DIX_HARNESS_APPROVER_ENABLED``
  is set to a truthy value (``1`` / ``true`` / ``yes`` / ``on``) **or**
  the caller passes ``enabled=True`` to the helper.
* When the gate is closed, the helper raises
  :class:`HarnessApproverDisabledError` loudly — no silent fallback,
  no default-permissive behaviour.
* ``ui.server`` opts in explicitly at startup (it is the harness, by
  definition); pytest sessions opt in via the conftest fixture.
* ``tools.authority_lint`` rule **B33** restricts callers of this
  module to ``ui.*`` and ``tests.*``; engines, adapters, and dashboard
  surfaces are forbidden from importing it.

Lint scope:
* B25 allows ``governance_engine.*`` to call the
  :func:`create_execution_intent` factory, so this module is the
  legitimate home for the approval shim.
* B1 — does not import any other engine package.
* B33 — no-implicit-approval; only ``ui.*`` (harness) and ``tests.*``
  may import this module.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Final, Protocol

from core.contracts.events import SignalEvent
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    ExecutionIntent,
    compute_content_hash,
    create_execution_intent,
    mark_approved,
)
from core.contracts.external_signal_trust import ExternalSignalTrustRegistry
from core.contracts.signal_trust import (
    SignalTrust,
    clamp_confidence,
    default_cap_for,
)
from core.contracts.source_trust_promotions import SourceTrustPromotionStore


class _IntentSigner(Protocol):
    """Duck-typed signer for harness-approved intents (AUDIT-P1.1).

    ``governance_engine.control_plane.decision_signer.DecisionSigner``
    satisfies this protocol structurally. Keeping the dependency
    duck-typed avoids an import cycle (``governance_engine`` already
    imports this module via ``governance_engine.engine``).
    """

    def sign(
        self, *, content_hash: str, governance_decision_id: str
    ) -> str: ...

__all__ = [
    "DEFAULT_HARNESS_ORIGIN",
    "HARNESS_APPROVER_ENV_VAR",
    "HARNESS_DECISION_ID_PREFIX",
    "HarnessApproverDisabledError",
    "apply_signal_trust_cap",
    "approve_signal_for_execution",
    "is_harness_approver_enabled",
]


DEFAULT_HARNESS_ORIGIN: Final[str] = (
    "intelligence_engine.signal_pipeline.orchestrator"
)
"""Default origin stamped on harness-approved intents.

Must be a member of :data:`AUTHORISED_INTENT_ORIGINS`. The default
matches the production signal-pipeline subsystem so the matrix's
intelligence actor row covers it without a per-harness exception."""


HARNESS_DECISION_ID_PREFIX: Final[str] = "harness:auto"
"""Prefix for synthetic governance_decision_ids emitted by the shim.

The auditor can grep this prefix to identify intents that bypassed
the live governance loop. Production governance never emits this
prefix."""


HARNESS_APPROVER_ENV_VAR: Final[str] = "DIX_HARNESS_APPROVER_ENABLED"
"""Env var that opts a process into the harness approval shim.

The shim is **off by default**: a process must explicitly set this to
a truthy value (``1``, ``true``, ``yes``, ``on``) before any caller
may invoke :func:`approve_signal_for_execution`. ``ui.server`` sets
this at startup; pytest sets it via the conftest fixture. Any other
production process touching this shim will trip the
:class:`HarnessApproverDisabledError` and fail closed."""


_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


class HarnessApproverDisabledError(RuntimeError):
    """Raised when the harness approval shim is invoked without opt-in.

    Hardening-S1 item 1 — no implicit approvals. The shim refuses to
    run unless the calling process explicitly opted in via the
    :data:`HARNESS_APPROVER_ENV_VAR` env var or the ``enabled=True``
    keyword. This guarantees that a developer who accidentally imports
    the shim from an engine, adapter, or dashboard surface gets a
    loud, traceable failure instead of silently bypassing the live
    governance loop.
    """


def is_harness_approver_enabled() -> bool:
    """Return ``True`` iff the env var opts this process into the shim.

    Pure helper used by callers that want to branch on the gate
    without triggering the loud failure path. Tests use this to skip
    cleanly on processes where the gate is not set.
    """

    raw = os.getenv(HARNESS_APPROVER_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY


def apply_signal_trust_cap(
    signal: SignalEvent,
    *,
    registry: ExternalSignalTrustRegistry | None = None,
    promotion_store: SourceTrustPromotionStore | None = None,
) -> SignalEvent:
    """Return a copy of *signal* with ``confidence`` clamped per Paper-S5/S6.

    The governance gate is the canonical place to enforce the
    per-source confidence cap (see
    :mod:`core.contracts.signal_trust` for the policy).

    Paper-S6 -- when a *promotion_store* is supplied, the helper first
    resolves the **effective** trust class for the producer:

    * ``INTERNAL`` always passes through unchanged (no cap).
    * If the operator promoted the source from ``EXTERNAL_LOW`` to
      ``EXTERNAL_MED`` (recorded in the store and replayed from the
      authority ledger at boot), the effective trust becomes the
      promoted target so the trust-class default rises from
      :data:`~core.contracts.signal_trust.DEFAULT_LOW_CAP` to
      :data:`~core.contracts.signal_trust.DEFAULT_MED_CAP`.
    * The overlay never *demotes* a producer-declared class
      (fail-closed): a producer that already declared
      ``EXTERNAL_MED`` is unaffected by an absent overlay.

    The cap itself is then looked up against the effective trust:

    1. If *registry* is provided, ``registry.cap_for(source_id, trust)``
       — which already takes the more-restrictive of the per-source
       row and the trust-class default (fail-closed when the row's
       declared trust disagrees with the effective trust).
    2. Otherwise, ``default_cap_for(effective_trust)`` — so an
       external signal is always clamped even when the registry is
       unavailable (fail-closed default).

    The clamp is monotone (the returned ``confidence`` is never
    larger than the input), so this helper can be applied multiple
    times without amplification.

    The returned signal is a dataclass copy with the same identity
    fields as the input; ``signal_trust`` and ``signal_source`` are
    preserved verbatim so the audit ledger can still trace the
    producer (the promotion overlay only widens the cap; the
    declared trust on the SignalEvent stays intact).
    """

    if signal.signal_trust is SignalTrust.INTERNAL:
        return signal
    if promotion_store is not None:
        effective_trust = promotion_store.effective_trust(
            signal.signal_source, signal.signal_trust
        )
    else:
        effective_trust = signal.signal_trust
    if registry is not None:
        cap = registry.cap_for(signal.signal_source, effective_trust)
    else:
        cap = default_cap_for(effective_trust)
    clamped = clamp_confidence(signal.confidence, cap)
    if clamped == signal.confidence:
        return signal
    return dataclasses.replace(signal, confidence=clamped)


def approve_signal_for_execution(
    signal: SignalEvent,
    *,
    ts_ns: int,
    origin: str = DEFAULT_HARNESS_ORIGIN,
    decision_id: str | None = None,
    enabled: bool | None = None,
    signer: _IntentSigner | None = None,
    registry: ExternalSignalTrustRegistry | None = None,
    promotion_store: SourceTrustPromotionStore | None = None,
) -> ExecutionIntent:
    """Build + approve an :class:`ExecutionIntent` in one deterministic call.

    Args:
        signal: The :class:`SignalEvent` to wrap. The signal's
            ``ts_ns`` and identity are preserved verbatim — the helper
            does not mutate the signal.
        ts_ns: Monotonic timestamp stamped onto the intent. The
            harness sources this from its replay clock so the result
            is reproducible.
        origin: Authorised intent origin. Defaults to
            :data:`DEFAULT_HARNESS_ORIGIN`. Must be a member of
            :data:`AUTHORISED_INTENT_ORIGINS`; the underlying factory
            raises :class:`UnauthorizedOriginError` otherwise.
        decision_id: Override the synthetic
            ``governance_decision_id``. When unset (the common case),
            a deterministic id is derived from ``ts_ns`` so two
            replays of the same input produce byte-identical intents
            (INV-15).
        enabled: Optional explicit override of the env-var gate.
            When ``None`` (the default), the gate is read from
            :data:`HARNESS_APPROVER_ENV_VAR`. When ``True``, the call
            proceeds even if the env var is unset (used by tests that
            want to opt in for one call without mutating process
            env). When ``False``, the call always fails closed.

    Returns:
        A frozen :class:`ExecutionIntent` with
        ``approved_by_governance=True``, ready for
        :meth:`ExecutionEngine.execute`.

    Raises:
        HarnessApproverDisabledError: when the gate is closed —
            i.e. ``enabled`` is ``False`` *or* ``enabled`` is ``None``
            and :data:`HARNESS_APPROVER_ENV_VAR` is not truthy. The
            error message names the env var so the operator can opt
            in explicitly.
        UnauthorizedOriginError: when ``origin`` is not a member of
            :data:`AUTHORISED_INTENT_ORIGINS`.

    Paper-S5 — when ``signal.signal_trust`` is not ``INTERNAL`` the
    helper applies the per-source confidence cap via
    :func:`apply_signal_trust_cap` *before* building the intent, so
    the resulting :class:`ExecutionIntent` (and any downstream
    :class:`DecisionTrace`) sees the clamped confidence. Internal
    signals pass through unchanged.
    """

    if enabled is False or (enabled is None and not is_harness_approver_enabled()):
        raise HarnessApproverDisabledError(
            "harness_approver is opt-in only (Hardening-S1 item 1, "
            "no-implicit-approval). Set "
            f"{HARNESS_APPROVER_ENV_VAR}=1 in the calling process or "
            "pass enabled=True to opt in for one call. Engines, "
            "adapters, and dashboard surfaces must NOT import this "
            "module — see authority_lint rule B33."
        )

    if origin not in AUTHORISED_INTENT_ORIGINS:
        # Re-raised here as well so the harness gets a stable error
        # path even if a caller passes a bare string. The factory
        # would raise too, but doing the check up-front keeps the
        # ``mark_approved`` call below from running with a bad value.
        from core.contracts.execution_intent import UnauthorizedOriginError

        raise UnauthorizedOriginError(
            f"unauthorised harness origin: {origin!r}"
        )
    # Paper-S5 governance gate -- clamp external-producer confidence
    # to the per-source cap before sealing the intent. Internal
    # producers pass through as a no-op. Paper-S6 -- the optional
    # *promotion_store* overlay can promote EXTERNAL_LOW sources to
    # EXTERNAL_MED so a higher class default applies.
    signal = apply_signal_trust_cap(
        signal,
        registry=registry,
        promotion_store=promotion_store,
    )
    intent = create_execution_intent(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
    )
    governance_decision_id = (
        decision_id
        if decision_id is not None
        else f"{HARNESS_DECISION_ID_PREFIX}:{ts_ns}"
    )
    decision_signature = ""
    if signer is not None:
        # AUDIT-P1.1 -- ``mark_approved`` re-hashes the intent with
        # ``approved_by_governance=True``, which is a different hash
        # than the un-approved ``intent.content_hash`` we just built.
        # The HMAC must therefore cover the *approved* content hash
        # so the AuthorityGuard verifier (which sees the post-approval
        # intent) can reproduce the signed input bit-for-bit.
        approved_content_hash = compute_content_hash(
            ts_ns=ts_ns,
            origin=origin,
            signal=signal,
            approved_by_governance=True,
            governance_decision_id=governance_decision_id,
        )
        decision_signature = signer.sign(
            content_hash=approved_content_hash,
            governance_decision_id=governance_decision_id,
        )
    return mark_approved(
        intent,
        governance_decision_id=governance_decision_id,
        decision_signature=decision_signature,
    )
