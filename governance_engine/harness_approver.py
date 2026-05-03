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

import os
from typing import Final

from core.contracts.events import SignalEvent
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    ExecutionIntent,
    create_execution_intent,
    mark_approved,
)

__all__ = [
    "DEFAULT_HARNESS_ORIGIN",
    "HARNESS_APPROVER_ENV_VAR",
    "HARNESS_DECISION_ID_PREFIX",
    "HarnessApproverDisabledError",
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


def approve_signal_for_execution(
    signal: SignalEvent,
    *,
    ts_ns: int,
    origin: str = DEFAULT_HARNESS_ORIGIN,
    decision_id: str | None = None,
    enabled: bool | None = None,
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
    intent = create_execution_intent(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
    )
    return mark_approved(
        intent,
        governance_decision_id=(
            decision_id
            if decision_id is not None
            else f"{HARNESS_DECISION_ID_PREFIX}:{ts_ns}"
        ),
    )
