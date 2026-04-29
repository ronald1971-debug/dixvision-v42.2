"""System Intent — read-only projection of the operator-set strategic vector.

Phase 6.T1d (v3.1 G1, INV-38). Companion read-only projection to
:mod:`core.coherence.belief_state` and
:mod:`core.coherence.performance_pressure`.

The :class:`SystemIntent` answers *what should the system want to do this
week?* — the strategic axis above the per-tick Mode FSM. It is **operator
written, system read**:

* Operator proposes via ``OperatorInterfaceBridge`` (GOV-CP-07) using a
  :class:`IntentTransitionRequest`.
* ``StateTransitionManager.propose_intent`` (GOV-CP-03) is the *only*
  writer of ``INTENT_TRANSITION`` ledger rows.
* This module exposes the pure-function projection
  :func:`derive_system_intent` that replays the ``INTENT_TRANSITION``
  rows of the authority ledger and returns the latest committed
  ``SystemIntent``. There is **no setter API** — INV-38.
* Meta-Controller / Indira / Execution / Learning / System read this
  projection through ``core.coherence`` and never write it back.

Authority constraints (lint rule **B8**):

* Imports only :mod:`core.contracts` — no engine packages.
* No clocks, no PRNG, no I/O. Pure function over a :class:`LedgerEntry`
  sequence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.contracts.governance import (
    IntentHorizon,
    IntentObjective,
    IntentRiskMode,
    LedgerEntry,
)

# Module version — bumped when the projection function changes shape.
# Recorded in every ``INTENT_TRANSITION`` ledger payload so the
# projection can disambiguate intent rows produced by older writers.
SYSTEM_INTENT_VERSION = "v3.4-T1d"

# Canonical ledger ``kind`` written by GOV-CP-03 for an approved intent
# transition. The rejected counterpart is ``INTENT_TRANSITION_REJECTED``
# but the projection only consumes accepted rows.
INTENT_TRANSITION_KIND = "INTENT_TRANSITION"

# Ledger payload key constants — every reader and writer must agree on
# these spellings. Keep them sorted; they are written in
# ``StateTransitionManager.propose_intent`` and read here.
INTENT_KEY_OBJECTIVE = "objective"
INTENT_KEY_RISK_MODE = "risk_mode"
INTENT_KEY_HORIZON = "horizon"
INTENT_KEY_FOCUS = "focus"
INTENT_KEY_REQUESTOR = "requestor"
INTENT_KEY_REASON = "reason"
INTENT_KEY_VERSION = "version"

# Focus list serialised as a single string with this delimiter so that
# the ledger payload remains a flat ``Mapping[str, str]``. The
# delimiter is a Unit Separator (0x1F), matching the canonical encoding
# used by ``LedgerAuthorityWriter._canonical_payload``.
_FOCUS_DELIMITER = "\x1f"


@dataclass(frozen=True, slots=True)
class SystemIntent:
    """Frozen read-only snapshot of the operator-committed intent vector.

    ``intent_id`` is the ``hash_chain`` of the ledger row that committed
    this intent (or :data:`GENESIS_INTENT_ID` for the boot default).
    ``set_at`` is the ledger ``seq`` at which the row was written
    (``-1`` for the boot default).
    """

    ts_ns: int
    objective: IntentObjective
    risk_mode: IntentRiskMode
    horizon: IntentHorizon
    focus: tuple[str, ...] = ()
    intent_id: str = ""
    set_at: int = -1
    version: str = SYSTEM_INTENT_VERSION


# Genesis identifier for the boot-default intent — chosen so that callers
# can distinguish "no operator intent has been committed yet" from "the
# operator has explicitly chosen capital preservation".
GENESIS_INTENT_ID: str = "genesis"

# Boot default — the system always starts in the most cautious posture.
# This matches the Mode FSM, which boots in ``SAFE``: until the operator
# explicitly raises both, the system runs cautiously.
DEFAULT_SYSTEM_INTENT: SystemIntent = SystemIntent(
    ts_ns=0,
    objective=IntentObjective.CAPITAL_PRESERVATION,
    risk_mode=IntentRiskMode.DEFENSIVE,
    horizon=IntentHorizon.INTRADAY,
    focus=(),
    intent_id=GENESIS_INTENT_ID,
    set_at=-1,
    version=SYSTEM_INTENT_VERSION,
)


def encode_focus(focus: Sequence[str]) -> str:
    """Encode a focus tuple for a flat ledger payload."""

    return _FOCUS_DELIMITER.join(focus)


def decode_focus(encoded: str) -> tuple[str, ...]:
    """Inverse of :func:`encode_focus`."""

    if encoded == "":
        return ()
    return tuple(encoded.split(_FOCUS_DELIMITER))


def _parse_objective(raw: str) -> IntentObjective | None:
    try:
        return IntentObjective(raw)
    except ValueError:
        return None


def _parse_risk_mode(raw: str) -> IntentRiskMode | None:
    try:
        return IntentRiskMode(raw)
    except ValueError:
        return None


def _parse_horizon(raw: str) -> IntentHorizon | None:
    try:
        return IntentHorizon(raw)
    except ValueError:
        return None


def _intent_from_row(entry: LedgerEntry) -> SystemIntent | None:
    """Project one ``INTENT_TRANSITION`` ledger row into a ``SystemIntent``.

    Returns ``None`` if the row is malformed (unknown enum value, missing
    keys). Malformed rows are skipped silently — the projection is
    deterministic on whatever rows survive validation, which mirrors how
    :func:`derive_belief_state` treats unparseable inputs.
    """

    payload: Mapping[str, str] = entry.payload
    objective = _parse_objective(payload.get(INTENT_KEY_OBJECTIVE, ""))
    risk_mode = _parse_risk_mode(payload.get(INTENT_KEY_RISK_MODE, ""))
    horizon = _parse_horizon(payload.get(INTENT_KEY_HORIZON, ""))
    if objective is None or risk_mode is None or horizon is None:
        return None
    focus = decode_focus(payload.get(INTENT_KEY_FOCUS, ""))
    return SystemIntent(
        ts_ns=entry.ts_ns,
        objective=objective,
        risk_mode=risk_mode,
        horizon=horizon,
        focus=focus,
        intent_id=entry.hash_chain or str(entry.seq),
        set_at=entry.seq,
        version=payload.get(INTENT_KEY_VERSION, SYSTEM_INTENT_VERSION),
    )


def derive_system_intent(
    rows: Sequence[LedgerEntry],
    *,
    default: SystemIntent = DEFAULT_SYSTEM_INTENT,
) -> SystemIntent:
    """Replay ``INTENT_TRANSITION`` rows in ledger order; return the last.

    Pure function. Same input → same output (INV-15). Rows whose ``kind``
    is not :data:`INTENT_TRANSITION_KIND` are ignored, which lets a
    caller pass an unfiltered ledger slice. If no committed intent
    exists yet, ``default`` is returned (the boot-default intent).
    """

    current = default
    for entry in rows:
        if entry.kind != INTENT_TRANSITION_KIND:
            continue
        projected = _intent_from_row(entry)
        if projected is None:
            continue
        current = projected
    return current


__all__ = [
    "DEFAULT_SYSTEM_INTENT",
    "GENESIS_INTENT_ID",
    "INTENT_KEY_FOCUS",
    "INTENT_KEY_HORIZON",
    "INTENT_KEY_OBJECTIVE",
    "INTENT_KEY_REASON",
    "INTENT_KEY_REQUESTOR",
    "INTENT_KEY_RISK_MODE",
    "INTENT_KEY_VERSION",
    "INTENT_TRANSITION_KIND",
    "SYSTEM_INTENT_VERSION",
    "SystemIntent",
    "decode_focus",
    "derive_system_intent",
    "encode_focus",
]
