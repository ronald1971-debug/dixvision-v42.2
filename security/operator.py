"""
security.operator — end-user authority tier (ABOVE governance).

Manifest addendum (operator-above-all):

    INDIRA / DYON / GOVERNANCE / DEVIN may *propose*.
    Only the OPERATOR may *commit*.

This module is the canonical register for operator-approval state.  Any
action that changes the live behaviour of the system (wallet live-signing,
daily-cap changes, autonomy-mode transitions, patch promotion to LIVE,
custom-strategy go-live, kill-switch overrides) must pass an
``OPERATOR/APPROVAL_GRANTED`` event for its action id before the
governance code-path writes ``GOVERNANCE/*_APPROVED`` and applies it.

Design invariants
-----------------
* Read paths are lock-free (``dataclass``-backed snapshots).
* Write paths serialise through ``_lock`` + single cached sqlite
  connection (same pattern as :mod:`security.wallet_policy`).
* Every request / grant / deny / revoke is an immutable event in the
  ledger ``OPERATOR/*`` stream.
* An outstanding ``request_id`` becomes stale after
  ``_DEFAULT_TTL_SEC`` and is auto-expired on read.
* Two-person gate: high-risk actions (``kind in _TWO_PERSON``) require
  *two* distinct ``operator_id`` approvals before they count as
  granted.  A single approval from the same operator never promotes.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from state.ledger.writer import get_writer
from system.time_source import utc_now


class ApprovalKind(str, Enum):
    WALLET_LIVE_SIGNING = "WALLET_LIVE_SIGNING"
    DAILY_CAP_CHANGE = "DAILY_CAP_CHANGE"
    AUTONOMY_MODE_CHANGE = "AUTONOMY_MODE_CHANGE"
    PATCH_PROMOTE_LIVE = "PATCH_PROMOTE_LIVE"
    CUSTOM_STRATEGY_GO_LIVE = "CUSTOM_STRATEGY_GO_LIVE"
    KILL_SWITCH_OVERRIDE = "KILL_SWITCH_OVERRIDE"
    FAST_PATH_AMEND = "FAST_PATH_AMEND"


class ApprovalState(str, Enum):
    PENDING = "PENDING"
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


# Kinds that require two distinct operator approvals before they count
# as GRANTED.  Matches manifest §1 "two-person sign-off" for fast-path
# amendments, and is extended to kill-switch overrides because that is
# the one action that can cancel every other safety gate.
_TWO_PERSON: frozenset[ApprovalKind] = frozenset({
    ApprovalKind.FAST_PATH_AMEND,
    ApprovalKind.KILL_SWITCH_OVERRIDE,
})

_DEFAULT_TTL_SEC = 24 * 3600

_DB_PATH = Path("data") / "operator.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    request_id TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    subject    TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    state      TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    ttl_sec    INTEGER NOT NULL,
    approvers  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_approvals_state ON approvals(state);
CREATE INDEX IF NOT EXISTS ix_approvals_kind  ON approvals(kind);
CREATE INDEX IF NOT EXISTS ix_approvals_subject ON approvals(subject);
"""


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    kind: ApprovalKind
    subject: str
    payload: dict
    state: ApprovalState
    created_utc: str
    ttl_sec: int
    approvers: list[str]

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "kind": self.kind.value,
            "subject": self.subject,
            "payload": self.payload,
            "state": self.state.value,
            "created_utc": self.created_utc,
            "ttl_sec": self.ttl_sec,
            "approvers": list(self.approvers),
        }


_lock = threading.RLock()
_conn_cached: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    """Cached sqlite connection. Caller MUST hold ``_lock``."""
    global _conn_cached
    if _conn_cached is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.executescript(_SCHEMA)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL;")
        _conn_cached = c
    return _conn_cached


def _now_iso() -> str:
    n = utc_now()
    return n.isoformat() if n else ""


def _now_epoch() -> float:
    n = utc_now()
    return n.timestamp() if n else 0.0


def _row_to_request(r: sqlite3.Row) -> ApprovalRequest:
    import json
    return ApprovalRequest(
        request_id=str(r["request_id"]),
        kind=ApprovalKind(str(r["kind"])),
        subject=str(r["subject"]),
        payload=json.loads(str(r["payload"] or "{}")),
        state=ApprovalState(str(r["state"])),
        created_utc=str(r["created_utc"]),
        ttl_sec=int(r["ttl_sec"]),
        approvers=json.loads(str(r["approvers"] or "[]")),
    )


def _maybe_expire_locked(c: sqlite3.Connection, r: sqlite3.Row) -> sqlite3.Row:
    """Auto-expire a row if its TTL has elapsed while still PENDING."""
    if str(r["state"]) != ApprovalState.PENDING.value:
        return r
    try:
        from datetime import datetime, timezone
        ts = str(r["created_utc"])
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        created = datetime.fromisoformat(ts)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = _now_epoch() - created.timestamp()
    except Exception:
        return r
    if age > float(r["ttl_sec"]):
        c.execute(
            "UPDATE approvals SET state=? WHERE request_id=?",
            (ApprovalState.EXPIRED.value, r["request_id"]),
        )
        c.commit()
        r2 = c.execute(
            "SELECT * FROM approvals WHERE request_id=?",
            (r["request_id"],),
        ).fetchone()
        return r2 or r
    return r


# -------- public API ---------------------------------------------------

def request_approval(kind: ApprovalKind, *, subject: str,
                     payload: dict | None = None,
                     ttl_sec: int = _DEFAULT_TTL_SEC,
                     requested_by: str = "system") -> ApprovalRequest:
    """Register a new pending approval request.

    ``subject`` is a stable identifier for the action (e.g. wallet key,
    autonomy mode name, strategy id, patch hash).  Returns the
    freshly-created :class:`ApprovalRequest` for UI display.  The
    request is ledger-logged as ``OPERATOR/APPROVAL_REQUESTED``.
    """
    import json
    request_id = str(uuid.uuid4())
    payload_s = json.dumps(payload or {}, sort_keys=True)
    created = _now_iso()
    with _lock:
        c = _connect()
        c.execute(
            "INSERT INTO approvals(request_id,kind,subject,payload,state,"
            "created_utc,ttl_sec,approvers) VALUES (?,?,?,?,?,?,?,?)",
            (request_id, kind.value, subject, payload_s,
             ApprovalState.PENDING.value, created, int(ttl_sec), "[]"),
        )
        c.commit()
    try:
        get_writer().write("OPERATOR", "APPROVAL_REQUESTED", "security.operator",
                           {"request_id": request_id, "kind": kind.value,
                            "subject": subject, "ttl_sec": int(ttl_sec),
                            "requested_by": requested_by})
    except Exception:
        pass
    return ApprovalRequest(
        request_id=request_id, kind=kind, subject=subject,
        payload=payload or {}, state=ApprovalState.PENDING,
        created_utc=created, ttl_sec=int(ttl_sec), approvers=[],
    )


def approve(request_id: str, *, operator_id: str) -> ApprovalRequest:
    """Operator click: grant the request.

    For two-person kinds, two distinct ``operator_id`` values are
    required before the state transitions to ``GRANTED``.  A second
    click by the same operator is idempotent (no-op).
    """
    import json
    with _lock:
        c = _connect()
        r = c.execute("SELECT * FROM approvals WHERE request_id=?",
                      (request_id,)).fetchone()
        if r is None:
            raise LookupError(f"unknown_request:{request_id}")
        r = _maybe_expire_locked(c, r)
        if str(r["state"]) not in (ApprovalState.PENDING.value,):
            return _row_to_request(r)
        approvers: list[str] = json.loads(str(r["approvers"] or "[]"))
        if operator_id not in approvers:
            approvers.append(operator_id)
        kind = ApprovalKind(str(r["kind"]))
        required = 2 if kind in _TWO_PERSON else 1
        new_state = (ApprovalState.GRANTED if len(approvers) >= required
                     else ApprovalState.PENDING)
        c.execute("UPDATE approvals SET state=?, approvers=? WHERE request_id=?",
                  (new_state.value, json.dumps(approvers), request_id))
        c.commit()
        r2 = c.execute("SELECT * FROM approvals WHERE request_id=?",
                       (request_id,)).fetchone()
    try:
        get_writer().write("OPERATOR", "APPROVAL_GRANTED" if new_state is
                           ApprovalState.GRANTED else "APPROVAL_CAST",
                           "security.operator",
                           {"request_id": request_id,
                            "kind": str(r["kind"]),
                            "subject": str(r["subject"]),
                            "operator_id": operator_id,
                            "approvers": approvers,
                            "required": required,
                            "state": new_state.value})
    except Exception:
        pass
    return _row_to_request(r2)


def deny(request_id: str, *, operator_id: str,
         reason: str = "") -> ApprovalRequest:
    with _lock:
        c = _connect()
        r = c.execute("SELECT * FROM approvals WHERE request_id=?",
                      (request_id,)).fetchone()
        if r is None:
            raise LookupError(f"unknown_request:{request_id}")
        c.execute("UPDATE approvals SET state=? WHERE request_id=?",
                  (ApprovalState.DENIED.value, request_id))
        c.commit()
        r2 = c.execute("SELECT * FROM approvals WHERE request_id=?",
                       (request_id,)).fetchone()
    try:
        get_writer().write("OPERATOR", "APPROVAL_DENIED", "security.operator",
                           {"request_id": request_id,
                            "kind": str(r["kind"]),
                            "subject": str(r["subject"]),
                            "operator_id": operator_id,
                            "reason": reason})
    except Exception:
        pass
    return _row_to_request(r2)


def revoke(request_id: str, *, operator_id: str) -> ApprovalRequest:
    """Revoke a previously-granted approval (e.g. rescind live-signing)."""
    with _lock:
        c = _connect()
        r = c.execute("SELECT * FROM approvals WHERE request_id=?",
                      (request_id,)).fetchone()
        if r is None:
            raise LookupError(f"unknown_request:{request_id}")
        c.execute("UPDATE approvals SET state=? WHERE request_id=?",
                  (ApprovalState.REVOKED.value, request_id))
        c.commit()
        r2 = c.execute("SELECT * FROM approvals WHERE request_id=?",
                       (request_id,)).fetchone()
    try:
        get_writer().write("OPERATOR", "APPROVAL_REVOKED", "security.operator",
                           {"request_id": request_id,
                            "kind": str(r["kind"]),
                            "subject": str(r["subject"]),
                            "operator_id": operator_id})
    except Exception:
        pass
    return _row_to_request(r2)


def is_granted(kind: ApprovalKind, subject: str) -> bool:
    """Cheap gate: returns True iff there is at least one
    non-expired, non-revoked, GRANTED approval for (kind, subject).

    Governance code-paths MUST call this before writing
    ``GOVERNANCE/*_APPROVED`` for the action.  Absent approval =>
    reject (fail-closed per manifest §1).
    """
    with _lock:
        c = _connect()
        rows: Iterable[sqlite3.Row] = c.execute(
            "SELECT * FROM approvals WHERE kind=? AND subject=? AND state=?",
            (kind.value, subject, ApprovalState.GRANTED.value),
        ).fetchall()
        for r in rows:
            r2 = _maybe_expire_locked(c, r)
            if str(r2["state"]) == ApprovalState.GRANTED.value:
                return True
    return False


def pending(kind: ApprovalKind | None = None) -> list[ApprovalRequest]:
    with _lock:
        c = _connect()
        if kind is None:
            rows = c.execute(
                "SELECT * FROM approvals WHERE state=? "
                "ORDER BY created_utc DESC",
                (ApprovalState.PENDING.value,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM approvals WHERE state=? AND kind=? "
                "ORDER BY created_utc DESC",
                (ApprovalState.PENDING.value, kind.value),
            ).fetchall()
        rows = [_maybe_expire_locked(c, r) for r in rows]
    return [
        _row_to_request(r) for r in rows
        if str(r["state"]) == ApprovalState.PENDING.value
    ]


def history(limit: int = 100) -> list[ApprovalRequest]:
    with _lock:
        c = _connect()
        rows = c.execute(
            "SELECT * FROM approvals ORDER BY created_utc DESC LIMIT ?",
            (int(max(1, min(limit, 1000))),),
        ).fetchall()
    return [_row_to_request(r) for r in rows]


__all__ = [
    "ApprovalKind", "ApprovalState", "ApprovalRequest",
    "request_approval", "approve", "deny", "revoke",
    "is_granted", "pending", "history",
]
