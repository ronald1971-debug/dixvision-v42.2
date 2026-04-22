"""
cockpit.pairing — per-device pairing tokens for phones / extra clients.

A pairing token is a short-lived, single-use credential the operator hands
to a phone by scanning a QR code. Once the phone calls POST /api/pair/claim
with the pairing token, it receives the long-lived bearer token and the
pairing is marked consumed (ledger-audited, never re-usable).

Backed by a tiny bounded SQLite table so tokens survive restarts without
needing an external database.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from state.ledger.writer import get_writer

_DB_PATH_ENV = "DIX_PAIRING_DB"
_DEFAULT_DB = "data/pairing.sqlite"
_LOCK = threading.Lock()
_DEFAULT_TTL_SEC = 900                                                          # 15 min


def _db_path() -> Path:
    return Path(os.environ.get(_DB_PATH_ENV, _DEFAULT_DB))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=5.0, isolation_level=None,
                          check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pairings (
            token TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            expires_utc TEXT NOT NULL,
            consumed_utc TEXT,
            revoked_utc TEXT,
            device_fingerprint TEXT
        )
        """
    )
    return con


@dataclass(frozen=True)
class Pairing:
    token: str
    label: str
    created_utc: str
    expires_utc: str
    consumed_utc: str | None
    revoked_utc: str | None
    device_fingerprint: str | None

    @property
    def active(self) -> bool:
        if self.consumed_utc or self.revoked_utc:
            return False
        # Parse both timestamps into tz-aware datetimes so comparison
        # is timezone-correct regardless of isoformat suffix quirks
        # (``Z`` vs ``+00:00``) or a manually-inserted expiry string.
        try:
            exp = datetime.fromisoformat(self.expires_utc.replace("Z", "+00:00"))
        except ValueError:
            return False
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < exp


def issue_pairing(label: str, ttl_sec: int = _DEFAULT_TTL_SEC) -> Pairing:
    """Mint a new pairing token valid for `ttl_sec` seconds."""
    with _LOCK, _connect() as con:
        tok = secrets.token_urlsafe(18)
        now = _utcnow_iso()
        expires_dt = datetime.now(timezone.utc).timestamp() + max(60, int(ttl_sec))
        exp = (datetime.fromtimestamp(expires_dt, tz=timezone.utc)
               .replace(microsecond=0).isoformat())
        con.execute(
            "INSERT INTO pairings(token,label,created_utc,expires_utc) "
            "VALUES(?,?,?,?)",
            (tok, label.strip()[:80] or "unnamed", now, exp),
        )
    get_writer().append_event(stream="SECURITY", kind="PAIRING_ISSUED",
                              payload={"label": label, "expires_utc": exp})
    return Pairing(token=tok, label=label, created_utc=now, expires_utc=exp,
                   consumed_utc=None, revoked_utc=None, device_fingerprint=None)


def list_pairings() -> list[Pairing]:
    with _LOCK, _connect() as con:
        rows = con.execute(
            "SELECT token,label,created_utc,expires_utc,consumed_utc,"
            "revoked_utc,device_fingerprint FROM pairings "
            "ORDER BY created_utc DESC LIMIT 200"
        ).fetchall()
    return [Pairing(*r) for r in rows]


def revoke_pairing(token: str) -> bool:
    with _LOCK, _connect() as con:
        cur = con.execute(
            "UPDATE pairings SET revoked_utc=? WHERE token=? AND revoked_utc IS NULL",
            (_utcnow_iso(), token),
        )
        changed = cur.rowcount > 0
    if changed:
        get_writer().append_event(stream="SECURITY", kind="PAIRING_REVOKED",
                                  payload={"token_prefix": token[:6]})
    return changed


def claim_pairing(token: str, *, bearer_token: str,
                  device_fingerprint: str) -> str | None:
    """Consume a pairing token; return the cockpit bearer token on success.

    The bearer token (cockpit auth) is passed in by the caller so pairing.py
    stays independent of cockpit.auth's token storage.

    Single-use is enforced by a conditional UPDATE whose WHERE clause
    includes every validity condition: only one of N racing writers can
    set consumed_utc from NULL, and SQLite's UPDATE rowcount tells us
    whether we won.  This holds across processes / async workers, not
    only within a single process — so the previous SELECT-then-UPDATE
    TOCTOU window is closed.
    """
    now = _utcnow_iso()
    with _LOCK, _connect() as con:
        cur = con.execute(
            """
            UPDATE pairings
               SET consumed_utc = ?, device_fingerprint = ?
             WHERE token = ?
               AND consumed_utc IS NULL
               AND revoked_utc IS NULL
               AND expires_utc > ?
            """,
            (now, device_fingerprint.strip()[:80], token, now),
        )
        if cur.rowcount <= 0:
            return None
    get_writer().append_event(stream="SECURITY", kind="PAIRING_CLAIMED",
                              payload={"token_prefix": token[:6],
                                       "device": device_fingerprint[:20]})
    return bearer_token


__all__ = ["Pairing", "issue_pairing", "list_pairings", "revoke_pairing",
           "claim_pairing"]
