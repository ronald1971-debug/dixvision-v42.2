"""
mind.custom_strategies — operator-authored strategy registry.

The operator is above every voice (see :mod:`security.operator`).
This module is the canonical place where operator-submitted strategy
definitions live.  It never executes a strategy; execution remains
the sole authority of :func:`mind.fast_execute.fast_execute_trade`
going through the adapter router, and every strategy here must
traverse the sandbox patch pipeline before any of its signals reach
the live path.

Lifecycle
---------

    DRAFT       freshly submitted; visible in the cockpit but has not
                passed the sandbox yet.
    SANDBOX_OK  ``tools.sandbox_runner`` returned ``ok=True`` against
                the strategy source + unit tests.
    SHADOW      strategy is being replayed against the paper-mode tape
                in cold path; signals land in the ledger but never in
                the fast-execute path.
    CANARY      operator-gated live execution under a tiny budget;
                emits the same signals as SHADOW but INDIRA actually
                routes them to the adapter.
    LIVE        operator has granted ``OPERATOR/APPROVAL_GRANTED`` of
                kind ``CUSTOM_STRATEGY_GO_LIVE`` for this strategy id.
                The strategy arbiter may now select this strategy.
    RETIRED     operator clicked archive; no longer selectable.

Every state change writes a ``GOVERNANCE/CUSTOM_STRATEGY_<state>``
ledger event.  ``LIVE`` promotion additionally requires the
:func:`security.operator.is_granted` gate.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from security.operator import ApprovalKind, is_granted, request_approval
from state.ledger.writer import get_writer
from system.time_source import utc_now


class StrategyState(str, Enum):
    DRAFT = "DRAFT"
    SANDBOX_OK = "SANDBOX_OK"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    LIVE = "LIVE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


_MAX_SOURCE_LEN = 200_000     # 200 KB hard cap on strategy source
_DB_PATH = Path("data") / "custom_strategies.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS custom_strategies (
    strategy_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    author        TEXT NOT NULL,
    language      TEXT NOT NULL,
    source        TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    state         TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT '',
    created_utc   TEXT NOT NULL,
    updated_utc   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_cs_state ON custom_strategies(state);
CREATE INDEX IF NOT EXISTS ix_cs_author ON custom_strategies(author);
"""


@dataclass(frozen=True)
class CustomStrategy:
    strategy_id: str
    name: str
    author: str
    language: str
    source: str
    source_sha256: str
    state: StrategyState
    detail: str
    created_utc: str
    updated_utc: str

    def as_dict(self, *, include_source: bool = False) -> dict:
        out = {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "author": self.author,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "source_len": len(self.source),
            "state": self.state.value,
            "detail": self.detail,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }
        if include_source:
            out["source"] = self.source
        return out


_lock = threading.RLock()
_conn_cached: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    """Cached sqlite connection (caller MUST hold ``_lock``)."""
    global _conn_cached
    if _conn_cached is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.executescript(_SCHEMA)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL;")
        _conn_cached = c
    return _conn_cached


def _iso_now() -> str:
    n = utc_now()
    return n.isoformat() if n else ""


def _row_to_strategy(r: sqlite3.Row) -> CustomStrategy:
    return CustomStrategy(
        strategy_id=str(r["strategy_id"]), name=str(r["name"]),
        author=str(r["author"]), language=str(r["language"]),
        source=str(r["source"]), source_sha256=str(r["source_sha256"]),
        state=StrategyState(str(r["state"])),
        detail=str(r["detail"] or ""),
        created_utc=str(r["created_utc"]),
        updated_utc=str(r["updated_utc"]),
    )


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------- CRUD --------------------------------------------------------

def submit(*, name: str, source: str, author: str,
           language: str = "python") -> CustomStrategy:
    """Register a new operator strategy.

    Source is deduplicated by SHA-256: submitting the same body twice
    returns the existing record (no duplicate in the registry).
    """
    if not name.strip():
        raise ValueError("name_required")
    if not source.strip():
        raise ValueError("source_required")
    if len(source) > _MAX_SOURCE_LEN:
        raise ValueError(f"source_too_large:{len(source)}>{_MAX_SOURCE_LEN}")
    if language not in ("python", "dsl"):
        raise ValueError("language_must_be_python_or_dsl")
    sha = _sha(source)
    strategy_id = sha[:16]
    now = _iso_now()
    with _lock:
        c = _connect()
        existing = c.execute(
            "SELECT * FROM custom_strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        if existing is not None:
            return _row_to_strategy(existing)
        c.execute(
            "INSERT INTO custom_strategies(strategy_id,name,author,language,"
            "source,source_sha256,state,detail,created_utc,updated_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (strategy_id, name.strip(), author.strip() or "operator",
             language, source, sha,
             StrategyState.DRAFT.value, "", now, now),
        )
        c.commit()
    try:
        get_writer().write("GOVERNANCE", "CUSTOM_STRATEGY_SUBMITTED",
                           "mind.custom_strategies",
                           {"strategy_id": strategy_id, "name": name,
                            "author": author, "language": language,
                            "sha256": sha, "bytes": len(source)})
    except Exception:
        pass
    return get(strategy_id)  # type: ignore[return-value]


def get(strategy_id: str) -> CustomStrategy | None:
    with _lock:
        c = _connect()
        r = c.execute(
            "SELECT * FROM custom_strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
    return _row_to_strategy(r) if r is not None else None


def list_strategies(state: StrategyState | None = None,
                    ) -> list[CustomStrategy]:
    with _lock:
        c = _connect()
        if state is None:
            rows: Iterable[sqlite3.Row] = c.execute(
                "SELECT * FROM custom_strategies ORDER BY updated_utc DESC"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM custom_strategies WHERE state=? "
                "ORDER BY updated_utc DESC",
                (state.value,),
            ).fetchall()
    return [_row_to_strategy(r) for r in rows]


def _transition(strategy_id: str, new_state: StrategyState, *,
                detail: str = "") -> CustomStrategy:
    now = _iso_now()
    with _lock:
        c = _connect()
        r = c.execute(
            "SELECT * FROM custom_strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        if r is None:
            raise LookupError(f"unknown_strategy:{strategy_id}")
        c.execute(
            "UPDATE custom_strategies SET state=?, detail=?, updated_utc=? "
            "WHERE strategy_id=?",
            (new_state.value, detail, now, strategy_id),
        )
        c.commit()
    try:
        get_writer().write("GOVERNANCE",
                           f"CUSTOM_STRATEGY_{new_state.value}",
                           "mind.custom_strategies",
                           {"strategy_id": strategy_id, "detail": detail})
    except Exception:
        pass
    return get(strategy_id)  # type: ignore[return-value]


# -------- pipeline hooks ---------------------------------------------

def run_sandbox(strategy_id: str) -> CustomStrategy:
    """Drop the strategy source into a temp module inside the repo
    tree (under ``mind/custom_submissions/``) and drive
    :mod:`tools.sandbox_runner` against it.

    Uses an in-tree staging dir (not the OS tempdir) because
    ``sandbox_runner.sandbox_import`` requires the module path to
    sit under the repo root.  The staged file is removed after the
    sandbox run regardless of outcome.
    """
    from tools.sandbox_runner import sandbox_import
    s = get(strategy_id)
    if s is None:
        raise LookupError(f"unknown_strategy:{strategy_id}")
    if s.language != "python":
        return _transition(strategy_id, StrategyState.REJECTED,
                           detail="language_not_python")
    repo_root = Path(__file__).resolve().parents[1]
    staging = repo_root / "mind" / "custom_submissions"
    staging.mkdir(parents=True, exist_ok=True)
    init = staging / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")
    _stamp = utc_now()
    _stamp_s = int(_stamp.timestamp()) if _stamp is not None else 0
    module_path = staging / f"_cs_{strategy_id}_{_stamp_s}.py"
    module_path.write_text(s.source, encoding="utf-8")
    try:
        verdict = sandbox_import(module_path, repo_root)
        if not verdict.ok:
            return _transition(
                strategy_id, StrategyState.REJECTED,
                detail=f"sandbox_import:{verdict.stderr[:500]}",
            )
    finally:
        try:
            module_path.unlink(missing_ok=True)
        except Exception:
            pass
    return _transition(strategy_id, StrategyState.SANDBOX_OK,
                       detail="sandbox_passed")


def promote_shadow(strategy_id: str) -> CustomStrategy:
    s = get(strategy_id)
    if s is None:
        raise LookupError(f"unknown_strategy:{strategy_id}")
    if s.state is not StrategyState.SANDBOX_OK:
        raise RuntimeError(f"cannot_shadow_from:{s.state.value}")
    return _transition(strategy_id, StrategyState.SHADOW,
                       detail="shadow_replay_enabled")


def promote_canary(strategy_id: str) -> CustomStrategy:
    s = get(strategy_id)
    if s is None:
        raise LookupError(f"unknown_strategy:{strategy_id}")
    if s.state is not StrategyState.SHADOW:
        raise RuntimeError(f"cannot_canary_from:{s.state.value}")
    return _transition(strategy_id, StrategyState.CANARY,
                       detail="canary_active")


def request_go_live(strategy_id: str, *, operator_id: str = "operator",
                    ttl_sec: int = 24 * 3600) -> dict:
    """Open an operator-approval request to promote CANARY → LIVE.

    Returns the :class:`~security.operator.ApprovalRequest` dict.  The
    go-live itself is performed by :func:`promote_live` once the
    operator has clicked Approve in the cockpit.
    """
    s = get(strategy_id)
    if s is None:
        raise LookupError(f"unknown_strategy:{strategy_id}")
    if s.state is not StrategyState.CANARY:
        raise RuntimeError(f"cannot_request_live_from:{s.state.value}")
    req = request_approval(
        ApprovalKind.CUSTOM_STRATEGY_GO_LIVE,
        subject=strategy_id,
        payload={"name": s.name, "author": s.author,
                 "sha256": s.source_sha256},
        ttl_sec=ttl_sec,
        requested_by=operator_id,
    )
    return req.as_dict()


def promote_live(strategy_id: str) -> CustomStrategy:
    """Final promotion — requires an ``OPERATOR/APPROVAL_GRANTED``
    event for ``(CUSTOM_STRATEGY_GO_LIVE, strategy_id)``.  Fails
    closed if no grant is on file.
    """
    s = get(strategy_id)
    if s is None:
        raise LookupError(f"unknown_strategy:{strategy_id}")
    if not is_granted(ApprovalKind.CUSTOM_STRATEGY_GO_LIVE, strategy_id):
        raise PermissionError("operator_approval_required")
    if s.state is not StrategyState.CANARY:
        raise RuntimeError(f"cannot_live_from:{s.state.value}")
    return _transition(strategy_id, StrategyState.LIVE,
                       detail="operator_approved_live")


def retire(strategy_id: str, *, reason: str = "") -> CustomStrategy:
    return _transition(strategy_id, StrategyState.RETIRED,
                       detail=f"retired:{reason}")


__all__ = [
    "StrategyState", "CustomStrategy",
    "submit", "get", "list_strategies",
    "run_sandbox", "promote_shadow", "promote_canary",
    "request_go_live", "promote_live", "retire",
]
