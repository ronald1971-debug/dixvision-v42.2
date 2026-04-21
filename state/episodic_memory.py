"""
state.episodic_memory \u2014 bounded ring-buffer of trade episodes.

Each episode captures (context, action, outcome, reward) for later replay
by the strategy arbiter / alpha-decay monitor. No vectordb, no embeddings.
SQLite WAL with rolling delete to keep size bounded.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from system.time_source import utc_now

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "episodes.sqlite"
_MAX_ROWS = 50_000                                              # hard cap
_LRU_CAP = 2_000                                                # in-memory tail

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    context_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    reward REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS ep_strategy ON episodes(strategy);
CREATE INDEX IF NOT EXISTS ep_symbol   ON episodes(symbol);
CREATE INDEX IF NOT EXISTS ep_ts       ON episodes(ts_utc);
"""


@dataclass
class Episode:
    id: int
    ts_utc: str
    strategy: str
    symbol: str
    side: str
    context: dict[str, object] = field(default_factory=dict)
    action: dict[str, object] = field(default_factory=dict)
    outcome: dict[str, object] = field(default_factory=dict)
    reward: float = 0.0


class EpisodicMemory:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = Path(db_path or _DB_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.commit()
        self._tail: deque[Episode] = deque(maxlen=_LRU_CAP)

    def record(self, *, strategy: str, symbol: str, side: str,
               context: dict[str, object], action: dict[str, object],
               outcome: dict[str, object], reward: float) -> Episode:
        ts = utc_now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes "
                "(ts_utc, strategy, symbol, side, context_json, "
                "action_json, outcome_json, reward) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, strategy, symbol, side,
                 json.dumps(context, default=str),
                 json.dumps(action, default=str),
                 json.dumps(outcome, default=str), float(reward)),
            )
            self._conn.commit()
            rid = int(cur.lastrowid or 0)
            self._trim_locked()
        ep = Episode(id=rid, ts_utc=ts, strategy=strategy, symbol=symbol,
                     side=side, context=context, action=action,
                     outcome=outcome, reward=float(reward))
        self._tail.append(ep)
        return ep

    def recent(self, *, strategy: str = "", symbol: str = "",
               limit: int = 200) -> list[Episode]:
        where, args = [], []
        if strategy:
            where.append("strategy = ?")
            args.append(strategy)
        if symbol:
            where.append("symbol = ?")
            args.append(symbol)
        sql = ("SELECT id, ts_utc, strategy, symbol, side, "
               "context_json, action_json, outcome_json, reward "
               "FROM episodes" +
               (" WHERE " + " AND ".join(where) if where else "") +
               " ORDER BY id DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        out: list[Episode] = []
        for r in rows:
            out.append(Episode(
                id=int(r[0]), ts_utc=r[1], strategy=r[2], symbol=r[3], side=r[4],
                context=json.loads(r[5] or "{}"),
                action=json.loads(r[6] or "{}"),
                outcome=json.loads(r[7] or "{}"),
                reward=float(r[8]),
            ))
        return out

    def reward_window(self, strategy: str, *, n: int = 100) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT reward FROM episodes WHERE strategy=? "
                "ORDER BY id DESC LIMIT ?", (strategy, int(n))).fetchall()
        return [float(r[0]) for r in rows]

    def count(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(r[0]) if r else 0

    def _trim_locked(self) -> None:
        r = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        total = int(r[0]) if r else 0
        if total <= _MAX_ROWS:
            return
        drop = total - _MAX_ROWS
        self._conn.execute(
            "DELETE FROM episodes WHERE id IN ("
            "SELECT id FROM episodes ORDER BY id ASC LIMIT ?)",
            (drop,),
        )
        self._conn.commit()


_singleton: EpisodicMemory | None = None
_singleton_lock = threading.Lock()


def get_episodic_memory() -> EpisodicMemory:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = EpisodicMemory()
    return _singleton


__all__ = ["Episode", "EpisodicMemory", "get_episodic_memory"]
