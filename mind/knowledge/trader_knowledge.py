"""
mind.knowledge.trader_knowledge \u2014 worldwide trader/strategy knowledge store.

Sqlite-backed, append-friendly, consumed by the cockpit chat ("what would
Druckenmiller do?"), the strategy arbiter (philosophy vectors), and the
language-agnostic retrieval layer.

Schema:
    traders     one row per person / fund / pseudonym
    strategies  one row per documented strategy style (value / macro / trend / etc.)
    statements  philosophical quotes + trades + signal snippets + source link + lang
    mentions    auto-incremented counter for organic growth / popularity ranking
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system.config import get_config


@dataclass
class Trader:
    id: int
    name: str
    era: str = ""           # e.g. "1923-1940", "present", "1990\u2013present"
    region: str = ""        # ISO country or "global"
    style_tags: str = ""    # csv: value, macro, momentum, quant, market-maker, ...
    cautionary: bool = False
    bio_summary: str = ""
    bio_lang: str = "en"
    source_url: str = ""


@dataclass
class Strategy:
    id: int
    name: str
    family: str              # "value" / "macro" / "trend" / "quant" / "mean_revert" / ...
    description: str
    source_url: str = ""


@dataclass
class Statement:
    id: int
    trader_id: int | None
    strategy_id: int | None
    kind: str               # "quote" | "trade" | "signal" | "fact"
    text: str
    lang: str = "en"
    en_summary: str = ""
    source_url: str = ""
    confidence: float = 0.3
    ledger_ref: int = 0


_DEFAULT_DB = "data/trader_knowledge.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    era          TEXT DEFAULT '',
    region       TEXT DEFAULT '',
    style_tags   TEXT DEFAULT '',
    cautionary   INTEGER DEFAULT 0,
    bio_summary  TEXT DEFAULT '',
    bio_lang     TEXT DEFAULT 'en',
    source_url   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS strategies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    family       TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    source_url   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS statements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trader_id    INTEGER,
    strategy_id  INTEGER,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    lang         TEXT DEFAULT 'en',
    en_summary   TEXT DEFAULT '',
    source_url   TEXT DEFAULT '',
    confidence   REAL DEFAULT 0.3,
    ledger_ref   INTEGER DEFAULT 0,
    FOREIGN KEY(trader_id)   REFERENCES traders(id),
    FOREIGN KEY(strategy_id) REFERENCES strategies(id)
);
CREATE TABLE IF NOT EXISTS mentions (
    trader_id    INTEGER PRIMARY KEY,
    count        INTEGER DEFAULT 0,
    last_seen_utc TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_statements_trader ON statements(trader_id);
CREATE INDEX IF NOT EXISTS idx_statements_strategy ON statements(strategy_id);
CREATE INDEX IF NOT EXISTS idx_traders_name ON traders(name);
CREATE INDEX IF NOT EXISTS idx_traders_style ON traders(style_tags);
"""


class TraderKnowledge:
    def __init__(self, db_path: str | None = None) -> None:
        try:
            cfg = get_config()
            path_val = cfg.get("TRADER_KB_DB", _DEFAULT_DB)
        except Exception:
            path_val = _DEFAULT_DB
        path = Path(db_path or path_val)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.commit()

    # ---- traders -----------------------------------------------------
    def upsert_trader(self, name: str, **fields: Any) -> int:
        with self._lock:
            row = self._conn.execute("SELECT id FROM traders WHERE name=?", (name,)).fetchone()
            if row:
                if fields:
                    cols = ", ".join(f"{k}=?" for k in fields.keys())
                    self._conn.execute(f"UPDATE traders SET {cols} WHERE id=?",
                                       tuple(list(fields.values()) + [row[0]]))
                    self._conn.commit()
                return int(row[0])
            cols = ["name"] + list(fields.keys())
            vals = [name] + list(fields.values())
            placeholders = ",".join("?" * len(cols))
            cur = self._conn.execute(
                f"INSERT INTO traders ({','.join(cols)}) VALUES ({placeholders})", tuple(vals)
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def get_trader(self, name: str) -> Trader | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT id,name,era,region,style_tags,cautionary,bio_summary,bio_lang,source_url "
                "FROM traders WHERE name=?", (name,)).fetchone()
        return None if not r else Trader(
            id=r[0], name=r[1], era=r[2] or "", region=r[3] or "",
            style_tags=r[4] or "", cautionary=bool(r[5]),
            bio_summary=r[6] or "", bio_lang=r[7] or "en", source_url=r[8] or "")

    def find_traders(self, q: str = "", style: str = "", region: str = "",
                     limit: int = 50) -> list[Trader]:
        where, args = [], []
        if q:
            where.append("name LIKE ?")
            args.append(f"%{q}%")
        if style:
            where.append("style_tags LIKE ?")
            args.append(f"%{style}%")
        if region:
            where.append("region LIKE ?")
            args.append(f"%{region}%")
        sql = ("SELECT id,name,era,region,style_tags,cautionary,bio_summary,bio_lang,source_url "
               "FROM traders" + (" WHERE " + " AND ".join(where) if where else "") +
               " ORDER BY name LIMIT ?")
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [Trader(id=r[0], name=r[1], era=r[2] or "", region=r[3] or "",
                       style_tags=r[4] or "", cautionary=bool(r[5]),
                       bio_summary=r[6] or "", bio_lang=r[7] or "en",
                       source_url=r[8] or "") for r in rows]

    def count(self) -> dict[str, int]:
        with self._lock:
            t = self._conn.execute("SELECT COUNT(*) FROM traders").fetchone()[0]
            s = self._conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
            st = self._conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0]
        return {"traders": int(t), "strategies": int(s), "statements": int(st)}

    # ---- strategies --------------------------------------------------
    def upsert_strategy(self, name: str, family: str = "",
                        description: str = "", source_url: str = "") -> int:
        with self._lock:
            row = self._conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()
            if row:
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO strategies (name,family,description,source_url) VALUES (?,?,?,?)",
                (name, family, description, source_url))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    # ---- statements --------------------------------------------------
    def add_statement(self, *, trader_id: int | None = None,
                      strategy_id: int | None = None, kind: str = "fact",
                      text: str = "", lang: str = "en", en_summary: str = "",
                      source_url: str = "", confidence: float = 0.3,
                      ledger_ref: int = 0) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO statements (trader_id,strategy_id,kind,text,lang,en_summary,source_url,confidence,ledger_ref) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (trader_id, strategy_id, kind, text, lang, en_summary, source_url, confidence, ledger_ref))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def statements_for(self, trader_id: int, limit: int = 20) -> list[Statement]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,trader_id,strategy_id,kind,text,lang,en_summary,source_url,confidence,ledger_ref "
                "FROM statements WHERE trader_id=? ORDER BY id DESC LIMIT ?",
                (trader_id, limit)).fetchall()
        return [Statement(id=r[0], trader_id=r[1], strategy_id=r[2], kind=r[3],
                          text=r[4], lang=r[5] or "en", en_summary=r[6] or "",
                          source_url=r[7] or "", confidence=float(r[8] or 0.0),
                          ledger_ref=int(r[9] or 0)) for r in rows]

    # ---- mentions ----------------------------------------------------
    def bump_mention(self, trader_id: int, now_utc: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO mentions (trader_id,count,last_seen_utc) VALUES (?,1,?) "
                "ON CONFLICT(trader_id) DO UPDATE SET count=count+1, last_seen_utc=excluded.last_seen_utc",
                (trader_id, now_utc))
            self._conn.commit()

    def top_mentioned(self, limit: int = 50) -> list[tuple[Trader, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.id,t.name,t.era,t.region,t.style_tags,t.cautionary,t.bio_summary,"
                "t.bio_lang,t.source_url,m.count "
                "FROM mentions m JOIN traders t ON t.id=m.trader_id "
                "ORDER BY m.count DESC LIMIT ?", (limit,)).fetchall()
        out: list[tuple[Trader, int]] = []
        for r in rows:
            out.append((Trader(id=r[0], name=r[1], era=r[2] or "", region=r[3] or "",
                               style_tags=r[4] or "", cautionary=bool(r[5]),
                               bio_summary=r[6] or "", bio_lang=r[7] or "en",
                               source_url=r[8] or ""), int(r[9] or 0)))
        return out


_kb: TraderKnowledge | None = None
_kb_lock = threading.Lock()


def get_trader_knowledge() -> TraderKnowledge:
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = TraderKnowledge()
    return _kb


def seed_default(kb: TraderKnowledge | None = None) -> dict[str, int]:
    """Populate the curated worldwide roster. Idempotent."""
    from mind.knowledge.trader_seed import CURATED_STRATEGIES, CURATED_TRADERS
    kb = kb or get_trader_knowledge()
    for s in CURATED_STRATEGIES:
        kb.upsert_strategy(**s)
    for t in CURATED_TRADERS:
        # Do NOT mutate the curated module-level dict; idempotency requires
        # the list to be re-readable across calls.  Copy, then split off the
        # statements key from the copy.
        trader = dict(t)
        statements = trader.pop("statements", [])
        tid = kb.upsert_trader(**trader)
        for st in statements:
            kb.add_statement(trader_id=tid, **st)
    return kb.count()


__all__ = [
    "Trader", "Strategy", "Statement",
    "TraderKnowledge", "get_trader_knowledge", "seed_default",
]
