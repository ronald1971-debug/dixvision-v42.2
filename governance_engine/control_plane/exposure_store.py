"""AUDIT-P0.4 — Durable exposure book + compliance daily caps.

The :class:`~governance_engine.control_plane.risk_evaluator.ExposureBook`
and :class:`~governance_engine.control_plane.compliance_validator.ComplianceValidator`
historically held their state in process memory. That meant:

* per-symbol exposure tracking (RiskEvaluator's
  :class:`MAX_POSITION_QTY` / :class:`MAX_NOTIONAL_USD` gate) reset
  to zero on every process restart, so an adversary could exit
  out-of-spec positions, restart the harness, and re-enter them
  without the cap noticing;

* per-domain daily caps (the MEMECOIN domain's
  ``max_daily_usd=1000`` cap is the security-critical example)
  reset to zero on every process restart, so a kill-9 plus
  relaunch would open up a fresh allowance regardless of how
  much the day had already spent.

Both of those are bypass channels the architecture-review audit
flagged as P0-4. This module closes them.

Storage shape
-------------

One SQLite database with two tables, opened against the same
``DIXVISION_LEDGER_PATH`` file as the authority ledger. WAL mode +
``synchronous=NORMAL`` matches the ledger's durability profile so
that restart-after-kill recovers consistently.

* ``exposure_book(symbol TEXT PRIMARY KEY, qty REAL NOT NULL,
  updated_ns INTEGER NOT NULL)`` — current per-symbol exposure.
  Updated in-place on every ``ExposureBook.set`` / ``apply``.

* ``compliance_daily(domain TEXT NOT NULL, day_iso TEXT NOT NULL,
  spent_usd REAL NOT NULL, updated_ns INTEGER NOT NULL,
  PRIMARY KEY(domain, day_iso))`` — running notional spend per
  (domain, UTC day). The day-key replaces the in-memory
  ``reset_daily()`` semantics: querying tomorrow's row returns
  zero automatically, so the cap "rolls over" without needing a
  cron / scheduler call.

Determinism
-----------

The store performs **no** clock reads of its own. Callers pass
``ts_ns`` and the corresponding ``day_iso`` so replay produces
identical rows. The single time-source dependency lives in
:mod:`ui.server` (and similar harness boots) which already owns
the ``wall_ns`` reference.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from threading import Lock

__all__ = [
    "ExposureStore",
    "day_iso_from_ns",
]


_DDL_EXPOSURE = """
CREATE TABLE IF NOT EXISTS exposure_book (
    symbol      TEXT PRIMARY KEY,
    qty         REAL NOT NULL,
    updated_ns  INTEGER NOT NULL
)
"""

_DDL_COMPLIANCE = """
CREATE TABLE IF NOT EXISTS compliance_daily (
    domain      TEXT NOT NULL,
    day_iso     TEXT NOT NULL,
    spent_usd   REAL NOT NULL,
    updated_ns  INTEGER NOT NULL,
    PRIMARY KEY(domain, day_iso)
)
"""

_NS_PER_DAY = 86_400 * 1_000_000_000


def day_iso_from_ns(ts_ns: int) -> str:
    """Return the ISO ``YYYY-MM-DD`` UTC day for a wall-ns timestamp.

    Implemented as integer arithmetic (no ``datetime.utcfromtimestamp``
    detour) so callers do not pay locale / DST / timezone-import cost
    on every order. The compliance daily-cap path is hot enough that
    keeping this branch-free matters.
    """

    if ts_ns < 0:
        raise ValueError(
            f"ts_ns must be non-negative, got {ts_ns!r}"
        )
    days = ts_ns // _NS_PER_DAY
    # 1970-01-01 was day 0; arithmetic over Julian day numbers
    # avoids any timezone library at all.
    a = days + 2_440_588  # JD of 1970-01-01 00:00:00 UTC
    b = a + 32044
    c = (4 * b + 3) // 146097
    d = b - (146097 * c) // 4
    e = (4 * d + 3) // 1461
    f = d - (1461 * e) // 4
    g = (5 * f + 2) // 153
    day = f - (153 * g + 2) // 5 + 1
    month = g + 3 - 12 * (g // 10)
    year = 100 * c + e - 4800 + g // 10
    return f"{year:04d}-{month:02d}-{day:02d}"


def _open_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_DDL_EXPOSURE)
    conn.execute(_DDL_COMPLIANCE)
    return conn


class ExposureStore:
    """SQLite-backed store for per-symbol exposures and daily caps.

    The store is a thin persistence sink — the *logic* (cap checks,
    side-signed exposure math) stays in :class:`ExposureBook` and
    :class:`ComplianceValidator`. Boot loads the latest snapshot
    into the in-memory dicts; every accept-side mutation writes
    through.

    Constructing an ``ExposureStore`` with ``db_path=None`` returns a
    no-op instance whose reads yield empty dicts and whose writes
    are silently dropped. That keeps unit tests and ephemeral
    harness modes (``DIXVISION_PERMIT_EPHEMERAL_LEDGER=1``) working
    without an on-disk SQLite file.
    """

    def __init__(self, *, db_path: Path | str | None = None) -> None:
        self._lock = Lock()
        if db_path is None:
            self._conn: sqlite3.Connection | None = None
        else:
            self._conn = _open_sqlite(Path(db_path))

    # ------------------------------------------------------------------
    # ExposureBook side
    # ------------------------------------------------------------------

    def load_exposures(self) -> dict[str, float]:
        """Hydrate every persisted ``(symbol, qty)`` row at boot."""

        if self._conn is None:
            return {}
        with self._lock:
            cur = self._conn.execute(
                "SELECT symbol, qty FROM exposure_book"
            )
            return {symbol: float(qty) for symbol, qty in cur.fetchall()}

    def write_exposure(
        self, *, symbol: str, qty: float, ts_ns: int
    ) -> None:
        """Upsert the current exposure for ``symbol``."""

        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO exposure_book(symbol, qty, updated_ns) "
                "VALUES(?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "  qty = excluded.qty, "
                "  updated_ns = excluded.updated_ns",
                (symbol, float(qty), int(ts_ns)),
            )

    # ------------------------------------------------------------------
    # ComplianceValidator side
    # ------------------------------------------------------------------

    def load_daily(self, *, day_iso: str) -> dict[str, float]:
        """Hydrate today's ``(domain, spent_usd)`` rows at boot.

        Rows for past days remain on disk for audit but are not
        loaded into memory — the in-memory map only ever tracks
        today's running totals, matching the historical semantics
        of :meth:`ComplianceValidator.reset_daily`.
        """

        if self._conn is None:
            return {}
        with self._lock:
            cur = self._conn.execute(
                "SELECT domain, spent_usd FROM compliance_daily "
                "WHERE day_iso = ?",
                (day_iso,),
            )
            return {
                domain: float(spent) for domain, spent in cur.fetchall()
            }

    def write_daily(
        self,
        *,
        domain: str,
        day_iso: str,
        spent_usd: float,
        ts_ns: int,
    ) -> None:
        """Upsert today's running spend total for ``domain``."""

        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO compliance_daily("
                "  domain, day_iso, spent_usd, updated_ns) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(domain, day_iso) DO UPDATE SET "
                "  spent_usd = excluded.spent_usd, "
                "  updated_ns = excluded.updated_ns",
                (
                    domain,
                    day_iso,
                    float(spent_usd),
                    int(ts_ns),
                ),
            )

    # ------------------------------------------------------------------
    # Test / observability surface
    # ------------------------------------------------------------------

    def snapshot_daily(self) -> Mapping[tuple[str, str], float]:
        """Return every persisted ``(domain, day) -> spent`` row.

        Used by tests + the operator dashboard's compliance widget.
        Not wired into the hot path.
        """

        if self._conn is None:
            return {}
        with self._lock:
            cur = self._conn.execute(
                "SELECT domain, day_iso, spent_usd FROM compliance_daily"
            )
            return {
                (domain, day): float(spent)
                for domain, day, spent in cur.fetchall()
            }

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None
