"""AUDIT-P0.4 regression tests — durable exposure book + daily caps.

These tests assert that the in-memory bypass channel flagged by
the architecture-review audit is closed:

* ``test_exposure_book_persists_across_restarts`` — apply a BUY,
  drop the book, re-open against the same SQLite file, and verify
  the new book reads the persisted exposure.
* ``test_compliance_daily_persists_across_restarts`` — accept four
  $200 MEMECOIN trades (under the $1000 daily cap), drop the
  validator, re-open against the same SQLite file, and verify the
  fifth trade is rejected — i.e. a kill-9 plus relaunch does NOT
  reopen a fresh allowance.
* ``test_compliance_daily_rolls_over_on_new_day`` — accept four
  $200 MEMECOIN trades on day N at the cap, then re-validate on
  day N+1 and verify spend resets to zero (durable rollover, no
  scheduler call required).
* ``test_governance_engine_wires_exposure_store`` — constructing a
  :class:`GovernanceEngine` with an ``exposure_store`` propagates
  the store into both the :class:`RiskEvaluator`'s book and the
  :class:`ComplianceValidator`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.governance import SystemMode
from governance_engine.control_plane.compliance_validator import (
    ComplianceValidator,
)
from governance_engine.control_plane.exposure_store import (
    ExposureStore,
    day_iso_from_ns,
)
from governance_engine.control_plane.risk_evaluator import ExposureBook
from governance_engine.engine import GovernanceEngine

DAY_NS = 86_400 * 1_000_000_000


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "exposure.db"


def test_exposure_book_persists_across_restarts(store_path: Path) -> None:
    store_a = ExposureStore(db_path=store_path)
    book_a = ExposureBook(store=store_a)
    book_a.apply("BTC-USD", "BUY", 1.5, ts_ns=1_000_000_000)
    book_a.apply("ETH-USD", "BUY", 4.0, ts_ns=1_000_000_001)
    book_a.apply("ETH-USD", "SELL", 1.0, ts_ns=1_000_000_002)
    store_a.close()

    store_b = ExposureStore(db_path=store_path)
    book_b = ExposureBook(store=store_b)
    assert book_b.get("BTC-USD") == pytest.approx(1.5)
    assert book_b.get("ETH-USD") == pytest.approx(3.0)
    store_b.close()


def test_compliance_daily_persists_across_restarts(
    store_path: Path,
) -> None:
    ts = 1_700_000_000_000_000_000  # arbitrary fixed ns timestamp

    store_a = ExposureStore(db_path=store_path)
    cv_a = ComplianceValidator(store=store_a)
    for _ in range(4):
        ok = cv_a.validate_order(
            domain="MEMECOIN",
            notional_usd=200.0,
            mode=SystemMode.LIVE,
            ts_ns=ts,
        )
        assert ok.passed is True
    store_a.close()

    # Simulated kill-9 + relaunch: brand-new validator + store
    # against the same SQLite file. The fifth trade must be
    # rejected — the audit invariant.
    store_b = ExposureStore(db_path=store_path)
    cv_b = ComplianceValidator(store=store_b)
    breach = cv_b.validate_order(
        domain="MEMECOIN",
        notional_usd=250.0,
        mode=SystemMode.LIVE,
        ts_ns=ts,
    )
    assert breach.passed is False
    assert any(
        v.startswith("COMPLIANCE_DAILY_CAP:MEMECOIN")
        for v in breach.violations
    )
    store_b.close()


def test_compliance_daily_rolls_over_on_new_day(
    store_path: Path,
) -> None:
    day_n_ns = 1_700_000_000_000_000_000
    day_n_plus_1_ns = day_n_ns + DAY_NS
    assert day_iso_from_ns(day_n_ns) != day_iso_from_ns(day_n_plus_1_ns)

    store = ExposureStore(db_path=store_path)
    cv = ComplianceValidator(store=store)

    # Day N — exhaust the cap.
    for _ in range(4):
        ok = cv.validate_order(
            domain="MEMECOIN",
            notional_usd=200.0,
            mode=SystemMode.LIVE,
            ts_ns=day_n_ns,
        )
        assert ok.passed is True
    breach = cv.validate_order(
        domain="MEMECOIN",
        notional_usd=250.0,
        mode=SystemMode.LIVE,
        ts_ns=day_n_ns,
    )
    assert breach.passed is False

    # Day N+1 — fresh allowance, no scheduler call.
    fresh = cv.validate_order(
        domain="MEMECOIN",
        notional_usd=200.0,
        mode=SystemMode.LIVE,
        ts_ns=day_n_plus_1_ns,
    )
    assert fresh.passed is True

    # Both days are persisted — the audit trail is durable.
    snapshot = dict(store.snapshot_daily())
    assert (
        snapshot.get(("MEMECOIN", day_iso_from_ns(day_n_ns))) == 800.0
    )
    assert (
        snapshot.get(
            ("MEMECOIN", day_iso_from_ns(day_n_plus_1_ns))
        )
        == 200.0
    )
    store.close()


def test_governance_engine_wires_exposure_store(
    store_path: Path,
) -> None:
    store = ExposureStore(db_path=store_path)
    eng = GovernanceEngine(exposure_store=store)
    # Both downstream primitives received the store reference --
    # asserting via the documented public surface.
    assert eng.risk.book._store is store  # type: ignore[attr-defined]
    assert eng.compliance._store is store  # type: ignore[attr-defined]
    store.close()


def test_compliance_validator_in_memory_default_unchanged() -> None:
    """Sanity: tests + ephemeral harness mode keep the legacy shape.

    Without an :class:`ExposureStore`, ``ComplianceValidator``
    must behave exactly as it did before AUDIT-P0.4 — no SQLite
    file, no day-key mutation between calls, ``reset_daily()``
    still clears the running map.
    """

    cv = ComplianceValidator()
    for _ in range(4):
        assert (
            cv.validate_order(
                domain="MEMECOIN",
                notional_usd=200.0,
                mode=SystemMode.LIVE,
            ).passed
            is True
        )
    breach = cv.validate_order(
        domain="MEMECOIN",
        notional_usd=250.0,
        mode=SystemMode.LIVE,
    )
    assert breach.passed is False

    cv.reset_daily()
    fresh = cv.validate_order(
        domain="MEMECOIN",
        notional_usd=200.0,
        mode=SystemMode.LIVE,
    )
    assert fresh.passed is True


def test_risk_evaluator_commit_propagates_ts_ns(
    store_path: Path,
) -> None:
    """``RiskEvaluator.commit`` must persist the assessment's ts_ns.

    Regression for Devin Review BUG_0001 on PR #196 -- the initial
    AUDIT-P0.4 commit threaded ``ts_ns`` through ``ExposureBook.set``
    and direct ``apply`` calls, but ``RiskEvaluator.commit`` still
    invoked ``self._book.apply(symbol, side, qty)`` without the
    timestamp, silently writing ``updated_ns=0`` for every
    governance-committed exposure. The audit row keeps the qty but
    loses the timestamp -- breaking the durable-audit invariant.
    """

    from core.contracts.governance import RiskAssessment
    from governance_engine.control_plane.risk_evaluator import RiskEvaluator

    store = ExposureStore(db_path=store_path)
    book = ExposureBook(store=store)
    evaluator = RiskEvaluator(exposure_book=book)

    assessment = RiskAssessment(
        ts_ns=1_700_000_000_000_000_000,
        symbol="BTC-USD",
        side="BUY",
        qty=2.5,
        approved=True,
        rejection_code="",
        breached_limits=(),
        exposure_after=2.5,
    )
    evaluator.commit(assessment)
    store.close()

    # Re-open the SQLite file with a raw connection to inspect the
    # persisted updated_ns -- the assertion the bug reporter
    # flagged.
    import sqlite3

    conn = sqlite3.connect(str(store_path))
    try:
        row = conn.execute(
            "SELECT symbol, qty, updated_ns FROM exposure_book "
            "WHERE symbol = ?",
            ("BTC-USD",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "BTC-USD"
    assert row[1] == pytest.approx(2.5)
    assert row[2] == 1_700_000_000_000_000_000


def test_day_iso_from_ns_known_dates() -> None:
    # 2023-11-14 22:13:20 UTC == 1700000000 s
    assert day_iso_from_ns(1_700_000_000_000_000_000) == "2023-11-14"
    # Epoch
    assert day_iso_from_ns(0) == "1970-01-01"
    # Day after epoch
    assert day_iso_from_ns(DAY_NS) == "1970-01-02"
    # Negative ns rejected
    with pytest.raises(ValueError):
        day_iso_from_ns(-1)
