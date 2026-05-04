"""Tests for ``governance_engine.control_plane.policy_hash_anchor``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.contracts.events import HazardSeverity
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_hash_anchor import (
    HAZARD_CODE_POLICY_DRIFT,
    LEDGER_KIND_POLICY_HASHES_BOUND,
    PolicyHashAnchor,
    PolicyHashEntry,
    compute_file_hash,
)


def _make_files(tmp_path: Path) -> tuple[tuple[str, Path], ...]:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_bytes(b"version: 1\n")
    b.write_bytes(b"rules: []\n")
    return (("a", a), ("b", b))


def test_bind_session_writes_one_row_with_all_hashes(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)

    entries = anchor.bind_session(ts_ns=100, requestor="test")

    assert len(entries) == 2
    rows = [r for r in ledger.read() if r.kind == LEDGER_KIND_POLICY_HASHES_BOUND]
    assert len(rows) == 1
    payload = rows[0].payload
    assert payload["requestor"] == "test"
    assert payload["a_sha256"] == compute_file_hash(files[0][1])
    assert payload["b_sha256"] == compute_file_hash(files[1][1])


def test_verify_no_drift_returns_none_when_files_unchanged(
    tmp_path: Path,
) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    assert anchor.verify_no_drift(ts_ns=200) is None


def test_verify_no_drift_emits_critical_hazard_on_mutation(
    tmp_path: Path,
) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    files[0][1].write_bytes(b"version: 2\n")  # mid-session edit

    hazard = anchor.verify_no_drift(ts_ns=200)
    assert hazard is not None
    assert hazard.severity is HazardSeverity.CRITICAL
    assert hazard.code == HAZARD_CODE_POLICY_DRIFT
    assert hazard.meta["a_status"] == "mismatch"
    assert hazard.meta["b_status"] == "ok"
    assert hazard.detail == "policy_hash_drift:a"


def test_verify_no_drift_reports_missing_file(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    files[1][1].unlink()

    hazard = anchor.verify_no_drift(ts_ns=200)
    assert hazard is not None
    assert hazard.severity is HazardSeverity.CRITICAL
    assert hazard.meta["b_status"] == "missing"


def test_verify_no_drift_reports_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any non-FileNotFoundError I/O failure must surface as drift, not raise.

    The "never raises" contract on ``verify_no_drift`` matters because the
    method is called on a hot path (periodic drift checks); an unhandled
    ``PermissionError`` from a chmod 000 / racing replace would otherwise
    crash the monitoring loop instead of producing the CRITICAL hazard.
    """

    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    real_read_bytes = Path.read_bytes

    def _explode(self: Path) -> bytes:
        if self == files[0][1]:
            raise PermissionError("simulated chmod 000")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _explode)

    hazard = anchor.verify_no_drift(ts_ns=200)
    assert hazard is not None
    assert hazard.severity is HazardSeverity.CRITICAL
    assert hazard.meta["a_status"] == "unreadable"
    assert hazard.meta["a_error"] == "PermissionError"
    assert hazard.meta["b_status"] == "ok"


def test_verify_no_drift_returns_hazard_when_never_bound(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)

    hazard = anchor.verify_no_drift(ts_ns=100)
    assert hazard is not None
    assert hazard.severity is HazardSeverity.CRITICAL
    assert hazard.detail == "policy_hash_anchor_not_bound"


def test_bind_session_raises_on_missing_file_at_boot(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    a.write_bytes(b"version: 1\n")
    b = tmp_path / "missing.yaml"  # never created
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=(("a", a), ("b", b)))

    with pytest.raises(FileNotFoundError):
        anchor.bind_session(ts_ns=100, requestor="test")


def test_replay_from_ledger_restores_bound_entries(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    fresh = PolicyHashAnchor(ledger=ledger, files=files)
    assert fresh.bound_entries() == ()
    fresh.replay_from_ledger()

    rebuilt = {e.name: e for e in fresh.bound_entries()}
    assert rebuilt["a"].sha256 == compute_file_hash(files[0][1])
    assert rebuilt["b"].sha256 == compute_file_hash(files[1][1])
    # And drift detection still works after replay.
    assert fresh.verify_no_drift(ts_ns=200) is None


def test_rebind_overwrites_cache_and_writes_fresh_row(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="boot")

    # Operator-approved policy reload: file content changes, then
    # rebind. After rebind, drift must be clear.
    files[0][1].write_bytes(b"version: 2\n")
    anchor.bind_session(ts_ns=200, requestor="reload")

    assert anchor.verify_no_drift(ts_ns=300) is None
    rows = [r for r in ledger.read() if r.kind == LEDGER_KIND_POLICY_HASHES_BOUND]
    assert len(rows) == 2
    assert rows[0].payload["requestor"] == "boot"
    assert rows[1].payload["requestor"] == "reload"


def test_compute_file_hash_is_deterministic_over_raw_bytes(
    tmp_path: Path,
) -> None:
    p = tmp_path / "x.yaml"
    p.write_bytes(b"hello\n")
    first = compute_file_hash(p)
    assert compute_file_hash(p) == first

    # Same bytes in different files -> same digest.
    p2 = tmp_path / "y.yaml"
    p2.write_bytes(b"hello\n")
    p3 = tmp_path / "z.yaml"
    p3.write_bytes(b"hello\n")
    assert compute_file_hash(p2) == compute_file_hash(p3)

    # Whitespace difference breaks the hash (raw-bytes contract).
    p.write_bytes(b"hello \n")
    assert compute_file_hash(p) != first


def test_policy_hash_entry_is_immutable() -> None:
    e = PolicyHashEntry(name="a", path=Path("/tmp/x"), sha256="deadbeef")
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        e.name = "b"  # type: ignore[misc]
