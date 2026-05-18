"""C-08 canonical adaptation tests — authority-ledger hash chain.

# ADAPTED FROM: https://github.com/EventStore/EventStore-Client-Python
# (stream hash-chain semantics — every event embeds the previous hash; any
#  in-place tamper is detectable by re-hashing the chain end-to-end).

Pins:

* ``state.ledger.hash_chain`` canonical row bytes are byte-identical to the
  writer's PR #164 ``_row_bytes`` (writer-compatibility lock — they must
  agree on every byte for INV-15 to hold).
* ``state.ledger.integrity.verify_entries`` accepts any chain the writer
  produces and rejects every kind of break.
* INV-15 byte-identical 3-run replay: the same input always yields the
  same canonical bytes and the same chain hash.
* B27 / B28 / INV-71: no typed-event constructors called from the module.
* B1: no runtime-tier imports (no ``intelligence_engine`` / ``execution_engine``
  / ``governance_engine`` / ``learning_engine`` / ``evolution_engine``).
* No top-level imports of ``time`` / ``datetime`` / ``random`` / ``asyncio``
  / ``os`` / ``esdbclient`` / ``numpy`` / ``torch`` / ``polars`` / ``requests``
  / ``httpx`` / ``aiohttp`` / ``tornado``.
* Lazy-seam discipline: ``NEW_PIP_DEPENDENCIES`` declared but the dep is
  never imported.
"""

from __future__ import annotations

import ast
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from core.contracts.governance import LedgerEntry
from state.ledger import hash_chain as hc
from state.ledger.hash_chain import (
    FIELD_SEPARATOR,
    GENESIS_PREV_HASH,
    HASH_CHAIN_VERSION,
    HEX_DIGEST_LENGTH,
    NEW_PIP_DEPENDENCIES,
    PAYLOAD_KV_SEPARATOR,
    PAYLOAD_SEPARATOR,
    canonical_payload,
    canonical_payload_bytes,
    canonical_row_bytes,
    compute_chain_hash,
    is_genesis_hash,
    is_valid_hash_hex,
    link,
)
from state.ledger.integrity import (
    INTEGRITY_VERIFIER_VERSION,
    BreakReason,
    ChainBreak,
    IntegrityResult,
    verify_entries,
    verify_sqlite,
)

# ---------------------------------------------------------------------------
# Module-surface constants
# ---------------------------------------------------------------------------


def test_hash_chain_version_is_one() -> None:
    assert HASH_CHAIN_VERSION == 1


def test_integrity_verifier_version_is_one() -> None:
    assert INTEGRITY_VERIFIER_VERSION == 1


def test_new_pip_dependencies_declared_but_not_imported() -> None:
    assert NEW_PIP_DEPENDENCIES == ("esdbclient",)
    assert "esdbclient" not in sys.modules


def test_separator_constants_are_control_bytes() -> None:
    assert FIELD_SEPARATOR == "\x1e"
    assert PAYLOAD_SEPARATOR == "\x1f"
    assert PAYLOAD_KV_SEPARATOR == "="


def test_hex_digest_length_matches_sha256() -> None:
    assert HEX_DIGEST_LENGTH == 64


def test_genesis_prev_hash_is_64_zeros() -> None:
    assert GENESIS_PREV_HASH == "0" * 64
    assert len(GENESIS_PREV_HASH) == HEX_DIGEST_LENGTH


# ---------------------------------------------------------------------------
# canonical_payload
# ---------------------------------------------------------------------------


def test_canonical_payload_empty() -> None:
    assert canonical_payload({}) == ""
    assert canonical_payload_bytes({}) == b""


def test_canonical_payload_single_pair() -> None:
    assert canonical_payload({"k": "v"}) == "k=v"


def test_canonical_payload_sorts_by_key() -> None:
    out = canonical_payload({"b": "2", "a": "1", "c": "3"})
    assert out == "a=1\x1fb=2\x1fc=3"


def test_canonical_payload_byte_stable_across_insertion_order() -> None:
    a = canonical_payload({"a": "1", "b": "2", "c": "3"})
    b = canonical_payload({"c": "3", "a": "1", "b": "2"})
    c = canonical_payload({"b": "2", "c": "3", "a": "1"})
    assert a == b == c


def test_canonical_payload_rejects_non_string_key() -> None:
    with pytest.raises(TypeError):
        canonical_payload({1: "v"})  # type: ignore[dict-item]


def test_canonical_payload_rejects_non_string_value() -> None:
    with pytest.raises(TypeError):
        canonical_payload({"k": 1})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# canonical_row_bytes
# ---------------------------------------------------------------------------


def test_canonical_row_bytes_form() -> None:
    body = canonical_row_bytes(
        seq=0,
        ts_ns=123,
        kind="MODE",
        payload={"to": "PAPER"},
        prev_hash=GENESIS_PREV_HASH,
    )
    expected = (f"0\x1e123\x1eMODE\x1eto=PAPER\x1e{GENESIS_PREV_HASH}").encode()
    assert body == expected


def test_canonical_row_bytes_rejects_negative_seq() -> None:
    with pytest.raises(ValueError):
        canonical_row_bytes(-1, 0, "MODE", {}, GENESIS_PREV_HASH)


def test_canonical_row_bytes_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError):
        canonical_row_bytes(0, -1, "MODE", {}, GENESIS_PREV_HASH)


def test_canonical_row_bytes_rejects_empty_kind() -> None:
    with pytest.raises(ValueError):
        canonical_row_bytes(0, 0, "", {}, GENESIS_PREV_HASH)


def test_canonical_row_bytes_rejects_bad_prev_hash() -> None:
    with pytest.raises(ValueError):
        canonical_row_bytes(0, 0, "MODE", {}, "deadbeef")
    with pytest.raises(ValueError):
        canonical_row_bytes(0, 0, "MODE", {}, "G" * 64)


def test_canonical_row_bytes_rejects_non_string_kind() -> None:
    with pytest.raises(TypeError):
        canonical_row_bytes(0, 0, 42, {}, GENESIS_PREV_HASH)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Writer compatibility lock — bytes match PR #164 helpers
# ---------------------------------------------------------------------------


def test_writer_compat_canonical_payload_matches() -> None:
    from governance_engine.control_plane.ledger_authority_writer import (
        _canonical_payload,
    )

    cases = [
        {},
        {"a": "1"},
        {"b": "2", "a": "1"},
        {"to": "PAPER", "by": "operator", "code": "MODE-CHANGE"},
    ]
    for payload in cases:
        assert canonical_payload(payload) == _canonical_payload(payload)


def test_writer_compat_row_bytes_match() -> None:
    from governance_engine.control_plane.ledger_authority_writer import _row_bytes

    cases = [
        (0, 1_000_000, "MODE", {"to": "PAPER"}, GENESIS_PREV_HASH),
        (1, 2_000_000, "STRATEGY", {"id": "s1", "to": "APPROVED"}, "a" * 64),
        (
            42,
            999_999_999,
            "POLICY",
            {"version": "v2", "hash": "deadbeef" * 8},
            "b" * 64,
        ),
    ]
    for seq, ts_ns, kind, payload, prev_hash in cases:
        mine = canonical_row_bytes(seq, ts_ns, kind, payload, prev_hash)
        theirs = _row_bytes(seq, ts_ns, kind, payload, prev_hash)
        assert mine == theirs


def test_writer_compat_chain_hash_matches() -> None:
    import hashlib

    body = canonical_row_bytes(0, 1_000_000, "MODE", {"to": "PAPER"}, GENESIS_PREV_HASH)
    mine = compute_chain_hash(GENESIS_PREV_HASH, body)
    theirs = hashlib.sha256(GENESIS_PREV_HASH.encode("ascii") + body).hexdigest()
    assert mine == theirs


# ---------------------------------------------------------------------------
# compute_chain_hash / link
# ---------------------------------------------------------------------------


def test_compute_chain_hash_form() -> None:
    body = canonical_row_bytes(0, 0, "MODE", {}, GENESIS_PREV_HASH)
    h = compute_chain_hash(GENESIS_PREV_HASH, body)
    assert is_valid_hash_hex(h)
    assert h != GENESIS_PREV_HASH


def test_compute_chain_hash_rejects_bad_prev_hash() -> None:
    with pytest.raises(ValueError):
        compute_chain_hash("not-hex", b"row")


def test_compute_chain_hash_rejects_non_bytes_body() -> None:
    with pytest.raises(TypeError):
        compute_chain_hash(GENESIS_PREV_HASH, "row")  # type: ignore[arg-type]


def test_link_returns_pair() -> None:
    body, h = link(0, 0, "MODE", {}, GENESIS_PREV_HASH)
    assert isinstance(body, bytes)
    assert is_valid_hash_hex(h)
    assert h != GENESIS_PREV_HASH


def test_link_byte_stable() -> None:
    a = link(7, 42, "STRATEGY", {"a": "1", "b": "2"}, "f" * 64)
    b = link(7, 42, "STRATEGY", {"b": "2", "a": "1"}, "f" * 64)
    c = link(7, 42, "STRATEGY", {"a": "1", "b": "2"}, "f" * 64)
    assert a == b == c


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def test_is_genesis_hash_only_for_zeros() -> None:
    assert is_genesis_hash(GENESIS_PREV_HASH)
    assert not is_genesis_hash("0" * 63 + "1")
    assert not is_genesis_hash("")


def test_is_valid_hash_hex_lowercase_only() -> None:
    assert is_valid_hash_hex("0" * 64)
    assert is_valid_hash_hex("abcdef0123456789" * 4)
    assert not is_valid_hash_hex("ABCDEF" + "0" * 58)
    assert not is_valid_hash_hex("0" * 63)
    assert not is_valid_hash_hex("0" * 65)
    assert not is_valid_hash_hex("g" + "0" * 63)
    assert not is_valid_hash_hex(0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Building a valid chain
# ---------------------------------------------------------------------------


def _build_chain(rows: list[tuple[int, str, dict[str, str]]]) -> list[LedgerEntry]:
    """Helper: mint a chain of LedgerEntry rows for ``[(ts_ns, kind, payload), ...]``."""
    entries: list[LedgerEntry] = []
    prev = GENESIS_PREV_HASH
    for i, (ts_ns, kind, payload) in enumerate(rows):
        body = canonical_row_bytes(i, ts_ns, kind, payload, prev)
        h = compute_chain_hash(prev, body)
        entries.append(
            LedgerEntry(
                seq=i,
                ts_ns=ts_ns,
                kind=kind,
                payload=dict(payload),
                prev_hash=prev,
                hash_chain=h,
            )
        )
        prev = h
    return entries


def test_verify_entries_empty_chain_is_ok() -> None:
    r = verify_entries([])
    assert r.ok
    assert r.total == 0
    assert r.breaks == ()


def test_verify_entries_single_genesis_row_is_ok() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"})])
    r = verify_entries(chain)
    assert r.ok
    assert r.total == 1
    assert r.breaks == ()


def test_verify_entries_long_chain_is_ok() -> None:
    chain = _build_chain([(1_000 + i, "MODE", {"to": "PAPER", "step": str(i)}) for i in range(50)])
    r = verify_entries(chain)
    assert r.ok
    assert r.total == 50


# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------


def test_verify_detects_bad_genesis() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"})])
    tampered = [
        LedgerEntry(
            seq=chain[0].seq,
            ts_ns=chain[0].ts_ns,
            kind=chain[0].kind,
            payload=chain[0].payload,
            prev_hash="a" * 64,
            hash_chain=chain[0].hash_chain,
        )
    ]
    r = verify_entries(tampered)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_GENESIS in reasons


def test_verify_detects_bad_prev_hash() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"}), (2_000, "STRATEGY", {"id": "s1"})])
    bad = list(chain)
    bad[1] = LedgerEntry(
        seq=chain[1].seq,
        ts_ns=chain[1].ts_ns,
        kind=chain[1].kind,
        payload=chain[1].payload,
        prev_hash="b" * 64,
        hash_chain=chain[1].hash_chain,
    )
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_PREV_HASH in reasons


def test_verify_detects_payload_tamper() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"})])
    bad = [
        LedgerEntry(
            seq=chain[0].seq,
            ts_ns=chain[0].ts_ns,
            kind=chain[0].kind,
            payload={"to": "AUTO"},
            prev_hash=chain[0].prev_hash,
            hash_chain=chain[0].hash_chain,
        )
    ]
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_CHAIN_HASH in reasons


def test_verify_detects_kind_tamper() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"}), (2_000, "STRATEGY", {"id": "s1"})])
    bad = list(chain)
    bad[1] = LedgerEntry(
        seq=chain[1].seq,
        ts_ns=chain[1].ts_ns,
        kind="POLICY",
        payload=chain[1].payload,
        prev_hash=chain[1].prev_hash,
        hash_chain=chain[1].hash_chain,
    )
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_CHAIN_HASH in reasons


def test_verify_detects_ts_tamper() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"})])
    bad = [
        LedgerEntry(
            seq=chain[0].seq,
            ts_ns=chain[0].ts_ns + 1,
            kind=chain[0].kind,
            payload=chain[0].payload,
            prev_hash=chain[0].prev_hash,
            hash_chain=chain[0].hash_chain,
        )
    ]
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_CHAIN_HASH in reasons


def test_verify_detects_seq_gap() -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"}), (2_000, "STRATEGY", {"id": "s1"})])
    bad = list(chain)
    bad[1] = LedgerEntry(
        seq=5,
        ts_ns=chain[1].ts_ns,
        kind=chain[1].kind,
        payload=chain[1].payload,
        prev_hash=chain[1].prev_hash,
        hash_chain=chain[1].hash_chain,
    )
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_SEQ in reasons


def test_verify_detects_bad_hash_form() -> None:
    bad = [
        LedgerEntry(
            seq=0,
            ts_ns=1_000,
            kind="MODE",
            payload={"to": "PAPER"},
            prev_hash="UPPERCASE" + "0" * 55,
            hash_chain="b" * 64,
        )
    ]
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_HASH_FORM in reasons


def test_verify_detects_bad_row_shape_empty_kind() -> None:
    bad = [
        LedgerEntry(
            seq=0,
            ts_ns=1_000,
            kind="",
            payload={},
            prev_hash=GENESIS_PREV_HASH,
            hash_chain="b" * 64,
        )
    ]
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_ROW_SHAPE in reasons


def test_verify_detects_bad_row_shape_negative_ts() -> None:
    bad = [
        LedgerEntry(
            seq=0,
            ts_ns=-1,
            kind="MODE",
            payload={},
            prev_hash=GENESIS_PREV_HASH,
            hash_chain="b" * 64,
        )
    ]
    r = verify_entries(bad)
    assert not r.ok
    reasons = {b.reason for b in r.breaks}
    assert BreakReason.BAD_ROW_SHAPE in reasons


def test_first_break_returns_earliest_by_seq() -> None:
    chain = _build_chain(
        [
            (1_000, "MODE", {"to": "PAPER"}),
            (2_000, "STRATEGY", {"id": "s1"}),
            (3_000, "POLICY", {"v": "1"}),
        ]
    )
    bad = list(chain)
    bad[2] = LedgerEntry(
        seq=2,
        ts_ns=chain[2].ts_ns,
        kind=chain[2].kind,
        payload={"v": "TAMPER"},
        prev_hash=chain[2].prev_hash,
        hash_chain=chain[2].hash_chain,
    )
    bad[1] = LedgerEntry(
        seq=1,
        ts_ns=chain[1].ts_ns,
        kind=chain[1].kind,
        payload={"id": "TAMPER"},
        prev_hash=chain[1].prev_hash,
        hash_chain=chain[1].hash_chain,
    )
    r = verify_entries(bad)
    first = r.first_break()
    assert first is not None
    assert first.seq == 1


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_inv15_canonical_row_bytes_three_run_equality() -> None:
    args = (3, 12_345, "POLICY", {"k": "v", "a": "b"}, "c" * 64)
    a = canonical_row_bytes(*args)
    b = canonical_row_bytes(*args)
    c = canonical_row_bytes(*args)
    assert a == b == c


def test_inv15_chain_hash_three_run_equality() -> None:
    args = (3, 12_345, "POLICY", {"k": "v", "a": "b"}, "c" * 64)
    a = link(*args)
    b = link(*args)
    c = link(*args)
    assert a == b == c


def test_inv15_full_chain_three_run_byte_identical() -> None:
    def build() -> list[LedgerEntry]:
        return _build_chain(
            [
                (1_000, "MODE", {"to": "PAPER"}),
                (2_000, "STRATEGY", {"id": "s1", "to": "APPROVED"}),
                (3_000, "POLICY", {"v": "2", "hash": "deadbeef" * 8}),
                (4_000, "POLICY-DRIFT", {"prev": "1", "next": "2"}),
                (5_000, "MODE", {"to": "AUTO"}),
            ]
        )

    a = build()
    b = build()
    c = build()
    assert a == b == c
    ra = verify_entries(a)
    rb = verify_entries(b)
    rc = verify_entries(c)
    assert ra == rb == rc
    assert ra.ok and ra.total == 5


# ---------------------------------------------------------------------------
# SQLite verifier
# ---------------------------------------------------------------------------


def _write_sqlite_chain(
    path: Path,
    entries: list[LedgerEntry],
) -> None:
    """Write ``entries`` into the same SQLite schema as the writer."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE authority_ledger (
                seq INTEGER PRIMARY KEY,
                ts_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash_chain TEXT NOT NULL
            )
            """
        )
        for e in entries:
            conn.execute(
                "INSERT INTO authority_ledger "
                "(seq, ts_ns, kind, payload, prev_hash, hash_chain) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    e.seq,
                    e.ts_ns,
                    e.kind,
                    json.dumps(dict(e.payload), sort_keys=True),
                    e.prev_hash,
                    e.hash_chain,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_verify_sqlite_clean_chain(tmp_path: Path) -> None:
    chain = _build_chain([(1_000 + i, "MODE", {"to": "PAPER", "step": str(i)}) for i in range(20)])
    db = tmp_path / "ledger.sqlite"
    _write_sqlite_chain(db, chain)
    r = verify_sqlite(db)
    assert r.ok
    assert r.total == 20


def test_verify_sqlite_tampered_payload(tmp_path: Path) -> None:
    chain = _build_chain([(1_000, "MODE", {"to": "PAPER"}), (2_000, "STRATEGY", {"id": "s1"})])
    db = tmp_path / "ledger.sqlite"
    _write_sqlite_chain(db, chain)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "UPDATE authority_ledger SET payload=? WHERE seq=1",
            (json.dumps({"id": "TAMPERED"}, sort_keys=True),),
        )
        conn.commit()
    finally:
        conn.close()
    r = verify_sqlite(db)
    assert not r.ok
    assert any(b.reason == BreakReason.BAD_CHAIN_HASH for b in r.breaks)


def test_verify_sqlite_three_run_byte_identical(tmp_path: Path) -> None:
    chain = _build_chain([(1_000 + i, "MODE", {"to": "PAPER", "step": str(i)}) for i in range(10)])
    db = tmp_path / "ledger.sqlite"
    _write_sqlite_chain(db, chain)
    a = verify_sqlite(db)
    b = verify_sqlite(db)
    c = verify_sqlite(db)
    assert a == b == c


def test_verify_sqlite_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        verify_sqlite(tmp_path / "no-such-file.sqlite")


# ---------------------------------------------------------------------------
# AST guardrails — pin tier rules at the source level
# ---------------------------------------------------------------------------


HASH_CHAIN_PATH = Path(hc.__file__)
INTEGRITY_PATH = Path(importlib.import_module("state.ledger.integrity").__file__ or "")

FORBIDDEN_TOPLEVEL_IMPORTS = {
    "time",
    "datetime",
    "random",
    "asyncio",
    "os",
    "esdbclient",
    "numpy",
    "torch",
    "polars",
    "requests",
    "httpx",
    "aiohttp",
    "tornado",
}

RUNTIME_TIERS = {
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "evolution_engine",
    "learning_engine",
}

TYPED_EVENT_CONSTRUCTORS = {
    "PatchProposal",
    "HazardEvent",
    "SignalEvent",
    "ExecutionEvent",
    "SystemEvent",
}


def _toplevel_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_hash_chain_no_forbidden_toplevel_imports() -> None:
    imported = _toplevel_imported_modules(HASH_CHAIN_PATH)
    bad = imported & FORBIDDEN_TOPLEVEL_IMPORTS
    assert not bad, f"forbidden top-level imports in hash_chain.py: {sorted(bad)}"


def test_integrity_no_forbidden_toplevel_imports() -> None:
    imported = _toplevel_imported_modules(INTEGRITY_PATH)
    bad = imported & FORBIDDEN_TOPLEVEL_IMPORTS
    assert not bad, f"forbidden top-level imports in integrity.py: {sorted(bad)}"


def test_hash_chain_no_runtime_tier_imports() -> None:
    imported = _toplevel_imported_modules(HASH_CHAIN_PATH)
    bad = imported & RUNTIME_TIERS
    assert not bad, f"runtime-tier imports in hash_chain.py: {sorted(bad)}"


def test_integrity_no_runtime_tier_imports() -> None:
    imported = _toplevel_imported_modules(INTEGRITY_PATH)
    bad = imported & RUNTIME_TIERS
    assert not bad, f"runtime-tier imports in integrity.py: {sorted(bad)}"


def _has_typed_event_ctor(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            if isinstance(func, ast.Name) and func.id in TYPED_EVENT_CONSTRUCTORS:
                found.add(func.id)
            if isinstance(func, ast.Attribute) and func.attr in TYPED_EVENT_CONSTRUCTORS:
                found.add(func.attr)
            self.generic_visit(node)

    V().visit(tree)
    return found


def test_hash_chain_no_typed_event_constructors() -> None:
    found = _has_typed_event_ctor(HASH_CHAIN_PATH)
    assert not found, f"typed-event ctors in hash_chain.py (B27/B28/INV-71): {found}"


def test_integrity_no_typed_event_constructors() -> None:
    found = _has_typed_event_ctor(INTEGRITY_PATH)
    assert not found, f"typed-event ctors in integrity.py (B27/B28/INV-71): {found}"


def test_module_reimports_clean() -> None:
    """Reimporting both modules must not pull in ``esdbclient``."""
    sys.modules.pop("esdbclient", None)
    importlib.reload(hc)
    importlib.reload(importlib.import_module("state.ledger.integrity"))
    assert "esdbclient" not in sys.modules


# ---------------------------------------------------------------------------
# Diagnostic structs
# ---------------------------------------------------------------------------


def test_chain_break_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    b = ChainBreak(seq=0, reason=BreakReason.BAD_GENESIS)
    with pytest.raises(FrozenInstanceError):
        b.seq = 1  # type: ignore[misc]


def test_integrity_result_first_break_clean() -> None:
    r = IntegrityResult(ok=True, total=0, breaks=())
    assert r.first_break() is None


def test_break_reason_string_set_pins_canonical_spelling() -> None:
    assert BreakReason.BAD_SEQ == "BAD_SEQ"
    assert BreakReason.BAD_GENESIS == "BAD_GENESIS"
    assert BreakReason.BAD_PREV_HASH == "BAD_PREV_HASH"
    assert BreakReason.BAD_HASH_FORM == "BAD_HASH_FORM"
    assert BreakReason.BAD_CHAIN_HASH == "BAD_CHAIN_HASH"
    assert BreakReason.BAD_ROW_SHAPE == "BAD_ROW_SHAPE"
