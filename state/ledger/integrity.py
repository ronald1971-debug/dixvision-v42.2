"""OFFLINE-only hash-chain verifier for the authority ledger (C-08).

# ADAPTED FROM: https://github.com/EventStore/EventStore-Client-Python
# (the read_stream / CatchupSubscription pattern — read every event in order,
#  re-hash, compare against the stored hash; any mismatch flags tamper).
#
# Tier: OFFLINE_ONLY — this module MUST NOT be imported from the hot path
# (``/api/tick`` / ``ExecutionEngine.execute`` / ``GovernanceEngine.process``).
# It walks the entire SQLite ``authority_ledger`` table and re-hashes every
# row. On a one-million-row ledger that is hundreds of milliseconds; way too
# expensive for the hot path. The canonical contract:
#
#   * The WRITER (governance_engine.control_plane.ledger_authority_writer)
#     computes the hash chain INLINE on every append using
#     :func:`state.ledger.hash_chain.canonical_row_bytes` /
#     :func:`compute_chain_hash`. The stored ``hash_chain`` column is the
#     source of truth at write time.
#
#   * This module re-runs the same canonical computation OFFLINE — typically
#     from a tool, a periodic background audit job, or an operator-triggered
#     ``/api/admin/ledger/verify`` endpoint — and compares the recomputed
#     hash against the stored hash. Any divergence flags a tampered or
#     corrupted row.
#
# Determinism (INV-15):
# --------------------
# Pure function of the SQLite file contents. Given two byte-identical files
# (or two byte-identical iterables of ``LedgerEntry``) this module returns
# byte-identical :class:`IntegrityResult` values. No clocks, no randomness.
#
# Authority discipline (B27 / B28 / INV-71):
# -----------------------------------------
# Never constructs typed events. Consumes opaque ``LedgerEntry`` rows that
# the writer already minted; never produces them. Only emits its own value
# objects (:class:`ChainBreak`, :class:`IntegrityResult`) which are pure
# diagnostic structs.
#
# Runtime-tier isolation (B1):
# ---------------------------
# Imports only from ``core.contracts`` (the ``LedgerEntry`` schema) and
# ``state.ledger.hash_chain`` (the canonical primitive). Imports nothing
# from ``intelligence_engine`` / ``execution_engine`` / ``governance_engine``
# / ``learning_engine`` / ``evolution_engine``. Walking the SQLite file is
# done via stdlib ``sqlite3`` against the writer's same on-disk file in
# read-only mode (``mode=ro``) so concurrent writers are not blocked.
#
# Returns / not raises:
# --------------------
# Verification never raises on a chain break — it returns an
# :class:`IntegrityResult` with ``ok=False`` and a list of
# :class:`ChainBreak` rows describing every mismatch. Callers (tools,
# admin endpoint) decide how to surface them. The only exceptions are
# I/O errors (missing file, locked database) which propagate naturally
# from ``sqlite3``.
#
# Outputs (C-08 declared in DIX_MASTER_CANONICAL.md):
#   1. state/ledger/hash_chain.py   ← canonical primitives
#   2. state/ledger/integrity.py    ← this module (OFFLINE verifier)
#   3. tests/test_ledger_hash_chain.py
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from core.contracts.governance import LedgerEntry
from state.ledger.hash_chain import (
    GENESIS_PREV_HASH,
    canonical_row_bytes,
    compute_chain_hash,
    is_valid_hash_hex,
)

__all__ = (
    "INTEGRITY_VERIFIER_VERSION",
    "BreakReason",
    "ChainBreak",
    "IntegrityResult",
    "verify_entries",
    "verify_sqlite",
)


INTEGRITY_VERIFIER_VERSION = 1


# ---------------------------------------------------------------------------
# Diagnostic value-objects
# ---------------------------------------------------------------------------


class BreakReason:
    """String constants for the kinds of chain breaks this verifier detects.

    Strings (not enums) so they serialise to JSON without ceremony for the
    audit-export tooling. Tests pin the exact spelling so a downstream
    operator dashboard widget can switch on them.
    """

    BAD_SEQ = "BAD_SEQ"
    """Row ``seq`` is not contiguous with the previous row (gap or repeat)."""

    BAD_GENESIS = "BAD_GENESIS"
    """Row 0 has a non-genesis ``prev_hash``."""

    BAD_PREV_HASH = "BAD_PREV_HASH"
    """Row N (N>0) has ``prev_hash`` that does not match row N-1's
    ``hash_chain``."""

    BAD_HASH_FORM = "BAD_HASH_FORM"
    """``prev_hash`` or ``hash_chain`` is not a valid 64-char lowercase
    hex string."""

    BAD_CHAIN_HASH = "BAD_CHAIN_HASH"
    """Recomputing the canonical row bytes and re-hashing yields a value
    different from the stored ``hash_chain`` — the row content was
    tampered with after the writer minted the hash."""

    BAD_ROW_SHAPE = "BAD_ROW_SHAPE"
    """Row fields are structurally invalid (negative seq, negative ts_ns,
    empty kind, payload not Mapping[str, str])."""


@dataclass(frozen=True, slots=True)
class ChainBreak:
    """One detected divergence between recomputed and stored chain state."""

    seq: int
    reason: str
    expected: str = ""
    actual: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    """Summary of a full chain walk."""

    ok: bool
    total: int
    breaks: tuple[ChainBreak, ...] = field(default_factory=tuple)

    def first_break(self) -> ChainBreak | None:
        """Return the earliest break by ``seq`` or ``None`` if clean."""
        if not self.breaks:
            return None
        return self.breaks[0]


# ---------------------------------------------------------------------------
# Pure entry-iterator verifier
# ---------------------------------------------------------------------------


def verify_entries(entries: Iterable[LedgerEntry]) -> IntegrityResult:
    """Walk ``entries`` in order; verify every link.

    ``entries`` must be already-sorted by ``seq`` ascending (which is how
    :class:`state.ledger.reader.LedgerReader.authority_entries` returns
    them). The verifier checks:

    * row 0's ``prev_hash`` equals :data:`GENESIS_PREV_HASH`;
    * row N's ``prev_hash`` equals row N-1's ``hash_chain``;
    * each row's stored ``hash_chain`` equals
      ``sha256(prev_hash || canonical_row_bytes)``;
    * every hash field is 64 lowercase hex chars;
    * ``seq`` is contiguous from 0;
    * structural validity (non-negative seq / ts_ns, non-empty kind,
      payload is ``Mapping[str, str]``).

    Returns an :class:`IntegrityResult`. ``breaks`` is ordered by ``seq``
    ascending. On a clean ledger the result is ``IntegrityResult(ok=True,
    total=N, breaks=())``.
    """
    breaks: list[ChainBreak] = []
    prev_hash = GENESIS_PREV_HASH
    expected_seq = 0
    total = 0
    for entry in entries:
        total += 1
        seq = entry.seq

        if seq != expected_seq:
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_SEQ,
                    expected=str(expected_seq),
                    actual=str(seq),
                )
            )
            expected_seq = seq
        expected_seq += 1

        if entry.ts_ns < 0 or not entry.kind:
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_ROW_SHAPE,
                    detail=(f"ts_ns={entry.ts_ns!r} kind={entry.kind!r}"),
                )
            )
            prev_hash = entry.hash_chain
            continue

        try:
            payload_check: dict[str, str] = {}
            for k, v in entry.payload.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError(f"non-string payload k={k!r} v={v!r}")
                payload_check[k] = v
        except TypeError as exc:
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_ROW_SHAPE,
                    detail=str(exc),
                )
            )
            prev_hash = entry.hash_chain
            continue

        if not is_valid_hash_hex(entry.prev_hash):
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_HASH_FORM,
                    actual=entry.prev_hash,
                    detail="prev_hash",
                )
            )
            prev_hash = entry.hash_chain
            continue
        if not is_valid_hash_hex(entry.hash_chain):
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_HASH_FORM,
                    actual=entry.hash_chain,
                    detail="hash_chain",
                )
            )
            prev_hash = entry.hash_chain
            continue

        if seq == 0:
            if entry.prev_hash != GENESIS_PREV_HASH:
                breaks.append(
                    ChainBreak(
                        seq=seq,
                        reason=BreakReason.BAD_GENESIS,
                        expected=GENESIS_PREV_HASH,
                        actual=entry.prev_hash,
                    )
                )
        else:
            if entry.prev_hash != prev_hash:
                breaks.append(
                    ChainBreak(
                        seq=seq,
                        reason=BreakReason.BAD_PREV_HASH,
                        expected=prev_hash,
                        actual=entry.prev_hash,
                    )
                )

        try:
            body = canonical_row_bytes(
                entry.seq,
                entry.ts_ns,
                entry.kind,
                payload_check,
                entry.prev_hash,
            )
        except (ValueError, TypeError) as exc:
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_ROW_SHAPE,
                    detail=str(exc),
                )
            )
            prev_hash = entry.hash_chain
            continue

        recomputed = compute_chain_hash(entry.prev_hash, body)
        if recomputed != entry.hash_chain:
            breaks.append(
                ChainBreak(
                    seq=seq,
                    reason=BreakReason.BAD_CHAIN_HASH,
                    expected=recomputed,
                    actual=entry.hash_chain,
                )
            )

        prev_hash = entry.hash_chain

    breaks.sort(key=lambda b: (b.seq, b.reason))
    return IntegrityResult(ok=not breaks, total=total, breaks=tuple(breaks))


# ---------------------------------------------------------------------------
# SQLite verifier
# ---------------------------------------------------------------------------


def verify_sqlite(db_path: str | Path) -> IntegrityResult:
    """Open ``db_path`` read-only, walk the ``authority_ledger`` table.

    The reader opens the file with ``mode=ro&immutable=0`` so a concurrent
    writer's WAL session is not blocked. Rows are pulled in ``seq ASC``
    order. The schema is the one minted by
    :mod:`governance_engine.control_plane.ledger_authority_writer`
    (PR #164):

        seq INTEGER PRIMARY KEY,
        ts_ns INTEGER NOT NULL,
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,   -- JSON of dict[str, str]
        prev_hash TEXT NOT NULL,
        hash_chain TEXT NOT NULL.

    Returns an :class:`IntegrityResult` (never raises on a chain break;
    raises ``sqlite3.OperationalError`` on missing file / locked db).
    """
    path = Path(db_path)
    uri = f"file:{path.as_posix()}?mode=ro&immutable=0"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(
            "SELECT seq, ts_ns, kind, payload, prev_hash, hash_chain "
            "FROM authority_ledger ORDER BY seq ASC"
        )
        entries: list[LedgerEntry] = []
        for seq, ts_ns, kind, payload_json, prev_hash, chain in cur.fetchall():
            entries.append(
                LedgerEntry(
                    seq=int(seq),
                    ts_ns=int(ts_ns),
                    kind=str(kind),
                    payload=dict(json.loads(payload_json)),
                    prev_hash=str(prev_hash),
                    hash_chain=str(chain),
                )
            )
    finally:
        conn.close()
    return verify_entries(entries)
