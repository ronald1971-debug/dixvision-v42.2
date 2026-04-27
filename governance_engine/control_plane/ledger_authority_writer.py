"""GOV-CP-05 — Ledger Authority Writer.

The **only** module permitted to append to the authority ledger
(``manifest.md`` §0.5 GOV-CP-05). Every approved governance decision
flows through ``append`` and is recorded as a hash-chained
:class:`LedgerEntry`.

Phase 1 stores the chain in memory (``list[LedgerEntry]``); the durable
hash-chain backend lands in Phase 4 (Dyon) reusing the same public
surface so the rest of the Control Plane is unaffected.

Determinism contract (INV-15 / TEST-01): given the same sequence of
``append`` calls, every entry — including the chain hash — is
bit-identical across runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from threading import Lock

from core.contracts.governance import LedgerEntry

# 32 zero hex digits — distinct from any sha256 output (which is 64 chars)
# so the genesis row is unambiguous when reading the chain.
GENESIS_PREV_HASH = "0" * 64


def _canonical_payload(payload: Mapping[str, str]) -> str:
    """Stable serialisation: sorted ``k=v`` pairs joined by ``\\x1f``.

    Used to build the bytes that go into the chain hash. ``\\x1f`` (Unit
    Separator) is a control character that cannot appear in a Python
    identifier or in a base16 digest, which keeps the encoding
    unambiguous.
    """

    parts = [f"{k}={payload[k]}" for k in sorted(payload)]
    return "\x1f".join(parts)


def _row_bytes(
    seq: int, ts_ns: int, kind: str, payload: Mapping[str, str], prev_hash: str
) -> bytes:
    canonical = (
        f"{seq}\x1e{ts_ns}\x1e{kind}\x1e{_canonical_payload(payload)}"
        f"\x1e{prev_hash}"
    )
    return canonical.encode("utf-8")


class LedgerAuthorityWriter:
    """Append-only hash-chained store of governance decisions.

    Thread-safe: ``append`` and ``read`` are guarded by a single lock,
    so multi-threaded operator bridges or hazard sensors cannot
    interleave rows.
    """

    name: str = "ledger_authority_writer"
    spec_id: str = "GOV-CP-05"

    def __init__(self) -> None:
        self._rows: list[LedgerEntry] = []
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        ts_ns: int,
        kind: str,
        payload: Mapping[str, str],
    ) -> LedgerEntry:
        """Append one row, returning the materialised :class:`LedgerEntry`.

        Empty ``kind`` is rejected — every authority row must declare
        what kind of decision it represents (e.g.
        ``"MODE_TRANSITION"``, ``"PLUGIN_LIFECYCLE"``,
        ``"OPERATOR_REJECTED"``).
        """

        if not kind:
            raise ValueError("kind must be non-empty")

        with self._lock:
            seq = len(self._rows)
            prev_hash = (
                self._rows[-1].hash_chain if self._rows else GENESIS_PREV_HASH
            )
            row_bytes = _row_bytes(seq, ts_ns, kind, payload, prev_hash)
            chain = hashlib.sha256(prev_hash.encode("ascii") + row_bytes).hexdigest()
            entry = LedgerEntry(
                seq=seq,
                ts_ns=ts_ns,
                kind=kind,
                payload=dict(payload),
                prev_hash=prev_hash,
                hash_chain=chain,
            )
            self._rows.append(entry)
            return entry

    def read(self) -> Sequence[LedgerEntry]:
        """Snapshot of the current chain (oldest first)."""

        with self._lock:
            return tuple(self._rows)

    def head_hash(self) -> str:
        """Hash of the most recently appended row (or genesis)."""

        with self._lock:
            return self._rows[-1].hash_chain if self._rows else GENESIS_PREV_HASH

    def verify(self) -> bool:
        """Recompute the chain top-to-bottom; return ``False`` on tamper."""

        with self._lock:
            prev = GENESIS_PREV_HASH
            for entry in self._rows:
                row_bytes = _row_bytes(
                    entry.seq, entry.ts_ns, entry.kind, entry.payload, prev
                )
                expected = hashlib.sha256(
                    prev.encode("ascii") + row_bytes
                ).hexdigest()
                if expected != entry.hash_chain or entry.prev_hash != prev:
                    return False
                prev = entry.hash_chain
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


__all__ = ["GENESIS_PREV_HASH", "LedgerAuthorityWriter"]
