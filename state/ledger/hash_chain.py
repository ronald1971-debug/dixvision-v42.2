"""Canonical hash-chain primitives for the authority ledger (C-08 / esdbclient).

# ADAPTED FROM: https://github.com/EventStore/EventStore-Client-Python
# (esdbclient/client.py — append_to_stream / read_stream / CatchupSubscription;
#  EventStoreDB stream hash-chain pattern where each event embeds the hash of
#  the previous event so any in-place tamper is detectable by re-hashing).
#
# Tier: PURE / OFFLINE-OR-WRITER — this module is consumed by
# :mod:`governance_engine.control_plane.ledger_authority_writer` (the only
# module allowed to APPEND to the chain) and by :mod:`state.ledger.integrity`
# (the only module allowed to VERIFY the chain). Both surfaces share the same
# canonical bytes definition so a writer and a verifier on different machines
# will agree on every chain hash byte-for-byte (INV-15).
#
# Determinism contract (INV-15):
# -----------------------------
# Given identical ``(seq, ts_ns, kind, payload, prev_hash)`` inputs, this
# module always produces the same canonical row bytes and the same
# ``hash_chain`` hex string. No clocks, no randomness, no env vars. Three
# independent runs from the same inputs produce byte-identical output.
#
# Authority discipline (B27 / B28 / INV-71):
# -----------------------------------------
# This module **never** constructs typed events (PatchProposal / HazardEvent /
# SignalEvent / ExecutionEvent / SystemEvent). It only operates on opaque
# ``(seq, ts_ns, kind, payload, prev_hash)`` tuples that the writer feeds in.
# The writer (in ``governance_engine.control_plane``) constructs the
# ``LedgerEntry`` rows; this module only computes the canonical bytes and the
# sha256 hash. Authority symmetry preserved.
#
# Runtime-tier isolation (B1):
# ---------------------------
# This module imports nothing from ``intelligence_engine``, ``execution_engine``,
# ``governance_engine``, ``evolution_engine``, ``learning_engine``. It is a
# leaf of the dependency graph; it lives in ``state.*`` which is allowed for
# both governance writers and offline verifiers per the existing L2 allow-list.
#
# Why a separate module from the writer:
# -------------------------------------
# The writer in PR #164 inlined ``_canonical_payload`` / ``_row_bytes`` /
# ``hashlib.sha256(...)``. Extracting them here lets:
#   * an OFFLINE verifier (``integrity.verify_*``) re-derive the chain on a
#     copy of the SQLite file without depending on the writer module;
#   * tooling (``tools/`` dump scripts, replay rigs) verify a row's hash
#     without importing governance code;
#   * an external auditor (e.g. an EventStoreDB-style mirror) port the same
#     canonical form to a different storage backend (SQL / object store /
#     append-only file) and stay byte-compatible.
#
# Canonical row form:
# ------------------
# Row bytes = UTF-8 encoding of
#
#     "{seq}\x1e{ts_ns}\x1e{kind}\x1e{canonical_payload}\x1e{prev_hash}"
#
# where ``canonical_payload`` is the sorted ``k=v`` pairs of ``payload``
# joined by ``\x1f`` (Unit Separator). ``\x1e`` (Record Separator) divides
# the top-level fields. Both separators are control bytes that cannot
# occur in valid kind / hex strings, so the canonical form is unambiguous.
#
# Chain hash = sha256(prev_hash_ascii || row_bytes).hexdigest().
#
# Outputs (C-08 declared in DIX_MASTER_CANONICAL.md):
#   1. state/ledger/hash_chain.py  ← this module (the canonical primitives)
#   2. state/ledger/integrity.py   ← offline verifier
#   3. tests/test_ledger_hash_chain.py
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Final

__all__ = (
    "HASH_CHAIN_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "FIELD_SEPARATOR",
    "PAYLOAD_SEPARATOR",
    "PAYLOAD_KV_SEPARATOR",
    "HEX_DIGEST_LENGTH",
    "GENESIS_PREV_HASH",
    "canonical_payload",
    "canonical_payload_bytes",
    "canonical_row_bytes",
    "compute_chain_hash",
    "link",
    "is_genesis_hash",
    "is_valid_hash_hex",
)


# ---------------------------------------------------------------------------
# Surface constants
# ---------------------------------------------------------------------------

#: Schema version of the canonical hash-chain form. Bumped when the row-bytes
#: layout changes in a non-backward-compatible way. The current form has been
#: stable since PR #164.
HASH_CHAIN_VERSION: Final[int] = 1

#: Declared dependency from the canonical C-08 prompt. Never imported here;
#: the verifier uses sqlite3 (stdlib) against the same SQLite file that the
#: writer maintains. ``esdbclient`` would only be needed if/when an
#: EventStoreDB mirror is wired as an alternative backend (governance research
#: gate — not in scope for OFFLINE_ONLY).
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("esdbclient",)

#: ``\x1e`` Record Separator — divides the top-level row fields.
FIELD_SEPARATOR: Final[str] = "\x1e"

#: ``\x1f`` Unit Separator — divides individual ``k=v`` pairs inside the
#: canonical payload.
PAYLOAD_SEPARATOR: Final[str] = "\x1f"

#: ``=`` divides the key from the value inside one canonical ``k=v`` pair.
#: A literal ``=`` in keys would corrupt the form; the writer constrains
#: payload to ``Mapping[str, str]`` where keys come from a fixed set of
#: governance schema names, none of which contain ``=``.
PAYLOAD_KV_SEPARATOR: Final[str] = "="

#: 64 hex digits = 256 bits. Matches sha256 output and the SQLite column.
HEX_DIGEST_LENGTH: Final[int] = 64

#: 64 ``0``s — distinct from any sha256 output, so the genesis row is
#: unambiguous when reading the chain. Pinned compatible with PR #164.
GENESIS_PREV_HASH: Final[str] = "0" * HEX_DIGEST_LENGTH


# ---------------------------------------------------------------------------
# Canonical bytes
# ---------------------------------------------------------------------------


def canonical_payload(payload: Mapping[str, str]) -> str:
    """Return the canonical ``k=v\\x1fk=v\\x1f…`` string for ``payload``.

    Stable: keys are sorted ascending by Python string order, then joined
    with :data:`PAYLOAD_SEPARATOR`. Empty mapping yields the empty string
    (NOT a leading separator), which matches the writer's PR #164 form.

    Raises
    ------
    TypeError
        If any key or value is not a ``str`` — the underlying contract
        ``Mapping[str, str]`` is strict at the canonical-form boundary.
    """
    items: list[str] = []
    for k in sorted(payload.keys()):
        v = payload[k]
        if not isinstance(k, str):
            raise TypeError(f"canonical_payload: key must be str, got {type(k).__name__}")
        if not isinstance(v, str):
            raise TypeError(
                f"canonical_payload: value for key {k!r} must be str, got {type(v).__name__}"
            )
        items.append(f"{k}{PAYLOAD_KV_SEPARATOR}{v}")
    return PAYLOAD_SEPARATOR.join(items)


def canonical_payload_bytes(payload: Mapping[str, str]) -> bytes:
    """UTF-8 bytes of :func:`canonical_payload`."""
    return canonical_payload(payload).encode("utf-8")


def canonical_row_bytes(
    seq: int,
    ts_ns: int,
    kind: str,
    payload: Mapping[str, str],
    prev_hash: str,
) -> bytes:
    """Return canonical row bytes for one ledger entry.

    Compatible with :func:`governance_engine.control_plane.ledger_authority_writer._row_bytes`
    (the test suite pins this — :func:`tests.test_ledger_hash_chain` compares
    every byte against the writer's helper to lock the form).

    Form
    ----

    ``"{seq}\\x1e{ts_ns}\\x1e{kind}\\x1e{canonical_payload}\\x1e{prev_hash}"``

    Then UTF-8 encoded. The four record separators are control bytes that
    cannot appear in any of the fields (seq / ts_ns are integers; kind is
    a governance enum string; payload values are constrained; prev_hash
    is 64 hex chars).

    Raises
    ------
    TypeError
        If ``kind`` or ``prev_hash`` are not strings, or if any payload key
        or value is not a string.
    ValueError
        If ``seq`` is negative, ``ts_ns`` is negative, ``kind`` is empty,
        or ``prev_hash`` is not 64 hex characters (including the genesis
        all-zero form).
    """
    if not isinstance(kind, str):
        raise TypeError(f"canonical_row_bytes: kind must be str, got {type(kind).__name__}")
    if not isinstance(prev_hash, str):
        raise TypeError(
            f"canonical_row_bytes: prev_hash must be str, got {type(prev_hash).__name__}"
        )
    if seq < 0:
        raise ValueError(f"canonical_row_bytes: seq must be >= 0, got {seq}")
    if ts_ns < 0:
        raise ValueError(f"canonical_row_bytes: ts_ns must be >= 0, got {ts_ns}")
    if not kind:
        raise ValueError("canonical_row_bytes: kind must be non-empty")
    if not is_valid_hash_hex(prev_hash):
        raise ValueError(
            f"canonical_row_bytes: prev_hash must be {HEX_DIGEST_LENGTH} "
            f"hex chars, got {prev_hash!r}"
        )
    body = (
        f"{seq}{FIELD_SEPARATOR}"
        f"{ts_ns}{FIELD_SEPARATOR}"
        f"{kind}{FIELD_SEPARATOR}"
        f"{canonical_payload(payload)}{FIELD_SEPARATOR}"
        f"{prev_hash}"
    )
    return body.encode("utf-8")


# ---------------------------------------------------------------------------
# Hash linking
# ---------------------------------------------------------------------------


def compute_chain_hash(prev_hash: str, row_bytes: bytes) -> str:
    """Compute ``sha256(prev_hash_ascii || row_bytes).hexdigest()``.

    The two-stage concatenation (prev_hash as a prefix, then the row body
    which itself ends with prev_hash) double-binds each row to its
    predecessor, matching the EventStoreDB pattern. Compatible with PR #164.

    Raises
    ------
    TypeError
        If ``row_bytes`` is not bytes.
    ValueError
        If ``prev_hash`` is not :data:`HEX_DIGEST_LENGTH` hex chars.
    """
    if not isinstance(row_bytes, bytes):
        raise TypeError(
            f"compute_chain_hash: row_bytes must be bytes, got {type(row_bytes).__name__}"
        )
    if not is_valid_hash_hex(prev_hash):
        raise ValueError(
            "compute_chain_hash: prev_hash must be "
            f"{HEX_DIGEST_LENGTH} hex chars, got {prev_hash!r}"
        )
    return hashlib.sha256(prev_hash.encode("ascii") + row_bytes).hexdigest()


def link(
    seq: int,
    ts_ns: int,
    kind: str,
    payload: Mapping[str, str],
    prev_hash: str,
) -> tuple[bytes, str]:
    """Convenience: return ``(canonical_row_bytes, chain_hash)`` for one row.

    Calls :func:`canonical_row_bytes` then :func:`compute_chain_hash`. The
    writer uses these two helpers separately (because it also persists the
    row bytes in the SQLite payload column) — this convenience exists for
    verifiers and tooling that only need the hash.
    """
    body = canonical_row_bytes(seq, ts_ns, kind, payload, prev_hash)
    chain = compute_chain_hash(prev_hash, body)
    return body, chain


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_genesis_hash(h: str) -> bool:
    """``True`` iff ``h`` is exactly :data:`GENESIS_PREV_HASH`."""
    return h == GENESIS_PREV_HASH


def is_valid_hash_hex(h: str) -> bool:
    """``True`` iff ``h`` is :data:`HEX_DIGEST_LENGTH` lowercase hex chars.

    Genesis (all-zero) form counts as valid. ``hexdigest()`` returns
    lowercase, so any uppercase form is rejected — a writer-side typo or
    an externally-injected tamper attempt that flips case is caught here.
    """
    if not isinstance(h, str):
        return False
    if len(h) != HEX_DIGEST_LENGTH:
        return False
    for ch in h:
        if ch not in "0123456789abcdef":
            return False
    return True
