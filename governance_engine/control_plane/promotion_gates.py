"""GOV-CP-08 -- Promotion-Gate hash anchor (Reviewer #4 finding 4).

The :class:`PromotionGates` adapter binds the SHA-256 hash of
``docs/promotion_gates.yaml`` to the authority ledger at PAPER
entry and refuses every forward transition into ``CANARY`` /
``LIVE`` / ``AUTO`` whose live file hash does not match the bound
hash. The mechanism converts the PAPER window from "a time during
which we observe results" into "a time during which we cannot
revise the rules", which is the only configuration in which
observed results can be trusted (see ``docs/promotion_gates.yaml``
header for the full discipline rationale).

SHADOW-DEMOLITION-02 collapsed the system-mode SHADOW tier into
PAPER, so the binding moment moved one ratchet earlier. The gated
targets (``CANARY`` / ``LIVE`` / ``AUTO``) are unchanged.

Determinism: the hash is computed over the **raw bytes** of the
file, not over the parsed YAML, so whitespace and comment edits
also break the bound hash. That is intentional -- mid-window edits
to comments are mid-window edits to the document of record.

This module is the **only** writer of ``PROMOTION_GATES_BOUND``
ledger rows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

LEDGER_KIND_PROMOTION_GATES_BOUND = "PROMOTION_GATES_BOUND"

# Forward modes that require a matching bound hash to enter.
_GATED_FORWARD_TARGETS: frozenset[str] = frozenset(
    {"CANARY", "LIVE", "AUTO"}
)


# Default location of the promotion-gates yaml relative to the repo
# root. Resolved once at import time so the rest of the file does not
# need to perform path arithmetic.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMOTION_GATES_PATH: Path = _REPO_ROOT / "docs" / "promotion_gates.yaml"


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the file's raw bytes.

    Pure (no IO outside the explicit ``read_bytes``); deterministic.
    Raises :class:`FileNotFoundError` if the file does not exist --
    callers must handle that, because a missing promotion-gates file
    is itself a governance failure (the document of record is gone).
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromotionGatesHashMismatchError(RuntimeError):
    """Raised when the live file hash differs from the bound hash."""


class PromotionGates:
    """Hash-anchored promotion-gate enforcement.

    Lifecycle:

    1. ``bind(ts_ns, requestor)`` -- called by
       :class:`StateTransitionManager` when PAPER is entered. Reads
       the live ``promotion_gates.yaml``, computes the SHA-256 hash,
       appends a ``PROMOTION_GATES_BOUND`` ledger row, caches the
       hash in memory.
    2. ``check(target_mode_name) -> (ok, code)`` -- called by
       :class:`StateTransitionManager` *before* every forward
       transition into a gated target (``CANARY`` / ``LIVE`` /
       ``AUTO``). Returns ``(False, "PROMOTION_GATES_*")`` if the
       file is missing, no bound hash exists, or the live hash does
       not match.

    The cached hash is reset every time ``bind`` is called -- so a
    new PAPER entry restarts the anchor.
    """

    name: str = "promotion_gates"
    spec_id: str = "GOV-CP-08"

    def __init__(
        self,
        *,
        ledger: LedgerAuthorityWriter,
        path: Path = DEFAULT_PROMOTION_GATES_PATH,
    ) -> None:
        self._ledger = ledger
        self._path = path
        self._bound_hash: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bound_hash(self) -> str | None:
        """Return the most recently bound hash (in-memory) or ``None``."""

        return self._bound_hash

    def bind(self, *, ts_ns: int, requestor: str) -> str:
        """Anchor the live file hash to the ledger.

        Returns the bound hash. Raises :class:`FileNotFoundError` if
        the file is missing -- this surfaces as a governance failure
        at PAPER entry rather than silently letting the system run
        without a document of record.
        """

        digest = compute_file_hash(self._path)
        # Try to record a repo-relative path so the ledger row stays
        # portable across machines; fall back to the absolute path when
        # callers point at a file outside the repo (notably tests using
        # ``tmp_path``).
        try:
            recorded_path = str(self._path.relative_to(_REPO_ROOT))
        except ValueError:
            recorded_path = str(self._path)
        payload: Mapping[str, str] = {
            "requestor": requestor,
            "promotion_gates_sha256": digest,
            "promotion_gates_path": recorded_path,
        }
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_PROMOTION_GATES_BOUND,
            payload=payload,
        )
        self._bound_hash = digest
        return digest

    def check(self, target_mode_name: str) -> tuple[bool, str]:
        """Verify the live hash matches the bound hash.

        Pure (other than the file read inside
        :func:`compute_file_hash`).

        Returns:
            ``(True, "")`` if the target is not a gated mode, or if
            the gate passes.
            ``(False, code)`` otherwise. ``code`` is one of:

            * ``"PROMOTION_GATES_NOT_BOUND"`` -- ``bind`` was never
              called (SHADOW was never entered cleanly).
            * ``"PROMOTION_GATES_FILE_MISSING"`` -- the document of
              record has been deleted.
            * ``"PROMOTION_GATES_HASH_MISMATCH"`` -- the live file
              hash differs from the bound hash (mid-window edit).
        """

        if target_mode_name not in _GATED_FORWARD_TARGETS:
            return True, ""

        if self._bound_hash is None:
            return False, "PROMOTION_GATES_NOT_BOUND"

        try:
            live = compute_file_hash(self._path)
        except FileNotFoundError:
            return False, "PROMOTION_GATES_FILE_MISSING"

        if live != self._bound_hash:
            return False, "PROMOTION_GATES_HASH_MISMATCH"

        return True, ""

    # ------------------------------------------------------------------
    # Replay (INV-15 / TEST-01)
    # ------------------------------------------------------------------

    def replay_from_ledger(self) -> None:
        """Reconstitute the bound hash from the ledger.

        Walks the ledger and adopts the most recent
        ``PROMOTION_GATES_BOUND`` row's ``promotion_gates_sha256`` as
        the in-memory bound hash. This is the determinism contract
        that lets governance bring up a fresh process and pick up the
        gate state as it was at the last SHADOW entry.
        """

        latest_hash: str | None = None
        for entry in self._ledger.read():
            if entry.kind == LEDGER_KIND_PROMOTION_GATES_BOUND:
                latest_hash = entry.payload.get("promotion_gates_sha256")
        self._bound_hash = latest_hash
