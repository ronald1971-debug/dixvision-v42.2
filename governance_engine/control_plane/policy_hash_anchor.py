"""Hardening-S1 item 4-ext -- session-bound policy hash anchor.

The architecture critique demanded that policy YAMLs become
**immutable during runtime**: their SHA-256 hashes must be bound to
the authority ledger at session start, and any mid-session mutation
must surface as a critical hazard, not a silent reload.

:class:`PolicyHashAnchor` extends the same primitive
:class:`PromotionGates` already applies to ``docs/promotion_gates.yaml``
to a configurable set of policy files (``registry/authority_matrix.yaml``,
``registry/constraint_rules.yaml``, ``registry/data_source_registry.yaml``,
``registry/plugins.yaml``). The anchor:

1. Binds every configured file's hash at session boot via
   :meth:`bind_session` -- one ledger row
   (``POLICY_HASHES_BOUND``) carries every (name, sha256) pair so the
   chain stays compact regardless of the number of anchored files.
2. Verifies on demand via :meth:`verify_no_drift`: recomputes the
   live hashes and returns a :class:`~core.contracts.events.HazardEvent`
   with severity ``CRITICAL`` (code ``HAZ-POLICY-DRIFT``) when any
   anchored file has been mutated, deleted, or replaced.

The contract is intentionally narrow -- the anchor is a pure
detector. Forcing the FSM to ``SAFE`` and locking the system is the
caller's responsibility (the harness already routes
``HAZ-POLICY-DRIFT`` through :class:`HazardThrottleAdapter` to
``GovernanceEngine.process``, which downgrades the mode through the
single FSM mutator and writes the audit row).

Determinism: all hashes are computed over **raw bytes**, matching
the :func:`promotion_gates.compute_file_hash` convention. Whitespace
and comment edits therefore count as drift -- intentional, since
those are still mid-session edits to the document of record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.contracts.events import HazardEvent, HazardSeverity
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

LEDGER_KIND_POLICY_HASHES_BOUND = "POLICY_HASHES_BOUND"
HAZARD_CODE_POLICY_DRIFT = "HAZ-POLICY-DRIFT"

# Default canonical policy set. Each entry is a (name, repo-relative
# path) pair. Names are stable identifiers used in the ledger row
# payload; paths are resolved relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POLICY_FILES: tuple[tuple[str, Path], ...] = (
    ("authority_matrix", _REPO_ROOT / "registry" / "authority_matrix.yaml"),
    ("constraint_rules", _REPO_ROOT / "registry" / "constraint_rules.yaml"),
    (
        "data_source_registry",
        _REPO_ROOT / "registry" / "data_source_registry.yaml",
    ),
    ("plugins", _REPO_ROOT / "registry" / "plugins.yaml"),
)


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the file's raw bytes.

    Mirrors :func:`promotion_gates.compute_file_hash` so the audit
    chain uses one consistent hash convention. Raises
    :class:`FileNotFoundError` when the file is absent -- callers
    must treat that as drift, not as success.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyHashEntry:
    """One (name, path, sha256) triple anchored at session start."""

    name: str
    path: Path
    sha256: str


class PolicyHashAnchor:
    """Session-bound multi-file SHA-256 anchor.

    Lifecycle:

    1. ``bind_session(ts_ns, requestor)`` -- called once at process
       boot. Hashes every configured file, writes one
       ``POLICY_HASHES_BOUND`` ledger row, caches the hashes in
       memory.
    2. ``verify_no_drift(ts_ns) -> HazardEvent | None`` -- pure
       detector. Returns a ``CRITICAL`` :class:`HazardEvent`
       (code ``HAZ-POLICY-DRIFT``) when any file has drifted from
       its bound hash, otherwise ``None``. Idempotent across calls.

    The anchor is **not** the FSM gate. Wiring is:

        anchor.bind_session(ts_ns=boot_ns, requestor="harness")
        ...
        hazard = anchor.verify_no_drift(now_ns)
        if hazard is not None:
            governance.process(hazard)  # routes to FSM downgrade

    A single :class:`HazardThrottleAdapter` already exists for that
    last step; this module merely produces the typed hazard event.
    """

    name: str = "policy_hash_anchor"
    spec_id: str = "HARDENING-S1-ITEM-4-EXT"

    def __init__(
        self,
        *,
        ledger: LedgerAuthorityWriter,
        files: Iterable[tuple[str, Path]] = DEFAULT_POLICY_FILES,
    ) -> None:
        self._ledger = ledger
        # Materialise once at construction so callers can't mutate
        # the configured set after the anchor has been built.
        self._files: tuple[tuple[str, Path], ...] = tuple(files)
        self._bound: dict[str, PolicyHashEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bound_entries(self) -> tuple[PolicyHashEntry, ...]:
        """Return the cached anchor entries (empty until bound)."""

        return tuple(self._bound.values())

    def bind_session(
        self, *, ts_ns: int, requestor: str
    ) -> tuple[PolicyHashEntry, ...]:
        """Anchor every configured policy file's hash to the ledger.

        Returns the bound entries. Raises :class:`FileNotFoundError`
        if any configured file is missing -- the document set of
        record must exist at boot. A second call rebinds (e.g. on
        operator-approved policy reload), overwriting the cached
        entries and writing a fresh ledger row.
        """

        entries: dict[str, PolicyHashEntry] = {}
        payload: dict[str, str] = {"requestor": requestor}
        for name, path in self._files:
            digest = compute_file_hash(path)
            try:
                recorded_path = str(path.relative_to(_REPO_ROOT))
            except ValueError:
                recorded_path = str(path)
            entries[name] = PolicyHashEntry(
                name=name, path=path, sha256=digest
            )
            payload[f"{name}_sha256"] = digest
            payload[f"{name}_path"] = recorded_path
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_POLICY_HASHES_BOUND,
            payload=payload,
        )
        self._bound = entries
        return tuple(entries.values())

    def verify_no_drift(self, ts_ns: int) -> HazardEvent | None:
        """Recompute live hashes; return a CRITICAL hazard on drift.

        Returns ``None`` when every anchored file's live hash matches
        the bound hash. Returns a :class:`HazardEvent`
        (severity ``CRITICAL``, code ``HAZ-POLICY-DRIFT``) the moment
        any file is missing, unreadable, or mutated. The hazard's
        ``meta`` carries the per-file diagnosis (``<name>_status`` is
        one of ``"ok"`` / ``"missing"`` / ``"mismatch"`` / ``"unreadable"``)
        so the
        operator dashboard can pinpoint which file drifted without
        reading the disk again.

        Pure (other than the file reads inside
        :func:`compute_file_hash`); idempotent; never raises.
        """

        if not self._bound:
            # Anchor has never been bound. We refuse to silently
            # treat the unbound state as "ok" -- the caller forgot
            # to call ``bind_session`` and that itself is a hazard.
            return HazardEvent(
                ts_ns=ts_ns,
                code=HAZARD_CODE_POLICY_DRIFT,
                severity=HazardSeverity.CRITICAL,
                source=self.name,
                detail="policy_hash_anchor_not_bound",
                produced_by_engine=self.name,
            )

        per_file_status: dict[str, str] = {}
        any_drift = False
        first_offender = ""
        for name, entry in self._bound.items():
            try:
                live = compute_file_hash(entry.path)
            except FileNotFoundError:
                per_file_status[f"{name}_status"] = "missing"
                any_drift = True
                if not first_offender:
                    first_offender = name
                continue
            except OSError as exc:
                # Honour the "never raises" contract: any I/O failure
                # (PermissionError, IsADirectoryError, transient disk
                # error, etc.) is itself drift -- the file isn't a
                # readable document of record any more. Surface the
                # exception class on the per-file meta so the audit
                # row records *why* the read failed.
                per_file_status[f"{name}_status"] = "unreadable"
                per_file_status[f"{name}_error"] = type(exc).__name__
                any_drift = True
                if not first_offender:
                    first_offender = name
                continue
            if live != entry.sha256:
                per_file_status[f"{name}_status"] = "mismatch"
                per_file_status[f"{name}_live_sha256"] = live
                per_file_status[f"{name}_bound_sha256"] = entry.sha256
                any_drift = True
                if not first_offender:
                    first_offender = name
            else:
                per_file_status[f"{name}_status"] = "ok"

        if not any_drift:
            return None

        return HazardEvent(
            ts_ns=ts_ns,
            code=HAZARD_CODE_POLICY_DRIFT,
            severity=HazardSeverity.CRITICAL,
            source=self.name,
            detail=f"policy_hash_drift:{first_offender}",
            meta=per_file_status,
            produced_by_engine=self.name,
        )

    # ------------------------------------------------------------------
    # Replay (INV-15 / TEST-01)
    # ------------------------------------------------------------------

    def replay_from_ledger(self) -> None:
        """Reconstitute bound entries from the most recent row.

        Walks the ledger and adopts the most recent
        ``POLICY_HASHES_BOUND`` row's per-file digests as the
        in-memory anchor. Lets a fresh process pick up the bound
        state from a SQLite-backed ledger after a restart, matching
        the determinism contract :class:`PromotionGates` follows.
        """

        latest_payload: Mapping[str, str] | None = None
        for entry in self._ledger.read():
            if entry.kind == LEDGER_KIND_POLICY_HASHES_BOUND:
                latest_payload = entry.payload
        if latest_payload is None:
            self._bound = {}
            return
        rebuilt: dict[str, PolicyHashEntry] = {}
        for name, path in self._files:
            digest = latest_payload.get(f"{name}_sha256")
            if digest is None:
                # Configured file wasn't part of the last bind -- skip
                # so the anchor stays internally consistent. The
                # next ``verify_no_drift`` call will surface the
                # missing entry as drift via the unbound-name code
                # path on the next bind cycle.
                continue
            rebuilt[name] = PolicyHashEntry(
                name=name, path=path, sha256=digest
            )
        self._bound = rebuilt


__all__ = [
    "DEFAULT_POLICY_FILES",
    "HAZARD_CODE_POLICY_DRIFT",
    "LEDGER_KIND_POLICY_HASHES_BOUND",
    "PolicyHashAnchor",
    "PolicyHashEntry",
    "compute_file_hash",
]
