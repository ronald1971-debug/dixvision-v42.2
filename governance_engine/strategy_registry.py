"""Governance-side strategy approval registry — Wave-04.6 PR-D.

The :class:`StrategyRegistry` maintains the current
:class:`StrategyRecord` for every strategy known to governance. Every
state mutation is durably appended to the authority ledger as a
``STRATEGY_LIFECYCLE`` row, which makes the registry deterministically
replayable (INV-15 / TEST-01).

This module is the **only** writer of ``STRATEGY_LIFECYCLE`` rows.
Other engines (intelligence, execution, learning) read the registry
through :meth:`StrategyRegistry.get` / :meth:`all_in` but cannot
mutate it. Wave-04.6 PR-E will introduce a separate
``UpdateValidator`` that proposes lifecycle transitions; the registry
owns the FSM transition itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from core.contracts.governance import LedgerEntry
from core.contracts.strategy_registry import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    StrategyLifecycle,
    StrategyLifecycleError,
    StrategyRecord,
    is_legal_transition,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

LEDGER_KIND_STRATEGY_LIFECYCLE = "STRATEGY_LIFECYCLE"


def _serialize_strategy_record(record: StrategyRecord) -> dict[str, str]:
    """Encode a :class:`StrategyRecord` as a ledger-friendly mapping.

    The ledger's payload schema is ``Mapping[str, str]``; this function
    is the canonical encoder used both at write time and at replay
    time. Keys are stable across versions so older ledger rows remain
    decodable.
    """

    payload: dict[str, str] = {
        "strategy_id": record.strategy_id,
        "version": str(record.version),
        "lifecycle": record.lifecycle.value,
        "created_ts_ns": str(record.created_ts_ns),
        "last_transition_ts_ns": str(record.last_transition_ts_ns),
        "composed_from": "\x1f".join(record.composed_from),
        "why": "\x1f".join(record.why),
    }
    for key in sorted(record.parameters):
        payload[f"param.{key}"] = record.parameters[key]
    return payload


def _deserialize_strategy_record(payload: Mapping[str, str]) -> StrategyRecord:
    """Inverse of :func:`_serialize_strategy_record`. Pure."""

    composed_raw = payload.get("composed_from", "")
    why_raw = payload.get("why", "")
    parameters = {
        k.removeprefix("param."): v
        for k, v in payload.items()
        if k.startswith("param.")
    }
    return StrategyRecord(
        strategy_id=payload["strategy_id"],
        version=int(payload["version"]),
        lifecycle=StrategyLifecycle(payload["lifecycle"]),
        parameters=parameters,
        composed_from=tuple(composed_raw.split("\x1f")) if composed_raw else (),
        why=tuple(why_raw.split("\x1f")) if why_raw else (),
        created_ts_ns=int(payload["created_ts_ns"]),
        last_transition_ts_ns=int(payload["last_transition_ts_ns"]),
    )


class StrategyRegistry:
    """In-memory, ledger-backed strategy approval registry.

    Invariants:

    * Every ``register_draft`` / ``transition`` call appends exactly
      one ``STRATEGY_LIFECYCLE`` row.
    * The current :class:`StrategyRecord` for any ``strategy_id`` is
      ``replay_from_ledger`` of the chain truncated at the row's
      sequence number (assuming no other writers touch the
      ``STRATEGY_LIFECYCLE`` kind).
    * Illegal transitions (per
      :data:`LEGAL_LIFECYCLE_TRANSITIONS`) raise
      :class:`StrategyLifecycleError` *before* the ledger row is
      written — i.e. the chain only ever contains legal histories.
    """

    name: str = "strategy_registry"
    spec_id: str = "GOV-SR-01"

    def __init__(self, *, ledger: LedgerAuthorityWriter) -> None:
        self._ledger = ledger
        self._records: dict[str, StrategyRecord] = {}

    # -- queries -------------------------------------------------------

    def get(self, strategy_id: str) -> StrategyRecord | None:
        return self._records.get(strategy_id)

    def all_in(
        self, lifecycle: StrategyLifecycle
    ) -> tuple[StrategyRecord, ...]:
        """All records currently in ``lifecycle`` (insertion order)."""
        return tuple(
            r for r in self._records.values() if r.lifecycle is lifecycle
        )

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    # -- mutations -----------------------------------------------------

    def register_draft(
        self,
        *,
        strategy_id: str,
        ts_ns: int,
        parameters: Mapping[str, str] | None = None,
        composed_from: Sequence[str] = (),
        why: Sequence[str] = (),
    ) -> StrategyRecord:
        """Register a brand-new strategy in the ``DRAFT`` state.

        Raises:
            ValueError: ``strategy_id`` is empty or already registered.
        """

        if not strategy_id:
            raise ValueError("strategy_id required")
        if strategy_id in self._records:
            raise ValueError(f"already registered: {strategy_id}")

        record = StrategyRecord(
            strategy_id=strategy_id,
            version=1,
            lifecycle=StrategyLifecycle.DRAFT,
            parameters=dict(parameters or {}),
            composed_from=tuple(composed_from),
            why=tuple(why),
            created_ts_ns=ts_ns,
            last_transition_ts_ns=ts_ns,
        )
        self._records[strategy_id] = record
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_STRATEGY_LIFECYCLE,
            payload=_serialize_strategy_record(record),
        )
        return record

    def transition(
        self,
        *,
        strategy_id: str,
        new_lifecycle: StrategyLifecycle,
        ts_ns: int,
        reason: str,
    ) -> StrategyRecord:
        """Move ``strategy_id`` to ``new_lifecycle`` and append a ledger row.

        Raises:
            KeyError: ``strategy_id`` is not registered.
            StrategyLifecycleError: transition is not in
                :data:`LEGAL_LIFECYCLE_TRANSITIONS`.
            ValueError: ``reason`` is empty (every transition must
                carry a human-readable rationale for audit).
        """

        if not reason:
            raise ValueError("reason required")
        record = self._records.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy: {strategy_id}")
        if not is_legal_transition(
            prev=record.lifecycle, new=new_lifecycle
        ):
            raise StrategyLifecycleError(
                f"illegal transition {record.lifecycle.value} → "
                f"{new_lifecycle.value} for {strategy_id}"
            )

        new_record = StrategyRecord(
            strategy_id=record.strategy_id,
            version=record.version + 1,
            lifecycle=new_lifecycle,
            parameters=dict(record.parameters),
            composed_from=record.composed_from,
            why=record.why,
            created_ts_ns=record.created_ts_ns,
            last_transition_ts_ns=ts_ns,
        )
        self._records[strategy_id] = new_record
        payload = _serialize_strategy_record(new_record)
        payload["reason"] = reason
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_STRATEGY_LIFECYCLE,
            payload=payload,
        )
        return new_record

    # -- replay --------------------------------------------------------

    def replay_from_ledger(self, entries: Iterable[LedgerEntry]) -> None:
        """Rebuild the in-memory registry from a chain of ledger entries.

        Only entries of kind :data:`LEDGER_KIND_STRATEGY_LIFECYCLE`
        are consumed; other rows are ignored. This makes replay
        composable with the rest of the authority ledger (which
        also carries ``MODE_TRANSITION``, ``PLUGIN_LIFECYCLE``, etc.).

        Replay is **idempotent** — calling on an already-populated
        registry first clears it.
        """

        self._records.clear()
        for entry in entries:
            if entry.kind != LEDGER_KIND_STRATEGY_LIFECYCLE:
                continue
            record = _deserialize_strategy_record(entry.payload)
            self._records[record.strategy_id] = record


__all__ = [
    "LEDGER_KIND_STRATEGY_LIFECYCLE",
    "StrategyRegistry",
]
