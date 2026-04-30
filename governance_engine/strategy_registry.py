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
    StrategyLifecycle,
    StrategyLifecycleError,
    StrategyRecord,
    is_legal_transition,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

LEDGER_KIND_STRATEGY_LIFECYCLE = "STRATEGY_LIFECYCLE"
LEDGER_KIND_STRATEGY_PARAMETER_UPDATE = "STRATEGY_PARAMETER_UPDATE"


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
        "mutable_parameters": "\x1f".join(record.mutable_parameters),
    }
    for key in sorted(record.parameters):
        payload[f"param.{key}"] = record.parameters[key]
    for key in sorted(record.parameter_bounds):
        lo, hi = record.parameter_bounds[key]
        payload[f"bound.{key}"] = f"{lo!r}|{hi!r}"
    return payload


def _deserialize_strategy_record(payload: Mapping[str, str]) -> StrategyRecord:
    """Inverse of :func:`_serialize_strategy_record`. Pure."""

    composed_raw = payload.get("composed_from", "")
    why_raw = payload.get("why", "")
    mutable_raw = payload.get("mutable_parameters", "")
    parameters = {
        k.removeprefix("param."): v
        for k, v in payload.items()
        if k.startswith("param.")
    }
    parameter_bounds: dict[str, tuple[float, float]] = {}
    for k, v in payload.items():
        if not k.startswith("bound."):
            continue
        lo_s, hi_s = v.split("|", 1)
        parameter_bounds[k.removeprefix("bound.")] = (
            float(lo_s),
            float(hi_s),
        )
    return StrategyRecord(
        strategy_id=payload["strategy_id"],
        version=int(payload["version"]),
        lifecycle=StrategyLifecycle(payload["lifecycle"]),
        parameters=parameters,
        composed_from=tuple(composed_raw.split("\x1f")) if composed_raw else (),
        why=tuple(why_raw.split("\x1f")) if why_raw else (),
        created_ts_ns=int(payload["created_ts_ns"]),
        last_transition_ts_ns=int(payload["last_transition_ts_ns"]),
        mutable_parameters=tuple(mutable_raw.split("\x1f"))
        if mutable_raw
        else (),
        parameter_bounds=parameter_bounds,
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
        mutable_parameters: Sequence[str] = (),
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> StrategyRecord:
        """Register a brand-new strategy in the ``DRAFT`` state.

        Raises:
            ValueError: ``strategy_id`` is empty or already registered,
                ``parameter_bounds`` references a parameter that is
                not in ``mutable_parameters``, or any bound has
                ``lo > hi``.
        """

        if not strategy_id:
            raise ValueError("strategy_id required")
        if strategy_id in self._records:
            raise ValueError(f"already registered: {strategy_id}")
        bounds = dict(parameter_bounds or {})
        mutable = tuple(mutable_parameters)
        for k in bounds:
            if k not in mutable:
                raise ValueError(
                    f"parameter_bounds key {k!r} not in mutable_parameters"
                )
        for k, (lo, hi) in bounds.items():
            if lo > hi:
                raise ValueError(
                    f"parameter_bounds[{k!r}]: lo {lo} > hi {hi}"
                )

        record = StrategyRecord(
            strategy_id=strategy_id,
            version=1,
            lifecycle=StrategyLifecycle.DRAFT,
            parameters=dict(parameters or {}),
            composed_from=tuple(composed_from),
            why=tuple(why),
            created_ts_ns=ts_ns,
            last_transition_ts_ns=ts_ns,
            mutable_parameters=mutable,
            parameter_bounds=bounds,
        )
        # Ledger-first: write the audit row before mutating in-memory
        # state. If the append raises, the registry stays clean and a
        # retry can succeed (Devin Review BUG_0001 on PR #113).
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_STRATEGY_LIFECYCLE,
            payload=_serialize_strategy_record(record),
        )
        self._records[strategy_id] = record
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
            mutable_parameters=record.mutable_parameters,
            parameter_bounds=dict(record.parameter_bounds),
        )
        payload = _serialize_strategy_record(new_record)
        payload["reason"] = reason
        # Ledger-first: see register_draft. State Transition Manager
        # (state_transition_manager.py:207-212) follows the same
        # pattern.
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_STRATEGY_LIFECYCLE,
            payload=payload,
        )
        self._records[strategy_id] = new_record
        return new_record

    def apply_parameter_update(
        self,
        *,
        strategy_id: str,
        parameter: str,
        new_value: str,
        ts_ns: int,
        reason: str,
    ) -> StrategyRecord:
        """Mutate a single parameter on an APPROVED record.

        This is the **only** entry point through which the closed
        learning loop (Wave-04.6 PR-E) is permitted to change
        strategy parameters. The ``UpdateApplier`` calls it after the
        ``UpdateValidator`` has ratified a ``LearningUpdate``.

        Lifecycle is preserved — only ``parameters[parameter]`` and
        ``version`` (+ ``last_transition_ts_ns``) change. A
        ``STRATEGY_PARAMETER_UPDATE`` ledger row is appended; the
        kind is distinct from ``STRATEGY_LIFECYCLE`` so
        :meth:`replay_from_ledger` can rebuild parameter history
        independently.

        Raises:
            KeyError: ``strategy_id`` is not registered.
            StrategyLifecycleError: ``strategy_id`` is not in the
                ``APPROVED`` state, or ``parameter`` is not in the
                strategy's ``mutable_parameters`` whitelist.
            ValueError: ``reason`` is empty.
        """

        if not reason:
            raise ValueError("reason required")
        record = self._records.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy: {strategy_id}")
        if record.lifecycle is not StrategyLifecycle.APPROVED:
            raise StrategyLifecycleError(
                f"parameter updates require APPROVED, got "
                f"{record.lifecycle.value} for {strategy_id}"
            )
        if parameter not in record.mutable_parameters:
            raise StrategyLifecycleError(
                f"parameter {parameter!r} is not in the mutable "
                f"whitelist for {strategy_id}"
            )

        next_params = dict(record.parameters)
        old_value = next_params.get(parameter, "")
        next_params[parameter] = new_value
        new_record = StrategyRecord(
            strategy_id=record.strategy_id,
            version=record.version + 1,
            lifecycle=record.lifecycle,
            parameters=next_params,
            composed_from=record.composed_from,
            why=record.why,
            created_ts_ns=record.created_ts_ns,
            last_transition_ts_ns=ts_ns,
            mutable_parameters=record.mutable_parameters,
            parameter_bounds=dict(record.parameter_bounds),
        )
        # Ledger-first: see register_draft.
        self._ledger.append(
            ts_ns=ts_ns,
            kind=LEDGER_KIND_STRATEGY_PARAMETER_UPDATE,
            payload={
                "strategy_id": strategy_id,
                "version": str(new_record.version),
                "parameter": parameter,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            },
        )
        self._records[strategy_id] = new_record
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
            if entry.kind == LEDGER_KIND_STRATEGY_LIFECYCLE:
                record = _deserialize_strategy_record(entry.payload)
                self._records[record.strategy_id] = record
            elif entry.kind == LEDGER_KIND_STRATEGY_PARAMETER_UPDATE:
                payload = entry.payload
                strategy_id = payload["strategy_id"]
                prev = self._records[strategy_id]
                next_params = dict(prev.parameters)
                next_params[payload["parameter"]] = payload["new_value"]
                self._records[strategy_id] = StrategyRecord(
                    strategy_id=prev.strategy_id,
                    version=int(payload["version"]),
                    lifecycle=prev.lifecycle,
                    parameters=next_params,
                    composed_from=prev.composed_from,
                    why=prev.why,
                    created_ts_ns=prev.created_ts_ns,
                    last_transition_ts_ns=entry.ts_ns,
                    mutable_parameters=prev.mutable_parameters,
                    parameter_bounds=dict(prev.parameter_bounds),
                )


__all__ = [
    "LEDGER_KIND_STRATEGY_LIFECYCLE",
    "LEDGER_KIND_STRATEGY_PARAMETER_UPDATE",
    "StrategyRegistry",
]
