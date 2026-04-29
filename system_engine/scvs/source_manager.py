"""Runtime source liveness manager for SCVS Phase 2.

Tracks per-source heartbeat + last-data timestamps and produces three
projections every observation tick:

* :class:`SourceLivenessReport` — current status for each enabled source
  (``LIVE`` / ``STALE`` / ``UNKNOWN``).
* :class:`SystemEvent` rows for ledger replay (``SOURCE_HEARTBEAT`` /
  ``SOURCE_STALE`` / ``SOURCE_RECOVERED``).
* :class:`HazardEvent` rows for governance escalation when a source
  marked ``critical: true`` in the registry transitions to ``STALE``
  (rule **SCVS-06** — critical-source fail-closed).

INV-15 — pure / deterministic. The manager owns no clock; every call
takes a caller-supplied ``now_ns`` so ledger replay reproduces every
liveness transition exactly.

Phase 2 enforces SCVS-03 / SCVS-05 / SCVS-06. Phase 3 (schema, AI,
duplicate, silent-fallback) layers on top in a follow-on PR.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from core.contracts.events import (
    HazardEvent,
    HazardSeverity,
    SystemEvent,
    SystemEventKind,
)
from system_engine.scvs.source_registry import SourceDeclaration, SourceRegistry

SOURCE = "system_engine.scvs.source_manager"
HAZ_CRITICAL_SOURCE_STALE = "HAZ-13"


class SourceStatus(StrEnum):
    """Runtime liveness status for one source."""

    UNKNOWN = "UNKNOWN"  # enabled but no heartbeat seen yet
    LIVE = "LIVE"  # heartbeat within threshold
    STALE = "STALE"  # heartbeat overdue


@dataclass(frozen=True, slots=True)
class SourceLivenessReport:
    """One immutable per-source liveness snapshot."""

    source_id: str
    status: SourceStatus
    last_heartbeat_ns: int  # 0 == never seen
    last_data_ns: int  # 0 == never seen
    gap_ns: int  # now_ns - last_heartbeat_ns; 0 when last == 0
    threshold_ns: int  # 0 == not liveness-checked
    critical: bool


@dataclass(slots=True)
class _SourceState:
    last_heartbeat_ns: int = 0
    last_data_ns: int = 0
    last_status: SourceStatus = SourceStatus.UNKNOWN


@dataclass(slots=True)
class SourceManager:
    """Pure runtime liveness tracker for the SCVS source registry.

    Caller-supplied timestamps everywhere. No clock, no PRNG, no I/O.
    """

    registry: SourceRegistry
    _state: dict[str, _SourceState] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Pre-seed state for every enabled source so reports are
        # deterministic even before the first heartbeat.
        for s in self.registry.sources:
            if s.enabled:
                self._state.setdefault(s.id, _SourceState())

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def record_heartbeat(self, source_id: str, ts_ns: int) -> None:
        """Record a heartbeat from ``source_id`` at ``ts_ns``.

        Raises ``KeyError`` if ``source_id`` is not in the registry, and
        ``ValueError`` if the source is not enabled — heartbeats from
        sources that are still placeholders are a configuration bug.
        """

        decl = self._lookup(source_id)
        if not decl.enabled:
            raise ValueError(
                f"source {source_id!r} is enabled=false; "
                "cannot record heartbeat for a placeholder row"
            )
        self._state.setdefault(source_id, _SourceState()).last_heartbeat_ns = ts_ns

    def record_data(self, source_id: str, ts_ns: int) -> None:
        """Record a data packet receipt from ``source_id`` at ``ts_ns``.

        Data flow is a stricter signal than heartbeat (SCVS-03): a
        source can heartbeat without producing data. Both are tracked
        independently; ``observe(...)`` uses ``last_heartbeat_ns`` for
        liveness so the data-flow seam can be lint-extended in Phase 3.
        """

        decl = self._lookup(source_id)
        if not decl.enabled:
            raise ValueError(
                f"source {source_id!r} is enabled=false; "
                "cannot record data for a placeholder row"
            )
        self._state.setdefault(source_id, _SourceState()).last_data_ns = ts_ns

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def reports(self, now_ns: int) -> tuple[SourceLivenessReport, ...]:
        """Return one liveness report per enabled, liveness-checked source."""

        out: list[SourceLivenessReport] = []
        for decl in self.registry.sources:
            if not decl.enabled:
                continue
            st = self._state.get(decl.id, _SourceState())
            threshold_ns = decl.liveness_threshold_ms * 1_000_000
            status = self._classify(st, now_ns, threshold_ns)
            gap = now_ns - st.last_heartbeat_ns if st.last_heartbeat_ns else 0
            out.append(
                SourceLivenessReport(
                    source_id=decl.id,
                    status=status,
                    last_heartbeat_ns=st.last_heartbeat_ns,
                    last_data_ns=st.last_data_ns,
                    gap_ns=gap,
                    threshold_ns=threshold_ns,
                    critical=decl.critical,
                )
            )
        return tuple(out)

    def observe(
        self, now_ns: int
    ) -> tuple[tuple[SystemEvent, ...], tuple[HazardEvent, ...]]:
        """Advance the FSM to ``now_ns`` and return the emitted events.

        * ``SystemEvent`` rows are emitted on every status transition
          (``SOURCE_HEARTBEAT`` on UNKNOWN→LIVE, ``SOURCE_STALE`` on
          LIVE→STALE, ``SOURCE_RECOVERED`` on STALE→LIVE).
        * ``HazardEvent`` rows are emitted only when a ``critical``
          source transitions to ``STALE`` (SCVS-06).

        Calling :meth:`observe` mutates the internal status memo so a
        second call at the same ``now_ns`` is a no-op.
        """

        sys_events: list[SystemEvent] = []
        hazards: list[HazardEvent] = []

        for decl in self.registry.sources:
            if not decl.enabled:
                continue
            st = self._state.setdefault(decl.id, _SourceState())
            threshold_ns = decl.liveness_threshold_ms * 1_000_000
            new_status = self._classify(st, now_ns, threshold_ns)
            if new_status == st.last_status:
                continue
            transition = (st.last_status, new_status)
            sys_events.append(
                _make_status_event(
                    now_ns=now_ns,
                    decl=decl,
                    transition=transition,
                    state=st,
                )
            )
            if new_status == SourceStatus.STALE and decl.critical:
                hazards.append(
                    HazardEvent(
                        ts_ns=now_ns,
                        code=HAZ_CRITICAL_SOURCE_STALE,
                        severity=HazardSeverity.HIGH,
                        source=SOURCE,
                        detail=(
                            f"critical source '{decl.id}' STALE "
                            f"(gap_ns={now_ns - st.last_heartbeat_ns}, "
                            f"threshold_ns={threshold_ns})"
                        ),
                        meta={
                            "source_id": decl.id,
                            "category": decl.category.value,
                        },
                        produced_by_engine="system_engine",
                    )
                )
            st.last_status = new_status

        return tuple(sys_events), tuple(hazards)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lookup(self, source_id: str) -> SourceDeclaration:
        decl = self.registry.by_id(source_id)
        if decl is None:
            raise KeyError(f"unknown source_id: {source_id!r}")
        return decl

    @staticmethod
    def _classify(
        st: _SourceState, now_ns: int, threshold_ns: int
    ) -> SourceStatus:
        if st.last_heartbeat_ns == 0:
            return SourceStatus.UNKNOWN
        if threshold_ns == 0:
            # Liveness disabled (e.g. synthetic replay). Once a heartbeat
            # has been seen we treat it as LIVE indefinitely.
            return SourceStatus.LIVE
        gap = now_ns - st.last_heartbeat_ns
        if gap <= threshold_ns:
            return SourceStatus.LIVE
        return SourceStatus.STALE


def _make_status_event(
    *,
    now_ns: int,
    decl: SourceDeclaration,
    transition: tuple[SourceStatus, SourceStatus],
    state: _SourceState,
) -> SystemEvent:
    prev, curr = transition
    if curr == SourceStatus.LIVE and prev == SourceStatus.UNKNOWN:
        sub = SystemEventKind.SOURCE_HEARTBEAT
    elif curr == SourceStatus.LIVE and prev == SourceStatus.STALE:
        sub = SystemEventKind.SOURCE_RECOVERED
    elif curr == SourceStatus.STALE:
        sub = SystemEventKind.SOURCE_STALE
    else:  # pragma: no cover — defensive: classifier never returns UNKNOWN
        sub = SystemEventKind.SOURCE_HEARTBEAT
    return SystemEvent(
        ts_ns=now_ns,
        sub_kind=sub,
        source=SOURCE,
        payload={
            "source_id": decl.id,
            "category": decl.category.value,
            "from": prev.value,
            "to": curr.value,
            "last_heartbeat_ns": str(state.last_heartbeat_ns),
        },
    )


def reports_to_iterable(
    reports: Iterable[SourceLivenessReport],
) -> tuple[SourceLivenessReport, ...]:
    """Trivial helper for callers that want a deterministic tuple."""

    return tuple(reports)


__all__ = [
    "HAZ_CRITICAL_SOURCE_STALE",
    "SourceLivenessReport",
    "SourceManager",
    "SourceStatus",
    "reports_to_iterable",
]
