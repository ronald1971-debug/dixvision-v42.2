"""Engine Status Grid — Phase 6 IMMUTABLE WIDGET 2 (DASH-EG-01).

Renders the 6 engines × {alive, degraded, halted, offline} grid by
calling ``check_self()`` on every registered engine and projecting the
:class:`HealthStatus` into a UI-friendly row.

This widget is *purely* a read projection. It performs no I/O, holds
no state of its own, and never mutates engine state. Each call returns
a fresh snapshot derived from each engine's latest reported health.

Authority constraints: read-only. The grid does not write the ledger,
emit events, or call into any control plane. (INV-08, INV-37)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.engine import (
    HealthState,
    HealthStatus,
)


@dataclass(frozen=True, slots=True)
class EngineHealthRow:
    """One row in the engine-status grid.

    The four-bucket UI vocabulary (``alive`` / ``degraded`` /
    ``halted`` / ``offline``) is the operator-facing lexicon from
    Build Compiler Spec §6. It is derived from the engine's
    :class:`HealthState` via :meth:`EngineStatusGrid._bucket`.
    """

    engine_name: str
    bucket: str
    detail: str
    plugin_states: tuple[tuple[str, str, str], ...]


class EngineStatusGrid:
    """DASH-EG-01 — Engine Status Grid widget backend.

    Constructed with a fixed mapping of engine name → engine instance
    so the grid order is stable across snapshots (UI layout
    requirement).
    """

    name: str = "engine_status_grid"
    spec_id: str = "DASH-EG-01"

    # Operator-facing buckets (Build Compiler Spec §6).
    _BUCKET_ALIVE = "alive"
    _BUCKET_DEGRADED = "degraded"
    _BUCKET_HALTED = "halted"
    _BUCKET_OFFLINE = "offline"

    def __init__(self, *, engines: Mapping[str, object]) -> None:
        self._engines: tuple[tuple[str, object], ...] = tuple(engines.items())

    def snapshot(self) -> tuple[EngineHealthRow, ...]:
        rows: list[EngineHealthRow] = []
        for name, engine in self._engines:
            rows.append(self._row_for(name, engine))
        return tuple(rows)

    def _row_for(self, engine_name: str, engine: object) -> EngineHealthRow:
        check = getattr(engine, "check_self", None)
        if not callable(check):
            return EngineHealthRow(
                engine_name=engine_name,
                bucket=self._BUCKET_OFFLINE,
                detail="engine missing check_self()",
                plugin_states=(),
            )
        try:
            status = check()
        except Exception as exc:
            return EngineHealthRow(
                engine_name=engine_name,
                bucket=self._BUCKET_OFFLINE,
                detail=f"check_self raised: {type(exc).__name__}",
                plugin_states=(),
            )
        if not isinstance(status, HealthStatus):
            return EngineHealthRow(
                engine_name=engine_name,
                bucket=self._BUCKET_OFFLINE,
                detail="check_self did not return HealthStatus",
                plugin_states=(),
            )
        return EngineHealthRow(
            engine_name=engine_name,
            bucket=self._bucket(status.state),
            detail=status.detail,
            plugin_states=self._plugin_rows(status),
        )

    @classmethod
    def _bucket(cls, state: HealthState) -> str:
        if state is HealthState.OK:
            return cls._BUCKET_ALIVE
        if state is HealthState.DEGRADED:
            return cls._BUCKET_DEGRADED
        if state is HealthState.FAIL:
            return cls._BUCKET_HALTED
        return cls._BUCKET_OFFLINE

    @staticmethod
    def _plugin_rows(
        status: HealthStatus,
    ) -> tuple[tuple[str, str, str], ...]:
        rows: list[tuple[str, str, str]] = []
        for slot, plugins in sorted(status.plugin_states.items()):
            for plugin_name, plugin_state in sorted(plugins.items()):
                rows.append((slot, plugin_name, plugin_state.value))
        return tuple(rows)
