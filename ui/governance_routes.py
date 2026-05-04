"""Tier-1 governance widget HTTP surface (read-only).

Six widgets specified in the Tier-1 dashboard plan, four of which
already have HTTP surfaces elsewhere in :mod:`ui.server` /
:mod:`ui.dashboard_routes`:

* PromotionGates panel    -> ``/api/governance/promotion_gates``
* DriftOracle panel       -> ``/api/governance/drift``
* SCVS source-liveness    -> ``/api/governance/sources``
* HazardMonitor           -> ``/api/governance/hazards``

(ApprovalQueue, AuditLedger / DecisionTrace, and StrategyRegistry FSM
already have endpoints under ``/api/cognitive/chat/approvals`` and
``/api/dashboard/{decisions,strategies}`` respectively; the Tier-1 UI
widgets consume those directly.)

These endpoints are deliberately *read-only* JSON projections of the
underlying contracts. They never mutate ledger state, never construct
governance decisions, and never write to disk. The dashboard surface
in :mod:`dashboard2026` polls them at low frequency for the operator
cockpit.

Wiring honesty: where a runtime instance is not yet constructed by
:mod:`ui.server` (drift monitor + SCVS source manager + hazard sensor
array — see ZIP analysis P0-2 / P0-4 / P0-7), the endpoint reports a
``backend_wired: false`` flag plus the static configuration that the
runtime would consume once it is wired. This makes the missing wiring
visible in the operator UI rather than silently faking data.

Authority lint: only imports from ``core.contracts`` and the
``governance_engine`` / ``system_engine`` *contract* surfaces (no
plugin or hot-path imports). B7-clean.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from core.contracts.api.governance import (
    DriftResponse,
    HazardsResponse,
    PromotionGatesResponse,
    SourcesResponse,
)
from governance_engine.control_plane.promotion_gates import (
    DEFAULT_PROMOTION_GATES_PATH,
    compute_file_hash,
)
from system_engine.scvs.source_registry import SourceRegistry

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

# The router uses a zero-arg provider returning whichever STATE-like
# object the host installs. We type it with ``Any`` here to avoid a
# circular import on ``ui.server.STATE``; concrete attribute access is
# guarded by ``getattr`` with sensible defaults.
_StateProvider = Callable[[], Any]


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


def _promotion_gates_payload(state: Any) -> dict[str, Any]:
    """Hash + bound-state read projection.

    Always reports the live file hash (computed from the on-disk
    yaml). Reports the bound hash if a :class:`PromotionGates` runtime
    instance is reachable on ``state``. Otherwise reports
    ``backend_wired=false`` so the UI can render the panel with a
    clear "no live binding" state.
    """

    path: Path = DEFAULT_PROMOTION_GATES_PATH
    file_present = path.exists()
    file_hash: str | None = None
    if file_present:
        try:
            file_hash = compute_file_hash(path)
        except OSError:
            file_hash = None

    gates = _safe_attr(state, "promotion_gates")
    bound_hash: str | None = None
    backend_wired = gates is not None
    if backend_wired:
        try:
            bound_hash = gates.bound_hash()
        except Exception:  # pragma: no cover -- diagnostic surface
            bound_hash = None

    matches: bool | None = None
    if backend_wired and bound_hash is not None and file_hash is not None:
        matches = bound_hash == file_hash

    return {
        "path": str(path),
        "file_present": file_present,
        "file_hash": file_hash,
        "bound_hash": bound_hash,
        "matches": matches,
        "backend_wired": backend_wired,
        "gated_targets": ["CANARY", "LIVE", "AUTO"],
        # The thresholds are pre-committed in the file; surface a
        # short summary so the operator can sanity-check defaults
        # without leaving the dashboard.
        "doc_url": (
            "https://github.com/ronald1971-debug/dixvision-v42.2/"
            "blob/main/docs/promotion_gates.yaml"
        ),
    }


# ---------------------------------------------------------------------------
# Drift oracle
# ---------------------------------------------------------------------------


def _drift_payload(state: Any) -> dict[str, Any]:
    """Composite drift projection (P0-7 oracle wiring).

    The :class:`DriftCompositeOracle` (``GOV-CP-08``) is constructed
    inside :class:`GovernanceEngine` and exposed to the dashboard as
    ``state.governance.drift_oracle``. The legacy
    ``state.drift_monitor`` attribute is still honoured for any
    in-tree caller that pre-dates the oracle.
    """

    governance = _safe_attr(state, "governance")
    oracle = _safe_attr(governance, "drift_oracle") if governance else None
    monitor = _safe_attr(state, "drift_monitor")
    backend_wired = oracle is not None or monitor is not None
    components: list[dict[str, Any]] = []
    composite: float | None = None
    if oracle is not None:
        composite = None
        components = []
    elif monitor is not None:
        try:
            components = list(monitor.components())  # type: ignore[attr-defined]
            composite = float(monitor.composite())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover -- diagnostic surface
            components = []
            composite = None

    return {
        "backend_wired": backend_wired,
        "composite": composite,
        # Spec components per Reviewer #4 finding 3 + Reviewer #5
        # AUTO safeguards (PR #125). Surfaced even when no live
        # reading exists so the operator can see the four axes.
        "expected_components": [
            {"id": "model", "label": "Model drift",
             "threshold": 0.25,
             "description": "feature distribution shift vs SHADOW window"},
            {"id": "exec", "label": "Execution drift",
             "threshold": 0.25,
             "description": "realised slippage / fill-rate vs SHADOW"},
            {"id": "latency", "label": "Latency drift",
             "threshold": 0.25,
             "description": "p99 round-trip vs SHADOW"},
            {"id": "causal", "label": "Causal drift",
             "threshold": 0.25,
             "description": "DecisionTrace coverage / why-layer ratio"},
        ],
        "downgrade_threshold": 0.25,
        "components": components,
    }


# ---------------------------------------------------------------------------
# SCVS source liveness
# ---------------------------------------------------------------------------


def _sources_payload(state: Any) -> dict[str, Any]:
    """Per-source liveness grid.

    Reads the static :class:`SourceRegistry` always (it is loaded at
    boot in ``ui.server``). When a :class:`SourceManager` runtime
    instance is reachable on ``state``, augments each row with
    runtime fields (status, gap_ns, last_heartbeat_ns).
    """

    registry: SourceRegistry | None = _safe_attr(state, "source_registry")
    manager = _safe_attr(state, "source_manager")
    runtime_wired = manager is not None

    rows: list[dict[str, Any]] = []
    if registry is not None:
        runtime_reports: dict[str, Any] = {}
        if runtime_wired:
            try:
                # Read-only GET: do NOT advance the global monotonic
                # counter. Use the read-only ``current_ts`` accessor so
                # the source-liveness gap can still be computed in the
                # harness's synthetic-time space without consuming a
                # ledger sequence number.
                now_ns = int(state.current_ts())
                for r in manager.reports(now_ns):  # type: ignore[attr-defined]
                    runtime_reports[r.source_id] = r
            except Exception:  # pragma: no cover -- diagnostic
                runtime_reports = {}
                runtime_wired = False
        for s in registry.sources:
            row: dict[str, Any] = {
                "source_id": s.id,
                "name": s.name,
                "category": str(s.category),
                "provider": s.provider,
                "auth": s.auth,
                "enabled": s.enabled,
                "critical": s.critical,
                "liveness_threshold_ms": s.liveness_threshold_ms,
            }
            if runtime_wired and s.id in runtime_reports:
                rep = runtime_reports[s.id]
                row.update(
                    {
                        "status": str(rep.status),
                        "last_heartbeat_ns": int(rep.last_heartbeat_ns),
                        "last_data_ns": int(rep.last_data_ns),
                        "gap_ns": int(rep.gap_ns),
                    }
                )
            else:
                row.update(
                    {
                        "status": "UNKNOWN",
                        "last_heartbeat_ns": 0,
                        "last_data_ns": 0,
                        "gap_ns": 0,
                    }
                )
            rows.append(row)

    return {
        "backend_wired": runtime_wired,
        "registry_loaded": registry is not None,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Hazard monitor
# ---------------------------------------------------------------------------


# Frozen HAZ taxonomy from system_engine.hazard_sensors. Surfaced even
# when no live sensor array is wired so the operator can see the full
# hazard universe.
_HAZ_TAXONOMY: tuple[dict[str, str], ...] = (
    {"code": "HAZ-01", "label": "WS_TIMEOUT",
     "description": "websocket feed silent past tolerance"},
    {"code": "HAZ-02", "label": "EXCHANGE_UNREACHABLE",
     "description": "adapter cannot reach venue"},
    {"code": "HAZ-03", "label": "CLOCK_DRIFT",
     "description": "TimeAuthority drift exceeds tolerance"},
    {"code": "HAZ-04", "label": "STALE_DATA",
     "description": "quote feed gap exceeds bar_window_ns"},
    {"code": "HAZ-05", "label": "MEMORY_OVERFLOW",
     "description": "RSS / heap budget breached"},
    {"code": "HAZ-06", "label": "LATENCY_SPIKE",
     "description": "round-trip exceeds latency budget"},
    {"code": "HAZ-07", "label": "HEARTBEAT_MISSED",
     "description": "engine heartbeat absent"},
    {"code": "HAZ-08", "label": "RISK_SNAPSHOT_STALE",
     "description": "fast risk cache version unchanged too long"},
    {"code": "HAZ-09", "label": "ORDER_FLOOD",
     "description": "order rate breaches per-window cap"},
    {"code": "HAZ-10", "label": "CIRCUIT_BREAKER_OPEN",
     "description": "runtime monitor opened venue/global breaker"},
    {"code": "HAZ-11", "label": "MARKET_ANOMALY",
     "description": "price/spread anomaly (statistical)"},
    {"code": "HAZ-12", "label": "SYSTEM_ANOMALY",
     "description": "process / cpu / fd resource anomaly"},
    {"code": "HAZ-13", "label": "CRITICAL_SOURCE_STALE",
     "description": "SCVS critical source transitioned to STALE"},
    {"code": "HAZ-NEWS-SHOCK", "label": "NEWS_SHOCK",
     "description": "news projection score breaches shock threshold"},
)


def _hazards_payload(state: Any) -> dict[str, Any]:
    """Hazard taxonomy + recent live events when wired."""

    array = _safe_attr(state, "hazard_sensor_array")
    backend_wired = array is not None

    recent: list[dict[str, Any]] = []
    if backend_wired:
        try:
            for evt in array.recent(limit=50):  # type: ignore[attr-defined]
                recent.append(
                    {
                        "code": str(evt.code),
                        "severity": str(evt.severity),
                        "ts_ns": int(evt.ts_ns),
                        "source": str(evt.source),
                        "summary": str(evt.summary),
                    }
                )
        except Exception:  # pragma: no cover -- diagnostic
            recent = []

    return {
        "backend_wired": backend_wired,
        "taxonomy": list(_HAZ_TAXONOMY),
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_governance_router(provider: _StateProvider) -> APIRouter:
    """Construct the read-only ``/api/governance/...`` router."""

    router = APIRouter(prefix="/api/governance", tags=["governance"])

    @router.get("/promotion_gates", response_model=PromotionGatesResponse)
    def get_promotion_gates() -> dict[str, Any]:
        return _promotion_gates_payload(provider())

    @router.get("/drift", response_model=DriftResponse)
    def get_drift() -> dict[str, Any]:
        return _drift_payload(provider())

    @router.get("/sources", response_model=SourcesResponse)
    def get_sources() -> dict[str, Any]:
        return _sources_payload(provider())

    @router.get("/hazards", response_model=HazardsResponse)
    def get_hazards() -> dict[str, Any]:
        return _hazards_payload(provider())

    return router


__all__ = ["build_governance_router"]
