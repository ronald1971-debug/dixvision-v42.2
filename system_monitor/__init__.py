"""system_monitor — DYON sensor layer (canonical §6).

This is the manifest-canonical home for Dyon. It re-exports the existing
concrete engine/detector/emitter (currently housed in ``execution/``) and adds
the checks/ + emitters/ subdirs required by the manifest.

Authority model: sense + report only. Cannot execute trades.
"""
from .anomaly_models import AnomalyWindow, is_anomalous
from .emitters.hazard_event_emitter import get_hazard_event_emitter
from .engine import get_system_monitor
from .hazard_detector import get_hazard_detector
from .heartbeat_monitor import get_heartbeat_monitor
from .telemetry_ingest import get_telemetry_ingest

__all__ = [
    "get_system_monitor",
    "get_hazard_detector",
    "get_telemetry_ingest",
    "get_heartbeat_monitor",
    "AnomalyWindow",
    "is_anomalous",
    "get_hazard_event_emitter",
]
