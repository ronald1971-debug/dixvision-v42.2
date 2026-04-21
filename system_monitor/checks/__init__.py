"""system_monitor.checks — Pluggable health checks (canonical §6)."""
from .clock_sync_check import check_clock_sync
from .connectivity_check import check_connectivity
from .data_integrity_check import check_data_integrity
from .latency_check import check_latency
from .process_health_check import check_process_health

__all__ = [
    "check_connectivity",
    "check_latency",
    "check_data_integrity",
    "check_clock_sync",
    "check_process_health",
]
