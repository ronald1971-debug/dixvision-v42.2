"""core.bootstrap — Deterministic startup / shutdown graph."""
from .dependency_graph import DependencyGraph
from .lifecycle import Lifecycle, LifecyclePhase
from .loader import load_module
from .shutdown_sequence import SHUTDOWN_SEQUENCE, run_shutdown
from .startup_sequence import STARTUP_SEQUENCE, run_startup

__all__ = [
    "load_module",
    "Lifecycle",
    "LifecyclePhase",
    "STARTUP_SEQUENCE",
    "run_startup",
    "SHUTDOWN_SEQUENCE",
    "run_shutdown",
    "DependencyGraph",
]
