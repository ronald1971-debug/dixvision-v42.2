"""RUNTIME-ENGINE-03 System (Phase E0 shell).

Dyon domain. Hazard sensors, health monitors, state plugins. Emits
``HAZARD_EVENT`` and ``SYSTEM_EVENT``. Subject to lint rules B1 and L3.
"""

from system_engine.engine import SystemEngine

__all__ = ["SystemEngine"]
