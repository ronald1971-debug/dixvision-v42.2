"""
system_monitor/emitters/hazard_event_emitter.py
Canonical hazard-event emitter façade. Concrete impl lives at
``execution.hazard.event_emitter`` and is shared by detector + interrupt path.
"""
from __future__ import annotations


def get_hazard_event_emitter(source: str = "dyon"):
    from execution.hazard.event_emitter import get_hazard_emitter

    return get_hazard_emitter(source)
