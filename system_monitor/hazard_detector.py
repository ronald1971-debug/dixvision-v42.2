"""
system_monitor/hazard_detector.py
Canonical re-export (see manifest §6). Concrete implementation in
``execution.hazard.detector``.
"""
from __future__ import annotations


def get_hazard_detector():
    from execution.hazard.detector import get_hazard_detector as _impl

    return _impl()
