"""
mind/risk_cache.py
Manifest-canonical mind/risk_cache — thin façade over the shared
``system.fast_risk_cache`` so both names resolve to the same singleton.
"""
from __future__ import annotations


def get_risk_cache():
    from system.fast_risk_cache import get_risk_cache as _impl

    return _impl()
