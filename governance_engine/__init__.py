"""RUNTIME-ENGINE-04 Governance (Phase E0 shell).

Sole authority layer. Hosts the 7-module Governance Control Plane
(GOV-CP-01..07) and the Plugin Activation Surface (PLUGIN-ACT-01..07). Subject
to lint rule B1; Governance must NEVER be imported by hot-path code (T1).
"""

from governance_engine.engine import GovernanceEngine

__all__ = ["GovernanceEngine"]
