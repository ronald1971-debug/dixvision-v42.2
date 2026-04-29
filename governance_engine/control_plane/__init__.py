"""GOV-CP — Governance Control Plane (Phase 1).

The Control Plane is the canonical seven-module pipeline that turns
inbound events and operator requests into authoritative system
changes. Per ``manifest.md`` §0.5 / Build Compiler Spec §0:

* GOV-CP-01 ``policy_engine``                 — constraint store + policy decisions
* GOV-CP-02 ``risk_evaluator``                — exposure + limit checks
* GOV-CP-03 ``state_transition_manager``      — Mode FSM (sole writer)
* GOV-CP-04 ``event_classifier``              — event → CP route
* GOV-CP-05 ``ledger_authority_writer``       — sole authority-ledger writer
* GOV-CP-06 ``compliance_validator``          — domain / jurisdiction
* GOV-CP-07 ``operator_interface_bridge``     — dashboard adapter

The pipeline is deterministic: same inputs → same ledger row → same
decision (INV-15, TEST-01).
"""

from governance_engine.control_plane.compliance_validator import (
    ComplianceValidator,
)
from governance_engine.control_plane.event_classifier import (
    EventClassifier,
    PipelineRoute,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.operator_interface_bridge import (
    OperatorInterfaceBridge,
)
from governance_engine.control_plane.policy_engine import (
    POLICY_TABLE_HASH_KEY,
    POLICY_TABLE_INSTALLED_KIND,
    PolicyEngine,
    install_policy_table,
    verify_policy_table_hash,
)
from governance_engine.control_plane.risk_evaluator import RiskEvaluator
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)

__all__ = [
    "ComplianceValidator",
    "EventClassifier",
    "LedgerAuthorityWriter",
    "OperatorInterfaceBridge",
    "PipelineRoute",
    "POLICY_TABLE_HASH_KEY",
    "POLICY_TABLE_INSTALLED_KIND",
    "PolicyEngine",
    "RiskEvaluator",
    "StateTransitionManager",
    "install_policy_table",
    "verify_policy_table_hash",
]
