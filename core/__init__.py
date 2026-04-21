"""core — Contracts, registry, bootstrap graph, runtime context, authority."""
from .authority import (
    AuthorityViolation,
    Domain,
    assert_domain,
    assert_no_adapter_import,
    control,
    market,
    requires,
    scope,
    security,
    system,
)
from .exceptions import (
    DomainViolation,
    GovernanceViolation,
    HardKillSwitchTriggered,
    StructuredError,
)
from .registry import Registry, get_registry, registry

__all__ = [
    "Registry",
    "get_registry",
    "registry",
    "StructuredError",
    "HardKillSwitchTriggered",
    "GovernanceViolation",
    "DomainViolation",
    "Domain",
    "AuthorityViolation",
    "assert_domain",
    "assert_no_adapter_import",
    "scope",
    "requires",
    "market",
    "system",
    "control",
    "security",
]
