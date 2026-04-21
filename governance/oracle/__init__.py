"""governance.oracle — Tiered approval oracles (fast → balanced → deep)."""
from .tier_l1_fast import approve_l1_fast
from .tier_l2_balanced import approve_l2_balanced
from .tier_l3_deep import approve_l3_deep

__all__ = ["approve_l1_fast", "approve_l2_balanced", "approve_l3_deep"]
