"""Intelligence engine plugin slots (Phase E2).

Concrete plugins land here as the system grows; Phase E2 introduces the
first one — a deterministic order-book microstructure scorer
(``microstructure.microstructure_v1.MicrostructureV1``) under the
``microstructure`` slot. The slot is a package so new versions
(``microstructure_v2.py`` …) can land alongside without churning
imports — see ``docs/directory_tree.md``.
"""

from intelligence_engine.plugins.footprint_delta import FootprintDeltaV1
from intelligence_engine.plugins.liquidity_physics import LiquidityPhysicsV1
from intelligence_engine.plugins.microstructure import MicrostructureV1
from intelligence_engine.plugins.order_book_pressure import OrderBookPressureV1
from intelligence_engine.plugins.regime_classifier import RegimeClassifierV1
from intelligence_engine.plugins.vpin_imbalance import VpinImbalanceV1

__all__ = [
    "FootprintDeltaV1",
    "LiquidityPhysicsV1",
    "MicrostructureV1",
    "OrderBookPressureV1",
    "RegimeClassifierV1",
    "VpinImbalanceV1",
]
