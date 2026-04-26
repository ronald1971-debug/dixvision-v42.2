"""Intelligence engine plugin slots (Phase E2).

Concrete plugins land here as the system grows; Phase E2 introduces the
first one — a deterministic order-book microstructure scorer
(``microstructure.MicrostructureV1``) under the ``microstructure`` slot.
"""

from intelligence_engine.plugins.microstructure import MicrostructureV1

__all__ = ["MicrostructureV1"]
