"""IND-L02 microstructure plugin slot.

The slot houses concrete microstructure plugins. Phase E2 ships the first
one (``microstructure_v1.MicrostructureV1``); future versions land here
as siblings (e.g. ``microstructure_v2.py``) per the canonical directory
tree (``docs/directory_tree.md`` →
``intelligence_engine/plugins/microstructure/``).
"""

from intelligence_engine.plugins.microstructure.microstructure_v1 import (
    MicrostructureV1,
)

__all__ = ["MicrostructureV1"]
