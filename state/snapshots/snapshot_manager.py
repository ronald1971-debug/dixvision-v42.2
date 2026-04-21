"""
state/snapshots/snapshot_manager.py
Manifest-canonical re-export. Concrete implementation lives at
``state.ledger.snapshot_manager`` (see §9).
"""
from state.ledger.snapshot_manager import restore_latest, save_incremental, save_snapshot

__all__ = ["save_snapshot", "save_incremental", "restore_latest"]
