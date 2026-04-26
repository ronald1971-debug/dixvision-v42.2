"""Authoritative state — append-only hash-chained ledger + snapshots.

The ``state.ledger.reader`` module is on the authority-lint allow-list:
offline engines may import it, but no engine may import any other
``state.ledger`` symbol directly. Writes go through Governance only
(GOV-CP-05).
"""
