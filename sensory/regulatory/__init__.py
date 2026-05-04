"""Regulatory / filings sub-package.

Houses the value-type contracts produced by regulatory adapters
(SEC EDGAR full-text search, FINRA, etc.). The canonical output shape
is :class:`sensory.regulatory.contracts.Filing`.

Authority discipline (see :mod:`sensory`): no engine imports, no FSM
mutation, no ledger writes.
"""
