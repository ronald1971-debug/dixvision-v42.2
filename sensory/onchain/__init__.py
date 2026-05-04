"""On-chain intelligence sub-package.

Houses the value-type contracts produced by on-chain data adapters
(Glassnode REST, Dune Analytics, Bitquery, Helius, etc.). The
canonical output shape is :class:`sensory.onchain.contracts.OnChainMetric`.

Authority discipline (see :mod:`sensory`): no engine imports, no FSM
mutation, no ledger writes. The only legal output of this sub-package
is a typed :class:`OnChainMetric` consumed by the intelligence engine.
"""
