"""Alternative / edge data sub-package.

Houses the value-type contracts produced by alternative-data adapters
(Polymarket, prediction markets, alt-data feeds, etc.). The canonical
output shape is :class:`sensory.alt.contracts.PredictionMarket`.

Authority discipline (see :mod:`sensory`): no engine imports, no FSM
mutation, no ledger writes.
"""
