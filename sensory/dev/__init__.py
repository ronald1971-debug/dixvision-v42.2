"""Developer / technical / code sub-package.

Houses the value-type contracts produced by developer-facing data
adapters (GitHub REST, GitLab, etc.). The canonical output shape is
:class:`sensory.dev.contracts.RepoEvent`.

Authority discipline (see :mod:`sensory`): no engine imports, no FSM
mutation, no ledger writes.
"""
