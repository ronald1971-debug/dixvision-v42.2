"""Top-level pytest fixtures.

Hardening-S1 item 1 — the harness approval shim
(``governance_engine.harness_approver``) is opt-in only. Production
processes must explicitly set
:data:`governance_engine.harness_approver.HARNESS_APPROVER_ENV_VAR`
to a truthy value before any caller may invoke
:func:`approve_signal_for_execution`. The pytest session is, by
definition, harness territory: replay tests, plugin tests, and the
dashboard backend smoke tests all construct synthetic intents through
the shim. Setting the env var here at session start is the single
canonical opt-in for the whole tree.
"""

from __future__ import annotations

import os

from governance_engine.harness_approver import HARNESS_APPROVER_ENV_VAR
from ui._ledger_boot import PERMIT_EPHEMERAL_LEDGER_ENV_VAR

# Set as early as possible so any import-time call paths (e.g. modules
# that build a synthetic intent at module load) see the gate already
# open. ``setdefault`` so an explicit override (e.g. a test that wants
# to verify the gate-closed behaviour) wins.
os.environ.setdefault(HARNESS_APPROVER_ENV_VAR, "1")

# AUDIT-P0.3 — the harness refuses to boot without a persistent
# ledger path unless ephemeral mode is explicitly opted into. The
# pytest session is, by definition, ephemeral; tests that need to
# verify the refusal path explicitly unset / override this in their
# own fixtures.
os.environ.setdefault(PERMIT_EPHEMERAL_LEDGER_ENV_VAR, "1")
