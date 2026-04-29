"""GOV-CP-01 — Policy Engine.

Owns the canonical constraint store and answers two yes/no questions:

* ``permit_mode_transition(req)``  — is the proposed Mode FSM edge
  legal under current policy?
* ``permit_operator_action(req)``  — is this dashboard-originated
  operator action allowed under the active mode?

Per Build Compiler Spec §1: the policy engine never **changes** state.
It only judges whether a proposal is acceptable. Mode writes go
through :class:`StateTransitionManager` (GOV-CP-03).

Constraints are loaded as :class:`Constraint` records; each carries a
``scope`` (GLOBAL / MODE / SYMBOL / DOMAIN) and a ``kind``. The kinds
relevant to mode transitions are:

* ``REQUIRE_OPERATOR`` — gate on ``request.operator_authorized``
* ``DOMAIN_ISOLATION`` — declarative; enforced elsewhere

Determinism contract: same constraint set + same request → same
verdict (INV-15).

Phase 7 / GOV-CP-01-PERF (I7 reframed): the operator-action gate is
**precompiled** into a frozen :pyattr:`_decision_table`
``Mapping[(OperatorAction, SystemMode, sub_kind), (bool, code)]`` at
``__init__``. ``permit_operator_action`` becomes one constant-time
dict lookup (worst case: two — exact key, then wildcard fallback).
The table is content-hashed via :pyattr:`table_hash`; replayers and
the bootstrap orchestrator install one ``POLICY_TABLE_INSTALLED``
ledger row per table and may re-verify on replay (SAFE-47).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from core.contracts.governance import (
    Constraint,
    ConstraintKind,
    ConstraintScope,
    LedgerEntry,
    ModeTransitionRequest,
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------

# A canonical wildcard sub-kind. The table always carries a wildcard
# entry for every (action, mode); a more specific entry (e.g. for
# ``REQUEST_PLUGIN_LIFECYCLE`` with ``target_status="ACTIVE"``) wins
# when present.
_WILDCARD_SUB_KIND = "*"

# Sub-kind dimensions that the decision table enumerates. Today only
# REQUEST_PLUGIN_LIFECYCLE has a payload-driven gate; everything else
# resolves through the wildcard entry.
_PLUGIN_LIFECYCLE_SUB_KINDS: tuple[str, ...] = ("ACTIVE",)

POLICY_TABLE_INSTALLED_KIND = "POLICY_TABLE_INSTALLED"
POLICY_TABLE_HASH_KEY = "table_hash"

DecisionTable = Mapping[tuple[OperatorAction, SystemMode, str], tuple[bool, str]]


def _extract_sub_kind(request: OperatorRequest) -> str:
    """Project the payload axis the table is keyed on."""

    if request.action is OperatorAction.REQUEST_PLUGIN_LIFECYCLE:
        return request.payload.get("target_status", "")
    return ""


def _compile_decision_table() -> DecisionTable:
    """Precompute every static (action, mode, sub_kind) verdict.

    The compilation is a pure function over the enum surface, so two
    PolicyEngines built from the same source always yield the same
    table — replay-determinism (INV-15) holds by construction.
    """

    table: dict[tuple[OperatorAction, SystemMode, str], tuple[bool, str]] = {}

    for mode in SystemMode:
        for action in OperatorAction:
            table[(action, mode, _WILDCARD_SUB_KIND)] = _compile_default(
                action, mode
            )

    # Lifecycle ACTIVE in SAFE is the one payload-keyed exception.
    for mode in SystemMode:
        for sub_kind in _PLUGIN_LIFECYCLE_SUB_KINDS:
            table[
                (OperatorAction.REQUEST_PLUGIN_LIFECYCLE, mode, sub_kind)
            ] = _compile_plugin_lifecycle(mode, sub_kind)

    return table


def _compile_default(
    action: OperatorAction, mode: SystemMode
) -> tuple[bool, str]:
    """Verdict for ``(action, mode, "*")``."""

    if mode is SystemMode.LOCKED:
        if action is OperatorAction.REQUEST_UNLOCK:
            return True, ""
        return False, "POLICY_LOCKED"

    if action is OperatorAction.REQUEST_KILL:
        return True, ""

    if action is OperatorAction.REQUEST_PLUGIN_LIFECYCLE:
        return True, ""

    if action is OperatorAction.REQUEST_MODE:
        return True, ""

    if action is OperatorAction.REQUEST_INTENT:
        return True, ""

    if action is OperatorAction.REQUEST_UNLOCK:
        # REQUEST_UNLOCK only makes sense from LOCKED.
        return False, "POLICY_UNLOCK_NOT_LOCKED"

    return False, "POLICY_UNKNOWN_ACTION"


def _compile_plugin_lifecycle(
    mode: SystemMode, sub_kind: str
) -> tuple[bool, str]:
    """Verdict for ``(REQUEST_PLUGIN_LIFECYCLE, mode, sub_kind)``."""

    if mode is SystemMode.LOCKED:
        return False, "POLICY_LOCKED"
    if sub_kind == "ACTIVE" and mode is SystemMode.SAFE:
        return False, "POLICY_LIFECYCLE_REQUIRES_NON_SAFE"
    return True, ""


def _hash_decision_table(table: DecisionTable) -> str:
    """Stable SHA-256 over the canonical-sorted decision table."""

    h = hashlib.sha256()
    for key in sorted(
        table, key=lambda k: (k[0].value, int(k[1]), k[2])
    ):
        action, mode, sub_kind = key
        allowed, code = table[key]
        h.update(
            f"{action.value}|{int(mode)}|{sub_kind}|{int(allowed)}|{code}\n".encode()
        )
    return h.hexdigest()


class PolicyEngine:
    name: str = "policy_engine"
    spec_id: str = "GOV-CP-01"

    def __init__(self, constraints: Iterable[Constraint] | None = None) -> None:
        self._constraints: tuple[Constraint, ...] = tuple(constraints or ())
        # GOV-CP-01-PERF — precompile the constant-time decision table.
        self._decision_table: DecisionTable = _compile_decision_table()
        self._table_hash: str = _hash_decision_table(self._decision_table)

    # ------------------------------------------------------------------
    # Constraint store
    # ------------------------------------------------------------------

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        return self._constraints

    @property
    def table_hash(self) -> str:
        """SHA-256 content hash of the precompiled decision table."""

        return self._table_hash

    @property
    def decision_table(self) -> DecisionTable:
        """Read-only view of the precompiled decision table."""

        return self._decision_table

    def load(self, constraints: Iterable[Constraint]) -> None:
        """Replace the constraint set (atomic)."""

        self._constraints = tuple(constraints)

    def for_kind(
        self, kind: ConstraintKind, *, scope: ConstraintScope | None = None
    ) -> tuple[Constraint, ...]:
        return tuple(
            c
            for c in self._constraints
            if c.kind is kind and (scope is None or c.scope is scope)
        )

    # ------------------------------------------------------------------
    # Mode transition gate
    # ------------------------------------------------------------------

    def permit_mode_transition(
        self, request: ModeTransitionRequest
    ) -> tuple[bool, str]:
        """Apply the policy half of the mode-transition gate.

        Returns ``(approved, rejection_code)``. The Mode FSM legality
        (legal edge set) is owned by ``StateTransitionManager``;
        ``PolicyEngine`` enforces *additional* policy gates layered on
        top:

        * AUTO and LIVE require ``operator_authorized`` when the
          transition is a *forward* ratchet (current rank below
          target rank). De-escalation toward LIVE/AUTO is always
          permitted by policy because the Mode FSM treats backward
          edges as safety operations (Build Compiler Spec §7).
        * REQUIRE_OPERATOR scoped to a target mode forces an explicit
          operator authorisation regardless of the requestor.
        """

        target = request.target_mode
        current = request.current_mode

        if target in (SystemMode.LIVE, SystemMode.AUTO):
            forward = (
                current is not SystemMode.LOCKED
                and int(target) > int(current)
            )
            if forward and not request.operator_authorized:
                return False, "POLICY_OPERATOR_REQUIRED"

        for c in self.for_kind(ConstraintKind.REQUIRE_OPERATOR):
            scoped_mode = c.params.get("mode")
            if scoped_mode and scoped_mode == target.name:
                if not request.operator_authorized:
                    return False, f"POLICY_OPERATOR_REQUIRED:{c.id}"

        return True, ""

    # ------------------------------------------------------------------
    # Operator action gate (constant-time after __init__)
    # ------------------------------------------------------------------

    def permit_operator_action(
        self, request: OperatorRequest, current_mode: SystemMode
    ) -> tuple[bool, str]:
        """Decide whether ``request`` is permitted in ``current_mode``.

        O(1) after construction: at most two dict lookups (an exact
        ``(action, mode, sub_kind)`` probe followed by the wildcard
        fallback). The verdict is identical to the pre-table logic for
        every (action, mode, payload) triple — the table is the
        source of truth.
        """

        sub_kind = _extract_sub_kind(request)
        verdict = self._decision_table.get(
            (request.action, current_mode, sub_kind)
        )
        if verdict is None:
            verdict = self._decision_table[
                (request.action, current_mode, _WILDCARD_SUB_KIND)
            ]
        return verdict


# ---------------------------------------------------------------------------
# Installation + replay verification (SAFE-47)
# ---------------------------------------------------------------------------


def install_policy_table(
    policy: PolicyEngine,
    ledger: LedgerAuthorityWriter,
    *,
    ts_ns: int,
) -> LedgerEntry:
    """Write one ``POLICY_TABLE_INSTALLED`` row capturing the table hash.

    The orchestrator (``GovernanceEngine``) calls this once at boot
    after constructing the PolicyEngine. Replay reconstructs the same
    table from source (it is enum-driven), so the same ledger replays
    to the same hash — :func:`verify_policy_table_hash` enforces this.
    """

    return ledger.append(
        ts_ns=ts_ns,
        kind=POLICY_TABLE_INSTALLED_KIND,
        payload={POLICY_TABLE_HASH_KEY: policy.table_hash},
    )


def verify_policy_table_hash(
    policy: PolicyEngine, ledger: LedgerAuthorityWriter
) -> None:
    """Fail-closed check that the most recent installed table matches.

    SAFE-47: if the ledger declares a different table hash than the
    PolicyEngine just compiled, the registry has been tampered with
    or the source code disagrees with the historical record. Either
    way, the engine refuses to proceed.

    Raises :class:`RuntimeError` on mismatch. Raises :class:`LookupError`
    if no installation row has been written yet.
    """

    last: LedgerEntry | None = None
    for row in ledger.read():
        if row.kind == POLICY_TABLE_INSTALLED_KIND:
            last = row
    if last is None:
        raise LookupError(
            "no POLICY_TABLE_INSTALLED row in ledger; install_policy_table "
            "must run before verify_policy_table_hash"
        )
    declared = last.payload.get(POLICY_TABLE_HASH_KEY, "")
    if declared != policy.table_hash:
        raise RuntimeError(
            "policy table hash mismatch: ledger declared "
            f"{declared!r}, runtime computed {policy.table_hash!r}"
        )


__all__ = [
    "PolicyEngine",
    "DecisionTable",
    "POLICY_TABLE_INSTALLED_KIND",
    "POLICY_TABLE_HASH_KEY",
    "install_policy_table",
    "verify_policy_table_hash",
]
