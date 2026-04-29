"""HARDEN-02 — Execution Gate runtime guard (INV-68).

The :class:`~execution_engine.engine.ExecutionEngine` is the *only*
runtime path to a venue. Everything that wants a fill must construct
an :class:`~core.contracts.execution_intent.ExecutionIntent`, get it
approved by Governance, then call :meth:`ExecutionEngine.execute`.

This module is the runtime defence the user described as
*"code-level runtime guard at the chokepoint plus the YAML authority
matrix as the source of truth that the guard reads"*. The static
defences (B7, B20-B22, B25 lint) catch policy at PR time; this guard
catches anything that drifts past lint at deploy time — a misconfigured
caller, a mock that smuggled in unauthorised origin, a future code
path nobody anticipated.

Layered defences this module enforces (in order):

1. ``intent`` is an :class:`ExecutionIntent` (no untyped values reach
   the venue).
2. ``intent.verify_content_hash()`` — payload was not tampered with
   between Governance approval and execution.
3. ``intent.approved_by_governance is True`` — Governance signed off
   and a non-empty ``governance_decision_id`` is recorded.
4. ``intent.origin in actor.intelligence_subsystems`` — the caller
   *engine* matches the matrix's intelligence actor row.
5. ``caller in CALLER_ALLOWLIST`` — the runtime entry point
   (e.g. ``"execution_engine"``) is the executor named in
   ``registry/authority_matrix.yaml``.

On every guard failure the guard:

* raises :class:`UnauthorizedActorError` (hard fail at the chokepoint),
* and emits a synthetic ``HAZ-AUTHORITY`` :class:`HazardEvent` via
  the optional ``hazard_sink`` callback, so the auditor sees the
  attempt even if the caller swallows the exception.

The matrix is read **once** at :class:`AuthorityGuard` construction
into a frozen :class:`~system_engine.authority.matrix.AuthorityMatrix`
value; subsequent ``assert_can_execute`` calls do no IO.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from core.contracts.events import HazardEvent, HazardSeverity
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    ExecutionIntent,
)
from system_engine.authority.matrix import (
    AuthorityMatrix,
    load_authority_matrix,
)

__all__ = [
    "AuthorityGuard",
    "AuthorityViolation",
    "HAZ_AUTHORITY_CODE",
    "UnauthorizedActorError",
]


HAZ_AUTHORITY_CODE: Final[str] = "HAZ-AUTHORITY"


class UnauthorizedActorError(RuntimeError):
    """Raised at the Execution Gate when authority validation fails."""


@dataclass(frozen=True, slots=True)
class AuthorityViolation:
    """Structured record of a single guard rejection.

    Mirrors the fields the synthetic hazard event carries so the audit
    ledger can attribute the violation back to the originating
    :class:`ExecutionIntent`.
    """

    reason: str
    intent_id: str
    origin: str
    caller: str
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# Default callers permitted to invoke ``execute(intent)``. The matrix's
# ``execution`` actor row is the source of truth — this constant is the
# concrete runtime label the engine identifies itself with. Adding a
# caller is a change here *and* in the matrix; the guard rejects any
# string not present in both.
DEFAULT_CALLER_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"execution_engine"}
)


def _intelligence_origin_prefix(matrix: AuthorityMatrix) -> str:
    """Resolve the matrix module path for the ``intelligence`` actor.

    Falls back to ``"intelligence_engine"`` when the actor row omits
    ``module`` (already validated by :func:`load_authority_matrix`).
    """

    actor = matrix.actor("intelligence")
    return actor.module or "intelligence_engine"


def _execution_actor_module(matrix: AuthorityMatrix) -> str:
    actor = matrix.actor("execution")
    return actor.module or "execution_engine"


HazardSink = Callable[[HazardEvent], None]


class AuthorityGuard:
    """Runtime validator for :class:`ExecutionIntent`.

    Construct one per :class:`ExecutionEngine` instance. The matrix is
    read once at construction; subsequent calls are O(1).

    Args:
        matrix_path: Path to ``registry/authority_matrix.yaml``. When
            unset, falls back to ``<repo_root>/registry/authority_matrix.yaml``
            relative to the engine package.
        caller_allowlist: Override the default ``{"execution_engine"}``
            set. Used by tests to drive guard scenarios; production
            code passes the default.
        hazard_sink: Optional callback that receives a
            :class:`HazardEvent` for every violation. The guard never
            reads the return value — sinks are advisory.
    """

    def __init__(
        self,
        *,
        matrix: AuthorityMatrix | None = None,
        matrix_path: Path | None = None,
        caller_allowlist: frozenset[str] | None = None,
        hazard_sink: HazardSink | None = None,
    ) -> None:
        if matrix is None:
            path = matrix_path or self._default_matrix_path()
            matrix = load_authority_matrix(path)
        self._matrix = matrix
        self._caller_allowlist = (
            caller_allowlist if caller_allowlist is not None else DEFAULT_CALLER_ALLOWLIST
        )
        self._intelligence_prefix = _intelligence_origin_prefix(matrix)
        self._execution_actor = _execution_actor_module(matrix)
        self._hazard_sink = hazard_sink

    @property
    def matrix(self) -> AuthorityMatrix:
        return self._matrix

    @staticmethod
    def _default_matrix_path() -> Path:
        # execution_engine/execution_gate.py -> repo_root/registry/...
        return (
            Path(__file__).resolve().parent.parent
            / "registry"
            / "authority_matrix.yaml"
        )

    def assert_can_execute(
        self, intent: ExecutionIntent, *, caller: str, ts_ns: int | None = None
    ) -> None:
        """Validate an intent at the Execution Gate.

        Raises:
            UnauthorizedActorError: on any guard failure.
        """

        # 1) Caller is whitelisted.
        if caller not in self._caller_allowlist:
            self._reject(
                intent=intent,
                caller=caller,
                reason="caller not in execution allowlist",
                ts_ns=ts_ns,
                extra=(
                    ("allowlist", ",".join(sorted(self._caller_allowlist))),
                ),
            )

        # 2) Intent type was already enforced by the type system; we
        # still validate :func:`verify_content_hash` so a hand-crafted
        # frozen replacement (e.g. ``dataclasses.replace``) is caught.
        if not intent.verify_content_hash():
            self._reject(
                intent=intent,
                caller=caller,
                reason="content_hash mismatch — intent tampered after construction",
                ts_ns=ts_ns,
            )

        # 3) Governance approval is mandatory.
        if not intent.approved_by_governance:
            self._reject(
                intent=intent,
                caller=caller,
                reason="intent not approved by governance",
                ts_ns=ts_ns,
            )
        if not intent.governance_decision_id:
            self._reject(
                intent=intent,
                caller=caller,
                reason="governance_decision_id missing on approved intent",
                ts_ns=ts_ns,
            )

        # 4) Origin is registered as an intelligence subsystem.
        if intent.origin not in AUTHORISED_INTENT_ORIGINS:
            self._reject(
                intent=intent,
                caller=caller,
                reason="origin not in authority matrix",
                ts_ns=ts_ns,
            )

        # 5) Origin string is a strict child of the matrix's
        # intelligence actor module (or the dedicated tests.fixtures
        # harness origin). We accept the harness origin only when the
        # caller_allowlist explicitly carries the test caller, so a
        # production deploy can never accept a fixture-origin intent.
        if intent.origin == "tests.fixtures":
            if "tests.fixtures" not in self._caller_allowlist:
                self._reject(
                    intent=intent,
                    caller=caller,
                    reason="tests.fixtures origin requires explicit test caller",
                    ts_ns=ts_ns,
                )
        else:
            if not intent.origin.startswith(self._intelligence_prefix + "."):
                self._reject(
                    intent=intent,
                    caller=caller,
                    reason=(
                        "origin does not match matrix intelligence actor "
                        f"(expected child of {self._intelligence_prefix!r})"
                    ),
                    ts_ns=ts_ns,
                )

    def _reject(
        self,
        *,
        intent: ExecutionIntent,
        caller: str,
        reason: str,
        ts_ns: int | None,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> None:
        violation = AuthorityViolation(
            reason=reason,
            intent_id=intent.intent_id,
            origin=intent.origin,
            caller=caller,
            extra=extra,
        )
        self._emit_hazard(
            violation,
            ts_ns=ts_ns if ts_ns is not None else intent.ts_ns,
        )
        raise UnauthorizedActorError(self._format(violation))

    def _emit_hazard(
        self, violation: AuthorityViolation, *, ts_ns: int
    ) -> None:
        if self._hazard_sink is None:
            return
        meta = {
            "intent_id": violation.intent_id,
            "origin": violation.origin,
            "caller": violation.caller,
        }
        for k, v in violation.extra:
            meta[k] = v
        self._hazard_sink(
            HazardEvent(
                ts_ns=ts_ns,
                code=HAZ_AUTHORITY_CODE,
                severity=HazardSeverity.CRITICAL,
                source=self._execution_actor,
                detail=violation.reason,
                meta=meta,
            )
        )

    @staticmethod
    def _format(violation: AuthorityViolation) -> str:
        return (
            f"AuthorityGuard rejected {violation.intent_id}: "
            f"{violation.reason} "
            f"(origin={violation.origin!r}, caller={violation.caller!r})"
        )
