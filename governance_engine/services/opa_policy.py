"""A-07 — Governance Policy Enforcement (OPA / Rego-shaped).

# ADAPTED FROM: open-policy-agent OPA policy evaluation contract
# (https://github.com/open-policy-agent/opa,
#  https://github.com/Turall/OPA-python-client — opa_client/opa.py
#  check_policy()/input-output mapping)

Tier classification: ``RUNTIME_SAFE`` (governance side; INV-71 / B27 /
B28 explicitly permit ``GovernanceDecision`` construction inside
``governance_engine.*`` — that is precisely why this module lives
under ``governance_engine/services/``).

Design goals
============

* Pure-Python, deterministic policy evaluation against a typed
  :class:`PolicyInput` envelope.
* No top-level OPA / network / clock dependency. Real OPA binary
  integration is deferred to a caller-supplied :class:`PolicyTransport`
  via :func:`opa_http_transport_factory`.
* A built-in :class:`InProcessPolicyTransport` evaluates a small
  Rego-shaped registry of pure Python predicates so the surface is
  exercisable offline.
* Fail-closed: any transport error, schema mismatch, or unknown
  ``rule_path`` becomes :class:`PolicyVerdict.REJECT`. The original
  cause is reported via ``rejection_code``.
* Decisions are content-addressed (BLAKE2b-16 ``policy_digest``) so
  replayers can verify policy outputs byte-for-byte across runs
  (INV-15).

Surface
=======

* :class:`PolicyVerdict` — three-valued verdict (APPROVE / REJECT /
  ESCALATE), mirrors the OPA decision semantics in spec line 1045.
* :class:`PolicyInput` — frozen envelope (``action`` / ``mode`` /
  ``subject`` / ``payload``) projected to canonical sorted-key JSON.
* :class:`PolicyDecision` — frozen verdict + ``policy_id`` +
  ``rule_path`` + ``rejection_code`` + ``policy_digest``.
* :class:`PolicyTransport` — Protocol with one ``evaluate`` method.
* :class:`PolicyRule` — frozen predicate envelope used by the in-process
  transport.
* :class:`InProcessPolicyTransport` — pure-Python evaluator over a
  tuple of :class:`PolicyRule`.
* :class:`OpaPolicyEvaluator` — top-level coordinator. Calls a
  caller-supplied transport, fail-closes on any error, returns a
  frozen :class:`PolicyDecision`.
* :func:`to_governance_decision` — maps a :class:`PolicyDecision` to a
  :class:`GovernanceDecision`. (This is the **only** place in the
  module that constructs ``GovernanceDecision`` — explicitly permitted
  under B27 / B28 because the file lives in ``governance_engine``.)
* :func:`opa_http_transport_factory` — lazy factory that builds a
  transport delegating to the ``opa-python-client`` Python SDK. The
  ``opa-python-client`` package is imported **only** inside the
  factory body so the module stays importable without the dependency.

INV-15 determinism
==================

* No top-level ``random`` / ``time`` / ``datetime`` / ``os`` /
  ``asyncio`` imports.
* All dataclasses are frozen and slotted.
* :meth:`PolicyInput.to_canonical_json` and
  :meth:`PolicyDecision.policy_digest` use sorted-key JSON for byte
  stability.
* Caller supplies the ``ts_ns`` used in any downstream
  :class:`GovernanceDecision`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.contracts.governance import (
    DecisionKind,
    GovernanceDecision,
)

__all__ = [
    "DEFAULT_POLICY_PACKAGE",
    "EMPTY_PAYLOAD",
    "MAX_PAYLOAD_KEYS",
    "MAX_PAYLOAD_VALUE_LEN",
    "MAX_REJECTION_CODE_LEN",
    "MAX_RULE_PATH_LEN",
    "MAX_SUMMARY_LEN",
    "NEW_PIP_DEPENDENCIES",
    "OPA_ADAPTER_VERSION",
    "InProcessPolicyTransport",
    "OpaPolicyEvaluator",
    "OpaPolicyError",
    "PolicyDecision",
    "PolicyEvaluationError",
    "PolicyInput",
    "PolicyRule",
    "PolicyTransport",
    "PolicyTransportError",
    "PolicyTransportResult",
    "PolicyVerdict",
    "build_baseline_rules",
    "opa_http_transport_factory",
    "to_governance_decision",
]


# ---------------------------------------------------------------------------
# Module identity + dependency manifest
# ---------------------------------------------------------------------------

# The opa-python-client SDK is the canonical Python entrypoint per
# spec line 1022. The package is lazy-imported only inside
# :func:`opa_http_transport_factory`; this module imports cleanly
# without it being installed.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("opa-python-client",)

OPA_ADAPTER_VERSION: str = "1"
DEFAULT_POLICY_PACKAGE: str = "dix.governance"

# Hard limits — keep policy inputs and outputs small so the in-process
# transport stays predictable and JSON serialisation is bounded.
MAX_PAYLOAD_KEYS: int = 32
MAX_PAYLOAD_VALUE_LEN: int = 1024
MAX_RULE_PATH_LEN: int = 256
MAX_REJECTION_CODE_LEN: int = 64
MAX_SUMMARY_LEN: int = 512

EMPTY_PAYLOAD: Mapping[str, str] = {}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OpaPolicyError(Exception):
    """Base class for OPA-adapter errors."""


class PolicyTransportError(OpaPolicyError):
    """Raised internally when a transport refuses to evaluate.

    The :class:`OpaPolicyEvaluator` catches this and fail-closes to
    :class:`PolicyVerdict.REJECT`. Callers wishing to surface the
    underlying error explicitly should call
    :meth:`PolicyTransport.evaluate` directly.
    """


class PolicyEvaluationError(OpaPolicyError):
    """Raised when a transport returns a malformed result envelope."""


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


class PolicyVerdict(StrEnum):
    """Three-valued OPA verdict (spec line 1045)."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------------
# PolicyInput
# ---------------------------------------------------------------------------


def _validate_string(*, value: Any, name: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__!r}")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} characters ({len(value)})")
    return value


def _validate_payload(payload: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Normalise the payload into a sorted tuple of (key, value) pairs.

    Values are restricted to JSON primitives (``str`` / ``int`` /
    ``float`` / ``bool`` / ``None``) so the canonical JSON projection
    is byte-stable across Python versions.
    """

    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a Mapping, got {type(payload).__name__!r}")
    if len(payload) > MAX_PAYLOAD_KEYS:
        raise ValueError(f"payload exceeds MAX_PAYLOAD_KEYS ({len(payload)} > {MAX_PAYLOAD_KEYS})")
    out: list[tuple[str, Any]] = []
    for key in sorted(payload):
        if not isinstance(key, str):
            raise TypeError(f"payload keys must be str, got {type(key).__name__!r}")
        if not key:
            raise ValueError("payload keys must be non-empty")
        value = payload[key]
        if isinstance(value, bool):
            out.append((key, value))
            continue
        if isinstance(value, (int, float)):
            out.append((key, value))
            continue
        if value is None:
            out.append((key, None))
            continue
        if isinstance(value, str):
            if len(value) > MAX_PAYLOAD_VALUE_LEN:
                raise ValueError(
                    f"payload[{key!r}] exceeds MAX_PAYLOAD_VALUE_LEN "
                    f"({len(value)} > {MAX_PAYLOAD_VALUE_LEN})"
                )
            out.append((key, value))
            continue
        raise TypeError(
            f"payload[{key!r}] has unsupported type "
            f"{type(value).__name__!r}; allowed: str/int/float/bool/None"
        )
    return tuple(out)


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyInput:
    """Frozen envelope sent to the OPA transport.

    ``action`` and ``mode`` carry the two axes the constraint engine
    keys on. ``subject`` identifies the request initiator (operator id,
    strategy id, etc). ``payload`` carries opaque key/value scalars.
    """

    action: str
    mode: str
    subject: str
    payload: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _validate_string(value=self.action, name="action", max_len=128)
        _validate_string(value=self.mode, name="mode", max_len=64)
        _validate_string(value=self.subject, name="subject", max_len=128)
        if not isinstance(self.payload, tuple):
            raise TypeError("payload must be a tuple of (key, value) pairs")
        seen: set[str] = set()
        for entry in self.payload:
            if not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str):
                raise TypeError("payload entries must be (str, value) tuples")
            if entry[0] in seen:
                raise ValueError(f"duplicate payload key: {entry[0]!r}")
            seen.add(entry[0])

    @classmethod
    def from_mapping(
        cls,
        *,
        action: str,
        mode: str,
        subject: str,
        payload: Mapping[str, Any] | None = None,
    ) -> PolicyInput:
        """Build a :class:`PolicyInput` from a dict-like payload."""

        normalised = _validate_payload(payload or EMPTY_PAYLOAD)
        return cls(action=action, mode=mode, subject=subject, payload=normalised)

    def as_payload(self) -> Mapping[str, Any]:
        """Project the payload back to a plain mapping (sorted keys)."""

        return dict(self.payload)

    def to_canonical_json(self) -> str:
        """Return the sorted-key JSON projection (INV-15 byte stable)."""

        blob = {
            "action": self.action,
            "mode": self.mode,
            "subject": self.subject,
            "payload": dict(self.payload),
        }
        return json.dumps(blob, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Transport contract
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyTransportResult:
    """Frozen result envelope returned by a :class:`PolicyTransport`."""

    verdict: PolicyVerdict
    policy_id: str
    rule_path: str
    rejection_code: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, PolicyVerdict):
            raise TypeError(f"verdict must be PolicyVerdict, got {type(self.verdict).__name__!r}")
        _validate_string(value=self.policy_id, name="policy_id", max_len=128)
        _validate_string(
            value=self.rule_path,
            name="rule_path",
            max_len=MAX_RULE_PATH_LEN,
        )
        if not isinstance(self.rejection_code, str):
            raise TypeError("rejection_code must be str")
        if len(self.rejection_code) > MAX_REJECTION_CODE_LEN:
            raise ValueError("rejection_code exceeds MAX_REJECTION_CODE_LEN")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be str")
        if len(self.summary) > MAX_SUMMARY_LEN:
            raise ValueError("summary exceeds MAX_SUMMARY_LEN")
        if self.verdict is PolicyVerdict.APPROVE and self.rejection_code:
            raise ValueError("rejection_code must be empty on APPROVE verdicts")
        if self.verdict is PolicyVerdict.REJECT and not self.rejection_code:
            raise ValueError("rejection_code is required on REJECT verdicts")


@runtime_checkable
class PolicyTransport(Protocol):
    """Per-call dispatch for one OPA policy evaluation."""

    def evaluate(
        self,
        policy_input: PolicyInput,
        /,
    ) -> PolicyTransportResult:
        """Evaluate ``policy_input`` and return a frozen result.

        Implementations MUST raise :class:`PolicyTransportError` on
        recoverable transport failures (timeouts, IO, schema mismatch).
        The :class:`OpaPolicyEvaluator` catches the error and
        fail-closes to :class:`PolicyVerdict.REJECT`.
        """


# ---------------------------------------------------------------------------
# In-process transport
# ---------------------------------------------------------------------------


Predicate = Callable[[PolicyInput], bool]


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyRule:
    """One Rego-shaped rule evaluated by the in-process transport."""

    policy_id: str
    rule_path: str
    predicate: Predicate
    rejection_code: str = ""
    summary: str = ""
    verdict_on_match: PolicyVerdict = PolicyVerdict.REJECT

    def __post_init__(self) -> None:
        _validate_string(value=self.policy_id, name="policy_id", max_len=128)
        _validate_string(
            value=self.rule_path,
            name="rule_path",
            max_len=MAX_RULE_PATH_LEN,
        )
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")
        if not isinstance(self.verdict_on_match, PolicyVerdict):
            raise TypeError("verdict_on_match must be PolicyVerdict")
        if not isinstance(self.rejection_code, str):
            raise TypeError("rejection_code must be str")
        if len(self.rejection_code) > MAX_REJECTION_CODE_LEN:
            raise ValueError("rejection_code exceeds MAX_REJECTION_CODE_LEN")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be str")
        if len(self.summary) > MAX_SUMMARY_LEN:
            raise ValueError("summary exceeds MAX_SUMMARY_LEN")
        if self.verdict_on_match is PolicyVerdict.REJECT and not self.rejection_code:
            raise ValueError("rejection_code required when verdict_on_match=REJECT")
        if self.verdict_on_match is PolicyVerdict.APPROVE and self.rejection_code:
            raise ValueError("rejection_code must be empty when verdict_on_match=APPROVE")


@dataclasses.dataclass(frozen=True, slots=True)
class InProcessPolicyTransport:
    """Pure-Python OPA stand-in.

    Walks the rule tuple in declaration order, returns the first
    matching rule's verdict, otherwise approves (default-allow). The
    rule list is the canonical Rego-mirror — the ``.rego`` files in
    ``governance_engine/policies/`` are the source-of-truth for the
    external OPA binary; the rules here mirror their semantics so the
    same input yields the same verdict offline.
    """

    rules: tuple[PolicyRule, ...]
    default_policy_id: str = DEFAULT_POLICY_PACKAGE
    default_rule_path: str = "default/allow"
    default_summary: str = "policy default allow"

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple):
            raise TypeError("rules must be a tuple of PolicyRule")
        for r in self.rules:
            if not isinstance(r, PolicyRule):
                raise TypeError(f"rule must be PolicyRule, got {type(r).__name__!r}")
        seen: set[tuple[str, str]] = set()
        for r in self.rules:
            key = (r.policy_id, r.rule_path)
            if key in seen:
                raise ValueError(f"duplicate rule {r.policy_id}/{r.rule_path}")
            seen.add(key)
        _validate_string(
            value=self.default_policy_id,
            name="default_policy_id",
            max_len=128,
        )
        _validate_string(
            value=self.default_rule_path,
            name="default_rule_path",
            max_len=MAX_RULE_PATH_LEN,
        )
        if not isinstance(self.default_summary, str):
            raise TypeError("default_summary must be str")
        if len(self.default_summary) > MAX_SUMMARY_LEN:
            raise ValueError("default_summary exceeds MAX_SUMMARY_LEN")

    def evaluate(
        self,
        policy_input: PolicyInput,
        /,
    ) -> PolicyTransportResult:
        if not isinstance(policy_input, PolicyInput):
            raise PolicyTransportError(
                f"policy_input must be PolicyInput, got {type(policy_input).__name__!r}"
            )
        for rule in self.rules:
            try:
                matched = bool(rule.predicate(policy_input))
            except (TypeError, ValueError, KeyError) as exc:
                raise PolicyTransportError(
                    f"predicate {rule.policy_id}/{rule.rule_path} raised: {exc}"
                ) from exc
            if matched:
                return PolicyTransportResult(
                    verdict=rule.verdict_on_match,
                    policy_id=rule.policy_id,
                    rule_path=rule.rule_path,
                    rejection_code=rule.rejection_code,
                    summary=rule.summary,
                )
        return PolicyTransportResult(
            verdict=PolicyVerdict.APPROVE,
            policy_id=self.default_policy_id,
            rule_path=self.default_rule_path,
            rejection_code="",
            summary=self.default_summary,
        )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Frozen verdict produced by :class:`OpaPolicyEvaluator`."""

    verdict: PolicyVerdict
    policy_id: str
    rule_path: str
    rejection_code: str
    summary: str
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, PolicyVerdict):
            raise TypeError("verdict must be PolicyVerdict")
        _validate_string(value=self.policy_id, name="policy_id", max_len=128)
        _validate_string(
            value=self.rule_path,
            name="rule_path",
            max_len=MAX_RULE_PATH_LEN,
        )
        if not isinstance(self.rejection_code, str):
            raise TypeError("rejection_code must be str")
        if len(self.rejection_code) > MAX_REJECTION_CODE_LEN:
            raise ValueError("rejection_code exceeds MAX_REJECTION_CODE_LEN")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be str")
        if len(self.summary) > MAX_SUMMARY_LEN:
            raise ValueError("summary exceeds MAX_SUMMARY_LEN")
        _validate_string(value=self.policy_digest, name="policy_digest", max_len=32)
        # Hex digest of BLAKE2b-16 -> 32 hex characters.
        if len(self.policy_digest) != 32:
            raise ValueError("policy_digest must be 32 hex characters (BLAKE2b-16)")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    policy_input: PolicyInput,
    transport_result: PolicyTransportResult,
) -> str:
    """Content-address the (input, result) pair (INV-15)."""

    blob = {
        "input": json.loads(policy_input.to_canonical_json()),
        "result": {
            "verdict": str(transport_result.verdict),
            "policy_id": transport_result.policy_id,
            "rule_path": transport_result.rule_path,
            "rejection_code": transport_result.rejection_code,
            "summary": transport_result.summary,
        },
        "version": OPA_ADAPTER_VERSION,
    }
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class OpaPolicyEvaluator:
    """Top-level coordinator. Calls a transport, fail-closes on errors.

    Per spec line 1046: OPA timeout or error → ``REJECT``. The
    fail-closed verdict carries ``rejection_code=fail_closed_code`` so
    operators can distinguish a transport failure from a real
    rule-driven rejection.
    """

    transport: PolicyTransport
    fail_closed_code: str = "POLICY_TRANSPORT_ERROR"
    fail_closed_policy_id: str = DEFAULT_POLICY_PACKAGE
    fail_closed_rule_path: str = "fail_closed/transport_error"

    def __post_init__(self) -> None:
        if not isinstance(self.transport, PolicyTransport):
            raise TypeError(
                f"transport must implement PolicyTransport, got {type(self.transport).__name__!r}"
            )
        _validate_string(
            value=self.fail_closed_code,
            name="fail_closed_code",
            max_len=MAX_REJECTION_CODE_LEN,
        )
        _validate_string(
            value=self.fail_closed_policy_id,
            name="fail_closed_policy_id",
            max_len=128,
        )
        _validate_string(
            value=self.fail_closed_rule_path,
            name="fail_closed_rule_path",
            max_len=MAX_RULE_PATH_LEN,
        )

    def evaluate(self, policy_input: PolicyInput) -> PolicyDecision:
        """Run a single fail-closed policy evaluation."""

        if not isinstance(policy_input, PolicyInput):
            raise TypeError(
                f"policy_input must be PolicyInput, got {type(policy_input).__name__!r}"
            )
        try:
            result = self.transport.evaluate(policy_input)
        except PolicyTransportError as exc:
            return self._fail_closed(policy_input=policy_input, reason=str(exc))
        if not isinstance(result, PolicyTransportResult):
            raise PolicyEvaluationError(
                f"transport returned non-PolicyTransportResult value: {type(result).__name__!r}"
            )
        digest = _compute_policy_digest(
            policy_input=policy_input,
            transport_result=result,
        )
        return PolicyDecision(
            verdict=result.verdict,
            policy_id=result.policy_id,
            rule_path=result.rule_path,
            rejection_code=result.rejection_code,
            summary=result.summary,
            policy_digest=digest,
        )

    def _fail_closed(self, *, policy_input: PolicyInput, reason: str) -> PolicyDecision:
        truncated = reason[:MAX_SUMMARY_LEN]
        synthetic = PolicyTransportResult(
            verdict=PolicyVerdict.REJECT,
            policy_id=self.fail_closed_policy_id,
            rule_path=self.fail_closed_rule_path,
            rejection_code=self.fail_closed_code,
            summary=f"transport error: {truncated}",
        )
        digest = _compute_policy_digest(
            policy_input=policy_input,
            transport_result=synthetic,
        )
        return PolicyDecision(
            verdict=synthetic.verdict,
            policy_id=synthetic.policy_id,
            rule_path=synthetic.rule_path,
            rejection_code=synthetic.rejection_code,
            summary=synthetic.summary,
            policy_digest=digest,
        )


# ---------------------------------------------------------------------------
# GovernanceDecision projection
# ---------------------------------------------------------------------------


def to_governance_decision(
    decision: PolicyDecision,
    *,
    ts_ns: int,
    kind: DecisionKind,
    ledger_seq: int = -1,
) -> GovernanceDecision:
    """Project a :class:`PolicyDecision` onto a :class:`GovernanceDecision`.

    ``approved`` is ``True`` iff ``decision.verdict`` is
    :class:`PolicyVerdict.APPROVE`. ``ESCALATE`` is mapped to
    ``approved=False`` with ``rejection_code`` preserved — the
    operator-attention layer is responsible for picking up the
    escalation.

    This is the **only** place the module emits a
    :class:`GovernanceDecision`. The file lives under
    ``governance_engine/`` precisely because B27 / B28 / INV-71 allow
    governance modules to construct governance decisions.
    """

    if not isinstance(decision, PolicyDecision):
        raise TypeError(f"decision must be PolicyDecision, got {type(decision).__name__!r}")
    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise TypeError("ts_ns must be int")
    if ts_ns < 0:
        raise ValueError("ts_ns must be non-negative")
    if not isinstance(kind, DecisionKind):
        raise TypeError("kind must be DecisionKind")
    if not isinstance(ledger_seq, int) or isinstance(ledger_seq, bool):
        raise TypeError("ledger_seq must be int")

    approved = decision.verdict is PolicyVerdict.APPROVE
    if approved:
        summary = decision.summary or "policy approved"
        rejection_code = ""
    else:
        summary = decision.summary or "policy rejected"
        rejection_code = decision.rejection_code

    return GovernanceDecision(
        ts_ns=ts_ns,
        kind=kind,
        approved=approved,
        summary=summary[:MAX_SUMMARY_LEN],
        rejection_code=rejection_code,
        ledger_seq=ledger_seq,
    )


# ---------------------------------------------------------------------------
# Baseline rules (Python mirror of governance_engine/policies/*.rego)
# ---------------------------------------------------------------------------


def _payload_value(policy_input: PolicyInput, key: str) -> Any:
    """Look up a payload value, returning ``None`` if absent."""

    for k, v in policy_input.payload:
        if k == key:
            return v
    return None


def _position_limit_violated(policy_input: PolicyInput) -> bool:
    """Mirror ``position_limits.rego``: notional > limit ⇒ REJECT."""

    notional = _payload_value(policy_input, "notional_usd")
    limit = _payload_value(policy_input, "notional_limit_usd")
    if notional is None or limit is None:
        return False
    if not isinstance(notional, (int, float)) or isinstance(notional, bool):
        return False
    if not isinstance(limit, (int, float)) or isinstance(limit, bool):
        return False
    return notional > limit


def _execution_gate_blocked(policy_input: PolicyInput) -> bool:
    """Mirror ``execution_gates.rego``: SAFE mode blocks all execution."""

    if policy_input.action != "EXECUTE_ORDER":
        return False
    return policy_input.mode == "SAFE"


def _autonomy_level_escalation(policy_input: PolicyInput) -> bool:
    """Mirror ``autonomy_levels.rego``: AUTO with hazards ⇒ ESCALATE."""

    if policy_input.mode != "AUTO":
        return False
    hazards = _payload_value(policy_input, "active_hazards")
    if hazards is None:
        return False
    if not isinstance(hazards, (int, float)) or isinstance(hazards, bool):
        return False
    return hazards > 0


def build_baseline_rules() -> tuple[PolicyRule, ...]:
    """Return the Python mirror of the three baseline Rego policies.

    Order is deterministic and matches the spec sub-output ordering
    (position_limits → execution_gates → autonomy_levels). The
    in-process transport stops at the first matching rule, so this
    order also defines precedence.
    """

    return (
        PolicyRule(
            policy_id="dix.governance.position_limits",
            rule_path="position_limits/notional_exceeded",
            predicate=_position_limit_violated,
            rejection_code="POLICY_POSITION_LIMIT",
            summary="notional exceeds configured limit",
            verdict_on_match=PolicyVerdict.REJECT,
        ),
        PolicyRule(
            policy_id="dix.governance.execution_gates",
            rule_path="execution_gates/safe_blocks_orders",
            predicate=_execution_gate_blocked,
            rejection_code="POLICY_EXECUTION_GATE",
            summary="SAFE mode blocks order execution",
            verdict_on_match=PolicyVerdict.REJECT,
        ),
        PolicyRule(
            policy_id="dix.governance.autonomy_levels",
            rule_path="autonomy_levels/auto_with_hazards",
            predicate=_autonomy_level_escalation,
            rejection_code="POLICY_AUTONOMY_ESCALATE",
            summary="AUTO mode requires zero active hazards",
            verdict_on_match=PolicyVerdict.ESCALATE,
        ),
    )


# ---------------------------------------------------------------------------
# Lazy HTTP transport factory (opa-python-client)
# ---------------------------------------------------------------------------


def opa_http_transport_factory(
    *,
    client: Any,
    policy_package: str = DEFAULT_POLICY_PACKAGE,
    decision_key: str = "result",
) -> PolicyTransport:
    """Wrap an ``opa-python-client`` ``OpaClient`` as a transport.

    The ``opa-python-client`` package is imported lazily inside this
    factory body. The factory returns a frozen
    :class:`_OpaClientTransport` that calls ``client.check_policy``
    and maps the returned JSON to a :class:`PolicyTransportResult`.

    Per spec line 1046, any exception raised by the client is
    converted to :class:`PolicyTransportError`, which
    :class:`OpaPolicyEvaluator` converts to a fail-closed
    :class:`PolicyVerdict.REJECT`.
    """

    if client is None:
        raise TypeError("client must be an opa-python-client OpaClient")
    _validate_string(value=policy_package, name="policy_package", max_len=128)
    _validate_string(value=decision_key, name="decision_key", max_len=64)

    # Lazy import — opa-python-client is optional. The factory is the
    # only place in the module that touches the package; module
    # import stays cheap and offline.
    try:  # pragma: no cover — exercised by integration setups.
        import opa_client.opa as _opa_module  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opa-python-client is required for opa_http_transport_factory; "
            "install via `pip install opa-python-client`"
        ) from exc

    return _OpaClientTransport(
        client=client,
        policy_package=policy_package,
        decision_key=decision_key,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _OpaClientTransport:
    """Frozen adapter over an ``opa-python-client`` instance.

    Kept private — callers obtain instances exclusively through
    :func:`opa_http_transport_factory` so the lazy import is
    enforced.
    """

    client: Any
    policy_package: str
    decision_key: str

    def evaluate(
        self,
        policy_input: PolicyInput,
        /,
    ) -> PolicyTransportResult:
        if not isinstance(policy_input, PolicyInput):
            raise PolicyTransportError("policy_input must be PolicyInput")
        try:
            raw = self.client.check_policy_rule(
                input_data=policy_input.as_payload(),
                package_path=self.policy_package,
                rule_name="allow",
            )
        except Exception as exc:  # pragma: no cover — defensive
            raise PolicyTransportError(f"opa client raised: {exc!r}") from exc
        if not isinstance(raw, Mapping):
            raise PolicyTransportError(
                f"opa client returned non-mapping payload: {type(raw).__name__!r}"
            )
        verdict_raw = raw.get(self.decision_key)
        if isinstance(verdict_raw, bool):
            verdict = PolicyVerdict.APPROVE if verdict_raw else PolicyVerdict.REJECT
        elif isinstance(verdict_raw, str):
            try:
                verdict = PolicyVerdict(verdict_raw.upper())
            except ValueError as exc:
                raise PolicyTransportError(f"unknown verdict {verdict_raw!r}") from exc
        else:
            raise PolicyTransportError(
                f"opa client returned unsupported decision type: {type(verdict_raw).__name__!r}"
            )
        rejection_code = ""
        summary = ""
        rule_path = f"{self.policy_package}/allow"
        meta = raw.get("metadata")
        if isinstance(meta, Mapping):
            rc = meta.get("rejection_code", "")
            if isinstance(rc, str):
                rejection_code = rc[:MAX_REJECTION_CODE_LEN]
            sm = meta.get("summary", "")
            if isinstance(sm, str):
                summary = sm[:MAX_SUMMARY_LEN]
            rp = meta.get("rule_path", "")
            if isinstance(rp, str) and rp:
                rule_path = rp[:MAX_RULE_PATH_LEN]
        if verdict is PolicyVerdict.REJECT and not rejection_code:
            rejection_code = "POLICY_REJECTED"
        return PolicyTransportResult(
            verdict=verdict,
            policy_id=self.policy_package,
            rule_path=rule_path,
            rejection_code=rejection_code,
            summary=summary,
        )
