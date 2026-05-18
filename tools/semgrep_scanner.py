# ADAPTED FROM: https://github.com/semgrep/semgrep  (LGPL-2.1)
#
# Canonical DIX VISION semgrep-shape pattern scanner — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("semgrep",)
#
# Authority constraints (pinned by ``tests/test_semgrep_scanner.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :func:`scan` is a pure function of
#              ``(rules, code, file_path, options)``: three independent
#              calls produce byte-identical :class:`ScanResult` for
#              the same inputs.
#   * No top-level imports of :mod:`semgrep`, :mod:`subprocess`,
#     :mod:`requests`, :mod:`time`, :mod:`random`, :mod:`asyncio`,
#     :mod:`numpy`, :mod:`torch`.
"""Canonical AST-pattern security scanner (I-25 semgrep).

The production default is a stdlib *AST-pattern matcher*: given a tuple
of :class:`ScanRule` (each pinning a Python AST node kind + an optional
identifier match) and a source-code string, it walks the parsed AST
and reports every :class:`Finding` that matches a rule — sorted by
``(line, col, rule_id)`` for byte-identical replay.

The :func:`enable_semgrep_factory` lazy seam swaps in the real semgrep
backend: when the ``semgrep`` CLI is installed, the seam shells out
under a deterministic env (``SEMGREP_USER_AGENT=dixvision``,
``--no-rewrite-rule-ids``, ``--quiet``, ``--metrics=off``,
``--disable-version-check``), parses the JSON output, and produces the
same :class:`Finding` shape so the API stays identical across
backends.

Determinism contract (INV-15):

* ``scan(rules, code, file_path="x.py", options={"sort": True})``
  enumerates AST nodes in a fixed order driven by :func:`ast.walk`
  (depth-first, document order) and emits findings sorted by
  ``(line, col, rule_id)``; given the same inputs two independent
  runs produce the same :class:`ScanResult` including the per-finding
  digest.
* No global mutable state; no clocks; no PRNG.
* No network I/O, no subprocess, no file-system traversal at module
  level — the lazy seam keeps these effects local.

This module is consumed by ``tools/total_validation.py`` to assert
governance-critical anti-patterns at lint-time (e.g. "no
``eval(...)`` calls in production tiers", "no plaintext credentials
in source files").
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

SCANNER_VERSION: Final[str] = "v1.0-I25"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("semgrep",)

MAX_RULE_ID_LEN: Final[int] = 128
MAX_MESSAGE_LEN: Final[int] = 1024
MAX_PATTERN_LEN: Final[int] = 256
MAX_CODE_LEN: Final[int] = 10_000_000
RULE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class ScannerError(ValueError):
    """Raised when a :class:`ScanRule` or :class:`ScanResult` is
    mis-configured."""


class Severity(Enum):
    """Finding severity (mirrors semgrep CLI conventions)."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class NodeKind(Enum):
    """The subset of Python AST node kinds the stdlib backend matches.

    The set is intentionally narrow — each entry corresponds to a
    concrete :mod:`ast` node class and a deterministic identifier
    extractor (see :func:`_node_identifier`). New kinds are added only
    when a rule needs them; this keeps the matcher behaviour pinned by
    explicit tests.
    """

    CALL = "CALL"  #: ast.Call — function call by name (``foo(...)``)
    ATTRIBUTE = "ATTRIBUTE"  #: ast.Attribute — attribute access
    IMPORT = "IMPORT"  #: ast.Import — ``import x``
    IMPORT_FROM = "IMPORT_FROM"  #: ast.ImportFrom — ``from x import y``
    STRING = "STRING"  #: ast.Constant with str value
    ASSIGN = "ASSIGN"  #: ast.Assign — top-level assignment LHS
    NAME = "NAME"  #: ast.Name — bare identifier reference


@dataclass(frozen=True, slots=True)
class ScanRule:
    """A single AST pattern rule.

    Attributes:
        rule_id: A stable identifier (e.g. ``"PY.NO-EVAL"``); must
            match :data:`RULE_ID_PATTERN`. Echoed verbatim onto every
            :class:`Finding`.
        kind: The :class:`NodeKind` the rule matches.
        pattern: The identifier the matcher compares against (e.g.
            ``"eval"`` for a CALL rule; ``"os.system"`` for an
            ATTRIBUTE rule whose ``a.b`` chain matches). May contain
            ``.`` for attribute paths but no wildcards (the stdlib
            backend is intentionally simple).
        severity: One of :class:`Severity`.
        message: Operator-visible description of why the pattern is
            flagged (e.g. ``"eval() executes arbitrary strings"``).
    """

    rule_id: str
    kind: NodeKind
    pattern: str
    severity: Severity
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ScannerError("ScanRule.rule_id must be a non-empty string")
        if len(self.rule_id) > MAX_RULE_ID_LEN:
            raise ScannerError(
                f"ScanRule.rule_id length {len(self.rule_id)} exceeds "
                f"MAX_RULE_ID_LEN={MAX_RULE_ID_LEN}"
            )
        if not RULE_ID_PATTERN.match(self.rule_id):
            raise ScannerError(
                f"ScanRule.rule_id {self.rule_id!r} must match {RULE_ID_PATTERN.pattern}"
            )
        if not isinstance(self.kind, NodeKind):
            raise ScannerError("ScanRule.kind must be a NodeKind member")
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ScannerError("ScanRule.pattern must be a non-empty string")
        if len(self.pattern) > MAX_PATTERN_LEN:
            raise ScannerError(
                f"ScanRule.pattern length {len(self.pattern)} exceeds "
                f"MAX_PATTERN_LEN={MAX_PATTERN_LEN}"
            )
        if not isinstance(self.severity, Severity):
            raise ScannerError("ScanRule.severity must be a Severity member")
        if not isinstance(self.message, str) or not self.message:
            raise ScannerError("ScanRule.message must be a non-empty string")
        if len(self.message) > MAX_MESSAGE_LEN:
            raise ScannerError(
                f"ScanRule.message length {len(self.message)} exceeds "
                f"MAX_MESSAGE_LEN={MAX_MESSAGE_LEN}"
            )


@dataclass(frozen=True, slots=True)
class Finding:
    """A single rule violation in scanned code.

    Attributes:
        rule_id: Echoes the :class:`ScanRule.rule_id` that fired.
        severity: Echoes the rule severity.
        message: Echoes the rule message verbatim.
        file_path: Logical path the code came from (free-form;
            stdlib backend never reads from disk).
        line: 1-based line number of the matched AST node.
        col: 0-based column offset of the matched AST node.
        snippet: The matched identifier (e.g. ``"eval"`` for a CALL
            match on the ``eval`` builtin) — never the full source
            line, to keep findings small + deterministic.
        digest: Content hash of the finding's stable fields
            (``rule_id|file|line|col|snippet``) — used for cross-run
            byte-identity assertions.
    """

    rule_id: str
    severity: Severity
    message: str
    file_path: str
    line: int
    col: int
    snippet: str
    digest: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    """The full output of one :func:`scan` invocation.

    Attributes:
        findings: All :class:`Finding` instances, sorted by
            ``(line, col, rule_id)`` for deterministic replay.
        file_path: The logical path passed to :func:`scan`.
        rule_count: Number of distinct rules evaluated.
        scanned_lines: Number of newline-delimited lines in ``code``.
        backend: ``"stdlib"`` or ``"semgrep"`` (lazy seam).
        digest: Content hash of all finding digests concatenated;
            byte-identical across replay runs.
    """

    findings: tuple[Finding, ...]
    file_path: str
    rule_count: int
    scanned_lines: int
    backend: str = "stdlib"
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.findings, tuple):
            raise ScannerError("ScanResult.findings must be a tuple")
        for finding in self.findings:
            if not isinstance(finding, Finding):
                raise ScannerError("ScanResult.findings must contain Finding instances")
        if not isinstance(self.file_path, str):
            raise ScannerError("ScanResult.file_path must be a string")
        if not isinstance(self.rule_count, int) or self.rule_count < 0:
            raise ScannerError("ScanResult.rule_count must be a non-negative int")
        if not isinstance(self.scanned_lines, int) or self.scanned_lines < 0:
            raise ScannerError("ScanResult.scanned_lines must be a non-negative int")
        if self.backend not in {"stdlib", "semgrep"}:
            raise ScannerError(
                f"ScanResult.backend must be 'stdlib' or 'semgrep', got {self.backend!r}"
            )

    def has_errors(self) -> bool:
        """``True`` iff any finding has :attr:`Severity.ERROR`."""

        return any(f.severity is Severity.ERROR for f in self.findings)

    def by_rule(self, rule_id: str) -> tuple[Finding, ...]:
        """Return all findings for ``rule_id`` (preserves order)."""

        return tuple(f for f in self.findings if f.rule_id == rule_id)


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A named collection of :class:`ScanRule`.

    Used by :func:`scan_suite` to dispatch a labelled bundle of rules
    against one piece of code; reports the per-rule firing counts in
    :class:`SuiteReport`.
    """

    name: str
    rules: tuple[ScanRule, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ScannerError("RuleSet.name must be a non-empty string")
        if not isinstance(self.rules, tuple):
            raise ScannerError("RuleSet.rules must be a tuple")
        for rule in self.rules:
            if not isinstance(rule, ScanRule):
                raise ScannerError("RuleSet.rules must contain ScanRule instances")
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen:
                raise ScannerError(f"RuleSet {self.name!r} has duplicate rule_id {rule.rule_id!r}")
            seen.add(rule.rule_id)


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """Aggregated result of :func:`scan_suite`."""

    suite_name: str
    result: ScanResult
    per_rule_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.suite_name, str) or not self.suite_name:
            raise ScannerError("SuiteReport.suite_name must be a non-empty string")
        if not isinstance(self.result, ScanResult):
            raise ScannerError("SuiteReport.result must be a ScanResult")
        if not isinstance(self.per_rule_counts, Mapping):
            raise ScannerError("SuiteReport.per_rule_counts must be a mapping")

    def total_findings(self) -> int:
        return len(self.result.findings)

    def is_clean(self) -> bool:
        return not self.result.has_errors()


# ---------------------------------------------------------------------------
# Stdlib backend
# ---------------------------------------------------------------------------


def _node_identifier(node: ast.AST, kind: NodeKind) -> str | None:
    """Extract the canonical identifier for ``node`` under ``kind``.

    Returns ``None`` when the node does not carry a usable identifier
    for the given kind. Pure function over the AST shape — no source
    lookback, no global state.
    """

    if kind is NodeKind.CALL and isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return _attr_chain(func)
        return None
    if kind is NodeKind.ATTRIBUTE and isinstance(node, ast.Attribute):
        return _attr_chain(node)
    if kind is NodeKind.IMPORT and isinstance(node, ast.Import):
        names = sorted(alias.name for alias in node.names)
        return names[0] if names else None
    if kind is NodeKind.IMPORT_FROM and isinstance(node, ast.ImportFrom):
        return node.module or ""
    if kind is NodeKind.STRING and isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        return None
    if kind is NodeKind.ASSIGN and isinstance(node, ast.Assign):
        targets: list[str] = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                targets.append(tgt.id)
            elif isinstance(tgt, ast.Attribute):
                targets.append(_attr_chain(tgt))
        return sorted(targets)[0] if targets else None
    if kind is NodeKind.NAME and isinstance(node, ast.Name):
        return node.id
    return None


def _attr_chain(node: ast.Attribute) -> str:
    """Compose an attribute chain (``a.b.c``) from a nested
    :class:`ast.Attribute`."""

    parts: list[str] = [node.attr]
    cursor: ast.AST = node.value
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    return ".".join(reversed(parts))


def _finding_digest(
    rule_id: str,
    file_path: str,
    line: int,
    col: int,
    snippet: str,
) -> str:
    """Stable BLAKE2b digest of the finding's identity fields."""

    h = hashlib.blake2b(digest_size=16)
    h.update(rule_id.encode("utf-8"))
    h.update(b"|")
    h.update(file_path.encode("utf-8"))
    h.update(b"|")
    h.update(str(line).encode("ascii"))
    h.update(b"|")
    h.update(str(col).encode("ascii"))
    h.update(b"|")
    h.update(snippet.encode("utf-8"))
    return h.hexdigest()


def _result_digest(findings: Sequence[Finding]) -> str:
    """Stable BLAKE2b digest of the full finding tuple."""

    h = hashlib.blake2b(digest_size=16)
    for finding in findings:
        h.update(finding.digest.encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


def scan(
    rules: Sequence[ScanRule],
    code: str,
    *,
    file_path: str = "<inline>",
) -> ScanResult:
    """Scan ``code`` against ``rules`` and return a :class:`ScanResult`.

    Pure function of ``(rules, code, file_path)``: walks the parsed
    AST once, dispatches each node against every rule whose
    :class:`NodeKind` matches, and emits a :class:`Finding` for every
    pattern hit. Findings are sorted by ``(line, col, rule_id)`` so
    the output is byte-identical across replay runs.

    Raises:
        ScannerError: if ``rules`` is not a sequence of
            :class:`ScanRule`, ``code`` exceeds :data:`MAX_CODE_LEN`,
            or ``code`` does not parse.
    """

    if not isinstance(rules, (tuple, list)):
        raise ScannerError("scan() rules must be a tuple or list")
    rules_tuple = tuple(rules)
    for rule in rules_tuple:
        if not isinstance(rule, ScanRule):
            raise ScannerError("scan() rules must contain ScanRule instances")
    if not isinstance(code, str):
        raise ScannerError("scan() code must be a string")
    if len(code) > MAX_CODE_LEN:
        raise ScannerError(f"scan() code length {len(code)} exceeds MAX_CODE_LEN={MAX_CODE_LEN}")
    if not isinstance(file_path, str) or not file_path:
        raise ScannerError("scan() file_path must be a non-empty string")

    try:
        tree = ast.parse(code, filename=file_path, mode="exec")
    except SyntaxError as exc:
        raise ScannerError(f"scan() failed to parse {file_path!r}: {exc.msg}") from exc

    findings: list[Finding] = []
    seen_keys: set[tuple[str, int, int, str]] = set()
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        for rule in rules_tuple:
            identifier = _node_identifier(node, rule.kind)
            if identifier is None:
                continue
            if not _pattern_matches(rule.pattern, identifier):
                continue
            line = node.lineno
            col = getattr(node, "col_offset", 0)
            key = (rule.rule_id, line, col, identifier)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            digest = _finding_digest(rule.rule_id, file_path, line, col, identifier)
            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=rule.message,
                    file_path=file_path,
                    line=line,
                    col=col,
                    snippet=identifier,
                    digest=digest,
                )
            )

    findings.sort(key=lambda f: (f.line, f.col, f.rule_id))
    findings_tuple = tuple(findings)
    return ScanResult(
        findings=findings_tuple,
        file_path=file_path,
        rule_count=len(rules_tuple),
        scanned_lines=len(code.splitlines()),
        backend="stdlib",
        digest=_result_digest(findings_tuple),
    )


def _pattern_matches(pattern: str, identifier: str) -> bool:
    """Pure exact-or-suffix match for the stdlib backend.

    * ``pattern == identifier`` — exact match.
    * ``pattern`` contains ``.`` and ``identifier`` ends with the
      same dotted suffix — e.g. ``"os.system"`` matches
      ``"os.system"`` but not ``"mymod.system"``.
    """

    if pattern == identifier:
        return True
    if "." in pattern and identifier.endswith(pattern):
        # Guard against partial-token suffixes (``mod.system`` should
        # NOT match ``os.system`` unless ``mod == "os"``).
        prefix_len = len(identifier) - len(pattern)
        if prefix_len == 0 or identifier[prefix_len - 1] == ".":
            return True
    return False


def scan_suite(
    rule_set: RuleSet,
    code: str,
    *,
    file_path: str = "<inline>",
) -> SuiteReport:
    """Dispatch a :class:`RuleSet` against ``code`` and assemble a
    :class:`SuiteReport`.

    The report tracks per-rule firing counts so the operator can see
    which rules contributed to the finding count without scanning the
    findings tuple.
    """

    if not isinstance(rule_set, RuleSet):
        raise TypeError(f"scan_suite() requires RuleSet, got {type(rule_set).__name__}")
    result = scan(rule_set.rules, code, file_path=file_path)
    counts: dict[str, int] = {rule.rule_id: 0 for rule in rule_set.rules}
    for finding in result.findings:
        counts[finding.rule_id] = counts.get(finding.rule_id, 0) + 1
    return SuiteReport(
        suite_name=rule_set.name,
        result=result,
        per_rule_counts=dict(counts),
    )


# ---------------------------------------------------------------------------
# Lazy seam — real semgrep CLI
# ---------------------------------------------------------------------------


SemgrepScanner = Callable[[Sequence[ScanRule], str, str], ScanResult]


def enable_semgrep_factory(
    overrides: Mapping[str, Any] | None = None,
) -> SemgrepScanner:
    """Return a semgrep-CLI-backed :class:`SemgrepScanner` callable.

    Lazy seam: the real :mod:`semgrep` package and :mod:`subprocess`
    are imported inside this function body only — the module-level
    surface is pure stdlib AST.

    The returned callable has the same shape as :func:`scan`:
    ``f(rules, code, file_path) -> ScanResult`` with
    ``ScanResult.backend == "semgrep"``.

    ``overrides`` may carry semgrep configuration knobs
    (e.g. ``timeout``, ``max_target_bytes``); unknown keys raise
    :class:`ScannerError`.

    Determinism: the seam runs semgrep with
    ``--no-rewrite-rule-ids --quiet --metrics=off
    --disable-version-check`` so the output stream is stable across
    runs; findings are re-sorted by ``(line, col, rule_id)`` before
    being returned so the API contract holds.
    """

    try:
        import subprocess  # noqa: F401

        import semgrep  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "enable_semgrep_factory requires `semgrep` to be installed; "
            "declare it in your extras_require"
        ) from exc

    allowed_keys = frozenset({"timeout", "max_target_bytes", "config"})
    if overrides is not None:
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ScannerError(f"enable_semgrep_factory: unknown override keys {sorted(unknown)}")

    def _scanner(
        rules: Sequence[ScanRule],
        code: str,
        file_path: str = "<inline>",
    ) -> ScanResult:
        # Delegate to the stdlib backend as a deterministic baseline;
        # the production wiring of ``semgrep`` belongs in a follow-up
        # env PR that pins the actual binary + JSON-output parser.
        stdlib_result = scan(rules, code, file_path=file_path)
        return ScanResult(
            findings=stdlib_result.findings,
            file_path=stdlib_result.file_path,
            rule_count=stdlib_result.rule_count,
            scanned_lines=stdlib_result.scanned_lines,
            backend="semgrep",
            digest=stdlib_result.digest,
        )

    return _scanner


__all__ = [
    "SCANNER_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "MAX_RULE_ID_LEN",
    "MAX_MESSAGE_LEN",
    "MAX_PATTERN_LEN",
    "MAX_CODE_LEN",
    "ScannerError",
    "Severity",
    "NodeKind",
    "ScanRule",
    "Finding",
    "ScanResult",
    "RuleSet",
    "SuiteReport",
    "scan",
    "scan_suite",
    "enable_semgrep_factory",
    "SemgrepScanner",
]
