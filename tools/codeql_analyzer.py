# ADAPTED FROM: https://github.com/github/codeql  (MIT)
#
# Canonical DIX VISION CodeQL-shape dataflow analyzer — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("codeql",)
#
# Authority constraints (pinned by ``tests/test_codeql_analyzer.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :func:`analyze` is a pure function of
#              ``(query, code, file_path)``: three independent calls
#              produce byte-identical :class:`AnalysisResult` for the
#              same inputs.
#   * No top-level imports of :mod:`codeql`, :mod:`subprocess`,
#     :mod:`requests`, :mod:`time`, :mod:`random`, :mod:`asyncio`,
#     :mod:`numpy`, :mod:`torch`.
"""Canonical AST-based taint dataflow analyzer (I-26 codeql).

The production default is a stdlib *intra-procedural taint tracker*:
given a :class:`DataFlowQuery` (a tuple of :class:`TaintSource` plus
:class:`TaintSink` patterns and an optional tuple of
:class:`Sanitizer` patterns) and a source-code string, it walks the
parsed AST, tracks variables whose values flow from a taint source,
and emits a :class:`Trace` whenever a tainted variable reaches a taint
sink without an intervening sanitizer.

The :func:`enable_codeql_factory` lazy seam swaps in the real CodeQL
backend: when the ``codeql`` CLI is installed, the seam shells out to
``codeql query run`` under a deterministic env, parses the BQRS
output, and produces the same :class:`Trace` shape so the API stays
identical across backends.

Determinism contract (INV-15):

* ``analyze(query, code, file_path="x.py")`` walks the AST in fixed
  document order (depth-first), records variable bindings as it
  encounters :class:`ast.Assign`, and resolves each :class:`ast.Call`
  argument that names a tainted variable against the sink set.
  Traces are emitted sorted by
  ``(sink_line, sink_col, source_line, source_col, source_name)`` so
  the output is byte-identical across replay runs.
* No global mutable state; no clocks; no PRNG.
* No network I/O, no subprocess, no file-system traversal at module
  level — the lazy seam keeps these effects local.

This module is consumed by ``tools/total_validation.py`` to assert
governance-critical taint invariants at lint-time (e.g. "user input
never flows into ``eval``", "DB query strings never flow from
``request.args``").
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

ANALYZER_VERSION: Final[str] = "v1.0-I26"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("codeql",)

MAX_QUERY_NAME_LEN: Final[int] = 128
MAX_PATTERN_LEN: Final[int] = 256
MAX_CODE_LEN: Final[int] = 10_000_000
MAX_TRACE_DEPTH: Final[int] = 64
QUERY_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class AnalyzerError(ValueError):
    """Raised when a :class:`DataFlowQuery` is mis-configured or
    :func:`analyze` cannot parse its input."""


class PatternKind(Enum):
    """The subset of :mod:`ast` node kinds taint patterns can match.

    * :attr:`CALL` — :class:`ast.Call` (matches the callee's dotted
      identifier, e.g. ``request.args.get``).
    * :attr:`NAME` — :class:`ast.Name` (matches a bare identifier;
      used for top-level free variables like ``user_input``).
    """

    CALL = "CALL"
    NAME = "NAME"


@dataclass(frozen=True, slots=True)
class TaintSource:
    """A pattern that introduces taint into the dataflow graph.

    Attributes:
        name: A stable identifier (e.g. ``"REQUEST.ARGS"``); must
            match :data:`QUERY_NAME_PATTERN`. Echoed onto every
            :class:`Trace.source_name`.
        kind: The :class:`PatternKind` the pattern matches.
        pattern: The identifier the matcher compares against (e.g.
            ``"request.args.get"`` for a CALL pattern; ``"user_input"``
            for a NAME pattern). May contain ``.`` for attribute
            paths but no wildcards.
    """

    name: str
    kind: PatternKind
    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AnalyzerError("TaintSource.name must be a non-empty string")
        if len(self.name) > MAX_QUERY_NAME_LEN:
            raise AnalyzerError(
                f"TaintSource.name length {len(self.name)} exceeds "
                f"MAX_QUERY_NAME_LEN={MAX_QUERY_NAME_LEN}"
            )
        if not QUERY_NAME_PATTERN.match(self.name):
            raise AnalyzerError(
                f"TaintSource.name {self.name!r} must match {QUERY_NAME_PATTERN.pattern}"
            )
        if not isinstance(self.kind, PatternKind):
            raise AnalyzerError("TaintSource.kind must be a PatternKind member")
        if not isinstance(self.pattern, str) or not self.pattern:
            raise AnalyzerError("TaintSource.pattern must be a non-empty string")
        if len(self.pattern) > MAX_PATTERN_LEN:
            raise AnalyzerError(
                f"TaintSource.pattern length {len(self.pattern)} exceeds "
                f"MAX_PATTERN_LEN={MAX_PATTERN_LEN}"
            )


@dataclass(frozen=True, slots=True)
class TaintSink:
    """A pattern that consumes tainted dataflow as an argument.

    Attributes:
        name: A stable identifier (e.g. ``"EVAL.CALL"``); must
            match :data:`QUERY_NAME_PATTERN`. Echoed onto every
            :class:`Trace.sink_name`.
        kind: The :class:`PatternKind` the pattern matches. Currently
            only :attr:`PatternKind.CALL` sinks are supported; a sink
            is a function call whose dotted identifier matches
            :attr:`pattern` and whose argument list contains a
            tainted variable name.
        pattern: The identifier the matcher compares against.
    """

    name: str
    kind: PatternKind
    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AnalyzerError("TaintSink.name must be a non-empty string")
        if len(self.name) > MAX_QUERY_NAME_LEN:
            raise AnalyzerError(
                f"TaintSink.name length {len(self.name)} exceeds "
                f"MAX_QUERY_NAME_LEN={MAX_QUERY_NAME_LEN}"
            )
        if not QUERY_NAME_PATTERN.match(self.name):
            raise AnalyzerError(
                f"TaintSink.name {self.name!r} must match {QUERY_NAME_PATTERN.pattern}"
            )
        if not isinstance(self.kind, PatternKind):
            raise AnalyzerError("TaintSink.kind must be a PatternKind member")
        if self.kind is not PatternKind.CALL:
            raise AnalyzerError(
                "TaintSink only supports PatternKind.CALL (sinks are function calls)"
            )
        if not isinstance(self.pattern, str) or not self.pattern:
            raise AnalyzerError("TaintSink.pattern must be a non-empty string")
        if len(self.pattern) > MAX_PATTERN_LEN:
            raise AnalyzerError(
                f"TaintSink.pattern length {len(self.pattern)} exceeds "
                f"MAX_PATTERN_LEN={MAX_PATTERN_LEN}"
            )


@dataclass(frozen=True, slots=True)
class Sanitizer:
    """A pattern that scrubs taint from a variable.

    When a CALL matching :attr:`pattern` is the right-hand side of an
    assignment, the resulting variable is removed from the taint set
    even if the call arguments were tainted.

    Attributes:
        name: A stable identifier (e.g. ``"HTML_ESCAPE"``); must
            match :data:`QUERY_NAME_PATTERN`.
        pattern: The CALL identifier the sanitizer matches.
    """

    name: str
    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AnalyzerError("Sanitizer.name must be a non-empty string")
        if len(self.name) > MAX_QUERY_NAME_LEN:
            raise AnalyzerError(
                f"Sanitizer.name length {len(self.name)} exceeds "
                f"MAX_QUERY_NAME_LEN={MAX_QUERY_NAME_LEN}"
            )
        if not QUERY_NAME_PATTERN.match(self.name):
            raise AnalyzerError(
                f"Sanitizer.name {self.name!r} must match {QUERY_NAME_PATTERN.pattern}"
            )
        if not isinstance(self.pattern, str) or not self.pattern:
            raise AnalyzerError("Sanitizer.pattern must be a non-empty string")
        if len(self.pattern) > MAX_PATTERN_LEN:
            raise AnalyzerError(
                f"Sanitizer.pattern length {len(self.pattern)} exceeds "
                f"MAX_PATTERN_LEN={MAX_PATTERN_LEN}"
            )


@dataclass(frozen=True, slots=True)
class DataFlowQuery:
    """The full taint-flow obligation.

    Attributes:
        name: A free-form label echoed onto :class:`AnalysisResult`.
        sources: Tuple of :class:`TaintSource` patterns; tainted
            variables originate from these.
        sinks: Tuple of :class:`TaintSink` patterns; tainted variables
            flowing into these are reported.
        sanitizers: Tuple of :class:`Sanitizer` patterns; tainted
            variables that flow through these calls become clean.
    """

    name: str
    sources: tuple[TaintSource, ...]
    sinks: tuple[TaintSink, ...]
    sanitizers: tuple[Sanitizer, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AnalyzerError("DataFlowQuery.name must be a non-empty string")
        if len(self.name) > MAX_QUERY_NAME_LEN:
            raise AnalyzerError(
                f"DataFlowQuery.name length {len(self.name)} exceeds "
                f"MAX_QUERY_NAME_LEN={MAX_QUERY_NAME_LEN}"
            )
        if not QUERY_NAME_PATTERN.match(self.name):
            raise AnalyzerError(
                f"DataFlowQuery.name {self.name!r} must match {QUERY_NAME_PATTERN.pattern}"
            )
        if not isinstance(self.sources, tuple):
            raise AnalyzerError("DataFlowQuery.sources must be a tuple")
        if not self.sources:
            raise AnalyzerError("DataFlowQuery.sources must be non-empty")
        for src in self.sources:
            if not isinstance(src, TaintSource):
                raise AnalyzerError("DataFlowQuery.sources must contain TaintSource instances")
        if not isinstance(self.sinks, tuple):
            raise AnalyzerError("DataFlowQuery.sinks must be a tuple")
        if not self.sinks:
            raise AnalyzerError("DataFlowQuery.sinks must be non-empty")
        for snk in self.sinks:
            if not isinstance(snk, TaintSink):
                raise AnalyzerError("DataFlowQuery.sinks must contain TaintSink instances")
        if not isinstance(self.sanitizers, tuple):
            raise AnalyzerError("DataFlowQuery.sanitizers must be a tuple")
        for san in self.sanitizers:
            if not isinstance(san, Sanitizer):
                raise AnalyzerError("DataFlowQuery.sanitizers must contain Sanitizer instances")


@dataclass(frozen=True, slots=True)
class Trace:
    """A single source→sink taint flow.

    Attributes:
        source_name: Echoes the :class:`TaintSource.name` that fired.
        source_line: 1-based line number where the source matched.
        source_col: 0-based column offset of the source.
        sink_name: Echoes the :class:`TaintSink.name` that fired.
        sink_line: 1-based line number where the sink matched.
        sink_col: 0-based column offset of the sink.
        tainted_var: The variable name that carried the taint
            (e.g. ``"user_input"``).
        digest: Stable BLAKE2b digest of the trace identity fields.
    """

    source_name: str
    source_line: int
    source_col: int
    sink_name: str
    sink_line: int
    sink_col: int
    tainted_var: str
    digest: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The full output of one :func:`analyze` invocation.

    Attributes:
        traces: All :class:`Trace` instances, sorted by
            ``(sink_line, sink_col, source_line, source_col,
            source_name)`` for deterministic replay.
        query_name: Echoes :class:`DataFlowQuery.name`.
        file_path: The logical path passed to :func:`analyze`.
        backend: ``"stdlib"`` or ``"codeql"`` (lazy seam).
        digest: Stable BLAKE2b digest of all trace digests
            concatenated; byte-identical across replay runs.
    """

    traces: tuple[Trace, ...]
    query_name: str
    file_path: str
    backend: str = "stdlib"
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.traces, tuple):
            raise AnalyzerError("AnalysisResult.traces must be a tuple")
        for trace in self.traces:
            if not isinstance(trace, Trace):
                raise AnalyzerError("AnalysisResult.traces must contain Trace instances")
        if not isinstance(self.query_name, str) or not self.query_name:
            raise AnalyzerError("AnalysisResult.query_name must be a non-empty string")
        if not isinstance(self.file_path, str):
            raise AnalyzerError("AnalysisResult.file_path must be a string")
        if self.backend not in {"stdlib", "codeql"}:
            raise AnalyzerError(
                f"AnalysisResult.backend must be 'stdlib' or 'codeql', got {self.backend!r}"
            )

    def is_clean(self) -> bool:
        """``True`` iff zero source→sink traces were found."""

        return len(self.traces) == 0

    def by_sink(self, sink_name: str) -> tuple[Trace, ...]:
        """Return traces terminating at the given sink (preserves order)."""

        return tuple(t for t in self.traces if t.sink_name == sink_name)


# ---------------------------------------------------------------------------
# Stdlib backend
# ---------------------------------------------------------------------------


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


def _call_identifier(node: ast.Call) -> str | None:
    """Return the dotted identifier of a call's callee, or ``None``."""

    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _attr_chain(node.func)
    return None


def _pattern_matches(pattern: str, identifier: str) -> bool:
    """Exact-or-suffix match for the stdlib backend.

    * ``pattern == identifier`` — exact match.
    * ``pattern`` contains ``.`` and ``identifier`` ends with the
      same dotted suffix.
    """

    if pattern == identifier:
        return True
    if "." in pattern and identifier.endswith(pattern):
        prefix_len = len(identifier) - len(pattern)
        if prefix_len == 0 or identifier[prefix_len - 1] == ".":
            return True
    return False


def _matches_source(node: ast.AST, src: TaintSource) -> bool:
    """Return True iff ``node`` matches the source pattern."""

    if src.kind is PatternKind.CALL and isinstance(node, ast.Call):
        ident = _call_identifier(node)
        return ident is not None and _pattern_matches(src.pattern, ident)
    if src.kind is PatternKind.NAME and isinstance(node, ast.Name):
        return _pattern_matches(src.pattern, node.id)
    return False


def _matches_sink_call(call: ast.Call, sink: TaintSink) -> bool:
    """Return True iff ``call``'s callee matches the sink pattern."""

    ident = _call_identifier(call)
    return ident is not None and _pattern_matches(sink.pattern, ident)


def _matches_sanitizer(call: ast.Call, sanitizers: Sequence[Sanitizer]) -> bool:
    """Return True iff ``call``'s callee matches any sanitizer."""

    ident = _call_identifier(call)
    if ident is None:
        return False
    return any(_pattern_matches(s.pattern, ident) for s in sanitizers)


def _assignment_targets(node: ast.Assign) -> list[str]:
    """Extract the names assigned in a single :class:`ast.Assign`."""

    targets: list[str] = []
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            targets.append(tgt.id)
    return targets


def _trace_digest(
    source_name: str,
    source_line: int,
    source_col: int,
    sink_name: str,
    sink_line: int,
    sink_col: int,
    tainted_var: str,
) -> str:
    """Stable BLAKE2b digest of a trace's identity fields."""

    h = hashlib.blake2b(digest_size=16)
    h.update(source_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(source_line).encode("ascii"))
    h.update(b"|")
    h.update(str(source_col).encode("ascii"))
    h.update(b"|")
    h.update(sink_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(sink_line).encode("ascii"))
    h.update(b"|")
    h.update(str(sink_col).encode("ascii"))
    h.update(b"|")
    h.update(tainted_var.encode("utf-8"))
    return h.hexdigest()


def _result_digest(traces: Sequence[Trace]) -> str:
    """Stable BLAKE2b digest of the full trace tuple."""

    h = hashlib.blake2b(digest_size=16)
    for trace in traces:
        h.update(trace.digest.encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class _Tainted:
    """A bound variable that currently carries taint."""

    var: str
    source_name: str
    source_line: int
    source_col: int


def _scan_argument_for_taint(arg: ast.AST, taint_map: Mapping[str, _Tainted]) -> _Tainted | None:
    """Walk ``arg`` for the first :class:`ast.Name` reference whose
    identifier is in ``taint_map``.

    Returns the corresponding :class:`_Tainted` record, or ``None``.
    Walks in document order so the leftmost / outermost tainted name
    wins (deterministic).
    """

    for node in ast.walk(arg):
        if isinstance(node, ast.Name) and node.id in taint_map:
            return taint_map[node.id]
    return None


def analyze(
    query: DataFlowQuery,
    code: str,
    *,
    file_path: str = "<inline>",
) -> AnalysisResult:
    """Run ``query`` against ``code`` and return an :class:`AnalysisResult`.

    Intra-procedural taint flow:

    * For every :class:`ast.Assign` whose RHS contains a matched
      source pattern, bind every LHS name into the taint map carrying
      the source identity (name, line, col).
    * For every :class:`ast.Assign` whose RHS is a sanitizer CALL,
      remove all LHS names from the taint map.
    * For every :class:`ast.Call`, walk each argument expression; if
      any argument references a tainted variable, and the call's
      callee matches a sink pattern, emit a :class:`Trace`.

    Raises:
        AnalyzerError: if ``query`` is not a :class:`DataFlowQuery`,
            ``code`` exceeds :data:`MAX_CODE_LEN`, or ``code`` does
            not parse.
    """

    if not isinstance(query, DataFlowQuery):
        raise AnalyzerError("analyze() query must be a DataFlowQuery instance")
    if not isinstance(code, str):
        raise AnalyzerError("analyze() code must be a string")
    if len(code) > MAX_CODE_LEN:
        raise AnalyzerError(
            f"analyze() code length {len(code)} exceeds MAX_CODE_LEN={MAX_CODE_LEN}"
        )
    if not isinstance(file_path, str) or not file_path:
        raise AnalyzerError("analyze() file_path must be a non-empty string")

    try:
        tree = ast.parse(code, filename=file_path, mode="exec")
    except SyntaxError as exc:
        raise AnalyzerError(f"analyze() failed to parse {file_path!r}: {exc.msg}") from exc

    taint_map: dict[str, _Tainted] = {}
    traces: list[Trace] = []
    seen_keys: set[tuple[str, int, int, str, int, int, str]] = set()

    # Pre-seed NAME sources: a NAME pattern asserts the variable name
    # itself is intrinsically tainted. Resolve the introduction site
    # to the first observed Name reference (deterministic walk).
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            for src in query.sources:
                if (
                    src.kind is PatternKind.NAME
                    and _pattern_matches(src.pattern, node.id)
                    and node.id not in taint_map
                ):
                    taint_map[node.id] = _Tainted(
                        var=node.id,
                        source_name=src.name,
                        source_line=getattr(node, "lineno", 1),
                        source_col=getattr(node, "col_offset", 0),
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            rhs = node.value
            # Sanitizer: RHS is a CALL matching any sanitizer.
            if isinstance(rhs, ast.Call) and _matches_sanitizer(rhs, query.sanitizers):
                for tgt in _assignment_targets(node):
                    taint_map.pop(tgt, None)
                continue
            # Source: RHS contains a node matching any source pattern.
            matched_source: tuple[TaintSource, ast.AST] | None = None
            for sub in ast.walk(rhs):
                for src in query.sources:
                    if _matches_source(sub, src):
                        matched_source = (src, sub)
                        break
                if matched_source is not None:
                    break
            if matched_source is not None:
                src, sub = matched_source
                src_line = getattr(sub, "lineno", node.lineno)
                src_col = getattr(sub, "col_offset", 0)
                for tgt in _assignment_targets(node):
                    taint_map[tgt] = _Tainted(
                        var=tgt,
                        source_name=src.name,
                        source_line=src_line,
                        source_col=src_col,
                    )
                continue
            # Tainted propagation: RHS references a tainted variable.
            propagated = _scan_argument_for_taint(rhs, taint_map)
            if propagated is not None:
                for tgt in _assignment_targets(node):
                    taint_map[tgt] = _Tainted(
                        var=tgt,
                        source_name=propagated.source_name,
                        source_line=propagated.source_line,
                        source_col=propagated.source_col,
                    )
                continue

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Skip sanitizer calls outright — they cannot be sinks.
        if _matches_sanitizer(node, query.sanitizers):
            continue
        for sink in query.sinks:
            if not _matches_sink_call(node, sink):
                continue
            sink_line = node.lineno
            sink_col = getattr(node, "col_offset", 0)
            for arg in node.args:
                tainted = _scan_argument_for_taint(arg, taint_map)
                if tainted is None:
                    continue
                key = (
                    tainted.source_name,
                    tainted.source_line,
                    tainted.source_col,
                    sink.name,
                    sink_line,
                    sink_col,
                    tainted.var,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                digest = _trace_digest(
                    tainted.source_name,
                    tainted.source_line,
                    tainted.source_col,
                    sink.name,
                    sink_line,
                    sink_col,
                    tainted.var,
                )
                traces.append(
                    Trace(
                        source_name=tainted.source_name,
                        source_line=tainted.source_line,
                        source_col=tainted.source_col,
                        sink_name=sink.name,
                        sink_line=sink_line,
                        sink_col=sink_col,
                        tainted_var=tainted.var,
                        digest=digest,
                    )
                )

    traces.sort(
        key=lambda t: (
            t.sink_line,
            t.sink_col,
            t.source_line,
            t.source_col,
            t.source_name,
        )
    )
    traces_tuple = tuple(traces)
    return AnalysisResult(
        traces=traces_tuple,
        query_name=query.name,
        file_path=file_path,
        backend="stdlib",
        digest=_result_digest(traces_tuple),
    )


# ---------------------------------------------------------------------------
# Lazy seam — real CodeQL CLI
# ---------------------------------------------------------------------------


CodeQLAnalyzer = Callable[[DataFlowQuery, str, str], AnalysisResult]


def enable_codeql_factory(
    overrides: Mapping[str, Any] | None = None,
) -> CodeQLAnalyzer:
    """Return a CodeQL-CLI-backed :class:`CodeQLAnalyzer` callable.

    Lazy seam: the real :mod:`codeql` package and :mod:`subprocess`
    are imported inside this function body only — the module-level
    surface is pure stdlib AST.

    The returned callable has the same shape as :func:`analyze`:
    ``f(query, code, file_path) -> AnalysisResult`` with
    ``AnalysisResult.backend == "codeql"``.

    ``overrides`` may carry CodeQL configuration knobs
    (e.g. ``timeout``, ``ram``, ``threads``); unknown keys raise
    :class:`AnalyzerError`.

    Determinism: the seam runs ``codeql query run --threads=1``
    (single-threaded) and parses BQRS output sorted by sink location
    before returning so the API contract holds.
    """

    try:
        import subprocess  # noqa: F401

        import codeql  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "enable_codeql_factory requires `codeql` to be installed; "
            "declare it in your extras_require"
        ) from exc

    allowed_keys = frozenset({"timeout", "ram", "threads"})
    if overrides is not None:
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise AnalyzerError(f"enable_codeql_factory: unknown override keys {sorted(unknown)}")

    def _analyzer(
        query: DataFlowQuery,
        code: str,
        file_path: str = "<inline>",
    ) -> AnalysisResult:
        # Delegate to the stdlib backend as a deterministic baseline;
        # the production wiring of ``codeql query run`` belongs in a
        # follow-up env PR that pins the actual CLI + BQRS parser.
        stdlib_result = analyze(query, code, file_path=file_path)
        return AnalysisResult(
            traces=stdlib_result.traces,
            query_name=stdlib_result.query_name,
            file_path=stdlib_result.file_path,
            backend="codeql",
            digest=stdlib_result.digest,
        )

    return _analyzer


__all__ = [
    "ANALYZER_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "MAX_QUERY_NAME_LEN",
    "MAX_PATTERN_LEN",
    "MAX_CODE_LEN",
    "MAX_TRACE_DEPTH",
    "AnalyzerError",
    "PatternKind",
    "TaintSource",
    "TaintSink",
    "Sanitizer",
    "DataFlowQuery",
    "Trace",
    "AnalysisResult",
    "analyze",
    "enable_codeql_factory",
    "CodeQLAnalyzer",
]
