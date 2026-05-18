# ADAPTED FROM: tree-sitter/tree-sitter + tree-sitter/tree-sitter-python
#   tree_sitter/binding.py — ``Parser``, ``Language``, ``Node.walk()``;
#   tree_sitter_python — Python grammar definition.
#
# License: MIT.
"""Static analysis stage — deterministic finding aggregation + AST audit.

Caller passes a list of findings produced by an external tool (e.g.
ruff, mypy, authority_lint). The stage classifies the patch by the
worst severity and emits a stage verdict.

C-26 enhancement (OFFLINE_ONLY):

* :class:`SemanticASTDiff` — pure stdlib ``ast`` walker that computes
  added / removed / changed top-level functions + classes between two
  source-file revisions. Mirrors the tree-sitter ``Node.walk()`` cursor
  pattern but stays in stdlib for replay determinism.
* :class:`ImportGraphExtractor` — collects top-level + nested ``import``
  / ``from … import`` statements; the ``forbidden`` predicate flags
  imports that violate authority tiers (e.g. a hot-path module
  importing ``torch`` / ``numpy`` / ``polars``).
* :class:`CrossFunctionCallAnalyzer` — extracts call sites between
  top-level functions; the ``runtime_tier_violation`` predicate flags
  calls that cross a tier boundary (e.g. a function tagged
  ``RUNTIME_SAFE`` calling a function tagged ``OFFLINE_ONLY``).
* :class:`PatchSafetyAnalyzer` — combines the three above + flags
  edits to files inside ``GOVERNANCE_BOUNDARY_PREFIXES``.
* :func:`tree_sitter_parser_factory` — lazy seam binding the live
  ``tree_sitter`` + ``tree_sitter_python`` packages. Used only for the
  "live AST" path; the in-memory analyzers fall back to the stdlib
  ``ast`` module so CI can run without the dep.

Authority constraints:

* OFFLINE_ONLY — never called from the hot path.
* No clock, no PRNG, no IO except the source-file read inside the
  ``analyze_files`` convenience entrypoint.
* Pure stdlib — no tree-sitter top-level import.
* ``NEW_PIP_DEPENDENCIES = ("tree-sitter", "tree-sitter-python")``.

INV-15 (replay determinism):

* Diff / graph entries are emitted in deterministic order (sorted
  alphabetically by name, then by location).
* All public ``*_to_dict`` projections sort their keys.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("tree-sitter", "tree-sitter-python")
TREE_SITTER_ADAPTER_VERSION: str = "1"
GOVERNANCE_BOUNDARY_PREFIXES: tuple[str, ...] = (
    "core/contracts/",
    "governance_engine/",
    "system_engine/",
)


# ---------------------------------------------------------------------------
# Existing FindingSeverity + StaticAnalysisFinding + Stage (preserved verbatim)
# ---------------------------------------------------------------------------
class FindingSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class StaticAnalysisFinding:
    rule: str
    severity: FindingSeverity
    location: str
    detail: str = ""


class StaticAnalysisStage:
    """GOV-G18-S2."""

    name: str = "static_analysis"
    spec_id: str = "GOV-G18-S2"

    __slots__ = ("_max_severity",)

    def __init__(
        self,
        *,
        max_severity: FindingSeverity = FindingSeverity.WARN,
    ) -> None:
        self._max_severity = max_severity

    def evaluate(
        self,
        *,
        ts_ns: int,
        findings: Sequence[StaticAnalysisFinding],
    ) -> StageVerdict:
        rank = {
            FindingSeverity.INFO: 0,
            FindingSeverity.WARN: 1,
            FindingSeverity.ERROR: 2,
        }
        worst = max(
            (rank[f.severity] for f in findings),
            default=-1,
        )
        passed = worst <= rank[self._max_severity]
        return StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.STATIC_ANALYSIS,
            passed=passed,
            detail=(
                f"{len(findings)} findings, worst="
                + ([k.value for k, v in rank.items() if v == worst][0] if worst >= 0 else "NONE")
            ),
            meta={
                "findings": str(len(findings)),
                "max_severity": self._max_severity.value,
            },
        )


# ---------------------------------------------------------------------------
# C-26 enhancement — semantic AST diff
# ---------------------------------------------------------------------------
class ASTDiffKind(StrEnum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    CHANGED = "CHANGED"


@dataclass(frozen=True, slots=True)
class ASTDiffEntry:
    kind: ASTDiffKind
    symbol: str
    symbol_kind: str  # "function" | "class"
    detail: str = ""


def _top_level_symbols(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Return ``{name: (kind, source-segment)}`` for top-level functions/classes."""
    out: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ("function", ast.unparse(node))
        elif isinstance(node, ast.ClassDef):
            out[node.name] = ("class", ast.unparse(node))
    return out


class SemanticASTDiff:
    """Compute added / removed / changed top-level functions + classes.

    The diff is computed against the **module body** only (top-level
    `def` / `async def` / `class`). Method bodies are included in the
    class's source segment so renames / signature changes inside a
    class still register as ``CHANGED`` at the class level.
    """

    __slots__ = ()

    def diff(
        self,
        *,
        before: str,
        after: str,
    ) -> tuple[ASTDiffEntry, ...]:
        if not isinstance(before, str):
            raise TypeError("before must be a str")
        if not isinstance(after, str):
            raise TypeError("after must be a str")
        try:
            before_tree = ast.parse(before)
        except SyntaxError as exc:
            raise ValueError(f"before source has syntax error: {exc!s}") from exc
        try:
            after_tree = ast.parse(after)
        except SyntaxError as exc:
            raise ValueError(f"after source has syntax error: {exc!s}") from exc
        before_syms = _top_level_symbols(before_tree)
        after_syms = _top_level_symbols(after_tree)
        entries: list[ASTDiffEntry] = []
        for name in sorted(set(before_syms) | set(after_syms)):
            in_before = name in before_syms
            in_after = name in after_syms
            if in_before and not in_after:
                kind_str, _ = before_syms[name]
                entries.append(
                    ASTDiffEntry(
                        kind=ASTDiffKind.REMOVED,
                        symbol=name,
                        symbol_kind=kind_str,
                    )
                )
            elif in_after and not in_before:
                kind_str, _ = after_syms[name]
                entries.append(
                    ASTDiffEntry(
                        kind=ASTDiffKind.ADDED,
                        symbol=name,
                        symbol_kind=kind_str,
                    )
                )
            else:
                before_kind, before_src = before_syms[name]
                after_kind, after_src = after_syms[name]
                if before_src != after_src or before_kind != after_kind:
                    entries.append(
                        ASTDiffEntry(
                            kind=ASTDiffKind.CHANGED,
                            symbol=name,
                            symbol_kind=after_kind,
                        )
                    )
        return tuple(entries)


# ---------------------------------------------------------------------------
# Import graph extractor
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImportEntry:
    module: str
    name: str  # "" for bare ``import x``
    location: str  # "lineno:col"
    is_top_level: bool


def _walk_imports(tree: ast.Module) -> Iterator[tuple[ast.AST, bool]]:
    """Yield every Import/ImportFrom node + a flag for top-level."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, True
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node in tree.body:
                continue
            yield node, False


class ImportGraphExtractor:
    """Extract every ``import`` / ``from … import`` statement.

    The ``forbidden`` predicate flags imports that match any module
    prefix in the supplied ``forbidden_modules`` set.
    """

    __slots__ = ("_forbidden",)

    def __init__(
        self,
        *,
        forbidden_modules: Iterable[str] = (),
    ) -> None:
        self._forbidden = tuple(sorted({m for m in forbidden_modules}))

    @property
    def forbidden_modules(self) -> tuple[str, ...]:
        return self._forbidden

    def extract(self, *, source: str) -> tuple[ImportEntry, ...]:
        if not isinstance(source, str):
            raise TypeError("source must be a str")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(f"source has syntax error: {exc!s}") from exc
        entries: list[ImportEntry] = []
        for node, is_top in _walk_imports(tree):
            loc = f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    entries.append(
                        ImportEntry(
                            module=alias.name,
                            name="",
                            location=loc,
                            is_top_level=is_top,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    entries.append(
                        ImportEntry(
                            module=module,
                            name=alias.name,
                            location=loc,
                            is_top_level=is_top,
                        )
                    )
        entries.sort(key=lambda e: (e.module, e.name, e.location, e.is_top_level))
        return tuple(entries)

    def forbidden_hits(self, *, source: str) -> tuple[ImportEntry, ...]:
        return tuple(
            e
            for e in self.extract(source=source)
            if any(self._is_match(e.module, m) for m in self._forbidden)
        )

    @staticmethod
    def _is_match(module: str, prefix: str) -> bool:
        if not module:
            return False
        return module == prefix or module.startswith(prefix + ".")


# ---------------------------------------------------------------------------
# Cross-function call analyzer
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CallEntry:
    caller: str
    callee: str
    location: str


class CrossFunctionCallAnalyzer:
    """Extract call sites between top-level functions.

    The ``runtime_tier_violation`` predicate flags calls where the
    caller tier is "more restricted" than the callee tier. Tier rank:

    * ``RUNTIME_SAFE`` = 0 (hot path, no IO, no clock)
    * ``OFFLINE_ONLY`` = 1 (CI / sandbox)
    * ``RESEARCH_SOURCE`` = 2 (notebook / analysis)

    A RUNTIME_SAFE function calling an OFFLINE_ONLY function is a tier
    violation; the inverse is not.
    """

    __slots__ = ()

    def extract(self, *, source: str) -> tuple[CallEntry, ...]:
        if not isinstance(source, str):
            raise TypeError("source must be a str")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(f"source has syntax error: {exc!s}") from exc
        entries: list[CallEntry] = []
        top_funcs: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_funcs[node.name] = node
        for caller, fn_node in top_funcs.items():
            for child in ast.walk(fn_node):
                if isinstance(child, ast.Call):
                    callee = self._call_name(child.func)
                    if callee is None:
                        continue
                    loc = f"{getattr(child, 'lineno', 0)}:{getattr(child, 'col_offset', 0)}"
                    entries.append(
                        CallEntry(
                            caller=caller,
                            callee=callee,
                            location=loc,
                        )
                    )
        entries.sort(key=lambda e: (e.caller, e.callee, e.location))
        return tuple(entries)

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = [node.attr]
            cur: ast.AST = node.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None

    def runtime_tier_violations(
        self,
        *,
        source: str,
        tier_map: Mapping[str, str],
    ) -> tuple[CallEntry, ...]:
        if not isinstance(tier_map, Mapping):
            raise TypeError("tier_map must be a Mapping")
        rank = {
            "RUNTIME_SAFE": 0,
            "OFFLINE_ONLY": 1,
            "RESEARCH_SOURCE": 2,
        }
        out: list[CallEntry] = []
        for entry in self.extract(source=source):
            caller_tier = tier_map.get(entry.caller)
            callee_tier = tier_map.get(entry.callee)
            if caller_tier is None or callee_tier is None:
                continue
            if caller_tier not in rank or callee_tier not in rank:
                continue
            if rank[caller_tier] < rank[callee_tier]:
                out.append(entry)
        return tuple(out)


# ---------------------------------------------------------------------------
# Patch safety analyzer
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PatchSafetyReport:
    boundary_touched: tuple[str, ...]
    ast_diff: tuple[ASTDiffEntry, ...]
    forbidden_imports: tuple[ImportEntry, ...]
    tier_violations: tuple[CallEntry, ...]

    @property
    def is_safe(self) -> bool:
        return not (self.boundary_touched or self.forbidden_imports or self.tier_violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_touched": list(self.boundary_touched),
            "ast_diff": [
                {
                    "kind": e.kind.value,
                    "symbol": e.symbol,
                    "symbol_kind": e.symbol_kind,
                    "detail": e.detail,
                }
                for e in self.ast_diff
            ],
            "forbidden_imports": [
                {
                    "module": e.module,
                    "name": e.name,
                    "location": e.location,
                    "is_top_level": e.is_top_level,
                }
                for e in self.forbidden_imports
            ],
            "tier_violations": [
                {
                    "caller": e.caller,
                    "callee": e.callee,
                    "location": e.location,
                }
                for e in self.tier_violations
            ],
        }


class PatchSafetyAnalyzer:
    """Combine AST diff + forbidden imports + tier violations + boundary check.

    Authority: OFFLINE_ONLY — CI / sandbox use only.
    """

    __slots__ = (
        "_ast_diff",
        "_boundary_prefixes",
        "_call_analyzer",
        "_import_extractor",
        "_tier_map",
    )

    def __init__(
        self,
        *,
        forbidden_modules: Iterable[str] = (),
        tier_map: Mapping[str, str] | None = None,
        boundary_prefixes: Iterable[str] | None = None,
    ) -> None:
        self._ast_diff = SemanticASTDiff()
        self._import_extractor = ImportGraphExtractor(
            forbidden_modules=forbidden_modules,
        )
        self._call_analyzer = CrossFunctionCallAnalyzer()
        self._tier_map: dict[str, str] = dict(tier_map) if tier_map is not None else {}
        if boundary_prefixes is None:
            self._boundary_prefixes = GOVERNANCE_BOUNDARY_PREFIXES
        else:
            self._boundary_prefixes = tuple(sorted(boundary_prefixes))

    @property
    def boundary_prefixes(self) -> tuple[str, ...]:
        return self._boundary_prefixes

    def analyze(
        self,
        *,
        path: str,
        before: str,
        after: str,
    ) -> PatchSafetyReport:
        if not isinstance(path, str):
            raise TypeError("path must be a str")
        boundary_touched = tuple(p for p in self._boundary_prefixes if path.startswith(p))
        ast_diff = self._ast_diff.diff(before=before, after=after)
        forbidden_imports = self._import_extractor.forbidden_hits(source=after)
        tier_violations = self._call_analyzer.runtime_tier_violations(
            source=after,
            tier_map=self._tier_map,
        )
        return PatchSafetyReport(
            boundary_touched=boundary_touched,
            ast_diff=ast_diff,
            forbidden_imports=forbidden_imports,
            tier_violations=tier_violations,
        )


# ---------------------------------------------------------------------------
# tree-sitter binding (lazy)
# ---------------------------------------------------------------------------
def tree_sitter_parser_factory() -> Any:
    """Lazy-bind the ``tree_sitter`` + ``tree_sitter_python`` packages.

    Both imports are confined to this function body.

    Returns a live ``tree_sitter.Parser`` configured for Python. The
    in-memory analyzers above do not depend on this; the live parser
    is provided for callers that want raw tree-sitter ``Node`` cursor
    access (e.g. for grammar-aware refactors).
    """
    try:
        import tree_sitter  # noqa: PLC0415
        import tree_sitter_python  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise RuntimeError(
            "tree-sitter / tree-sitter-python is not installed; see NEW_PIP_DEPENDENCIES"
        ) from exc
    language = tree_sitter.Language(tree_sitter_python.language())
    parser = tree_sitter.Parser(language)
    return parser


__all__ = [
    "ASTDiffEntry",
    "ASTDiffKind",
    "CallEntry",
    "CrossFunctionCallAnalyzer",
    "FindingSeverity",
    "GOVERNANCE_BOUNDARY_PREFIXES",
    "ImportEntry",
    "ImportGraphExtractor",
    "NEW_PIP_DEPENDENCIES",
    "PatchSafetyAnalyzer",
    "PatchSafetyReport",
    "SemanticASTDiff",
    "StaticAnalysisFinding",
    "StaticAnalysisStage",
    "TREE_SITTER_ADAPTER_VERSION",
    "tree_sitter_parser_factory",
]
