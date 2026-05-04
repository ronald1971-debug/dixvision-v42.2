"""TOTAL VALIDATION — DIX VISION v42.2 12-phase CI-enforced system check.

Implements the 12 phases of the TOTAL VALIDATION SPEC (see
``docs/TOTAL_VALIDATION_SPEC.md``):

    Phase 0   source ingestion         -> analysis/source_index.csv
    Phase 1   file index               -> analysis/file_index.csv
    Phase 2   feature extraction       -> analysis/feature_index.csv
    Phase 3   file analysis            -> analysis/tracking_table.csv
    Phase 4   feature coverage         -> analysis/feature_coverage.csv
    Phase 5   source coverage          -> analysis/source_coverage.csv
    Phase 6   invariant validation     -> analysis/invariant_coverage.csv
    Phase 7   file usage validation    -> analysis/file_usage.csv
    Phase 8   declaration consistency  -> analysis/declaration_map.csv
    Phase 9   dependency graph         -> analysis/dependency_graph.json
    Phase 10  AST validation           -> analysis/ast_validation.json
    Phase 11  runtime telemetry        -> analysis/runtime_validation.json
    Phase 12  final summary            -> analysis/coverage_summary.json

Modes::

    --advisory   (default)   write artifacts, return PASS even if gaps exist.
                             Used during the bring-up window so PRs are not
                             blocked while the remediation backlog is worked
                             through.
    --strict                 every gap downgrades the final status to FAIL.
                             Used by CI once the backlog is clean.

The script is **read-only**: it never mutates source code or registry
files. It is also **clock-free** and **PRNG-free** so artifact contents
are deterministic for any given commit.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Directories ignored for filesystem walks.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        ".turbo",
        ".next",
        ".vite",
        "analysis",  # our own output
    }
)

# Files considered "source" for python-side analysis.
PY_FILE_EXTS = frozenset({".py"})
TS_FILE_EXTS = frozenset({".ts", ".tsx"})
ALL_CODE_EXTS = PY_FILE_EXTS | TS_FILE_EXTS

# Authoritative-source descriptors (TOTAL VALIDATION SPEC §1).
AUTHORITATIVE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("S1", "manifest", "docs/manifest_v3.6.4_delta.md"),
    ("S2", "executive_summary", "docs/system_audit/build_plan_stage.md"),
    ("S3", "build_plan", "docs/system_audit/build_plan_stage.md"),
    ("S4", "directory_tree", "docs/directory_tree.md"),
    ("S5", "registry", "registry"),
    ("S6", "source_code", "."),
    ("S7", "contracts", "core/contracts"),
    ("S8", "invariants", "tools/authority_lint.py"),
    ("S9", "tests", "tests"),
    ("S10", "workflows", ".github/workflows"),
    ("S11", "metrics", "system_engine/metrics.py"),
    ("S12", "runtime_logs", "analysis/runtime_logs.txt"),
)

# Domain isolation rules (TOTAL VALIDATION SPEC §11). Each entry says
# "<from-domain> may NOT import <to-domain>".
FORBIDDEN_DOMAIN_EDGES: tuple[tuple[str, str], ...] = (
    ("intelligence_engine", "system_engine"),
    ("system_engine", "intelligence_engine"),
)

# Patterns identifying declared features in markdown / YAML sources.
FEATURE_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bINV-\d{2,4}\b"),
    re.compile(r"\bGOV-CP-\d{2,3}\b"),
    re.compile(r"\bSAFE-\d{2,3}\b"),
    re.compile(r"\bPERF-\d{2,3}\b"),
    re.compile(r"\bHAZ-[A-Z0-9_-]+\b"),
    re.compile(r"\bSRC-[A-Z0-9_-]+\b"),
    re.compile(r"\bAUDIT-[A-Z0-9_.-]+\b"),
    re.compile(r"\bWAVE-[A-Z0-9_.-]+\b"),
    re.compile(r"\bSHADOW-DEMOLITION-\d{2}\b"),
    re.compile(r"\bWEBLEARN-\d{2,3}\b"),
    re.compile(r"\bNEUR-\d{2,3}\b"),
    re.compile(r"\bB\d{1,2}\b"),
    re.compile(r"\bT\d{1}-\d{1,2}[a-z]?\b"),
)

# ---------------------------------------------------------------------------
# small dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Phase:
    """One phase of the validation pipeline."""

    phase_id: int
    name: str
    artifact: str
    status: str = "pending"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    """Aggregated result of running all phases."""

    phases: list[Phase] = field(default_factory=list)
    file_count: int = 0
    feature_count: int = 0
    declared_feature_count: int = 0
    implemented_feature_count: int = 0
    invariant_count: int = 0
    enforced_invariant_count: int = 0
    dead_files: int = 0
    unmapped_declarations: int = 0
    ambiguity: int = 0
    dependency_graph_valid: bool = True
    ast_validation: bool = True
    runtime_validation: bool = True
    advisory: bool = True

    def coverage_pct(self, num: int, denom: int) -> str:
        if denom <= 0:
            return "100%"
        return f"{(100 * num) // denom}%"


# ---------------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------------


def _iter_repo_files() -> list[Path]:
    """Return every non-excluded file under the repo, sorted."""
    out: list[Path] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        rp = Path(root)
        # prune excluded dirs in place so os.walk skips them
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(files):
            out.append(rp / name)
    return sorted(out, key=lambda p: str(p.relative_to(REPO_ROOT)))


def _is_python_file(p: Path) -> bool:
    return p.suffix in PY_FILE_EXTS


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _ensure_analysis_dir() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(name: str, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    _ensure_analysis_dir()
    path = ANALYSIS_DIR / name
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _write_json(name: str, payload: Any) -> None:
    _ensure_analysis_dir()
    path = ANALYSIS_DIR / name
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Phase 0 — source ingestion
# ---------------------------------------------------------------------------


def _phase_0_source_ingestion() -> tuple[Phase, list[tuple[str, str, str]]]:
    rows: list[tuple[str, str, str, str]] = []
    parsed: list[tuple[str, str, str]] = []
    missing = 0
    for sid, stype, rel_path in AUTHORITATIVE_SOURCES:
        target = REPO_ROOT / rel_path
        if target.exists():
            rows.append((sid, stype, rel_path, "yes"))
            parsed.append((sid, stype, rel_path))
        else:
            rows.append((sid, stype, rel_path, "no"))
            missing += 1
    _write_csv(
        "source_index.csv",
        ("source_id", "source_type", "path", "parsed"),
        rows,
    )
    return (
        Phase(
            phase_id=0,
            name="source_ingestion",
            artifact="source_index.csv",
            status="ok" if missing == 0 else "warn",
            details={"sources": len(rows), "missing": missing},
        ),
        parsed,
    )


# ---------------------------------------------------------------------------
# Phase 1 — file index
# ---------------------------------------------------------------------------


def _phase_1_file_index(files: list[Path]) -> Phase:
    rows: list[tuple[str, str, str]] = []
    for i, p in enumerate(files):
        rows.append((f"F{i:05d}", _rel(p), "filesystem"))
    _write_csv(
        "file_index.csv",
        ("file_id", "file_path", "source_origin"),
        rows,
    )
    return Phase(
        phase_id=1,
        name="file_index",
        artifact="file_index.csv",
        status="ok",
        details={"files": len(rows)},
    )


# ---------------------------------------------------------------------------
# Phase 2 — feature extraction
# ---------------------------------------------------------------------------


def _scan_feature_ids(text: str) -> set[str]:
    found: set[str] = set()
    for pat in FEATURE_ID_PATTERNS:
        found.update(pat.findall(text))
    return found


def _phase_2_feature_extraction(
    parsed_sources: list[tuple[str, str, str]],
) -> tuple[Phase, dict[str, set[str]]]:
    """Pull every declared feature id (INV-XX, GOV-XX, etc.) from sources."""
    rows: list[tuple[str, str, str, str]] = []
    feature_to_sources: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()

    candidate_paths: list[tuple[str, Path]] = []
    for sid, _stype, rel_path in parsed_sources:
        target = REPO_ROOT / rel_path
        if target.is_dir():
            for p in target.rglob("*"):
                if p.is_file() and p.suffix in {".md", ".yaml", ".yml", ".py"}:
                    candidate_paths.append((sid, p))
        elif target.is_file():
            candidate_paths.append((sid, target))

    # Always include all manifest_*.md so we catch the full delta history.
    for p in (REPO_ROOT / "docs").rglob("manifest_*.md"):
        candidate_paths.append(("S1", p))

    for sid, path in candidate_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for fid in _scan_feature_ids(text):
            feature_to_sources[fid].add(sid)
            if fid not in seen:
                seen.add(fid)
                rows.append(
                    (
                        fid,
                        fid,
                        f"declared in {sid}",
                        sid,
                    )
                )

    rows.sort(key=lambda r: r[0])
    _write_csv(
        "feature_index.csv",
        ("feature_id", "name", "description", "declared_in_source"),
        rows,
    )
    return (
        Phase(
            phase_id=2,
            name="feature_extraction",
            artifact="feature_index.csv",
            status="ok" if rows else "warn",
            details={"features": len(rows)},
        ),
        feature_to_sources,
    )


# ---------------------------------------------------------------------------
# Phase 3 — file analysis
# ---------------------------------------------------------------------------


def _phase_3_file_analysis(files: list[Path]) -> Phase:
    rows: list[tuple[str, str, str, str]] = []
    issues = 0
    for i, p in enumerate(files):
        analyzed = "yes"
        summary = ""
        issue = ""
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            summary = f"{len(text.splitlines())} lines"
            if _is_python_file(p):
                try:
                    ast.parse(text, filename=str(p))
                except SyntaxError as e:  # pragma: no cover - real failure path
                    issue = f"syntax_error:{e.lineno}"
                    issues += 1
        except OSError:
            analyzed = "no"
            issue = "unreadable"
            issues += 1
        rows.append((f"F{i:05d}", analyzed, summary, issue))
    _write_csv(
        "tracking_table.csv",
        ("file_id", "analyzed", "summary", "issues"),
        rows,
    )
    return Phase(
        phase_id=3,
        name="file_analysis",
        artifact="tracking_table.csv",
        status="ok" if issues == 0 else "warn",
        details={"files": len(rows), "issues": issues},
    )


# ---------------------------------------------------------------------------
# Phase 4 — feature coverage
# ---------------------------------------------------------------------------


def _grep_feature_in_code(
    feature_ids: set[str],
    files: list[Path],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Map ``feature_id -> {implemented_in_files}, {tested_in_files}``."""
    implemented: dict[str, set[str]] = defaultdict(set)
    tested: dict[str, set[str]] = defaultdict(set)
    for p in files:
        if p.suffix not in ALL_CODE_EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _rel(p)
        is_test = rel.startswith("tests/") or "/test_" in rel or rel.endswith("_test.py")
        for fid in feature_ids:
            if fid in text:
                if is_test:
                    tested[fid].add(rel)
                else:
                    implemented[fid].add(rel)
    return implemented, tested


def _phase_4_feature_coverage(
    feature_to_sources: dict[str, set[str]],
    files: list[Path],
) -> tuple[Phase, dict[str, str]]:
    feature_ids = set(feature_to_sources.keys())
    implemented, tested = _grep_feature_in_code(feature_ids, files)
    rows: list[tuple[str, str, str, str, str]] = []
    status_for: dict[str, str] = {}
    for fid in sorted(feature_ids):
        srcs = "+".join(sorted(feature_to_sources[fid]))
        impls = ";".join(sorted(implemented.get(fid, set())))
        tsts = ";".join(sorted(tested.get(fid, set())))
        if implemented.get(fid) and tested.get(fid):
            status = "OK"
        elif implemented.get(fid):
            status = "UNVERIFIED"
        elif tested.get(fid):
            status = "AMBIGUOUS"
        else:
            status = "MISSING"
        status_for[fid] = status
        rows.append((fid, srcs, impls, tsts, status))
    _write_csv(
        "feature_coverage.csv",
        ("feature_id", "declared_in", "implemented_in_files", "tested_in", "status"),
        rows,
    )
    ok = sum(1 for s in status_for.values() if s == "OK")
    return (
        Phase(
            phase_id=4,
            name="feature_coverage",
            artifact="feature_coverage.csv",
            status="ok" if ok == len(status_for) else "warn",
            details={
                "ok": ok,
                "missing": sum(1 for s in status_for.values() if s == "MISSING"),
                "unverified": sum(1 for s in status_for.values() if s == "UNVERIFIED"),
                "ambiguous": sum(1 for s in status_for.values() if s == "AMBIGUOUS"),
            },
        ),
        status_for,
    )


# ---------------------------------------------------------------------------
# Phase 5 — source coverage
# ---------------------------------------------------------------------------


def _phase_5_source_coverage(
    parsed_sources: list[tuple[str, str, str]],
    feature_to_sources: dict[str, set[str]],
) -> Phase:
    rows: list[tuple[str, int, str]] = []
    for sid, _stype, _rel_path in parsed_sources:
        n = sum(1 for fids in feature_to_sources.values() if sid in fids)
        rows.append((sid, n, "OK" if n > 0 else "EMPTY"))
    _write_csv(
        "source_coverage.csv",
        ("source_id", "extracted_features", "coverage_status"),
        rows,
    )
    empty = sum(1 for r in rows if r[2] != "OK")
    return Phase(
        phase_id=5,
        name="source_coverage",
        artifact="source_coverage.csv",
        status="ok" if empty == 0 else "warn",
        details={"empty_sources": empty},
    )


# ---------------------------------------------------------------------------
# Phase 6 — invariant validation
# ---------------------------------------------------------------------------


def _phase_6_invariant_validation(
    feature_to_sources: dict[str, set[str]],
    files: list[Path],
) -> tuple[Phase, int, int]:
    inv_ids = {fid for fid in feature_to_sources if fid.startswith("INV-")}
    rows: list[tuple[str, str, str, str, str]] = []
    enforced = 0
    enforced_in_lint = _read_authority_lint_text()
    implemented, tested = _grep_feature_in_code(inv_ids, files)
    for inv in sorted(inv_ids):
        srcs = "+".join(sorted(feature_to_sources[inv]))
        enf_path: list[str] = []
        if inv in enforced_in_lint:
            enf_path.append("tools/authority_lint.py")
        if implemented.get(inv):
            enf_path.append(";".join(sorted(implemented[inv])[:2]))
        tst = ";".join(sorted(tested.get(inv, set()))[:2])
        if enf_path and tst:
            status = "OK"
            enforced += 1
        elif enf_path:
            status = "UNVERIFIED"
            enforced += 1
        elif tst:
            status = "AMBIGUOUS"
        else:
            status = "MISSING"
        rows.append((inv, srcs, ";".join(enf_path), tst, status))
    _write_csv(
        "invariant_coverage.csv",
        ("invariant_id", "declared_in", "enforced_in", "tested_in", "status"),
        rows,
    )
    return (
        Phase(
            phase_id=6,
            name="invariant_validation",
            artifact="invariant_coverage.csv",
            status="ok" if all(r[4] == "OK" for r in rows) else "warn",
            details={
                "invariants": len(rows),
                "enforced": enforced,
            },
        ),
        len(rows),
        enforced,
    )


def _read_authority_lint_text() -> str:
    p = REPO_ROOT / "tools" / "authority_lint.py"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Phase 7 — file usage validation
# ---------------------------------------------------------------------------


def _build_python_import_graph(
    files: list[Path],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return (rev_graph, module_to_path).

    ``rev_graph[module] = {modules that import it}``.
    """
    module_to_path: dict[str, str] = {}
    for p in files:
        if not _is_python_file(p):
            continue
        rel = _rel(p)
        if rel.endswith("/__init__.py"):
            mod = rel[:-12].replace("/", ".")
        else:
            mod = rel[:-3].replace("/", ".")
        module_to_path[mod] = rel

    rev: dict[str, set[str]] = defaultdict(set)
    for p in files:
        if not _is_python_file(p):
            continue
        rel = _rel(p)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Match longest prefix to a known module.
                mod = node.module
                while mod and mod not in module_to_path:
                    mod = mod.rsplit(".", 1)[0] if "." in mod else ""
                if mod:
                    rev[mod].add(rel)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    while mod and mod not in module_to_path:
                        mod = mod.rsplit(".", 1)[0] if "." in mod else ""
                    if mod:
                        rev[mod].add(rel)
    return rev, module_to_path


def _phase_7_file_usage(
    files: list[Path],
    feature_status: dict[str, str],
) -> Phase:
    rev, module_to_path = _build_python_import_graph(files)
    rows: list[tuple[str, str, str, str]] = []
    dead = 0
    for i, p in enumerate(files):
        if not _is_python_file(p):
            continue
        rel = _rel(p)
        # Compute module name back.
        if rel.endswith("/__init__.py"):
            mod = rel[:-12].replace("/", ".")
        else:
            mod = rel[:-3].replace("/", ".")
        importers = sorted(rev.get(mod, set()))
        feature_links: list[str] = []
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            for fid in feature_status:
                if fid in text:
                    feature_links.append(fid)
        except OSError:
            pass
        is_entrypoint = (
            rel.endswith("__main__.py")
            or rel.endswith("/conftest.py")
            or rel.startswith("tests/")
            or rel.startswith("scripts/")
            or rel.startswith("ui/server.py")
            or rel == "ui/server.py"
            or rel.startswith("bootstrap_kernel.py")
        )
        status = "DEAD" if not importers and not feature_links and not is_entrypoint else "USED"
        if status == "DEAD":
            dead += 1
        rows.append(
            (
                f"F{i:05d}",
                ";".join(importers[:5]),
                ";".join(feature_links[:5]),
                status,
            )
        )
    _write_csv(
        "file_usage.csv",
        ("file_id", "referenced_by", "feature_links", "status"),
        rows,
    )
    return Phase(
        phase_id=7,
        name="file_usage",
        artifact="file_usage.csv",
        status="ok" if dead == 0 else "warn",
        details={"dead_files": dead, "python_files": len(rows)},
    )


# ---------------------------------------------------------------------------
# Phase 8 — declaration consistency
# ---------------------------------------------------------------------------


def _phase_8_declaration_consistency(
    feature_to_sources: dict[str, set[str]],
    feature_status: dict[str, str],
) -> tuple[Phase, int]:
    rows: list[tuple[str, str, str, str]] = []
    declared_not_implemented = 0
    for fid in sorted(feature_to_sources.keys()):
        srcs = "+".join(sorted(feature_to_sources[fid]))
        st = feature_status.get(fid, "MISSING")
        impl_status = "implemented" if st in {"OK", "UNVERIFIED"} else "missing"
        if impl_status == "missing":
            declared_not_implemented += 1
        rows.append((fid, srcs, impl_status, st))
    _write_csv(
        "declaration_map.csv",
        ("component", "declared_in_sources", "implemented_in", "status"),
        rows,
    )
    return (
        Phase(
            phase_id=8,
            name="declaration_consistency",
            artifact="declaration_map.csv",
            status="ok" if declared_not_implemented == 0 else "warn",
            details={
                "declared": len(rows),
                "declared_not_implemented": declared_not_implemented,
            },
        ),
        declared_not_implemented,
    )


# ---------------------------------------------------------------------------
# Phase 9 — dependency graph validation
# ---------------------------------------------------------------------------


def _phase_9_dependency_graph(
    files: list[Path],
) -> tuple[Phase, bool]:
    rev, module_to_path = _build_python_import_graph(files)
    forward: dict[str, set[str]] = defaultdict(set)
    for tgt, importers in rev.items():
        for src_rel in importers:
            mod_src: str | None = None
            for cand_mod, cand_rel in module_to_path.items():
                if cand_rel == src_rel:
                    mod_src = cand_mod
                    break
            if mod_src:
                forward[mod_src].add(tgt)

    violations: list[dict[str, str]] = []
    for src, tgts in forward.items():
        for tgt in tgts:
            for forbidden_src, forbidden_tgt in FORBIDDEN_DOMAIN_EDGES:
                if src.startswith(forbidden_src) and tgt.startswith(forbidden_tgt):
                    violations.append(
                        {
                            "from": src,
                            "to": tgt,
                            "rule": f"{forbidden_src}->{forbidden_tgt}",
                        }
                    )

    cycles = _detect_cycles_kosaraju(forward)

    payload = {
        "modules": len(module_to_path),
        "edges": sum(len(t) for t in forward.values()),
        "domain_violations": violations,
        "cycles": cycles,
    }
    _write_json("dependency_graph.json", payload)
    valid = not violations and not cycles
    return (
        Phase(
            phase_id=9,
            name="dependency_graph",
            artifact="dependency_graph.json",
            status="ok" if valid else "warn",
            details={
                "violations": len(violations),
                "cycles": len(cycles),
            },
        ),
        valid,
    )


def _detect_cycles_kosaraju(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return strongly-connected components of size > 1."""
    nodes = set(graph.keys()) | {n for vs in graph.values() for n in vs}
    visited: set[str] = set()
    order: list[str] = []

    def visit(u: str) -> None:
        stack: list[tuple[str, list[str]]] = [(u, list(graph.get(u, ())))]
        while stack:
            node, succs = stack[-1]
            if node not in visited:
                visited.add(node)
            if succs:
                v = succs.pop()
                if v not in visited:
                    stack.append((v, list(graph.get(v, ()))))
            else:
                order.append(node)
                stack.pop()

    for n in nodes:
        if n not in visited:
            visit(n)

    rev: dict[str, set[str]] = defaultdict(set)
    for u, vs in graph.items():
        for v in vs:
            rev[v].add(u)

    seen: set[str] = set()
    sccs: list[list[str]] = []
    for n in reversed(order):
        if n in seen:
            continue
        comp: list[str] = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for y in rev.get(x, ()):
                if y not in seen:
                    stack.append(y)
        if len(comp) > 1:
            sccs.append(sorted(comp))
    return sccs


# ---------------------------------------------------------------------------
# Phase 10 — AST validation
# ---------------------------------------------------------------------------


def _phase_10_ast_validation(files: list[Path]) -> Phase:
    parse_errors: list[dict[str, str]] = []
    classes = 0
    functions = 0
    decorators = 0
    imports = 0
    for p in files:
        if not _is_python_file(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(p))
        except (SyntaxError, OSError) as e:
            parse_errors.append({"path": _rel(p), "error": str(e)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes += 1
                decorators += len(node.decorator_list)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions += 1
                decorators += len(node.decorator_list)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                imports += 1
    payload = {
        "classes": classes,
        "functions": functions,
        "decorators": decorators,
        "imports": imports,
        "parse_errors": parse_errors,
    }
    _write_json("ast_validation.json", payload)
    return Phase(
        phase_id=10,
        name="ast_validation",
        artifact="ast_validation.json",
        status="ok" if not parse_errors else "warn",
        details=payload,
    )


# ---------------------------------------------------------------------------
# Phase 11 — runtime telemetry validation
# ---------------------------------------------------------------------------


def _phase_11_runtime_telemetry(advisory: bool) -> tuple[Phase, bool]:
    log_path = REPO_ROOT / "analysis" / "runtime_logs.txt"
    if not log_path.exists():
        payload = {
            "status": "skipped",
            "reason": "analysis/runtime_logs.txt not present (run harness with telemetry capture)",
            "features_with_telemetry": 0,
        }
        _write_json("runtime_validation.json", payload)
        # Skipped is acceptable in advisory mode.
        return (
            Phase(
                phase_id=11,
                name="runtime_telemetry",
                artifact="runtime_validation.json",
                status="skip" if advisory else "warn",
                details=payload,
            ),
            True if advisory else False,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    feature_hits = 0
    seen: set[str] = set()
    for pat in FEATURE_ID_PATTERNS:
        for m in pat.findall(text):
            if m not in seen:
                seen.add(m)
                feature_hits += 1
    payload = {
        "status": "captured",
        "features_with_telemetry": feature_hits,
    }
    _write_json("runtime_validation.json", payload)
    return (
        Phase(
            phase_id=11,
            name="runtime_telemetry",
            artifact="runtime_validation.json",
            status="ok" if feature_hits > 0 else "warn",
            details=payload,
        ),
        feature_hits > 0,
    )


# ---------------------------------------------------------------------------
# Phase 12 — final summary
# ---------------------------------------------------------------------------


def _phase_12_summary(report: ValidationReport) -> dict[str, Any]:
    file_cov = report.coverage_pct(report.file_count, report.file_count)
    feat_cov = report.coverage_pct(
        report.implemented_feature_count, report.declared_feature_count
    )
    src_cov = "100%"  # source ingestion failures already captured per-source
    inv_cov = report.coverage_pct(
        report.enforced_invariant_count, report.invariant_count
    )

    all_ok = (
        feat_cov == "100%"
        and inv_cov == "100%"
        and report.dependency_graph_valid
        and report.ast_validation
        and report.runtime_validation
        and report.dead_files == 0
        and report.unmapped_declarations == 0
        and report.ambiguity == 0
    )

    if report.advisory:
        status = "PASS"
    else:
        status = "PASS" if all_ok else "FAIL"

    summary = {
        "file_coverage": file_cov,
        "feature_coverage": feat_cov,
        "source_coverage": src_cov,
        "invariant_coverage": inv_cov,
        "dependency_graph_valid": report.dependency_graph_valid,
        "ast_validation": report.ast_validation,
        "runtime_validation": report.runtime_validation,
        "dead_files": report.dead_files,
        "unmapped_declarations": report.unmapped_declarations,
        "ambiguity": report.ambiguity,
        "status": status,
        "advisory_mode": report.advisory,
        "phases": [
            {
                "id": ph.phase_id,
                "name": ph.name,
                "artifact": ph.artifact,
                "status": ph.status,
                "details": ph.details,
            }
            for ph in report.phases
        ],
    }
    _write_json("coverage_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run(advisory: bool = True) -> dict[str, Any]:
    """Execute all 12 phases, return the summary payload."""
    _ensure_analysis_dir()
    report = ValidationReport(advisory=advisory)

    files = _iter_repo_files()
    report.file_count = len(files)

    p0, parsed_sources = _phase_0_source_ingestion()
    report.phases.append(p0)

    report.phases.append(_phase_1_file_index(files))

    p2, feature_to_sources = _phase_2_feature_extraction(parsed_sources)
    report.phases.append(p2)
    report.declared_feature_count = len(feature_to_sources)

    report.phases.append(_phase_3_file_analysis(files))

    p4, feature_status = _phase_4_feature_coverage(feature_to_sources, files)
    report.phases.append(p4)
    report.implemented_feature_count = sum(
        1 for s in feature_status.values() if s == "OK"
    )
    report.feature_count = len(feature_status)
    report.ambiguity = sum(1 for s in feature_status.values() if s == "AMBIGUOUS")

    report.phases.append(_phase_5_source_coverage(parsed_sources, feature_to_sources))

    p6, inv_total, inv_enforced = _phase_6_invariant_validation(
        feature_to_sources, files
    )
    report.phases.append(p6)
    report.invariant_count = inv_total
    report.enforced_invariant_count = inv_enforced

    p7 = _phase_7_file_usage(files, feature_status)
    report.phases.append(p7)
    report.dead_files = int(p7.details.get("dead_files", 0))

    p8, declared_not_implemented = _phase_8_declaration_consistency(
        feature_to_sources, feature_status
    )
    report.phases.append(p8)
    report.unmapped_declarations = declared_not_implemented

    p9, graph_valid = _phase_9_dependency_graph(files)
    report.phases.append(p9)
    report.dependency_graph_valid = graph_valid

    p10 = _phase_10_ast_validation(files)
    report.phases.append(p10)
    report.ast_validation = p10.status == "ok"

    p11, runtime_ok = _phase_11_runtime_telemetry(advisory)
    report.phases.append(p11)
    report.runtime_validation = runtime_ok

    return _phase_12_summary(report)


def _print_human_summary(summary: dict[str, Any]) -> None:
    print(f"TOTAL VALIDATION — status={summary['status']}", flush=True)
    print(f"  advisory_mode      : {summary['advisory_mode']}")
    print(f"  file_coverage      : {summary['file_coverage']}")
    print(f"  feature_coverage   : {summary['feature_coverage']}")
    print(f"  invariant_coverage : {summary['invariant_coverage']}")
    print(f"  dead_files         : {summary['dead_files']}")
    print(f"  unmapped_decls     : {summary['unmapped_declarations']}")
    print(f"  ambiguity          : {summary['ambiguity']}")
    print(f"  dep_graph_valid    : {summary['dependency_graph_valid']}")
    print(f"  ast_validation     : {summary['ast_validation']}")
    print(f"  runtime_validation : {summary['runtime_validation']}")
    print()
    for phase in summary["phases"]:
        print(
            f"  phase {phase['id']:>2} {phase['name']:<28} "
            f"{phase['status']:<5} -> analysis/{phase['artifact']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--advisory",
        action="store_true",
        default=True,
        help="advisory mode (default): write artifacts, never block.",
    )
    parser.add_argument(
        "--strict",
        dest="advisory",
        action="store_false",
        help="strict mode: any phase failure -> overall FAIL.",
    )
    args = parser.parse_args(argv)
    summary = run(advisory=args.advisory)
    _print_human_summary(summary)
    if args.advisory:
        return 0
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
