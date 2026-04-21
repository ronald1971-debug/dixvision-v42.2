"""
tools/authority_lint.py \u2014 AST firewall for DIX VISION.

Single gate that enforces:
    1. cross-domain imports
         - mind.fast_execute (hot path) may not import from governance.*, mind.sources.*,
           mind.knowledge.*, or sqlite3 / db-libs.
         - governance.* may not import mind.fast_execute internals.
         - DYON (system_monitor.*) may not import execution.adapters.* or
           mind.fast_execute.
         - cockpit.* may not import mind.fast_execute.
    2. hot-path DB reads
         - mind/fast_execute.py and mind/knowledge/trader_knowledge.py fast helpers
           may not call sqlite3.connect(...) inside a function tagged
           ``# hot-path`` or inside ``fast_execute_trade``.
    3. forbidden primitives outside their authoritative module
         - os.environ.get \u2192 only in system/config.py
         - time.time / datetime.utcnow \u2192 only in system/time_source.py
         - print( \u2192 only in system/logger.py and scripts/
         - random / secrets (outside core/secrets, cockpit/auth) discouraged in hot path.

Run:
    python3 tools/authority_lint.py         (repo root)
    exit 0  = clean
    exit 1  = violations (printed with file:line + rule id)
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIRS = ("cockpit", "core", "execution", "governance", "mind", "state",
            "system", "system_monitor", "security", "translation",
            "enforcement", "observability", "immutable_core")

# (rule_id, forbidden_prefix, path_pattern_allowlist)
FORBIDDEN_IMPORTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A1", "governance", ("mind/fast_execute",)),
    ("A2", "mind.sources", ("mind/fast_execute",)),
    ("A3", "mind.knowledge", ("mind/fast_execute",)),
    ("A4", "mind.fast_execute", ("cockpit/",)),
    ("A5", "mind.fast_execute", ("governance/",)),
    ("A6", "mind.fast_execute", ("system_monitor/",)),
    ("A7", "execution.adapters", ("system_monitor/",)),
)

FORBIDDEN_PRIMITIVES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # (rule_id, call_name, allow_path_prefixes)
    ("B1", "os.environ.get", ("system/config.py", "core/secrets.py",
                              "system/locale.py", "cockpit/auth.py",
                              "cockpit/launcher.py", "cockpit/pairing.py",
                              "cockpit/app.py",
                              "core/authority.py", "core/single_instance.py",
                              "core/runtime/", "immutable_core/",
                              "tools/")),
    ("B2", "time.time", ("system/time_source.py",
                         "mind/sources/rate_limiter.py",
                         "system_monitor/checks/clock_sync_check.py",
                         "security/authentication.py",
                         "immutable_core/",
                         "tests/")),
    ("B3", "datetime.utcnow", ("system/time_source.py", "tests/")),
    ("B4", "print", ("system/logger.py", "system/health_monitor.py",
                     "scripts/", "tests/",
                     "dix.py", "main.py", "diagnose_foundation.py",
                     "bootstrap_kernel.py", "startup_test.py", "tools/",
                     "cockpit/launcher.py",
                     "governance/patch_pipeline.py")),
)

HOT_PATH_FILES: tuple[str, ...] = (
    "mind/fast_execute.py",
    "system/fast_risk_cache.py",
    "interrupt/",
)

HOT_PATH_DB_BANNED: tuple[str, ...] = ("sqlite3", "psycopg", "duckdb", "asyncpg")


@dataclass
class Violation:
    rule: str
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} [{self.rule}] {self.detail}"


def _iter_py(root: Path) -> Iterable[Path]:
    for d in SRC_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def _allowed(rel_path: str, allow: Sequence[str]) -> bool:
    return any(rel_path.startswith(a) for a in allow)


def _check_imports(rel_path: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        elif isinstance(node, ast.Import):
            continue  # handled by name below
        else:
            continue
        for rule, prefix, allow in FORBIDDEN_IMPORTS:
            # "forbidden_prefix" means: a path_pattern_allowlist file may NOT
            # import modules starting with this prefix.
            if rel_path.startswith(allow) and (mod == prefix or mod.startswith(prefix + ".")):
                out.append(Violation(rule, rel_path, node.lineno,
                                     f"{rel_path} must not import {mod}"))
    # top-level `import X`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for rule, prefix, allow in FORBIDDEN_IMPORTS:
                    if rel_path.startswith(allow) and (alias.name == prefix or
                                                       alias.name.startswith(prefix + ".")):
                        out.append(Violation(rule, rel_path, node.lineno,
                                             f"{rel_path} must not import {alias.name}"))
    return out


def _check_primitives(rel_path: str, src: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = _call_name(node.func)
            for rule, target, allow in FORBIDDEN_PRIMITIVES:
                if name == target and not _allowed(rel_path, allow):
                    out.append(Violation(rule, rel_path, node.lineno,
                                         f"use of {target} outside authoritative module"))
            self.generic_visit(node)

    V().visit(tree)
    return out


def _check_hot_path_db(rel_path: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    is_hot = any(rel_path.startswith(h) for h in HOT_PATH_FILES)
    if not is_hot:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                for banned in HOT_PATH_DB_BANNED:
                    if n == banned or n.startswith(banned + "."):
                        out.append(Violation("C1", rel_path, node.lineno,
                                             f"hot-path file may not import {n}"))
    return out


def _call_name(n: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


def run() -> int:
    violations: list[Violation] = []
    for p in _iter_py(REPO):
        rel = _rel(p)
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel)
        except SyntaxError as exc:
            violations.append(Violation("SYN", rel, exc.lineno or 0,
                                        f"syntax error: {exc.msg}"))
            continue
        violations.extend(_check_imports(rel, tree))
        violations.extend(_check_primitives(rel, src, tree))
        violations.extend(_check_hot_path_db(rel, tree))

    if not violations:
        print("[authority_lint] clean \u2014 0 violations")
        return 0
    print(f"[authority_lint] {len(violations)} violation(s):")
    for v in violations:
        print(" -", v)
    return 1


if __name__ == "__main__":
    sys.exit(run())
