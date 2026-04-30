"""Authority Lint — CORE-31 / CI-05 (Phase E0).

Static AST scan over Python imports. Enforces the architectural seams
declared in ``manifest.md`` §0.8 and ``docs/total_recall_index.md`` §40.

Rule set (ZIP v4):

* **T1**  Fast-path purity: ``hot path`` modules must not import
  ``governance`` (INV-17).
* **C2**  Neuromorphic isolation (NEUR-04 / SAFE-18).
* **C3**  Web-autolearn isolation (SAFE-15 / WEBLEARN-07).
* **W1**  Burner-wallet (memecoin adapters cannot import main wallet).
* **L1**  Learning ↔ Evolution direct imports forbidden in BOTH directions.
* **L2**  Offline engines may not import runtime engines.
* **L3**  Runtime engines may not import ``learning_engine`` or
  ``evolution_engine``.
* **B1**  Cross-runtime-engine direct imports forbidden — generalises T1.
* **B7**  Dashboard isolation (Build Compiler Spec §6 + INV-37). The
  ``dashboard/`` package is the dashboard *control plane*. It may only
  import ``core.contracts``, ``core.coherence`` (read-only),
  ``governance_engine.control_plane`` (Protocol surfaces / GOV-CP-07
  bridge), ``state.ledger.reader``, and ``intelligence_engine``
  read-only public surfaces (``check_self``-style health + the
  strategy lifecycle FSM). Private engine modules and any
  ``learning_engine`` / ``evolution_engine`` imports are forbidden.
* **B8**  System Intent isolation (INV-38). The
  ``core/coherence/system_intent.py`` projection is operator-written
  via GOV-CP-07 + GOV-CP-03 only — the projection module itself must
  never import any ``*_engine`` package or any other writable surface.
  Allowed imports: ``core.contracts``, ``core.coherence``, and
  ``state.ledger.reader``.
* **B17** Shadow meta-controller is non-acting (INV-52). The
  ``intelligence_engine.meta_controller.policy.shadow_policy`` module
  may not import ``governance_engine``.
* **B20** Triad Lock — Governance is order-blind (INV-56).
  ``governance_engine`` may not import any ``execution_engine``
  surface. Complements B1 with an explicit triad-lock message.
* **B21** Triad Lock — only ``execution_engine`` may construct
  ``ExecutionEvent`` (INV-56). Tests + ``contracts/`` are exempt.
* **B22** Triad Lock — only ``intelligence_engine`` (and the ``ui/``
  dev-harness) may construct ``SignalEvent`` (INV-56). Tests +
  ``contracts/`` are exempt.
* **B23** Registry-driven AI providers (Dashboard-2026 wave-01). Chat
  widget files (``ui/static/chat_widget.js``,
  ``ui/static/indira_chat.html``, ``ui/static/dyon_chat.html``, and
  any future ``intelligence_engine.cognitive.chat.*`` Python module)
  may not contain any string literal naming a specific AI vendor.
  The single source of truth is ``registry/data_source_registry.yaml``
  (rows with ``category: ai``); chat widgets must read that registry
  via ``GET /api/ai/providers`` and surface whatever it returns.
  Adding a new provider is a registry-only change — no widget edit.
* **B25** Execution Gate origin restriction (HARDEN-01 / INV-68).
  Only ``intelligence_engine.*`` and ``governance_engine.*`` may call
  ``create_execution_intent`` / ``mark_approved`` / ``mark_rejected``.
  Tests use the dedicated ``tests.fixtures`` origin so production
  code paths stay tight.

* **B24** LangGraph / LangChain import containment (Dashboard-2026
  wave-03 prep, INV-67). Only ``intelligence_engine.cognitive.*``
  and ``evolution_engine.dyon.*`` may import ``langgraph``,
  ``langchain*``, or ``langsmith``. Hot-path engines
  (``execution_engine``, ``governance_engine``, ``system_engine``)
  and the deterministic core (``core``) must never import any of
  these surfaces — graph orchestration is non-deterministic and is
  quarantined as advisory-only. Rule fires defensively even before
  any module imports LangGraph (currently none) so future work
  cannot drift past the boundary unnoticed.

Allow-list applies to every rule:

* ``core`` / ``core.contracts``
* ``state.ledger.reader``
* the standard library and approved third-party packages

Usage::

    python tools/authority_lint.py [--strict] [<repo_root>]

Exits non-zero on any violation. Designed to run in pre-commit and CI
(``.github/workflows/lint.yml``).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from collections.abc import Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-set definitions
# ---------------------------------------------------------------------------

RUNTIME_ENGINE_PACKAGES: tuple[str, ...] = (
    "intelligence_engine",
    "execution_engine",
    "system_engine",
    "governance_engine",
)

OFFLINE_ENGINE_PACKAGES: tuple[str, ...] = (
    "learning_engine",
    "evolution_engine",
)

ALL_ENGINE_PACKAGES: tuple[str, ...] = (
    RUNTIME_ENGINE_PACKAGES + OFFLINE_ENGINE_PACKAGES
)

# Common allow-list — these may be imported from any engine package.
ALLOWED_SHARED_PREFIXES: tuple[str, ...] = (
    "core",
    "core.contracts",
    "core.contracts.events",
    "core.contracts.engine",
    "state.ledger.reader",
)

# Hot-path modules subject to T1.
HOT_PATH_MODULES: tuple[str, ...] = (
    "mind.fast_execute",
    "execution_engine.hot_path",
)

# Neuromorphic files subject to C2.
NEUROMORPHIC_PREFIXES: tuple[str, ...] = (
    "mind.neuromorphic",
)

NEUROMORPHIC_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "execution_engine",
    "governance_engine",
    "intelligence_engine.alpha",
    "mind.fast_execute",
    "mind.execute",
    "execution.adapters",
    "wallet",
)

# Web-autolearn modules subject to C3.
WEB_AUTOLEARN_PREFIXES: tuple[str, ...] = (
    "mind.web_autolearn",
)

WEB_AUTOLEARN_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "execution",
    "execution_engine",
    "governance",
    "governance_engine",
    "mind.fast_execute",
    "wallet",
)

# Memecoin adapters subject to W1.
MEMECOIN_ADAPTER_PREFIXES: tuple[str, ...] = (
    "execution_engine.adapters.memecoin",
    "execution.adapters.memecoin",
)

MAIN_WALLET_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "wallet.main_wallet",
)

# Dashboard isolation (B7).
DASHBOARD_PREFIXES: tuple[str, ...] = ("dashboard",)

# Imports the dashboard control-plane is permitted to make beyond the
# common allow-list. Each entry is matched as a dotted prefix.
DASHBOARD_ALLOWED_PREFIXES: tuple[str, ...] = (
    "core",
    "core.contracts",
    "core.coherence",
    "state.ledger.reader",
    "governance_engine.control_plane",
    # Read-only public surfaces of intelligence_engine that the
    # dashboard projects (strategy lifecycle FSM types + health).
    "intelligence_engine.strategy_runtime.state_machine",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    file: Path
    line: int
    importer: str
    imported: str
    detail: str

    def format(self, repo_root: Path) -> str:
        try:
            rel = self.file.relative_to(repo_root)
        except ValueError:
            rel = self.file
        return (
            f"{self.rule} {rel}:{self.line}: "
            f"{self.importer!r} imports {self.imported!r} -- {self.detail}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_name_for(path: Path, repo_root: Path) -> str:
    """Convert a path inside the repo into a dotted module name."""
    rel = path.relative_to(repo_root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _iter_python_files(repo_root: Path) -> Iterable[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        ".tox",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "rust",
        "target",
    }
    for p in repo_root.rglob("*.py"):
        try:
            rel_parts = p.relative_to(repo_root).parts
        except ValueError:
            rel_parts = p.parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        yield p


def _iter_imports(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """Yield ``(lineno, dotted_module_name)`` for every import target."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                # Relative imports are local — we do not enforce against them
                # because they are by definition in-package.
                continue
            yield node.lineno, mod


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _check_allow_list(target: str) -> bool:
    """Return True when ``target`` is unconditionally allowed."""
    return _starts_with_any(target, ALLOWED_SHARED_PREFIXES)


def _check_t1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, HOT_PATH_MODULES):
        return None
    if _starts_with_any(target, ("governance", "governance_engine")):
        return Violation(
            "T1",
            file,
            line,
            importer,
            target,
            "hot-path module must not import governance (INV-17)",
        )
    return None


def _check_c2(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, NEUROMORPHIC_PREFIXES):
        return None
    if _starts_with_any(target, NEUROMORPHIC_FORBIDDEN_PREFIXES):
        return Violation(
            "C2",
            file,
            line,
            importer,
            target,
            "neuromorphic isolation (NEUR-04 / SAFE-18)",
        )
    return None


def _check_c3(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, WEB_AUTOLEARN_PREFIXES):
        return None
    if _starts_with_any(target, WEB_AUTOLEARN_FORBIDDEN_PREFIXES):
        return Violation(
            "C3",
            file,
            line,
            importer,
            target,
            "web-autolearn isolation (SAFE-15 / WEBLEARN-07)",
        )
    return None


def _check_w1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, MEMECOIN_ADAPTER_PREFIXES):
        return None
    if _starts_with_any(target, MAIN_WALLET_FORBIDDEN_PREFIXES):
        return Violation(
            "W1",
            file,
            line,
            importer,
            target,
            "memecoin adapters must use burner wallet (INV-20 / SAFE-12)",
        )
    return None


def _check_l1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if _starts_with_any(importer, ("learning_engine",)) and _starts_with_any(
        target, ("evolution_engine",)
    ):
        return Violation(
            "L1",
            file,
            line,
            importer,
            target,
            "Learning ↔ Evolution domain boundary (sharing a process is not "
            "sharing a domain)",
        )
    if _starts_with_any(importer, ("evolution_engine",)) and _starts_with_any(
        target, ("learning_engine",)
    ):
        return Violation(
            "L1",
            file,
            line,
            importer,
            target,
            "Learning ↔ Evolution domain boundary (sharing a process is not "
            "sharing a domain)",
        )
    return None


def _check_l2(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, OFFLINE_ENGINE_PACKAGES):
        return None
    if _starts_with_any(target, RUNTIME_ENGINE_PACKAGES):
        return Violation(
            "L2",
            file,
            line,
            importer,
            target,
            "offline engine must not import runtime engines (INV-15)",
        )
    return None


def _check_l3(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, RUNTIME_ENGINE_PACKAGES):
        return None
    if _starts_with_any(target, OFFLINE_ENGINE_PACKAGES):
        return Violation(
            "L3",
            file,
            line,
            importer,
            target,
            "runtime engine must not import offline engines (INV-15)",
        )
    return None


def _check_b1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    importer_pkg = next(
        (p for p in RUNTIME_ENGINE_PACKAGES if _starts_with_any(importer, (p,))),
        None,
    )
    if importer_pkg is None:
        return None
    target_pkg = next(
        (p for p in RUNTIME_ENGINE_PACKAGES if _starts_with_any(target, (p,))),
        None,
    )
    if target_pkg is None or target_pkg == importer_pkg:
        return None
    if _check_allow_list(target):
        return None
    return Violation(
        "B1",
        file,
        line,
        importer,
        target,
        f"cross-runtime-engine direct import {importer_pkg} → {target_pkg} "
        "(events bus only; INV-08 / INV-11)",
    )


def _check_b7(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, DASHBOARD_PREFIXES):
        return None
    # Imports inside the dashboard package itself are always fine.
    if _starts_with_any(target, DASHBOARD_PREFIXES):
        return None
    if _check_allow_list(target):
        return None
    if _starts_with_any(target, DASHBOARD_ALLOWED_PREFIXES):
        return None
    # Block any other engine import — runtime or offline.
    if _starts_with_any(target, ALL_ENGINE_PACKAGES):
        return Violation(
            "B7",
            file,
            line,
            importer,
            target,
            "dashboard isolation: only core.contracts, core.coherence, "
            "state.ledger.reader, governance_engine.control_plane, and "
            "intelligence_engine.strategy_runtime.state_machine are allowed "
            "(Build Compiler Spec §6 + INV-37)",
        )
    return None


SYSTEM_INTENT_MODULE: str = "core.coherence.system_intent"

# System Intent (INV-38) may import only these prefixes beyond the
# common allow-list; every engine package is forbidden.
SYSTEM_INTENT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "core",
    "core.contracts",
    "core.coherence",
    "state.ledger.reader",
)


def _check_b8(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    """B8 — System Intent projection isolation (INV-38)."""
    if importer != SYSTEM_INTENT_MODULE:
        return None
    if _check_allow_list(target):
        return None
    if _starts_with_any(target, SYSTEM_INTENT_ALLOWED_PREFIXES):
        return None
    if _starts_with_any(target, ALL_ENGINE_PACKAGES):
        return Violation(
            "B8",
            file,
            line,
            importer,
            target,
            "System Intent projection isolation: "
            "core/coherence/system_intent.py may only import core.*, "
            "core.coherence.*, and state.ledger.reader. Any *_engine "
            "import would let the system write its own intent (INV-38).",
        )
    return None


SHADOW_POLICY_PREFIXES: tuple[str, ...] = (
    "intelligence_engine.meta_controller.policy.shadow_policy",
)

SHADOW_POLICY_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "governance_engine",
)


def _check_b17(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, SHADOW_POLICY_PREFIXES):
        return None
    if _starts_with_any(target, SHADOW_POLICY_FORBIDDEN_PREFIXES):
        return Violation(
            "B17",
            file,
            line,
            importer,
            target,
            "shadow meta-controller is non-acting; governance_engine "
            "imports forbidden (INV-52)",
        )
    return None


# ---------------------------------------------------------------------------
# Triad Lock (INV-56) — B20 / B21 / B22
#
# Three explicit rules that codify the triad invariant:
#
#   * Decider  = intelligence_engine (signals + meta-controller)
#   * Executor = execution_engine (orders + fills)
#   * Approver = governance_engine (approves / rejects / constrains; never
#                trades)
#
# B20 is import-based and complements B1 with an explicit "governance is
# order-blind" message. B21 / B22 are construction-based and walk Call
# nodes to ensure the producing engine is the only one that *creates* the
# typed bus events.
# ---------------------------------------------------------------------------

GOVERNANCE_PREFIXES: tuple[str, ...] = ("governance_engine",)
GOVERNANCE_FORBIDDEN_TARGET_PREFIXES: tuple[str, ...] = ("execution_engine",)


def _check_b20(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    """B20 — Governance is order-blind (INV-56 Triad Lock)."""
    if not _starts_with_any(importer, GOVERNANCE_PREFIXES):
        return None
    if _starts_with_any(target, GOVERNANCE_FORBIDDEN_TARGET_PREFIXES):
        return Violation(
            "B20",
            file,
            line,
            importer,
            target,
            "Triad Lock: governance is order-blind — governance_engine "
            "must never import any execution_engine surface "
            "(INV-56)",
        )
    return None


# Modules that may construct ExecutionEvent / SignalEvent directly.
EXECUTION_EVENT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "execution_engine",
)
SIGNAL_EVENT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "intelligence_engine",
    # Dev / operator harness: ui/server.py exposes a synthetic-signal
    # POST endpoint that flows through Intelligence → Execution. Treated
    # as a non-runtime fixture; production trading never goes through it.
    "ui",
)

# Path prefixes (relative parts) that are exempt from B21/B22 entirely.
TRIAD_CONSTRUCTOR_TEST_EXEMPT_PARTS: tuple[tuple[str, ...], ...] = (
    ("tests",),
    ("contracts",),
)


def _is_triad_constructor_test_exempt(
    path: Path, repo_root: Path
) -> bool:
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        return False
    for exempt in TRIAD_CONSTRUCTOR_TEST_EXEMPT_PARTS:
        if rel_parts[: len(exempt)] == exempt:
            return True
    return False


def _iter_named_calls(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """Yield ``(lineno, callee_name)`` for every plain ``Name(...)`` call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node.lineno, node.func.id


def _check_triad_event_constructions(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B21 / B22 — typed-event constructor restrictions (INV-56)."""
    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    out: list[Violation] = []
    for line, name in _iter_named_calls(tree):
        if name == "ExecutionEvent" and not _starts_with_any(
            importer, EXECUTION_EVENT_ALLOWED_PREFIXES
        ):
            out.append(
                Violation(
                    "B21",
                    file,
                    line,
                    importer,
                    "ExecutionEvent",
                    "Triad Lock: only execution_engine may construct "
                    "ExecutionEvent — outside callers must request a "
                    "fill via the typed bus (INV-56)",
                )
            )
        elif name == "SignalEvent" and not _starts_with_any(
            importer, SIGNAL_EVENT_ALLOWED_PREFIXES
        ):
            out.append(
                Violation(
                    "B22",
                    file,
                    line,
                    importer,
                    "SignalEvent",
                    "Triad Lock: only intelligence_engine may construct "
                    "SignalEvent — outside callers must publish via "
                    "the typed bus (INV-56)",
                )
            )
    return out


def _check_b24(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    """B24 — LangGraph / LangChain import containment (INV-67).

    Only the cognitive subsystems may pull in graph-orchestration or
    LangChain surfaces. Hot-path engines, deterministic core, and the
    governance layer must stay free of non-deterministic ML
    dependencies so INV-15 replay determinism is preserved on the
    typed bus.
    """

    if not _starts_with_any(
        target, ("langgraph", "langchain", "langsmith")
    ):
        return None
    if _starts_with_any(importer, COGNITIVE_ALLOWED_PREFIXES):
        return None
    return Violation(
        "B24",
        file,
        line,
        importer,
        target,
        "LangGraph / LangChain / LangSmith imports are quarantined to"
        " intelligence_engine.cognitive.* and evolution_engine.dyon.*"
        " (INV-67) — non-deterministic graph orchestration may not"
        " enter the hot path or governance.",
    )


# Module prefixes allowed to import langgraph / langchain* / langsmith.
# ``tests`` is included because the cognitive subsystems' unit tests
# must exercise those imports to pin the contract (e.g. asserting a
# ``BaseCheckpointSaver`` subclass round-trips through real LangGraph
# types, or that ``BaseChatModel`` invocations forward parameters
# correctly). Production runtime is still confined by the prefixes
# above — ``tests`` is a non-runtime quarantine of its own.
COGNITIVE_ALLOWED_PREFIXES: tuple[str, ...] = (
    "intelligence_engine.cognitive",
    "evolution_engine.dyon",
    "tests",
)


RULE_CHECKS = (
    _check_t1,
    _check_c2,
    _check_c3,
    _check_w1,
    _check_l1,
    _check_l2,
    _check_l3,
    _check_b1,
    _check_b7,
    _check_b8,
    _check_b17,
    _check_b20,
    _check_b24,
)


# ---------------------------------------------------------------------------
# B25 — Execution Gate origin restriction (HARDEN-01 / INV-68)
# ---------------------------------------------------------------------------

# Modules that may call the ExecutionIntent factory functions. Tests
# and the contract module itself are exempt (the dataclass lives in
# ``core.contracts.execution_intent`` so it can reference its own
# helpers).
B25_ALLOWED_PREFIXES: tuple[str, ...] = (
    "intelligence_engine",
    "governance_engine",
    "core.contracts.execution_intent",
)

B25_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {"create_execution_intent", "mark_approved", "mark_rejected"}
)


def _check_b25(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B25 — ExecutionIntent factory restriction (INV-68 / HARDEN-01)."""

    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    if _starts_with_any(importer, B25_ALLOWED_PREFIXES):
        return []
    out: list[Violation] = []
    for line, name in _iter_named_calls(tree):
        if name in B25_FORBIDDEN_NAMES:
            out.append(
                Violation(
                    "B25",
                    file,
                    line,
                    importer,
                    name,
                    "Execution Gate (INV-68): only intelligence_engine.*"
                    " and governance_engine.* may construct or mutate"
                    " an ExecutionIntent — outside callers must request"
                    " approval through the typed Governance bridge.",
                )
            )
    return out


# ---------------------------------------------------------------------------
# B23 — registry-driven AI providers (Dashboard-2026 wave-01)
# ---------------------------------------------------------------------------

# Chat widget files that B23 lints. Static files (HTML / JS) are scanned
# byte-by-byte (case-insensitive substring); Python files matching
# CHAT_WIDGET_PYTHON_PREFIXES are scanned via AST (string constants only)
# so that hot-path docstrings explaining the rule itself don't trip it.
CHAT_WIDGET_STATIC_RELATIVES: tuple[tuple[str, ...], ...] = (
    ("ui", "static", "chat_widget.js"),
    ("ui", "static", "indira_chat.html"),
    ("ui", "static", "dyon_chat.html"),
)

# Python module prefixes that B23 also lints. Empty for wave-01 (the
# router itself is in core.cognitive_router and is registry-driven by
# construction; chat widget *backends* land in wave-02). Listed here
# so wave-02 doesn't have to re-edit the lint to add coverage.
CHAT_WIDGET_PYTHON_PREFIXES: tuple[str, ...] = (
    "intelligence_engine.cognitive.chat",
    "ui.cognitive.chat",
)

# Provider tokens that are forbidden in chat widget code. Curated list
# of known AI vendors / brand names as of 2026 — any string match
# (case-insensitive) trips the rule. Adding a new token here is FINE;
# the rule only fails if it appears in chat widget source. The list
# is intentionally long so it covers names that aren't yet in the
# registry but might be added later (Anthropic / Claude / Qwen / etc.).
FORBIDDEN_AI_PROVIDER_TOKENS: tuple[str, ...] = (
    "openai",
    "chatgpt",
    "gpt-4",
    "gpt4",
    "gemini",
    "grok",
    "deepseek",
    "anthropic",
    "claude",
    "qwen",
    "mistral",
    "llama",
    "xai",
    "cognition",
    # NOTE: the brand name "Devin" is also forbidden but is omitted
    # from this list because the substring "devin" appears in
    # repo-internal Devin-Review comments and branch names that may
    # legitimately be quoted in chat widget audit text. Treat it as a
    # known limitation: humans must still avoid hard-coding "devin"
    # in chat widget code by convention.
)


def _check_b23_static(repo_root: Path) -> list[Violation]:
    """B23 — scan chat widget static files for forbidden vendor tokens."""

    out: list[Violation] = []
    for parts in CHAT_WIDGET_STATIC_RELATIVES:
        path = repo_root.joinpath(*parts)
        if not path.exists():
            # Wave-01 may ship subset of these. A missing file is fine;
            # the rule fires on existing content only.
            continue
        text = path.read_text(encoding="utf-8")
        haystack = text.lower()
        for token in FORBIDDEN_AI_PROVIDER_TOKENS:
            idx = haystack.find(token)
            if idx == -1:
                continue
            # Compute 1-based line number of the first hit for clarity.
            line_no = haystack.count("\n", 0, idx) + 1
            out.append(
                Violation(
                    "B23",
                    path,
                    line_no,
                    "ui.static." + path.stem,
                    token,
                    "chat widget files must be registry-driven; the"
                    " literal token "
                    f"{token!r} is forbidden — read providers from"
                    " /api/ai/providers (SCVS registry).",
                )
            )
    return out


def _check_b23_python(
    importer: str, file: Path, tree: ast.AST
) -> list[Violation]:
    """B23 — scan chat widget Python modules' string literals."""

    if not _starts_with_any(importer, CHAT_WIDGET_PYTHON_PREFIXES):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        haystack = node.value.lower()
        for token in FORBIDDEN_AI_PROVIDER_TOKENS:
            if token in haystack:
                out.append(
                    Violation(
                        "B23",
                        file,
                        getattr(node, "lineno", 0),
                        importer,
                        token,
                        "chat widget Python modules must be"
                        " registry-driven; the literal token "
                        f"{token!r} is forbidden — read providers"
                        " from the SCVS registry.",
                    )
                )
                break  # one violation per node is enough
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def lint_repo(repo_root: Path) -> list[Violation]:
    repo_root = repo_root.resolve()
    violations: list[Violation] = []
    for path in _iter_python_files(repo_root):
        # Ignore the lint tool itself and its tests' synthetic violation
        # fixtures (those are evaluated through targeted helpers in the test
        # suite, not by linting the repo tree).
        rel = path.relative_to(repo_root)
        if rel.parts[:1] == ("tools",) and rel.name == "authority_lint.py":
            continue
        if rel.parts[:2] == ("tests", "fixtures"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(
                Violation(
                    "SYNTAX",
                    path,
                    exc.lineno or 0,
                    _module_name_for(path, repo_root),
                    "",
                    f"unparseable: {exc.msg}",
                )
            )
            continue
        importer = _module_name_for(path, repo_root)
        for line, target in _iter_imports(tree):
            if not target:
                continue
            for check in RULE_CHECKS:
                v = check(importer, target, path, line)
                if v is not None:
                    violations.append(v)
        # INV-56 Triad Lock — typed-event constructor restrictions.
        violations.extend(
            _check_triad_event_constructions(importer, path, repo_root, tree)
        )
        # B25 — Execution Gate intent factory restriction (INV-68).
        violations.extend(_check_b25(importer, path, repo_root, tree))
        # B23 — chat widget Python modules must be registry-driven.
        violations.extend(_check_b23_python(importer, path, tree))
    # B23 — chat widget static files (HTML / JS).
    violations.extend(_check_b23_static(repo_root))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DIX VISION authority lint (Phase E0)."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="repo root (default: current directory)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any violation (default behaviour anyway)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve()
    violations = lint_repo(repo_root)

    if not violations:
        print(f"authority_lint: 0 violations in {repo_root}")
        return 0

    print(f"authority_lint: {len(violations)} violation(s) in {repo_root}")
    for v in violations:
        print(v.format(repo_root))
    return 1


if __name__ == "__main__":
    sys.exit(main())
