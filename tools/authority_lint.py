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
* **B26** Operator-approval edge restriction (Wave-03 PR-5). Only
  ``intelligence_engine.cognitive.approval_edge`` may construct a
  ``SignalEvent`` carrying the cognitive ``produced_by_engine`` stamp
  (``"intelligence_engine.cognitive"``). Every other path that needs
  to surface a cognitive proposal must go through the operator-
  approval queue and the typed approve / reject endpoints so the
  audit ledger captures the operator click before any event
  reaches the bus.
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

# B1-only allow-list — extra prefixes that runtime engines may import
# directly from each other but that *must not* relax the dashboard (B7)
# or system-intent (B8) isolation rules. Each entry needs an explicit
# justification.
#
# * ``system_engine.authority`` — the authority matrix
#   (``registry/authority_matrix.yaml`` + the frozen
#   :class:`AuthorityMatrix` value type) is the *source of truth* every
#   engine consults when proving its own role. It is data, not
#   behaviour; the loader has no side effects beyond reading a
#   registry YAML. The Execution Gate
#   (``execution_engine.execution_gate``) loads it once at
#   construction to validate every :class:`ExecutionIntent`
#   (HARDEN-02 / INV-68). Scoped to B1 so the dashboard and the
#   system-intent module remain blocked from importing it.
B1_EXTRA_ALLOWED_PREFIXES: tuple[str, ...] = (
    "system_engine.authority",
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

# Dashboard isolation (B7). Wave-Live PR-3 (#106) renamed the
# server-side widget package from ``dashboard`` to ``dashboard_backend``
# to disambiguate it from the React/Vite SPA in ``dashboard2026/``.
# B7's enforcement key must follow the rename or the rule silently
# matches no modules and the architectural isolation it protects
# (Build Compiler Spec §6 / INV-37: dashboard may not import private
# engine internals) becomes a no-op.
DASHBOARD_PREFIXES: tuple[str, ...] = ("dashboard_backend",)

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
    if _starts_with_any(target, B1_EXTRA_ALLOWED_PREFIXES):
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
# B26 — Operator-approval edge restriction (Wave-03 PR-5)
# ---------------------------------------------------------------------------

# The single module allowed to stamp a SignalEvent with the cognitive
# producer string. Every other code path that wants to surface a
# cognitive proposal must enqueue it for operator approval; the
# operator click is what flips it into a real SignalEvent on the bus.
B26_COGNITIVE_PRODUCER: str = "intelligence_engine.cognitive"
B26_ALLOWED_MODULES: tuple[str, ...] = (
    "intelligence_engine.cognitive.approval_edge",
)


def _signal_event_produced_by_engine(node: ast.Call) -> str | None:
    """Return the literal value of ``produced_by_engine=...`` on this Call.

    Returns ``None`` if the kwarg is absent or non-literal (covers
    constants, captures, registry lookups, etc.). B26 only fires on
    *literal* string matches — dynamic dispatch through a constant
    is by-construction routed through the approval edge anyway.
    """

    for kw in node.keywords:
        if kw.arg != "produced_by_engine":
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(
            kw.value.value, str
        ):
            return kw.value.value
    return None


def _check_b26(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B26 — only the approval edge may stamp the cognitive producer."""

    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    if _starts_with_any(importer, B26_ALLOWED_MODULES):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SignalEvent"
        ):
            continue
        stamped = _signal_event_produced_by_engine(node)
        if stamped != B26_COGNITIVE_PRODUCER:
            continue
        out.append(
            Violation(
                "B26",
                file,
                node.lineno,
                importer,
                'SignalEvent(produced_by_engine="intelligence_engine.cognitive")',
                "Operator-approval edge (Wave-03 PR-5): only "
                "intelligence_engine.cognitive.approval_edge may "
                "construct a SignalEvent stamped with the cognitive "
                "producer — every other cognitive proposal must flow "
                "through the approval queue + typed approve/reject "
                "endpoints so the operator click hits the audit ledger "
                "before the event reaches the bus.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# B27 — LearningUpdate construction restriction (HARDEN-06 / INV-71)
# ---------------------------------------------------------------------------
#
# Authority symmetry with B25/B26: only the learning subsystem may
# construct a :class:`core.contracts.learning.LearningUpdate`. Outside
# callers must observe LearningUpdates on the typed bus rather than
# synthesise them — otherwise a non-learning engine could indirectly
# trigger parameter mutations under the cover of a "looks legitimate"
# event class.

B27_ALLOWED_PREFIXES: tuple[str, ...] = (
    "learning_engine",
    "core.contracts.learning",
)

B27_FORBIDDEN_NAMES: frozenset[str] = frozenset({"LearningUpdate"})


def _check_b27(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B27 — LearningUpdate construction restriction (INV-71)."""

    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    if _starts_with_any(importer, B27_ALLOWED_PREFIXES):
        return []
    out: list[Violation] = []
    for line, name in _iter_named_calls(tree):
        if name in B27_FORBIDDEN_NAMES:
            out.append(
                Violation(
                    "B27",
                    file,
                    line,
                    importer,
                    name,
                    "Authority symmetry (INV-71 / HARDEN-06): only "
                    "learning_engine.* may construct a LearningUpdate "
                    "— outside callers must observe parameter "
                    "mutations on the typed bus.",
                )
            )
    return out


# ---------------------------------------------------------------------------
# B28 — PatchProposal construction restriction (HARDEN-06 / INV-71)
# ---------------------------------------------------------------------------
#
# Symmetric to B27 for the evolution subsystem. ``PatchProposal``
# carries structural mutations into the patch pipeline; the lint
# guarantees the only legitimate producer is ``evolution_engine.*``.

B28_ALLOWED_PREFIXES: tuple[str, ...] = (
    "evolution_engine",
    "core.contracts.learning",
)

B28_FORBIDDEN_NAMES: frozenset[str] = frozenset({"PatchProposal"})


def _check_b28(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B28 — PatchProposal construction restriction (INV-71)."""

    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    if _starts_with_any(importer, B28_ALLOWED_PREFIXES):
        return []
    out: list[Violation] = []
    for line, name in _iter_named_calls(tree):
        if name in B28_FORBIDDEN_NAMES:
            out.append(
                Violation(
                    "B28",
                    file,
                    line,
                    importer,
                    name,
                    "Authority symmetry (INV-71 / HARDEN-06): only "
                    "evolution_engine.* may construct a PatchProposal "
                    "— outside callers must observe structural "
                    "mutations on the typed bus.",
                )
            )
    return out


# ---------------------------------------------------------------------------
# B29 — TraderObservation construction restriction (Wave-04 PR-1 / INV-71)
# ---------------------------------------------------------------------------
#
# Authority symmetry with B27 (LearningUpdate) and B28 (PatchProposal).
# :class:`core.contracts.trader_intelligence.TraderObservation` is the
# bus-transport record for the Trader-Intelligence layer. It must only
# be constructed inside the dedicated trader-modeling subsystem
# (``intelligence_engine.trader_modeling.*``) — outside callers must
# observe rows on the typed bus rather than synthesising them. Without
# this rule, a non-modeling engine could indirectly inject philosophy /
# performance attributions into the strategy composition pipeline,
# bypassing the SCVS source-liveness FSM and the operator-approval gate
# that Wave-04 PR-4 will route compositions through.

B29_ALLOWED_PREFIXES: tuple[str, ...] = (
    "intelligence_engine.trader_modeling",
    "core.contracts.trader_intelligence",
)

B29_FORBIDDEN_NAMES: frozenset[str] = frozenset({"TraderObservation"})


def _check_b29(
    importer: str, file: Path, repo_root: Path, tree: ast.AST
) -> list[Violation]:
    """B29 — TraderObservation construction restriction (INV-71)."""

    if _is_triad_constructor_test_exempt(file, repo_root):
        return []
    if _starts_with_any(importer, B29_ALLOWED_PREFIXES):
        return []
    out: list[Violation] = []
    for line, name in _iter_named_calls(tree):
        if name in B29_FORBIDDEN_NAMES:
            out.append(
                Violation(
                    "B29",
                    file,
                    line,
                    importer,
                    name,
                    "Authority symmetry (INV-71 / Wave-04 PR-1): only "
                    "intelligence_engine.trader_modeling.* may construct "
                    "a TraderObservation — outside callers must observe "
                    "trader rows on the typed bus.",
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
#
# Wave-Live PR-2 retired the actual files at these paths in favour of
# the React SPA at ``/dash2/#/chat``. The allowlist is preserved on
# purpose: ``_check_b23_static`` skips paths that don't exist, so the
# rule is a no-op while the files are gone, but the moment anyone
# re-adds a chat widget at one of these canonical locations the rule
# starts scanning again. That's defence in depth — we don't want a
# future contributor to drop ``chat_widget.js`` back into the repo and
# silently bypass the registry-driven invariant.
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

# Modules that are exempt from B23's Python-string scan because they
# are *the* registry-to-backend translation point. Vendor names
# necessarily appear here as dispatch keys (e.g. ``"openai"``,
# ``"google"``, ``"cognition"``) — that is the file's whole job.
# The exemption is module-exact (not prefix-based) so anything else
# in ``intelligence_engine.cognitive.chat`` still has to stay
# registry-driven.
B23_PYTHON_EXEMPT_MODULES: frozenset[str] = frozenset(
    {
        "intelligence_engine.cognitive.chat.http_chat_transport",
    }
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
    if importer in B23_PYTHON_EXEMPT_MODULES:
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
        # B26 — only the approval edge may stamp the cognitive producer.
        violations.extend(_check_b26(importer, path, repo_root, tree))
        # B27/B28/B29 — authority symmetry for learning + evolution +
        # trader-modeling origins.
        violations.extend(_check_b27(importer, path, repo_root, tree))
        violations.extend(_check_b28(importer, path, repo_root, tree))
        violations.extend(_check_b29(importer, path, repo_root, tree))
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
