"""C-19 AutoHedge — portfolio-decomposition pattern catalog (PATTERN_ONLY).

# ADAPTED FROM: The-Swarm-Corporation/AutoHedge
#   - autohedge/agents/market_analyst.py    (market regime / context)
#   - autohedge/agents/technical_analyst.py (signal / indicator scoring)
#   - autohedge/agents/risk_manager.py      (per-position risk checks)
#   - autohedge/agents/portfolio_optimizer.py (allocation / sizing)
#   - autohedge/agents/execution_manager.py (broker dispatch)

What this module is:

* A **frozen pattern catalog** describing AutoHedge's five-role
  portfolio decomposition. Each role is captured as an
  :class:`AutoHedgePatternRole` value object recording:

  - the role tag (closed enum),
  - the role's "I do X" responsibility statement,
  - the **canonical DIX module** that already fulfils that role
    (anchored by string path; pinned by an existence test so
    refactors that move the module surface a CI failure here).

What this module is **not**:

* **Not** a framework adaptation. C-18 (TradingAgents,
  ``trading_agents_bridge.py``) is the production committee
  surface; this module is a read-only design reference plus a
  short helper :func:`autohedge_role_for_dix_module` that lets
  the operator console answer the question "which AutoHedge role
  does ``governance_engine/control_plane/risk_evaluator.py`` map
  to?"
* **Not** a runtime path. No ``SignalEvent`` / ``ExecutionIntent``
  / ``GovernanceDecision`` / ``PatchProposal`` is ever
  constructed here (B27 / B28 / INV-71).

Authority discipline (OFFLINE_ONLY — advisory):

* **B27 / B28 / INV-71** — pure value-object catalog; never
  constructs any typed bus event.
* **B1 engine isolation** — no ``execution_engine.*`` /
  ``governance_engine.*`` / ``system_engine.*`` /
  ``evolution_engine.*`` *import*. The module references those
  packages **only as string anchors**, so the catalog can name
  the canonical DIX module without coupling the
  intelligence-engine tier to the runtime tiers. Pinned by AST
  tests.
* **INV-15 determinism** — module imports no ``random`` /
  ``time`` / ``datetime`` / ``secrets`` / ``os`` / ``asyncio``.

Tier: PATTERN_ONLY (advisory, never executes). The output of
:func:`autohedge_pattern_catalog` is a frozen tuple of
:class:`AutoHedgePatternRole`. The output of
:func:`autohedge_role_for_dix_module` is the canonical
:class:`AutoHedgeRole` (or ``None`` if the DIX path is not
mapped).
"""

# ADAPTED FROM: The-Swarm-Corporation/AutoHedge
#   - autohedge/* (role decomposition + consensus flow)

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

__all__ = (
    "AUTOHEDGE_PATTERN_CATALOG",
    "AutoHedgePatternError",
    "AutoHedgePatternRole",
    "AutoHedgeRole",
    "NEW_PIP_DEPENDENCIES",
    "autohedge_pattern_catalog",
    "autohedge_role_for_dix_module",
    "canonical_consensus_flow",
)

# C-19 directive: ``pip: source`` — no published package.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AutoHedgePatternError(ValueError):
    """Raised on contract violations (bad role / unknown DIX path)."""


# ---------------------------------------------------------------------------
# Closed role roster — matches AutoHedge upstream verbatim
# ---------------------------------------------------------------------------


class AutoHedgeRole(StrEnum):
    """The five-role portfolio decomposition from AutoHedge."""

    MARKET_ANALYST = "MARKET_ANALYST"
    TECHNICAL_ANALYST = "TECHNICAL_ANALYST"
    RISK_MANAGER = "RISK_MANAGER"
    PORTFOLIO_OPTIMIZER = "PORTFOLIO_OPTIMIZER"
    EXECUTION_MANAGER = "EXECUTION_MANAGER"


# ---------------------------------------------------------------------------
# Pattern value object
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class AutoHedgePatternRole:
    """Frozen catalog entry for one AutoHedge role.

    ``role`` is the AutoHedge role tag. ``responsibility`` is a
    short "I do X" statement (mirrors the upstream module
    docstring). ``dix_module`` is the canonical DIX module path
    (string anchor — not imported) that already fulfils this
    role. ``dix_summary`` is a short summary of how the DIX
    module fulfils the role."""

    role: AutoHedgeRole
    responsibility: str
    dix_module: str
    dix_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, AutoHedgeRole):
            raise AutoHedgePatternError("AutoHedgePatternRole.role must be an AutoHedgeRole member")
        if not isinstance(self.responsibility, str):
            raise AutoHedgePatternError("AutoHedgePatternRole.responsibility must be str")
        if not self.responsibility.strip():
            raise AutoHedgePatternError("AutoHedgePatternRole.responsibility must be non-empty")
        if not isinstance(self.dix_module, str):
            raise AutoHedgePatternError("AutoHedgePatternRole.dix_module must be str")
        if not self.dix_module.strip():
            raise AutoHedgePatternError("AutoHedgePatternRole.dix_module must be non-empty")
        if not isinstance(self.dix_summary, str):
            raise AutoHedgePatternError("AutoHedgePatternRole.dix_summary must be str")
        if not self.dix_summary.strip():
            raise AutoHedgePatternError("AutoHedgePatternRole.dix_summary must be non-empty")


# ---------------------------------------------------------------------------
# Canonical catalog
# ---------------------------------------------------------------------------


AUTOHEDGE_PATTERN_CATALOG: tuple[AutoHedgePatternRole, ...] = (
    AutoHedgePatternRole(
        role=AutoHedgeRole.MARKET_ANALYST,
        responsibility=(
            "Read market regime and macro context; produce a regime label and confidence."
        ),
        dix_module="intelligence_engine/macro/regime_engine.py",
        dix_summary=(
            "MacroRegimeEngine classifies the current MacroSnapshot into a "
            "MacroRegime label with confidence, satisfying the AutoHedge "
            "market_analyst role."
        ),
    ),
    AutoHedgePatternRole(
        role=AutoHedgeRole.TECHNICAL_ANALYST,
        responsibility=("Score price action / indicators / order-flow into a signal."),
        dix_module="intelligence_engine/plugins/",
        dix_summary=(
            "IND-L01..L12 plugin family (momentum / order-book-pressure / "
            "liquidity-physics / regime-classifier / VPIN / footprint / "
            "sentiment / on-chain / trader-imitation / news-reaction) "
            "produces SignalEvents — the AutoHedge technical_analyst "
            "responsibility, decomposed across many specialist plugins."
        ),
    ),
    AutoHedgePatternRole(
        role=AutoHedgeRole.RISK_MANAGER,
        responsibility=(
            "Enforce per-position and account-level risk gates before "
            "any position update is allowed."
        ),
        dix_module="governance_engine/control_plane/risk_evaluator.py",
        dix_summary=(
            "RiskEvaluator + RiskSnapshot.halted + the GOV-CP-07 hazard "
            "throttle adapter together fulfil the AutoHedge risk_manager "
            "role; risk gates run on every ExecutionIntent."
        ),
    ),
    AutoHedgePatternRole(
        role=AutoHedgeRole.PORTFOLIO_OPTIMIZER,
        responsibility=("Decide allocation / position sizing across the candidate set."),
        dix_module="intelligence_engine/portfolio/",
        dix_summary=(
            "PortfolioAllocator + ExposureManager (E-track) compute "
            "allocations and exposure caps from the SignalEvent set, "
            "fulfilling the AutoHedge portfolio_optimizer role."
        ),
    ),
    AutoHedgePatternRole(
        role=AutoHedgeRole.EXECUTION_MANAGER,
        responsibility=("Dispatch the chosen orders to the broker / venue."),
        dix_module="execution_engine/engine.py",
        dix_summary=(
            "ExecutionEngine.execute(intent) is the single chokepoint for "
            "all order dispatch (HARDEN-02), fulfilling the AutoHedge "
            "execution_manager role."
        ),
    ),
)


def autohedge_pattern_catalog() -> tuple[AutoHedgePatternRole, ...]:
    """Return the canonical AutoHedge → DIX pattern catalog."""

    return AUTOHEDGE_PATTERN_CATALOG


# ---------------------------------------------------------------------------
# Reverse lookup — DIX path → AutoHedge role
# ---------------------------------------------------------------------------


_DIX_MODULE_TO_ROLE: Mapping[str, AutoHedgeRole] = MappingProxyType(
    {entry.dix_module: entry.role for entry in AUTOHEDGE_PATTERN_CATALOG}
)


def autohedge_role_for_dix_module(path: str) -> AutoHedgeRole | None:
    """Return the AutoHedge role that the given DIX module fulfils.

    ``path`` is a repo-relative module path (e.g.
    ``governance_engine/control_plane/risk_evaluator.py``). The
    lookup is exact-match against the canonical catalog; returns
    ``None`` when the path is not a registered role anchor."""

    if not isinstance(path, str):
        raise AutoHedgePatternError("autohedge_role_for_dix_module: path must be str")
    return _DIX_MODULE_TO_ROLE.get(path)


# ---------------------------------------------------------------------------
# Canonical consensus flow — read-only sequence of role tags
# ---------------------------------------------------------------------------


_CONSENSUS_FLOW: tuple[AutoHedgeRole, ...] = (
    AutoHedgeRole.MARKET_ANALYST,
    AutoHedgeRole.TECHNICAL_ANALYST,
    AutoHedgeRole.RISK_MANAGER,
    AutoHedgeRole.PORTFOLIO_OPTIMIZER,
    AutoHedgeRole.EXECUTION_MANAGER,
)


def canonical_consensus_flow() -> tuple[AutoHedgeRole, ...]:
    """Return the canonical AutoHedge consensus flow sequence.

    The upstream framework runs roles in a fixed order:
    market analyst → technical analyst → risk manager →
    portfolio optimizer → execution manager. DIX mirrors this
    flow across separate engines (intelligence → intelligence →
    governance → intelligence → execution); the returned tuple
    is the read-only spec the operator console / docs reference."""

    return _CONSENSUS_FLOW


# ---------------------------------------------------------------------------
# Catalog invariants
# ---------------------------------------------------------------------------


def _verify_catalog_invariants() -> None:
    members = set(AutoHedgeRole)
    catalog_roles = {entry.role for entry in AUTOHEDGE_PATTERN_CATALOG}
    if members != catalog_roles:
        raise AutoHedgePatternError(
            "AUTOHEDGE_PATTERN_CATALOG must cover every AutoHedgeRole member exactly once"
        )
    catalog_paths = [entry.dix_module for entry in AUTOHEDGE_PATTERN_CATALOG]
    if len(set(catalog_paths)) != len(catalog_paths):
        raise AutoHedgePatternError("AUTOHEDGE_PATTERN_CATALOG dix_module anchors must be unique")
    flow_set = set(_CONSENSUS_FLOW)
    if flow_set != members:
        raise AutoHedgePatternError(
            "canonical_consensus_flow() must visit every AutoHedgeRole exactly once"
        )


_verify_catalog_invariants()
