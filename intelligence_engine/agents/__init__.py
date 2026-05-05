"""Stateful intelligence agents (INV-54, B19, AGT-XX family).

Concrete agents live in this package and implement
:class:`core.contracts.agent.AgentIntrospection` via the abstract
base in :mod:`intelligence_engine.agents._base`.
"""

from intelligence_engine.agents._base import AgentBase
from intelligence_engine.agents.macro import MacroAgent
from intelligence_engine.agents.scalper import ScalperAgent
from intelligence_engine.agents.swing import SwingAgent

__all__ = ["AgentBase", "MacroAgent", "ScalperAgent", "SwingAgent"]
