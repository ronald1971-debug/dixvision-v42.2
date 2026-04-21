"""
mind.knowledge — INDIRA's knowledge tree (manifest §4).

Consolidates previously scattered top-level stubs (drift_monitor,
edge_case_memory, feedback_cleaner, knowledge_validator, memory_index,
source_conflict_graph) into a single coherent subsystem.
"""
from __future__ import annotations

from mind.knowledge.drift_monitor import DriftMonitor, get_drift_monitor
from mind.knowledge.edge_case_memory import EdgeCaseMemory, get_edge_case_memory
from mind.knowledge.feedback_cleaner import FeedbackCleaner, get_feedback_cleaner
from mind.knowledge.knowledge_validator import KnowledgeValidator, get_knowledge_validator
from mind.knowledge.memory_index import MemoryIndex, get_memory_index
from mind.knowledge.source_conflict_graph import (
    SourceConflictGraph,
    get_source_conflict_graph,
)

__all__ = [
    "DriftMonitor",
    "get_drift_monitor",
    "EdgeCaseMemory",
    "get_edge_case_memory",
    "FeedbackCleaner",
    "get_feedback_cleaner",
    "KnowledgeValidator",
    "get_knowledge_validator",
    "MemoryIndex",
    "get_memory_index",
    "SourceConflictGraph",
    "get_source_conflict_graph",
]
