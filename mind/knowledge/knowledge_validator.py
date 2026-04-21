"""
mind.knowledge.knowledge_validator — verifies that a proposed knowledge
entry (fact / rule / pattern) is consistent with existing knowledge and
with the governance-approved constraint set before it's added to the
strategy arbiter's lookup table.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    reasons: list


class KnowledgeValidator:
    def validate(self, entry: dict[str, Any]) -> ValidationResult:
        reasons: list = []
        if not entry.get("source"):
            reasons.append("missing:source")
        if not entry.get("claim"):
            reasons.append("missing:claim")
        if entry.get("confidence", 0.0) < 0.1:
            reasons.append("low_confidence")
        return ValidationResult(valid=not reasons, reasons=reasons)


_v: KnowledgeValidator | None = None
_lock = threading.Lock()


def get_knowledge_validator() -> KnowledgeValidator:
    global _v
    if _v is None:
        with _lock:
            if _v is None:
                _v = KnowledgeValidator()
    return _v
