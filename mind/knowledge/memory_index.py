"""
mind.knowledge.memory_index — lightweight keyword index over remembered
trade contexts / edge cases / cleaned feedback. Used by the chat layer
(``cockpit.chat``) to look up relevant history when answering user
questions.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndexedRecord:
    id: str
    kind: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


class MemoryIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, IndexedRecord] = {}
        self._by_word: dict[str, set] = defaultdict(set)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.strip().lower() for t in text.replace(",", " ").split() if t.strip()]

    def add(self, record: IndexedRecord) -> None:
        with self._lock:
            self._records[record.id] = record
            for w in self._tokenize(record.text):
                self._by_word[w].add(record.id)

    def query(self, q: str, limit: int = 20) -> list[IndexedRecord]:
        with self._lock:
            tokens = self._tokenize(q)
            ids: dict[str, int] = {}
            for t in tokens:
                for rid in self._by_word.get(t, ()):
                    ids[rid] = ids.get(rid, 0) + 1
            ranked = sorted(ids.items(), key=lambda x: -x[1])[:limit]
            return [self._records[rid] for rid, _ in ranked if rid in self._records]


_idx: MemoryIndex | None = None
_lock = threading.Lock()


def get_memory_index() -> MemoryIndex:
    global _idx
    if _idx is None:
        with _lock:
            if _idx is None:
                _idx = MemoryIndex()
    return _idx
