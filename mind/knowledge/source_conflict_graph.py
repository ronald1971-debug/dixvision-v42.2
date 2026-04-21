"""
mind.knowledge.source_conflict_graph — tracks disagreement between
signal sources (e.g. trend-follower vs mean-reverter on the same asset).
The strategy arbiter reads this to prefer the side with historically
higher resolved-reward when two signals point opposite directions.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class ConflictEdge:
    source_a: str
    source_b: str
    disagreements: int = 0
    a_wins: int = 0
    b_wins: int = 0


class SourceConflictGraph:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._edges: dict[tuple[str, str], ConflictEdge] = {}

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def observe_disagreement(self, source_a: str, source_b: str) -> None:
        with self._lock:
            k = self._key(source_a, source_b)
            e = self._edges.setdefault(k, ConflictEdge(*k))
            e.disagreements += 1

    def observe_winner(self, source: str, loser: str) -> None:
        with self._lock:
            k = self._key(source, loser)
            e = self._edges.setdefault(k, ConflictEdge(*k))
            if k[0] == source:
                e.a_wins += 1
            else:
                e.b_wins += 1

    def win_ratio(self, source: str, other: str) -> float:
        with self._lock:
            k = self._key(source, other)
            e = self._edges.get(k)
            if not e:
                return 0.5
            total = e.a_wins + e.b_wins
            if total == 0:
                return 0.5
            wins = e.a_wins if k[0] == source else e.b_wins
            return wins / total


_g: SourceConflictGraph | None = None
_lock = threading.Lock()


def get_source_conflict_graph() -> SourceConflictGraph:
    global _g
    if _g is None:
        with _lock:
            if _g is None:
                _g = SourceConflictGraph()
    return _g
