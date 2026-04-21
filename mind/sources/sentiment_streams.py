"""
mind/sources/sentiment_streams.py
Structured sentiment source registry (social media / analyst polarity). All
consumers read precomputed (polarity, confidence) — never raw text.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SentimentPoint:
    source: str
    polarity: float = 0.0
    confidence: float = 0.0
    window_seconds: int = 60


SentimentFn = Callable[[], list[SentimentPoint]]


class SentimentStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, SentimentFn] = {}

    def register(self, name: str, fn: SentimentFn) -> None:
        with self._lock:
            self._sources[name] = fn

    def pull_all(self) -> list[SentimentPoint]:
        out: list[SentimentPoint] = []
        with self._lock:
            srcs = list(self._sources.items())
        for _, fn in srcs:
            try:
                out.extend(fn())
            except Exception:
                continue
        return out


_registry: SentimentStreamRegistry | None = None


def get_sentiment_streams() -> SentimentStreamRegistry:
    global _registry
    if _registry is None:
        _registry = SentimentStreamRegistry()
    return _registry
