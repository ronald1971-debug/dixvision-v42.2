"""
mind/sources/news_streams.py
Registry of structured news feeds. Each registered source returns a list of
(headline, polarity, confidence) tuples — never raw text.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class NewsItem:
    source: str
    headline: str
    polarity: float = 0.0
    confidence: float = 0.0
    timestamp_utc: str = ""


NewsFn = Callable[[], list[NewsItem]]


class NewsStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, NewsFn] = {}

    def register(self, name: str, fn: NewsFn) -> None:
        with self._lock:
            self._sources[name] = fn

    def pull_all(self) -> list[NewsItem]:
        out: list[NewsItem] = []
        with self._lock:
            srcs = list(self._sources.items())
        for _, fn in srcs:
            try:
                out.extend(fn())
            except Exception:
                continue
        return out


_registry: NewsStreamRegistry | None = None


def get_news_streams() -> NewsStreamRegistry:
    global _registry
    if _registry is None:
        _registry = NewsStreamRegistry()
    return _registry
