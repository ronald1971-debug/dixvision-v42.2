"""News-fusion subsystem (Wave-News-Fusion PR-1).

Closes the news-to-signal gap reviewer #3 (audit v3, item 2) called
out: the CoinDesk RSS adapter ships :class:`core.contracts.news.NewsItem`
rows into the system, but until this package nothing folded them into
a :class:`core.coherence.belief_state.BeliefState`-bound signal. The
public entry point is :func:`project_news`.

Authority constraints:

* Only :mod:`core.contracts` and :mod:`core.coherence.belief_state` are
  imported. No ``*_engine`` package, no ledger writers.
* The module honours **B30** by importing ``BeliefState`` directly so
  every news-driven :class:`SignalEvent` it constructs is anchored in
  the unified belief projection.
* Pure function — no clocks, no PRNG, no I/O. Replay determinism
  (INV-15) is preserved end-to-end.
"""

from intelligence_engine.news.news_projection import (
    NEWS_PROJECTION_VERSION,
    project_news,
)

__all__ = ["NEWS_PROJECTION_VERSION", "project_news"]
