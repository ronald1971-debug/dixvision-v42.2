"""sensory.web_autolearn — WEBLEARN-01..10.

Web autolearn is the loop:

    seeds.yaml  --(WEBLEARN-01)-->  Crawler  --(RawDocument)-->
    AIFilter (WEBLEARN-02)  --(FilteredItem)-->
    Curator  (WEBLEARN-03)  --(CuratedItem)-->
    PendingBuffer (WEBLEARN-04, HITL-07)
    --(operator approves at /api/web_autolearn/approve)-->
    SignalEvent on the canonical bus.

Every value is a frozen dataclass. Every transform is a pure function
or a small class with no I/O coupling beyond its declared inputs. The
only stateful component is :class:`PendingBuffer`, which holds at
most ``capacity`` items and is FIFO-evicting.

The Crawler is a :class:`Protocol`; production wiring will inject a
Playwright-backed implementation while tests inject a deterministic
in-memory crawler. This keeps replay determinism (INV-15) intact.
"""

from sensory.web_autolearn.ai_filter import (
    AIFilter,
    FilterDecision,
    KeywordAIFilter,
)
from sensory.web_autolearn.contracts import (
    CuratedItem,
    FilteredItem,
    NewsItem,
    RawDocument,
    SocialPost,
)
from sensory.web_autolearn.crawler import (
    Crawler,
    DeterministicCrawler,
)
from sensory.web_autolearn.curator import (
    Curator,
    CuratorRules,
)
from sensory.web_autolearn.pending_buffer import (
    HitlBufferFull,
    PendingBuffer,
    PendingItem,
)

__all__ = [
    "AIFilter",
    "Crawler",
    "CuratedItem",
    "Curator",
    "CuratorRules",
    "DeterministicCrawler",
    "FilterDecision",
    "FilteredItem",
    "HitlBufferFull",
    "KeywordAIFilter",
    "NewsItem",
    "PendingBuffer",
    "PendingItem",
    "RawDocument",
    "SocialPost",
]
