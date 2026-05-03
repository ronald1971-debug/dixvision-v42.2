"""Wave-News-Fusion PR-3 — single-sink fanout for news pumps.

Closes the news→signal loop reviewer #3 (audit v3, "three things that
need honest scrutiny", item 2) called out: PR #118 shipped the
projection (NewsItem → SignalEvent | None) and PR #119 shipped the
shock classifier (NewsItem → tuple[HazardEvent, ...]). On their own
they are leaves — neither runs unless something hands a NewsItem to
both. This module composes them into one sink so a news pump (e.g.
:class:`ui.feeds.coindesk_rss.CoinDeskRSSPump`) only has to be wired
once.

Design constraints:

* INV-08 / INV-11 — only typed events cross engine boundaries. The
  fanout itself lives in ``ui/feeds`` (the harness / orchestration
  layer) so it can compose ``intelligence_engine.news`` and
  ``system_engine.hazard_sensors`` outputs without violating the
  lint-enforced engine isolation in either domain. Both leaves keep
  their own producer stamp (``produced_by_engine='intelligence_engine'``
  for the projected ``SignalEvent``; ``'system_engine'`` for the
  ``HazardEvent``); the fanout never re-stamps.
* INV-15 — caller supplies the optional ``current_belief`` callable
  that returns the latest :class:`BeliefState` snapshot, so the
  projection's regime damping path stays purely a function of inputs.
  No ``time.time``; no PRNG; no I/O.
* HARDEN-03 producer split — sensor + projector outputs preserve
  their own ``produced_by_engine`` so the receiver-side
  :func:`assert_event_provenance` checks remain effective.

Dispatch order is **hazard first, signal second**: this gives
Governance's hazard-throttle layer (INV-64) a head start on
registering the throttle window before the directional signal arrives
through the regular signal pipeline. Both sinks are caller-supplied;
either may be a no-op.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.coherence.belief_state import BeliefState
from core.contracts.events import HazardEvent, SignalEvent
from core.contracts.news import NewsItem
from intelligence_engine.news import project_news
from system_engine.hazard_sensors import NewsShockSensor

SignalSink = Callable[[SignalEvent], None]
HazardSink = Callable[[HazardEvent], None]
NewsItemSink = Callable[[NewsItem], None]
BeliefStateView = Callable[[], BeliefState | None]


@dataclass(frozen=True, slots=True)
class NewsFanout:
    """Compose the news projection + shock sensor into one ``NewsItem``
    sink suitable for the news pump.

    Wire-up sketch::

        fanout = NewsFanout(
            signal_sink=signal_bus.publish,
            hazard_sink=governance_bus.publish_hazard,
            sensor=NewsShockSensor(),
            current_belief=lambda: harness.belief_state,
        )
        pump = CoinDeskRSSPump(sink=fanout, clock_ns=time_authority.ns)

    Attributes:
        signal_sink: Callable invoked with a non-``None`` projected
            :class:`SignalEvent`. Skipped silently when the projector
            yields ``None`` (e.g. unresolved symbol, tied score, no
            keyword hits).
        hazard_sink: Callable invoked once per emitted
            :class:`HazardEvent` (typically zero or one per
            ``NewsItem``).
        sensor: A :class:`NewsShockSensor` instance. Held by reference
            so threshold tuning at construction time is visible to
            every subsequent dispatch.
        current_belief: Optional zero-arg callable returning the
            current :class:`BeliefState` projection (or ``None`` when
            no belief is available yet). Used solely as the
            ``current_belief`` argument of :func:`project_news` for
            regime damping. Not invoked for the hazard path.
    """

    signal_sink: SignalSink
    hazard_sink: HazardSink
    sensor: NewsShockSensor
    current_belief: BeliefStateView | None = None
    index_sink: NewsItemSink | None = None

    def __call__(self, news: NewsItem) -> None:
        # Index first so the knowledge store records the raw item even
        # when the projection / sensor short-circuit. The index is a
        # pure in-memory append; it does not affect downstream logic.
        if self.index_sink is not None:
            self.index_sink(news)

        for hazard in self.sensor.on_news(news):
            self.hazard_sink(hazard)

        belief = (
            self.current_belief()
            if self.current_belief is not None
            else None
        )
        signal = project_news(news, current_belief=belief)
        if signal is not None:
            self.signal_sink(signal)


__all__ = [
    "BeliefStateView",
    "HazardSink",
    "NewsFanout",
    "NewsItemSink",
    "SignalSink",
]
