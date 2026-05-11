"""Harness internals — extracted from ``ui.server`` (P1.2+).

The harness god-object refactor splits the historically monolithic
``ui.server._State.__init__`` into domain-organised managers:

* :class:`ui.harness.boot_manager.HarnessBootManager` — owns
  construction order across intelligence / execution / governance /
  system / learning / evolution sections.

INV-15 byte-identical replay is preserved bit-for-bit: each manager
exposes a ``populate(state)`` (or equivalent) entry point that
mutates the harness state object in the same order the previous
inline ``__init__`` did. The managers add zero new behaviour — they
are pure code-organisation.
"""

from ui.harness.background_task_manager import HarnessBackgroundTaskManager
from ui.harness.boot_manager import HarnessBootManager
from ui.harness.source_trust_replay import replay_source_trust_promotions

__all__ = (
    "HarnessBackgroundTaskManager",
    "HarnessBootManager",
    "replay_source_trust_promotions",
)
