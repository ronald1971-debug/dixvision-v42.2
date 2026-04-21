"""
core.charter \u2014 self-knowledge contract for every voice in DIX VISION v42.2.

Each voice (INDIRA / DYON / GOVERNANCE / DEVIN) declares WHAT it is, HOW
it operates, WHY (manifest \u00a7 citations), what it MUST NOT do, and how
it is AUDITED. The chat introspection API ("what/how/why are you doing
X?") reads this to answer grounded in each voice's declared role.

Charters are LOADED, not WRITTEN, at runtime. A charter can only be
amended via a governance-gated patch (SYSTEM/CHARTER_AMENDED event).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.authority import Domain


class Voice(str, Enum):
    INDIRA = "INDIRA"
    DYON = "DYON"
    GOVERNANCE = "GOVERNANCE"
    DEVIN = "DEVIN"


@dataclass(frozen=True)
class Charter:
    voice: Voice
    domain: Domain
    what: str
    how: list[str]
    why: list[str]                    # manifest citations, e.g. "\u00a75 execution authority"
    not_do: list[str]
    accountability: list[str]         # ledger streams each action writes
    tools: list[str] = field(default_factory=list)
    peers_readable: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "voice": self.voice.value,
            "domain": self.domain.value,
            "what": self.what,
            "how": list(self.how),
            "why": list(self.why),
            "not_do": list(self.not_do),
            "accountability": list(self.accountability),
            "tools": list(self.tools),
        }


_REGISTRY: dict[Voice, Charter] = {}


def register_charter(c: Charter) -> None:
    _REGISTRY[c.voice] = c


def get_charter(v: Voice) -> Charter | None:
    return _REGISTRY.get(v)


def all_charters() -> dict[Voice, Charter]:
    return dict(_REGISTRY)


__all__ = ["Voice", "Charter", "register_charter", "get_charter", "all_charters"]
