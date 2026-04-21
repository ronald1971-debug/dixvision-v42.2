"""
core.introspection \u2014 charter-grounded "what / how / why" responder.

Every voice (INDIRA / DYON / GOVERNANCE / DEVIN) can answer operator
questions about its role by reading its own Charter + the most recent
ledger events tagged to its accountability streams. No LLM required;
if one is configured, the router paraphrases the structured answer.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from core.charter import Voice, get_charter


@dataclass
class Introspection:
    voice: Voice
    question: str
    what: str
    how: list[str]
    why: list[str]
    not_do: list[str]
    accountability: list[str]
    tools: list[str]
    ledger_refs: list[int] = field(default_factory=list)
    peers: dict[str, dict[str, object]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"[{self.voice.value}] \u2014 introspection",
            f"Q: {self.question}",
            "",
            "WHAT I AM:",
            f"  {self.what}",
            "HOW I OPERATE:",
            *[f"  - {h}" for h in self.how],
            "WHY (manifest):",
            *[f"  - {w}" for w in self.why],
            "I DO NOT:",
            *[f"  - {n}" for n in self.not_do],
            "ACCOUNTABILITY (ledger streams):",
            f"  {', '.join(self.accountability)}",
            "TOOLS:",
            f"  {', '.join(self.tools)}",
        ]
        if self.ledger_refs:
            lines += ["LEDGER_REFS:", f"  {self.ledger_refs}"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "voice": self.voice.value,
            "question": self.question,
            "what": self.what,
            "how": self.how,
            "why": self.why,
            "not_do": self.not_do,
            "accountability": self.accountability,
            "tools": self.tools,
            "ledger_refs": self.ledger_refs,
            "peers": self.peers,
        }


def _classify(question: str) -> list[str]:
    q = question.lower()
    out: list[str] = []
    if any(k in q for k in ("what", "who are you", "role", "rol")):
        out.append("what")
    if any(k in q for k in ("how", "hoe", "way", "approach")):
        out.append("how")
    if any(k in q for k in ("why", "waarom", "reason", "because", "ground")):
        out.append("why")
    if not out:
        out = ["what", "how", "why"]
    return out


def introspect(voice: Voice, question: str,
               peers: Sequence[Voice] | None = None,
               recent_ledger_refs: Iterable[int] | None = None) -> Introspection:
    c = get_charter(voice)
    if c is None:
        return Introspection(voice=voice, question=question,
                             what=f"(no charter registered for {voice.value})",
                             how=[], why=[], not_do=[], accountability=[], tools=[])
    tags = _classify(question)
    how = list(c.how) if "how" in tags else []
    why = list(c.why) if "why" in tags else []
    not_do = list(c.not_do)
    accountability = list(c.accountability)
    tools = list(c.tools)
    refs = list(recent_ledger_refs or [])
    peers_map: dict[str, dict[str, object]] = {}
    for pv in peers or ():
        pc = get_charter(pv)
        if pc is not None and pc.peers_readable:
            peers_map[pv.value] = {
                "domain": pc.domain.value,
                "what": pc.what,
                "tools": list(pc.tools),
            }
    return Introspection(
        voice=voice, question=question,
        what=c.what, how=how, why=why, not_do=not_do,
        accountability=accountability, tools=tools,
        ledger_refs=refs, peers=peers_map,
    )


__all__ = ["Introspection", "introspect"]
