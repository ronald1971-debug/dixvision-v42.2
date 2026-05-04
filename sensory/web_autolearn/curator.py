"""WEBLEARN-03 — Curator.

The curator is the second filter in the pipeline. Where the AIFilter
(WEBLEARN-02) decides *relevance* (does the document mention the
seed's topic at all), the curator decides *admissibility* — whether
the operator wants this seed to surface items above a configurable
score threshold, with seed-specific allow/deny tags applied on top.

Architecturally the curator is a pure function; the only state it
carries is the :class:`CuratorRules` value loaded from
:file:`seeds.yaml` (WEBLEARN-10). All decisions are deterministic
given the same rules + same input.

Authority discipline: no engine imports, no FSM mutation, no ledger
writes. The curator's only output is a sequence of
:class:`CuratedItem` instances destined for the
:class:`PendingBuffer` HITL gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sensory.web_autolearn.contracts import (
    CuratedItem,
    FilteredItem,
)


@dataclass(frozen=True, slots=True)
class _SeedRule:
    """One seed row's curator rule.

    Attributes:
        topic: Topic label carried into :class:`CuratedItem.seed_topic`
            (e.g. ``"crypto"``, ``"macro"``). Empty string is rejected.
        min_score: Minimum filter score for this seed to admit. Items
            below are dropped at the curator (not at the filter).
        allow_substrings: If non-empty, the document title+body must
            contain at least one substring (case-insensitive) for the
            item to admit. Empty tuple = no allow filter.
        deny_substrings: If any substring is present (case-insensitive)
            in title+body the item is dropped. Empty tuple = no deny
            filter.
        tags: Carry-through tags applied to admitted items.
    """

    topic: str
    min_score: float = 0.0
    allow_substrings: tuple[str, ...] = ()
    deny_substrings: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("seed rule topic must be non-empty")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                "seed rule min_score must be in [0.0, 1.0]"
            )


@dataclass(frozen=True, slots=True)
class CuratorRules:
    """Aggregate of all seed-level curator rules, keyed by ``seed_id``.

    Build with :meth:`from_mapping` or directly from a parsed YAML
    document.
    """

    rules: Mapping[str, _SeedRule] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Mapping[str, object]],
    ) -> CuratorRules:
        """Build :class:`CuratorRules` from a parsed YAML mapping.

        Expected shape::

            seed_a:
              topic: crypto
              min_score: 0.4
              allow: ["bitcoin", "btc"]
              deny: ["sponsored"]
              tags: ["spot", "majors"]

        ``allow`` / ``deny`` / ``tags`` default to empty.
        ``min_score`` defaults to 0.0.
        """

        out: dict[str, _SeedRule] = {}
        for seed_id, body in raw.items():
            if not isinstance(body, Mapping):
                raise ValueError(
                    f"seed {seed_id!r} body must be a mapping"
                )
            topic_raw = body.get("topic", "")
            if not isinstance(topic_raw, str):
                raise ValueError(
                    f"seed {seed_id!r} topic must be a string"
                )
            min_score_raw = body.get("min_score", 0.0)
            if not isinstance(min_score_raw, (int, float)):
                raise ValueError(
                    f"seed {seed_id!r} min_score must be a number"
                )
            allow_raw = body.get("allow", ()) or ()
            deny_raw = body.get("deny", ()) or ()
            tags_raw = body.get("tags", ()) or ()
            allow_tuple = tuple(str(s) for s in allow_raw)
            deny_tuple = tuple(str(s) for s in deny_raw)
            tags_tuple = tuple(str(s) for s in tags_raw)
            out[str(seed_id)] = _SeedRule(
                topic=topic_raw,
                min_score=float(min_score_raw),
                allow_substrings=allow_tuple,
                deny_substrings=deny_tuple,
                tags=tags_tuple,
            )
        return cls(rules=out)


@dataclass(frozen=True, slots=True)
class Curator:
    """Apply :class:`CuratorRules` to a sequence of :class:`FilteredItem`.

    The curator is intentionally synchronous and pure — :meth:`curate`
    returns a fresh tuple of :class:`CuratedItem` for every call. It
    never accumulates state; the :class:`PendingBuffer` is the only
    component in the pipeline allowed to retain items across calls.
    """

    rules: CuratorRules

    def curate(
        self,
        items: Sequence[FilteredItem],
    ) -> tuple[CuratedItem, ...]:
        """Return only the items the operator wants to review.

        Drops in order of evaluation:

          1. Item's ``seed_id`` has no rule -> drop (unknown seed).
          2. Item's ``score`` < ``rule.min_score`` -> drop.
          3. Any of ``rule.deny_substrings`` present -> drop.
          4. ``rule.allow_substrings`` non-empty AND none present
             -> drop.

        Otherwise the item is promoted to :class:`CuratedItem` with
        the rule's ``topic`` and ``tags``.
        """

        out: list[CuratedItem] = []
        for item in items:
            rule = self.rules.rules.get(item.seed_id)
            if rule is None:
                continue
            if item.score < rule.min_score:
                continue
            haystack = f"{item.title} {item.body}".lower()
            if any(
                d.lower() in haystack
                for d in rule.deny_substrings
                if d
            ):
                continue
            if rule.allow_substrings:
                if not any(
                    a.lower() in haystack
                    for a in rule.allow_substrings
                    if a
                ):
                    continue
            tags = tuple(
                sorted({t for t in rule.tags if t})
            )
            out.append(
                CuratedItem(
                    ts_ns=item.ts_ns,
                    seed_id=item.seed_id,
                    url=item.url,
                    title=item.title,
                    body=item.body,
                    score=item.score,
                    seed_topic=rule.topic,
                    curator_tags=tags,
                    meta=dict(item.meta),
                )
            )
        return tuple(out)


__all__ = [
    "Curator",
    "CuratorRules",
]
