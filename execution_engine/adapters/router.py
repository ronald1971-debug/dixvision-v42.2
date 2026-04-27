"""Adapter router — EXEC-01 / Phase 2.

Routes a :class:`SignalEvent` to the correct :class:`BrokerAdapter` by
domain (NORMAL / COPY_TRADING / MEMECOIN) and venue. Pure dispatch:
no IO, no allocations beyond a dict lookup.

The router is the boundary that enforces *hard 3-domain isolation*
declared in the Build Compiler Spec §7 / docs/directory_tree.md
``execution_engine/domains/``: a memecoin signal can only ever reach a
memecoin-bound adapter, never a NORMAL adapter — and vice versa.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from core.contracts.events import SignalEvent
from execution_engine.adapters.base import BrokerAdapter


class TradingDomain(StrEnum):
    NORMAL = "NORMAL"
    COPY_TRADING = "COPY_TRADING"
    MEMECOIN = "MEMECOIN"


class RouterError(LookupError):
    """No adapter registered for the requested key."""


class AdapterRouter:
    """Domain + venue → adapter lookup.

    Args:
        adapters: Pre-registered ``{(domain, venue): adapter}`` mapping.
        default_domain: Domain when ``signal.meta`` doesn't carry one.

    Hard isolation guarantees:

    * MEMECOIN signals are never returned a non-MEMECOIN adapter.
    * NORMAL signals are never returned a MEMECOIN adapter.
    * COPY_TRADING signals are never returned a NORMAL or MEMECOIN
      adapter.
    """

    name: str = "adapter_router"
    spec_id: str = "EXEC-01"

    def __init__(
        self,
        adapters: (
            Mapping[tuple[TradingDomain, str], BrokerAdapter] | None
        ) = None,
        *,
        default_domain: TradingDomain = TradingDomain.NORMAL,
    ) -> None:
        self._adapters: dict[tuple[TradingDomain, str], BrokerAdapter] = (
            dict(adapters or {})
        )
        self._default_domain = default_domain

    # -- registration ------------------------------------------------------

    def register(
        self,
        *,
        domain: TradingDomain,
        venue: str,
        adapter: BrokerAdapter,
    ) -> None:
        if not venue:
            raise ValueError("venue required")
        key = (domain, venue)
        if key in self._adapters:
            raise ValueError(f"adapter already registered: {domain.name}/{venue}")
        self._adapters[key] = adapter

    def venues(self, domain: TradingDomain) -> tuple[str, ...]:
        return tuple(
            sorted(v for (d, v) in self._adapters if d == domain)
        )

    def __len__(self) -> int:
        return len(self._adapters)

    # -- lookup ------------------------------------------------------------

    def domain_of(self, signal: SignalEvent) -> TradingDomain:
        raw = signal.meta.get("domain")
        if raw is None:
            return self._default_domain
        normalized = raw.upper()
        try:
            return TradingDomain(normalized)
        except ValueError as exc:
            raise RouterError(f"unknown domain: {raw!r}") from exc

    def adapter_for(
        self,
        signal: SignalEvent,
        *,
        venue: str | None = None,
    ) -> BrokerAdapter:
        domain = self.domain_of(signal)
        chosen_venue = venue or signal.meta.get("venue")
        if not chosen_venue:
            raise RouterError(
                "venue required (pass venue=... or signal.meta['venue'])"
            )
        key = (domain, chosen_venue)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise RouterError(
                f"no adapter for {domain.name}/{chosen_venue}"
            )
        return adapter


__all__ = ["AdapterRouter", "RouterError", "TradingDomain"]
