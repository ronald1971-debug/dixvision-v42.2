"""Adapter registry — single source of truth for live-adapter status.

The :class:`AdapterRouter` already owns the (domain, venue) → adapter
mapping needed to *route* an approved signal. The registry adds a
parallel introspection surface so the operator dashboard can answer
"which adapters are live, which are scaffolds, which are halted".

Two pieces:

1. :class:`AdapterRegistry` — keeps a list of ``LiveAdapterBase``
   instances and exposes ``snapshot()`` returning ``AdapterStatus``
   tuples for each.
2. :func:`default_registry()` — returns a process-wide singleton
   pre-populated with the three D-track scaffolds (Hummingbot,
   PumpFun, UniswapX). The defaults are *all in DISCONNECTED state* —
   no credential is read from the environment until the operator
   explicitly invokes ``adapter.connect()``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from execution_engine.adapters._live_base import (
    AdapterStatus,
    LiveAdapterBase,
)
from execution_engine.adapters.hummingbot import HummingbotAdapter
from execution_engine.adapters.pumpfun import PumpFunAdapter

_log = logging.getLogger(__name__)


class AdapterRegistry:
    """Process-wide registry of live adapters."""

    def __init__(self, adapters: Iterable[LiveAdapterBase] = ()) -> None:
        self._adapters: list[LiveAdapterBase] = list(adapters)

    def add(self, adapter: LiveAdapterBase) -> None:
        for existing in self._adapters:
            if existing.name == adapter.name:
                raise ValueError(
                    f"adapter already registered: {adapter.name}"
                )
        self._adapters.append(adapter)

    def get(self, name: str) -> LiveAdapterBase | None:
        for a in self._adapters:
            if a.name == name:
                return a
        return None

    def snapshot(self) -> tuple[AdapterStatus, ...]:
        return tuple(a.status() for a in self._adapters)

    def __len__(self) -> int:
        return len(self._adapters)


_DEFAULT: AdapterRegistry | None = None


def default_registry() -> AdapterRegistry:
    """Return the process-wide singleton registry.

    Pre-populated with three scaffolds so the operator dashboard has
    something to render before any credentials are wired. All three
    start in ``DISCONNECTED`` — no environment is read.
    """
    global _DEFAULT
    if _DEFAULT is None:
        reg = AdapterRegistry()
        reg.add(HummingbotAdapter(connector="paper"))
        reg.add(PumpFunAdapter())
        # UniswapX needs eth-account for EIP-712 signing. That dep lives in
        # the optional ``[evm]`` / ``[dev]`` extras so the base launcher can
        # boot without it. Skip the adapter cleanly if eth-account is not
        # installed; the operator can install it on demand to surface the
        # adapter in the dashboard registry.
        try:
            from execution_engine.adapters.uniswapx import UniswapXAdapter
        except ImportError as exc:  # pragma: no cover - exercised by hotfix test
            _log.warning(
                "UniswapX adapter skipped from default registry "
                "(missing dependency: %s). Install with `pip install -e '.[evm]'` "
                "to enable.",
                exc.name or exc,
            )
        else:
            reg.add(UniswapXAdapter())
        _DEFAULT = reg
    return _DEFAULT


__all__ = ["AdapterRegistry", "default_registry"]
