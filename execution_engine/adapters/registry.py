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

C-1 / P1-4 — credential pipeline. UniswapX needs an EVM private key
to sign EIP-712 Permit2 witnesses. The key location and the JSON-RPC
endpoint are looked up via :func:`system_engine.credentials.storage.resolve_env`
(``os.environ`` precedence then ``.env``), not via raw
``os.environ.get`` calls. This keeps the read shim consistent with
every other credential in the system (Reuters, FRED, BLS, GitHub,
Glassnode, Dune, ...) and ensures the dashboard ``/credentials``
surface and the audit ledger see the same value the adapter sees.
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
from system_engine.credentials.storage import resolve_env

_log = logging.getLogger(__name__)


class AdapterRegistry:
    """Process-wide registry of live adapters."""

    def __init__(self, adapters: Iterable[LiveAdapterBase] = ()) -> None:
        self._adapters: list[LiveAdapterBase] = list(adapters)

    def add(self, adapter: LiveAdapterBase) -> None:
        for existing in self._adapters:
            if existing.name == adapter.name:
                raise ValueError(f"adapter already registered: {adapter.name}")
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
            # C-1 / P1-4 — read EVM credentials through the canonical
            # ``resolve_env`` shim (os.environ > .env > None) so a
            # freshly-saved ``.env`` line is visible without a server
            # restart, and so the dashboard credential view reflects
            # exactly what the adapter sees. Both keys are optional;
            # ``UniswapXAdapter.connect`` already handles ``None`` by
            # staying in DISCONNECTED scaffold mode.
            env = resolve_env()
            rpc_url = env.get("DIX_EVM_RPC_URL") or None
            private_key_path = env.get("DIX_EVM_PRIVATE_KEY_PATH") or None
            reg.add(
                UniswapXAdapter(
                    rpc_url=rpc_url,
                    private_key_path=private_key_path,
                )
            )
        _DEFAULT = reg
    return _DEFAULT


__all__ = ["AdapterRegistry", "default_registry"]
