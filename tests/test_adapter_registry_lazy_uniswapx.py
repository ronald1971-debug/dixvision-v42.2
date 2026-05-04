"""Regression test: adapter registry tolerates missing ``eth-account``.

The Windows launcher only installs base runtime deps. ``eth-account``
lives in the optional ``[evm]``/``[dev]`` extras. Before this hotfix,
``execution_engine.adapters.__init__`` and ``...registry`` eagerly
imported :class:`UniswapXAdapter`, which transitively imported
``eth_account``. A user without the extras installed therefore
crashed at server boot with ``ModuleNotFoundError: No module named
'eth_account'``.

These tests verify the lazy-import fallback in:

* :mod:`execution_engine.adapters.__init__` — sets ``UniswapXAdapter``
  to ``None`` on ``ImportError`` rather than re-raising;
* :mod:`execution_engine.adapters.registry` — skips registering
  UniswapX in :func:`default_registry` when its import chain is
  broken, while still registering the always-available
  :class:`HummingbotAdapter` and :class:`PumpFunAdapter`.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Callable

import pytest


def _force_eth_account_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make ``import eth_account`` (and submodules) raise ImportError."""
    real_import = builtins.__import__

    def _raise_for_eth_account(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "eth_account" or name.startswith("eth_account."):
            raise ModuleNotFoundError(
                "No module named 'eth_account'", name="eth_account"
            )
        return real_import(name, globals_, locals_, fromlist, level)

    # Drop any cached eth_account modules so the next import attempt
    # actually invokes ``builtins.__import__`` (which we just patched).
    for cached in list(sys.modules):
        if cached == "eth_account" or cached.startswith("eth_account."):
            monkeypatch.delitem(sys.modules, cached, raising=False)

    # Drop adapters cached around eth_account so they re-import too.
    for cached in (
        "execution_engine.adapters",
        "execution_engine.adapters.uniswapx",
        "execution_engine.adapters._uniswapx_signer",
        "execution_engine.adapters._uniswapx_quote",
        "execution_engine.adapters.registry",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    monkeypatch.setattr(builtins, "__import__", _raise_for_eth_account)


def _reload_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Callable[[], object]]:
    """Reload the adapters package under the patched importer."""
    _force_eth_account_missing(monkeypatch)
    adapters = importlib.import_module("execution_engine.adapters")
    return adapters, adapters.default_registry  # type: ignore[attr-defined]


def test_adapters_module_imports_without_eth_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters, _ = _reload_adapters(monkeypatch)
    # Must not raise. The public ``UniswapXAdapter`` symbol is still
    # exported (per ``__all__``), but is ``None`` so callers can
    # detect the absence without an AttributeError.
    assert adapters.UniswapXAdapter is None  # type: ignore[attr-defined]
    # Other adapters remain importable.
    assert adapters.PaperBroker is not None  # type: ignore[attr-defined]
    assert adapters.HummingbotAdapter is not None  # type: ignore[attr-defined]
    assert adapters.PumpFunAdapter is not None  # type: ignore[attr-defined]


def test_default_registry_skips_uniswapx_without_eth_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, default_registry = _reload_adapters(monkeypatch)
    reg = default_registry()
    snap = reg.snapshot()
    names = {s.name for s in snap}
    # Always-available adapters are still registered.
    assert "hummingbot:paper" in names
    assert "pumpfun" in names
    # UniswapX is skipped — not registered, not crashed.
    assert not any(n.startswith("uniswapx:") for n in names)
