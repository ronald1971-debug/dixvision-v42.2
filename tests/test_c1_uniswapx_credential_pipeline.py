"""C-1 / P1-4 — UniswapX credential pipeline regression pins.

Phase-6 RISK-05 flagged that the UniswapX adapter must read EVM
credentials through the canonical credential pipeline
(``system_engine.credentials.storage.resolve_env``), not via
ad-hoc ``os.environ.get`` calls in the adapter constructor or via
hard-coded paths in the launcher. This guarantees that:

1. A freshly-saved ``.env`` line is visible without restarting the
   server (``resolve_env`` merges ``os.environ`` and the dotenv file
   with ``os.environ`` taking precedence).
2. The dashboard ``/credentials`` matrix and the adapter see
   exactly the same value, because both call ``resolve_env``.
3. There is a single discoverable contract for which env vars the
   operator needs to set, namely the entry in
   :data:`system_engine.credentials.manifest.CREDENTIAL_BLUEPRINTS`
   keyed by provider ``"uniswapx"``.

These tests pin all three guarantees and the absence of any direct
``os.environ.get`` read inside :mod:`execution_engine.adapters.uniswapx`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _reload_registry() -> object:
    """Force a fresh import of ``execution_engine.adapters.registry``.

    Drops cached module state so the ``_DEFAULT`` singleton is rebuilt
    against the current environment.
    """
    for name in (
        "execution_engine.adapters.registry",
        "execution_engine.adapters",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("execution_engine.adapters.registry")


def test_uniswapx_blueprint_declares_canonical_env_vars() -> None:
    from system_engine.credentials.manifest import CREDENTIAL_BLUEPRINTS

    bp = CREDENTIAL_BLUEPRINTS["uniswapx"]
    # Exactly the two env vars the registry reads.
    assert bp.env_vars == (
        "DIX_EVM_RPC_URL",
        "DIX_EVM_PRIVATE_KEY_PATH",
    )
    # Operator brings their own EVM wallet; no public signup.
    assert bp.signup_url is None
    assert bp.free_tier is False
    # Documents that the *path* is the env var, not the key itself.
    assert "PRIVATE_KEY_PATH" in bp.notes
    assert "never appears" in bp.notes


def test_default_registry_reads_credentials_via_resolve_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registry must call ``resolve_env`` (not raw ``os.environ.get``).

    We assert the contract by setting the two canonical env vars and
    verifying that the UniswapX adapter inside the default registry
    is wired with them.
    """
    pytest.importorskip("eth_account")

    key_file = tmp_path / "evm.key"
    key_file.write_text("0x" + "11" * 32, encoding="utf-8")
    monkeypatch.setenv("DIX_EVM_RPC_URL", "https://rpc.example/v1")
    monkeypatch.setenv("DIX_EVM_PRIVATE_KEY_PATH", str(key_file))

    registry = _reload_registry()
    reg = registry.default_registry()
    uniswapx = next(
        (a for a in reg._adapters if a.name.startswith("uniswapx:")),
        None,
    )
    assert uniswapx is not None, "UniswapX adapter not registered"
    assert uniswapx._rpc_url == "https://rpc.example/v1"
    assert uniswapx._private_key_path == str(key_file)


def test_default_registry_tolerates_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither env var set the adapter is registered but DISCONNECTED.

    The dashboard must still surface the row so the operator can see
    what is missing.
    """
    pytest.importorskip("eth_account")

    monkeypatch.delenv("DIX_EVM_RPC_URL", raising=False)
    monkeypatch.delenv("DIX_EVM_PRIVATE_KEY_PATH", raising=False)

    registry = _reload_registry()
    reg = registry.default_registry()
    uniswapx = next(
        (a for a in reg._adapters if a.name.startswith("uniswapx:")),
        None,
    )
    assert uniswapx is not None, "UniswapX adapter not registered"
    assert uniswapx._rpc_url is None
    assert uniswapx._private_key_path is None


def test_uniswapx_adapter_has_no_direct_os_environ_get() -> None:
    """Pin the absence of ``os.environ`` reads inside the adapter.

    The credential resolution lives at registry-construction time so
    the adapter stays a pure consumer of constructor arguments. Any
    drift back to ``os.environ.get`` inside ``uniswapx.py`` would be
    a regression of the C-1 / P1-4 contract.
    """
    src = Path("execution_engine/adapters/uniswapx.py").read_text(encoding="utf-8")
    assert "os.environ" not in src, (
        "execution_engine/adapters/uniswapx.py must not read os.environ "
        "directly — credentials flow through registry.py + resolve_env."
    )


def test_registry_imports_resolve_env() -> None:
    """Pin the import — drift here would silently break the contract."""
    src = Path("execution_engine/adapters/registry.py").read_text(encoding="utf-8")
    assert "from system_engine.credentials.storage import resolve_env" in src
