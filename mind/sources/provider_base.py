"""
mind.sources.provider_base — common interface for every external API.

Each provider declares:
    name             canonical provider id (e.g. "binance", "cryptopanic")
    kind             MARKET | NEWS | SENTIMENT | ONCHAIN
    api_key_env      env-var name (or "" if key-free)
    api_key_required True if provider does not work without a key
    rate_limit_rps   max requests / second
    poll()           returns list of normalized domain objects

Missing-but-required key is NOT fatal: the provider just reports
``enabled=False`` and is skipped by the registry. This keeps the system
bootable on a fresh install with zero API keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.secrets import get_secret
from mind.sources.rate_limiter import TokenBucket
from mind.sources.source_types import SourceKind


@dataclass
class ProviderStatus:
    name: str
    kind: str
    enabled: bool
    has_key: bool
    last_poll_ok: bool = False
    last_poll_count: int = 0
    last_error: str = ""


class Provider:
    name: str = "base"
    kind: SourceKind = SourceKind.REST
    api_key_env: str = ""
    api_key_required: bool = False
    rate_limit_rps: float = 5.0
    rate_limit_burst: float = 10.0

    def __init__(self) -> None:
        self._bucket = TokenBucket(rate_per_sec=self.rate_limit_rps, capacity=self.rate_limit_burst)
        self._status = ProviderStatus(
            name=self.name, kind=self.kind.value,
            enabled=self._enabled(), has_key=bool(self._api_key()),
        )

    def _api_key(self) -> str:
        if not self.api_key_env:
            return ""
        try:
            return get_secret(self.api_key_env, default="")
        except Exception:
            return ""

    def _enabled(self) -> bool:
        if self.api_key_required and not self._api_key():
            return False
        return True

    def enabled(self) -> bool:
        return self._enabled()

    def status(self) -> ProviderStatus:
        self._status.enabled = self._enabled()
        self._status.has_key = bool(self._api_key())
        return self._status

    def poll(self) -> list[Any]:
        return []

    def _mark_ok(self, count: int) -> None:
        self._status.last_poll_ok = True
        self._status.last_poll_count = count
        self._status.last_error = ""

    def _mark_err(self, err: str) -> None:
        self._status.last_poll_ok = False
        self._status.last_error = err[:200]
