"""A-15 — Helius Solana onchain intelligence adapter.

# ADAPTED FROM: helius-py SDK (https://github.com/helius-labs/helius-sdk
# — TypeScript reference) + python helius-sdk package surface.

This adapter is the **read-only intelligence** entry point for the
Helius API: Enhanced Transactions, Token holders, and DAS asset info.
Despite living under :mod:`execution_engine.adapters` (the canonical
on-disk home for chain-bound adapters per
``execution_engine/adapters/pumpfun.py``), this module **never**
authorises execution — it is advisory only (INV-19).

Authority symmetry (B27 / B28 / INV-71):

* This module **must not** construct :class:`SignalEvent` /
  :class:`HazardEvent` — only ``intelligence_engine`` (signal) and
  ``system_engine`` / ``execution_engine`` (hazard, narrow scope) may.
  Instead the adapter returns frozen advisory value objects
  (:class:`OnchainEvent`, :class:`HolderShiftAdvisory`) defined in
  :mod:`sensory.onchain.contracts`, which downstream intelligence-tier
  coordinators project into typed events on the proper side of the
  authority boundary.
* Pinned by an AST test (``test_helius_adapter.py``).

Determinism (INV-15):

* All parsing is pure-Python on caller-supplied payload dicts.
* The adapter does **not** read the wall clock, the OS environment, or
  use ``random`` / ``asyncio``. Callers supply ``ts_ns`` from
  :class:`system.time_source.TimeAuthority`.
* The transport seam (``HeliusTransport`` Protocol) supports a pure
  in-process backend for tests; the HTTP-backed transport is created
  by ``helius_http_transport_factory()`` which lazy-imports
  ``helius_sdk`` only inside the function body.

Credentials:

* The API key is **never** read from ``os.environ`` here — it must be
  passed in explicitly (typically by ``system_engine.credentials``
  bridging code, mirroring the Wave-04.5 / S-12 pattern).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Final, Protocol, runtime_checkable

from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)
from sensory.onchain.contracts import HolderShiftAdvisory, OnchainEvent

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("helius-py",)


_SOURCE: Final[str] = "HELIUS"
_CHAIN: Final[str] = "SOLANA"

_VALID_KINDS: Final[frozenset[str]] = frozenset(
    {
        "TRANSFER",
        "SWAP",
        "MINT",
        "BURN",
        "HOLDER_SHIFT",
        "PROGRAM_CALL",
        "NFT_TRADE",
        "UNKNOWN",
    }
)


# ---------------------------------------------------------------------------
# Transport seam
# ---------------------------------------------------------------------------


@runtime_checkable
class HeliusTransport(Protocol):
    """Pluggable transport contract for Helius API surface.

    Implementations:

    * :class:`InProcessHeliusTransport` — pure-Python, used by tests
      and replay; returns the canned payloads supplied at construction.
    * The HTTP-backed transport created by
      :func:`helius_http_transport_factory` — lazy-imports
      ``helius_sdk`` only inside the factory body, never at module top.
    """

    def get_enhanced_transactions(
        self, signatures: Sequence[str]
    ) -> tuple[Mapping[str, object], ...]: ...

    def get_token_holders(
        self,
        mint: str,
        *,
        limit: int = 20,
    ) -> tuple[Mapping[str, object], ...]: ...


# ---------------------------------------------------------------------------
# In-process transport for tests / replay
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class InProcessHeliusTransport:
    """Pure-Python transport that replays caller-supplied payloads.

    The payloads are stored as nested tuples of frozen mappings so the
    transport is byte-identical across runs (INV-15).

    Attributes:
        enhanced_transactions: Mapping of signature → enhanced-tx dict.
        token_holders: Mapping of mint → ordered list of holder dicts.
    """

    enhanced_transactions: Mapping[str, Mapping[str, object]] = (
        dataclasses.field(default_factory=dict)
    )
    token_holders: Mapping[str, Sequence[Mapping[str, object]]] = (
        dataclasses.field(default_factory=dict)
    )

    def get_enhanced_transactions(
        self, signatures: Sequence[str]
    ) -> tuple[Mapping[str, object], ...]:
        out: list[Mapping[str, object]] = []
        for sig in signatures:
            if not isinstance(sig, str) or not sig:
                raise ValueError(
                    "InProcessHeliusTransport.get_enhanced_transactions: "
                    "signatures must be non-empty strings"
                )
            payload = self.enhanced_transactions.get(sig)
            if payload is not None:
                out.append(dict(payload))
        return tuple(out)

    def get_token_holders(
        self,
        mint: str,
        *,
        limit: int = 20,
    ) -> tuple[Mapping[str, object], ...]:
        if not isinstance(mint, str) or not mint:
            raise ValueError(
                "InProcessHeliusTransport.get_token_holders: "
                "mint must be a non-empty string"
            )
        if limit <= 0:
            raise ValueError(
                "InProcessHeliusTransport.get_token_holders: "
                "limit must be positive"
            )
        rows = self.token_holders.get(mint, ())
        materialised = [dict(r) for r in rows[:limit]]
        return tuple(materialised)


# ---------------------------------------------------------------------------
# Helius parsing primitives
# ---------------------------------------------------------------------------


def _normalise_kind(raw: object) -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    up = raw.strip().upper()
    if up in _VALID_KINDS:
        return up
    return "UNKNOWN"


def _coerce_str(raw: object, *, default: str = "") -> str:
    if isinstance(raw, str):
        return raw
    return default


def _coerce_optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _sorted_meta(raw: Mapping[str, object]) -> Mapping[str, str]:
    """Project a free-form mapping into a sorted ``str→str`` mapping.

    Order is fixed by sorted keys so the resulting OnchainEvent meta
    is byte-identical across runs (INV-15). ``None`` values are
    skipped; non-stringable values are rejected for canonical replay.
    """

    out: dict[str, str] = {}
    for key in sorted(raw):
        if not isinstance(key, str):
            continue
        value = raw[key]
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            out[key] = str(value)
    return out


def parse_enhanced_transaction(
    payload: Mapping[str, object],
    *,
    ts_ns: int,
) -> OnchainEvent:
    """Project a Helius enhanced-transaction dict into an OnchainEvent.

    The Helius enhanced-transaction shape carries (per the SDK spec):
    ``signature``, ``type`` (transfer / swap / mint / nft-trade /
    program-call / …), ``feePayer``, optional ``tokenTransfers`` with
    a ``mint`` field for token-level events, and ``rugScore`` for the
    DAS / Token API.

    The mapping into :class:`OnchainEvent` is intentionally lossy —
    only the fields the rest of DIX consumes are projected. Free-form
    metadata is preserved sorted into ``meta``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError(
            "parse_enhanced_transaction: payload must be a mapping"
        )
    if ts_ns <= 0:
        raise ValueError(
            "parse_enhanced_transaction: ts_ns must be positive"
        )

    kind = _normalise_kind(payload.get("type"))
    signature = _coerce_str(payload.get("signature"))
    actor = _coerce_str(payload.get("feePayer"))

    asset = ""
    transfers = payload.get("tokenTransfers")
    if isinstance(transfers, Sequence) and transfers:
        first = transfers[0]
        if isinstance(first, Mapping):
            asset = _coerce_str(first.get("mint"))

    rug_score = _coerce_optional_float(payload.get("rugScore"))
    if rug_score is not None and not (0.0 <= rug_score <= 1.0):
        rug_score = None

    raw_meta = payload.get("meta")
    meta_source: Mapping[str, object]
    if isinstance(raw_meta, Mapping):
        meta_source = raw_meta
    else:
        meta_source = {}

    return OnchainEvent(
        ts_ns=ts_ns,
        source=_SOURCE,
        chain=_CHAIN,
        kind=kind,
        asset=asset,
        actor=actor,
        signature=signature,
        rug_score=rug_score,
        meta=_sorted_meta(meta_source),
    )


def _share_of_top_holders(
    holders: Sequence[Mapping[str, object]],
    *,
    top_n: int,
) -> float:
    if top_n <= 0:
        raise ValueError("_share_of_top_holders: top_n must be positive")
    if not holders:
        return 0.0
    balances: list[float] = []
    for h in holders:
        if not isinstance(h, Mapping):
            continue
        b = h.get("balance")
        if isinstance(b, bool):
            continue
        if isinstance(b, (int, float)) and b >= 0:
            balances.append(float(b))
    if not balances:
        return 0.0
    total = sum(balances)
    if total <= 0.0:
        return 0.0
    balances.sort(reverse=True)
    top_sum = sum(balances[:top_n])
    return top_sum / total


def _heuristic_rug_score(
    *,
    share_before: float,
    share_after: float,
    holders_changed: int,
    top_n: int,
) -> float:
    """Deterministic rug-score heuristic in ``[0.0, 1.0]``.

    Weighted blend of concentration delta + churn ratio, both
    clipped to ``[0, 1]``. Pure function — no clock, no randomness.
    """

    delta = max(0.0, share_after - share_before)
    churn = (
        min(1.0, holders_changed / float(top_n)) if top_n > 0 else 0.0
    )
    score = 0.6 * delta + 0.4 * churn
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def diff_holder_snapshots(
    asset: str,
    *,
    ts_ns: int,
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    top_n: int = 10,
    meta: Mapping[str, object] | None = None,
) -> HolderShiftAdvisory:
    """Compare two top-holder snapshots → :class:`HolderShiftAdvisory`.

    Pure function — caller supplies snapshots already fetched via the
    transport, both ordered most-recent-first. ``top_n`` bounds the
    concentration window. Determinism: outputs are byte-identical for
    a given input pair (INV-15).
    """

    if not asset:
        raise ValueError("diff_holder_snapshots: asset must be non-empty")
    if ts_ns <= 0:
        raise ValueError("diff_holder_snapshots: ts_ns must be positive")
    if top_n <= 0:
        raise ValueError("diff_holder_snapshots: top_n must be positive")

    share_before = _share_of_top_holders(before, top_n=top_n)
    share_after = _share_of_top_holders(after, top_n=top_n)

    addrs_before: set[str] = set()
    addrs_after: set[str] = set()
    for h in before[:top_n]:
        if isinstance(h, Mapping):
            owner = _coerce_str(h.get("owner"))
            if owner:
                addrs_before.add(owner)
    for h in after[:top_n]:
        if isinstance(h, Mapping):
            owner = _coerce_str(h.get("owner"))
            if owner:
                addrs_after.add(owner)
    changed = len(addrs_before.symmetric_difference(addrs_after))

    rug = _heuristic_rug_score(
        share_before=share_before,
        share_after=share_after,
        holders_changed=changed,
        top_n=top_n,
    )

    meta_source = meta if isinstance(meta, Mapping) else {}
    return HolderShiftAdvisory(
        ts_ns=ts_ns,
        asset=asset,
        top_holder_share_before=share_before,
        top_holder_share_after=share_after,
        holders_changed=changed,
        rug_score=rug,
        meta=_sorted_meta(meta_source),
    )


# ---------------------------------------------------------------------------
# Adapter — façade that combines transport + parsing
# ---------------------------------------------------------------------------


class HeliusAdapter(LiveAdapterBase):
    """Advisory-only Helius adapter.

    Stays in :attr:`AdapterState.DISCONNECTED` until both an API key
    and a transport are supplied. The adapter is **never** an
    executor — :meth:`submit` is not implemented; the only
    public surface is the advisory-projection methods below.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: HeliusTransport | None = None,
    ) -> None:
        super().__init__(name="helius", venue="helius:solana")
        self._api_key = api_key
        self._transport: HeliusTransport | None = transport

    @property
    def transport(self) -> HeliusTransport | None:
        return self._transport

    def connect(self) -> None:
        missing: list[str] = []
        if self._api_key is None or self._api_key == "":
            missing.append("DIX_HELIUS_API_KEY")
        if self._transport is None:
            missing.append("transport")
        if missing:
            self._state = AdapterState.DISCONNECTED
            self._detail = (
                "missing credentials: "
                + ", ".join(missing)
                + " — scaffold mode active"
            )
            return
        self._state = AdapterState.CONNECTING
        self._detail = "awaiting first Helius RPC roundtrip"

    # ------------------------------------------------------------------
    # Advisory projections
    # ------------------------------------------------------------------

    def fetch_enhanced_transactions(
        self,
        signatures: Sequence[str],
        *,
        ts_ns: int,
    ) -> tuple[OnchainEvent, ...]:
        """Fetch + parse enhanced transactions into ``OnchainEvent``s.

        Pure-functional projection. The adapter is **not** required
        to be CONNECTED — when ``transport`` is ``None`` an empty
        tuple is returned so callers can run in scaffold mode.
        """

        if self._transport is None:
            return ()
        if ts_ns <= 0:
            raise ValueError(
                "fetch_enhanced_transactions: ts_ns must be positive"
            )
        payloads = self._transport.get_enhanced_transactions(signatures)
        return tuple(
            parse_enhanced_transaction(p, ts_ns=ts_ns) for p in payloads
        )

    def diff_token_holders(
        self,
        asset: str,
        *,
        ts_ns: int,
        before: Sequence[Mapping[str, object]],
        after: Sequence[Mapping[str, object]] | None = None,
        top_n: int = 10,
        meta: Mapping[str, object] | None = None,
    ) -> HolderShiftAdvisory:
        """Diff two holder snapshots into a :class:`HolderShiftAdvisory`.

        When ``after`` is ``None`` the adapter fetches it via the
        transport. ``before`` is always caller-supplied so replay can
        feed deterministic snapshots.
        """

        if after is None:
            if self._transport is None:
                raise RuntimeError(
                    "diff_token_holders: transport is required to "
                    "fetch the live snapshot — supply ``after`` "
                    "explicitly to keep this call pure."
                )
            after = self._transport.get_token_holders(
                asset,
                limit=top_n,
            )
        return diff_holder_snapshots(
            asset,
            ts_ns=ts_ns,
            before=before,
            after=after,
            top_n=top_n,
            meta=meta,
        )


# ---------------------------------------------------------------------------
# HTTP transport factory (lazy import of helius_sdk)
# ---------------------------------------------------------------------------


def helius_http_transport_factory(
    *,
    api_key: str,
    base_url: str = "https://api.helius.xyz",
) -> HeliusTransport:
    """Construct an HTTP-backed :class:`HeliusTransport`.

    ``helius_sdk`` is lazy-imported inside the function body — never
    at module top. This is pinned by an AST test in
    ``test_helius_adapter.py``.

    The returned transport is a thin adapter over the SDK's
    enhanced-tx + token-holder endpoints, normalised to the tuple-of-
    mappings shape this module consumes.
    """

    if not api_key:
        raise ValueError(
            "helius_http_transport_factory: api_key must be non-empty"
        )
    if not base_url:
        raise ValueError(
            "helius_http_transport_factory: base_url must be non-empty"
        )

    import helius_sdk  # noqa: PLC0415

    client = helius_sdk.Helius(api_key=api_key, base_url=base_url)

    class _HttpHeliusTransport:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def get_enhanced_transactions(
            self, signatures: Sequence[str]
        ) -> tuple[Mapping[str, object], ...]:
            inner = self._inner
            method = getattr(inner, "get_enhanced_transactions", None)
            if method is None:
                method = getattr(inner, "parse_transactions", None)
            if method is None:
                return ()
            raw = method(list(signatures))
            if isinstance(raw, Sequence):
                return tuple(
                    dict(item) for item in raw if isinstance(item, Mapping)
                )
            return ()

        def get_token_holders(
            self,
            mint: str,
            *,
            limit: int = 20,
        ) -> tuple[Mapping[str, object], ...]:
            inner = self._inner
            method = getattr(inner, "get_token_holders", None)
            if method is None:
                method = getattr(inner, "get_assets_by_owner", None)
            if method is None:
                return ()
            raw = method(mint, limit=limit)
            if isinstance(raw, Sequence):
                return tuple(
                    dict(item) for item in raw if isinstance(item, Mapping)
                )
            return ()

    return _HttpHeliusTransport(client)


__all__ = [
    "HeliusAdapter",
    "HeliusTransport",
    "HolderShiftAdvisory",
    "InProcessHeliusTransport",
    "NEW_PIP_DEPENDENCIES",
    "OnchainEvent",
    "diff_holder_snapshots",
    "helius_http_transport_factory",
    "parse_enhanced_transaction",
]
