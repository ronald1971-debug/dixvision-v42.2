"""Loader for ``registry/external_signal_trust.yaml`` (Paper-S1).

Pure I/O on the registry filesystem; no clock, no PRNG, no
cross-engine import. Returns an immutable in-memory snapshot
(:class:`ExternalSignalTrustRegistry`) that the governance gate
reads to decide whether to apply a per-source cap on top of the
built-in default for the SignalEvent's trust class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from core.contracts.signal_trust import SignalTrust, default_cap_for

DEFAULT_REGISTRY_PATH = "registry/external_signal_trust.yaml"


@dataclass(frozen=True, slots=True)
class ExternalSignalSource:
    """One row of ``external_signal_trust.yaml``.

    Attributes:
        source_id: Stable identifier (e.g. ``"tradingview.public"``).
            Producers stamp this onto :attr:`SignalEvent.signal_source`.
        trust: Trust class for this source.
        cap: Per-source confidence cap; ``None`` means "use the
            built-in default for the trust class"
            (:func:`core.contracts.signal_trust.default_cap_for`).
        note: Free-form audit string; no behavioural effect.
    """

    source_id: str
    trust: SignalTrust
    cap: float | None
    note: str = ""


@dataclass(frozen=True, slots=True)
class ExternalSignalTrustRegistry:
    """Immutable snapshot of the on-disk registry."""

    version: int
    sources: Mapping[str, ExternalSignalSource]

    def cap_for(self, source_id: str, trust: SignalTrust) -> float | None:
        """Return the cap to apply for ``(source_id, trust)``.

        Falls back to the built-in default for the trust class when
        *source_id* has no explicit row, so an unregistered EXTERNAL_LOW
        producer is automatically clamped to
        :data:`core.contracts.signal_trust.DEFAULT_LOW_CAP` instead of
        riding the upstream confidence.
        """

        row = self.sources.get(source_id)
        if row is None:
            return default_cap_for(trust)
        # Honour the row's cap, but only if its declared trust class
        # matches the producer-declared trust. If they disagree we
        # take the *more restrictive* of the two — fail-closed.
        cap_from_row = row.cap if row.cap is not None else default_cap_for(row.trust)
        cap_from_class = default_cap_for(trust)
        if cap_from_row is None:
            return cap_from_class
        if cap_from_class is None:
            return cap_from_row
        return min(cap_from_row, cap_from_class)


def load_external_signal_trust(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ExternalSignalTrustRegistry:
    """Read and validate ``external_signal_trust.yaml`` at *path*."""

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping; got {type(raw).__name__}")
    version = raw.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"{path}: 'version' must be a positive int; got {version!r}")
    raw_sources = raw.get("sources", {}) or {}
    if not isinstance(raw_sources, dict):
        raise ValueError(f"{path}: 'sources' must be a mapping; got {type(raw_sources).__name__}")
    sources: dict[str, ExternalSignalSource] = {}
    for source_id, body in raw_sources.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path}: source_id must be a non-empty string")
        if not isinstance(body, dict):
            raise ValueError(f"{path}: source {source_id!r} body must be a mapping")
        trust_raw = body.get("trust")
        if not isinstance(trust_raw, str):
            raise ValueError(f"{path}: source {source_id!r}: 'trust' must be a string")
        try:
            trust = SignalTrust(trust_raw)
        except ValueError as exc:
            raise ValueError(
                f"{path}: source {source_id!r}: unknown trust class {trust_raw!r}"
            ) from exc
        cap_raw = body.get("cap", None)
        if cap_raw is None:
            cap: float | None = None
        elif isinstance(cap_raw, (int, float)):
            cap = float(cap_raw)
            if not (0.0 <= cap <= 1.0):
                raise ValueError(
                    f"{path}: source {source_id!r}: 'cap' must be in [0.0, 1.0]; got {cap}"
                )
        else:
            raise ValueError(f"{path}: source {source_id!r}: 'cap' must be a number or null")
        if trust is SignalTrust.INTERNAL and cap is not None:
            raise ValueError(
                f"{path}: source {source_id!r}: INTERNAL sources must not "
                f"declare a cap (got {cap}); only governance/hazard gates "
                f"may clamp internal confidence"
            )
        note_raw = body.get("note", "")
        if note_raw is not None and not isinstance(note_raw, str):
            raise ValueError(f"{path}: source {source_id!r}: 'note' must be a string")
        sources[source_id] = ExternalSignalSource(
            source_id=source_id,
            trust=trust,
            cap=cap,
            note=note_raw or "",
        )
    return ExternalSignalTrustRegistry(
        version=version,
        sources=MappingProxyType(sources),
    )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ExternalSignalSource",
    "ExternalSignalTrustRegistry",
    "load_external_signal_trust",
]
