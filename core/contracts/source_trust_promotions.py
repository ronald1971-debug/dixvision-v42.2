"""Paper-S6 -- in-memory store for operator-approved source-trust promotions.

The Paper-S5 governance gate clamps every external SignalEvent's
``confidence`` to the per-source cap loaded from
``registry/external_signal_trust.yaml``. Paper-S6 lets the operator
*promote* a specific source from ``EXTERNAL_LOW`` to ``EXTERNAL_MED``
without redeploying the YAML registry: the promotion is recorded as
an authority-ledger row and lives in this in-memory overlay until
demoted (or the harness is restarted -- in which case the boot-time
ledger replay rebuilds the overlay so promotions survive across
restarts without any out-of-band state).

Design constraints:

* INTERNAL signals are never promoted or demoted by operator action;
  the in-process intelligence path is the canonical source of truth
  for ``INTERNAL`` confidence.
* Only ``EXTERNAL_LOW -> EXTERNAL_MED`` promotion is supported. The
  default cap stays the more restrictive of (per-source row,
  trust-class default), so a promoted source still respects any
  explicit cap pinned in the YAML.
* The store is thread-safe: ``ui.server`` reads it on the harness hot
  path while operator routes mutate it. All access goes through a
  single :class:`threading.Lock` so reads observe a consistent
  snapshot.
* The store has no clock and no I/O. Timestamps are passed in by the
  caller (``ui.server`` sources them from :func:`ui.server.wall_ns`),
  keeping replay deterministic (INV-15).

Authority:

* No cross-engine imports -- this module only re-exports primitives.
* No clock, no PRNG; all values are caller-supplied.
* The harness approver consults the store at cap-application time;
  this module just records the operator's decision.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.signal_trust import SignalTrust

PROMOTION_LEDGER_KIND = "OPERATOR_SOURCE_TRUST_PROMOTED"
"""Authority-ledger ``kind`` for promotion rows.

Replayed at boot by ``ui.server._State`` to rebuild the in-memory
promotion overlay. Production governance never emits this kind
outside the operator promotion route.
"""

DEMOTION_LEDGER_KIND = "OPERATOR_SOURCE_TRUST_DEMOTED"
"""Authority-ledger ``kind`` for demotion (revert) rows."""

_ALLOWED_PROMOTION_TARGETS: frozenset[SignalTrust] = frozenset(
    {SignalTrust.EXTERNAL_MED}
)


__all__ = [
    "DEMOTION_LEDGER_KIND",
    "PROMOTION_LEDGER_KIND",
    "SourceTrustPromotion",
    "SourceTrustPromotionStore",
    "is_promotable_target",
]


def is_promotable_target(trust: SignalTrust) -> bool:
    """Return ``True`` iff *trust* is a valid promotion target.

    Only ``EXTERNAL_MED`` is currently allowed; ``INTERNAL`` is
    reserved for in-process producers and ``EXTERNAL_LOW`` is the
    default for unregistered external producers.
    """

    return trust in _ALLOWED_PROMOTION_TARGETS


@dataclass(frozen=True, slots=True)
class SourceTrustPromotion:
    """One operator-approved promotion record.

    Attributes:
        source_id: ``SignalEvent.signal_source`` value the promotion
            applies to (e.g. ``"tradingview.public"``).
        target_trust: Promoted trust class. Currently only
            :data:`SignalTrust.EXTERNAL_MED` is allowed.
        requestor: Free-form operator identifier (audit field).
        reason: Free-form justification (audit field).
        ts_ns: Monotonic ns timestamp at promotion time -- supplied
            by the caller for INV-15 replay determinism.
    """

    source_id: str
    target_trust: SignalTrust
    requestor: str
    reason: str
    ts_ns: int


class SourceTrustPromotionStore:
    """Thread-safe in-memory map of source_id -> promotion record.

    The harness approver consults
    :meth:`effective_trust` at cap-application time to decide which
    trust-class default applies. The mutator routes
    (:meth:`promote` / :meth:`demote`) are called from
    ``ui.server`` operator endpoints and must be paired with an
    authority-ledger row write so a restart can replay the overlay.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._promotions: dict[str, SourceTrustPromotion] = {}

    def promote(
        self,
        *,
        source_id: str,
        target_trust: SignalTrust,
        requestor: str,
        reason: str,
        ts_ns: int,
    ) -> SourceTrustPromotion:
        """Record a promotion (idempotent for same target).

        Raises:
            ValueError: ``source_id`` is empty, or ``target_trust``
                is not a permitted promotion target.
        """

        if not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not is_promotable_target(target_trust):
            raise ValueError(
                "only EXTERNAL_MED is a valid promotion target; "
                f"got {target_trust!r}"
            )
        record = SourceTrustPromotion(
            source_id=source_id,
            target_trust=target_trust,
            requestor=requestor,
            reason=reason,
            ts_ns=ts_ns,
        )
        with self._lock:
            self._promotions[source_id] = record
        return record

    def demote(self, source_id: str) -> SourceTrustPromotion | None:
        """Drop the overlay row for *source_id*.

        Returns the record that was removed, or ``None`` when there
        was nothing to demote (idempotent).
        """

        with self._lock:
            return self._promotions.pop(source_id, None)

    def get(self, source_id: str) -> SourceTrustPromotion | None:
        """Return the promotion record for *source_id* or ``None``."""

        with self._lock:
            return self._promotions.get(source_id)

    def is_promoted(self, source_id: str) -> bool:
        """Return ``True`` iff a promotion overlay exists."""

        with self._lock:
            return source_id in self._promotions

    def effective_trust(
        self, source_id: str, declared_trust: SignalTrust
    ) -> SignalTrust:
        """Compute the effective trust class for ``(source_id, declared)``.

        Rules (fail-closed -- never *demote* via overlay):

        * ``INTERNAL`` always passes through. The in-process
          intelligence path is canonical and operator promotion
          cannot affect it.
        * If a promotion overlay exists *and* the declared trust is
          ``EXTERNAL_LOW``, the promoted target wins.
        * Otherwise, the declared trust wins (a producer that
          already declared ``EXTERNAL_MED`` is never demoted by an
          absent overlay; a producer that declared a class higher
          than the overlay target is never lowered).
        """

        if declared_trust is SignalTrust.INTERNAL:
            return declared_trust
        promotion = self.get(source_id)
        if promotion is None:
            return declared_trust
        if declared_trust is SignalTrust.EXTERNAL_LOW:
            return promotion.target_trust
        return declared_trust

    def list_all(self) -> Mapping[str, SourceTrustPromotion]:
        """Return a snapshot of the current overlay (defensive copy)."""

        with self._lock:
            return dict(self._promotions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._promotions)
