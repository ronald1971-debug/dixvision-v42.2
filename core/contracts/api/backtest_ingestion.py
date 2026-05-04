"""Backtest ingestion seam — Paper-S3.

Defines the :class:`BacktestIngester` Protocol implemented by adapters
under ``sensory/external/<platform>/`` (TradingView / MT5 historical)
and ``system_engine/backtest_ingest/<platform>/`` (QuantConnect).

An ingester is **stateless / pure**:

* takes a raw payload (whatever the source platform speaks — dict,
  CSV row dict, parsed JSON, …),
* returns a single :class:`~core.contracts.backtest_result.BacktestResult`
  or raises :class:`BacktestIngestionError` if the payload is malformed
  / missing required fields.

Ingesters MUST NOT:

* perform IO (HTTP / file IO belongs in the adapter shell that *calls*
  the ingester),
* read system clocks,
* keep mutable state across calls,
* emit :class:`~execution_engine.contracts.ExecutionIntent` (read-only
  ingestion only — execution authority is granted exclusively by the
  governance gate).

The Protocol carries a :data:`source` class-level attribute so the
governance / SCVS layer can look up the per-source confidence cap in
``registry/external_signal_trust.yaml`` without instantiating the
ingester.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from core.contracts.backtest_result import BacktestResult


class BacktestIngestionError(ValueError):
    """Raised by a :class:`BacktestIngester` for malformed payloads.

    Subclasses :class:`ValueError` so existing input-validation
    handlers (and the Pydantic boundary in ``ui/server.py``) catch
    it via ``except ValueError``.
    """


@runtime_checkable
class BacktestIngester(Protocol):
    """Read-only adapter from a source-specific backtest payload to
    a canonical :class:`BacktestResult`.

    Implementations are typically frozen dataclasses or module-level
    functions wrapped in a thin class. They are registered on the
    :class:`BacktestIngesterRegistry` (Paper-S4 / S5) keyed by
    ``source``.
    """

    source: ClassVar[str]

    def ingest(self, payload: Mapping[str, Any]) -> BacktestResult:
        """Validate *payload* and return a canonical :class:`BacktestResult`.

        Raises:
            BacktestIngestionError: payload is malformed / missing
                required fields. Field-validation errors raised by the
                :class:`BacktestResult` constructor (``ValueError``)
                are re-raised as :class:`BacktestIngestionError` so
                callers can handle them uniformly.
        """


__all__ = [
    "BacktestIngester",
    "BacktestIngestionError",
]
