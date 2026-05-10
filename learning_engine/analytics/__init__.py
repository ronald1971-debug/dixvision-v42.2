"""High-performance OFFLINE analytics — polars LazyFrame batch tier.

This package is the OFFLINE_ONLY analytics tier adapted from the
``pola-rs/polars`` Python SDK (lazy-API surface). Modules here run in
the slow-cadence learning loop only; ``tools/authority_lint.py`` will
ban ``import polars`` from any module under ``execution_engine/``,
``governance_engine/``, ``system_engine/``, ``core/``, or
``intelligence_engine/meta_controller/hot_path.py``.
"""
