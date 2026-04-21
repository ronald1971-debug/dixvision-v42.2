"""
mind.sources.providers.code_search \u2014 DYON enhancement-scout lane.

Pluggable code/dataset/model search providers (GitHub, GitLab, Codeberg,
SourceHut, Bitbucket, PyPI, npm, crates.io, HuggingFace, Papers-with-Code,
Kaggle, Google Dataset Search). Candidates are scored and emitted as
``SYSTEM/DISCOVERY_CANDIDATE`` ledger events.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from mind.sources.provider_base import Provider
from mind.sources.source_types import SourceKind


@dataclass
class DiscoveryCandidate:
    name: str
    url: str
    source: str                  # which provider found it
    stars: int = 0
    license: str = ""
    author: str = ""
    last_activity: str = ""
    summary: str = ""
    language: str = ""
    score: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "url": self.url, "source": self.source,
            "stars": self.stars, "license": self.license, "author": self.author,
            "last_activity": self.last_activity, "summary": self.summary,
            "language": self.language, "score": round(self.score, 3),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class CodeSearchSpec:
    name: str
    url: str
    key_env: str = ""
    key_required: bool = False
    rps: float = 0.2
    weight: float = 1.0                                      # default score weight


_SPECS: tuple[CodeSearchSpec, ...] = (
    CodeSearchSpec("github_search",
                   "https://api.github.com/search/repositories",
                   key_env="GITHUB_TOKEN", rps=0.5, weight=1.0),
    CodeSearchSpec("github_trending",
                   "https://api.github.com/search/repositories?sort=stars",
                   key_env="GITHUB_TOKEN", rps=0.25, weight=0.9),
    CodeSearchSpec("gitlab_search",
                   "https://gitlab.com/api/v4/projects",
                   key_env="GITLAB_TOKEN", rps=0.5, weight=0.7),
    CodeSearchSpec("codeberg_search",
                   "https://codeberg.org/api/v1/repos/search", rps=0.5, weight=0.5),
    CodeSearchSpec("sourcehut_search",
                   "https://git.sr.ht/api/projects", rps=0.25, weight=0.4),
    CodeSearchSpec("bitbucket_search",
                   "https://api.bitbucket.org/2.0/repositories", rps=0.25, weight=0.4),
    CodeSearchSpec("pypi_json",
                   "https://pypi.org/pypi/", rps=1.0, weight=0.8),
    CodeSearchSpec("npm_registry",
                   "https://registry.npmjs.org/-/v1/search", rps=1.0, weight=0.6),
    CodeSearchSpec("crates_io",
                   "https://crates.io/api/v1/crates", rps=1.0, weight=0.7),
    CodeSearchSpec("huggingface_models",
                   "https://huggingface.co/api/models",
                   key_env="HUGGINGFACE_TOKEN", rps=0.5, weight=0.9),
    CodeSearchSpec("huggingface_datasets",
                   "https://huggingface.co/api/datasets",
                   key_env="HUGGINGFACE_TOKEN", rps=0.5, weight=0.7),
    CodeSearchSpec("papers_with_code",
                   "https://paperswithcode.com/api/v1/", rps=0.2, weight=0.7),
    CodeSearchSpec("kaggle_datasets",
                   "https://www.kaggle.com/api/v1/datasets/list",
                   key_env="KAGGLE_API_KEY", key_required=True, rps=0.2, weight=0.6),
    CodeSearchSpec("google_dataset_search",
                   "https://datasetsearch.research.google.com/search",
                   rps=0.1, weight=0.3),
    CodeSearchSpec("awesome_quant",
                   "https://raw.githubusercontent.com/wilsonfreitas/awesome-quant/master/README.md",
                   rps=0.05, weight=0.5),
)


_TRADING_SEEDS: tuple[str, ...] = (
    "execution algorithm", "market impact", "regime detection", "tca",
    "mev protection", "orderbook aggregation", "cross-exchange arb",
    "slippage model", "funding rate", "basis trade", "volatility targeting",
    "var", "expected shortfall", "almgren-chriss", "vwap", "twap",
)
_CODING_SEEDS: tuple[str, ...] = (
    "asyncio retry backoff", "sqlite wal best practice", "ast transformer",
    "fastapi auth middleware", "websocket reconnect strategy",
    "rust python interop", "safe credential storage", "cross-platform desktop",
    "pyo3 bindings", "ccxt adapter", "token bucket rate limiter",
)


def relevance_score(text: str, *, seeds: Iterable[str]) -> float:
    t = (text or "").lower()
    hits = sum(1 for s in seeds if s in t)
    return min(1.0, hits / 6.0)


def score_candidate(text: str) -> float:
    a = relevance_score(text, seeds=_TRADING_SEEDS)
    b = relevance_score(text, seeds=_CODING_SEEDS)
    return max(a, b)


def _make_class(spec: CodeSearchSpec) -> type:
    cls_name = "".join(p.capitalize() for p in spec.name.split("_")) + "Provider"
    attrs: dict[str, object] = {
        "name": spec.name,
        "kind": SourceKind.REST,
        "api_key_env": spec.key_env,
        "api_key_required": spec.key_required,
        "rate_limit_rps": spec.rps,
        "rate_limit_burst": max(2.0, spec.rps * 4),
        "homepage": spec.url,
        "weight": spec.weight,
        "poll": lambda self: [],
    }
    return type(cls_name, (Provider,), attrs)


CODE_SEARCH_PROVIDERS: list[type] = [_make_class(s) for s in _SPECS]


def propose(text: str, *, url: str, source: str,
            stars: int = 0, license: str = "",
            author: str = "", last_activity: str = "",
            summary: str = "", language: str = "") -> DiscoveryCandidate:
    s = score_candidate(text)
    c = DiscoveryCandidate(name=text[:120], url=url, source=source,
                           stars=stars, license=license, author=author,
                           last_activity=last_activity, summary=summary,
                           language=language, score=s)
    try:
        from state.ledger.writer import get_writer
        get_writer().write("SYSTEM", "DISCOVERY_CANDIDATE", "DYON", c.to_dict())
    except Exception:
        pass
    return c


__all__ = ["CODE_SEARCH_PROVIDERS", "DiscoveryCandidate",
           "score_candidate", "propose"]
