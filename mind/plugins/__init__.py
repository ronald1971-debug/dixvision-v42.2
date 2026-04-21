"""mind.plugins — Pluggable intelligence strategies. Contract: IIntelligence."""
from typing import Any, Dict


class _BasePlugin:
    """Common skeleton used by all default plugins."""

    name: str = "plugin"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - override
        raise NotImplementedError

    def learn(self, sample: Any) -> None:  # pragma: no cover - override
        return None
