"""GOV-CP-06 — Compliance Validator.

Validates proposed actions against domain-isolation and operational
compliance rules. In particular it enforces the hard 3-domain
isolation called out in Build Compiler Spec §7:

* NORMAL_TRADING domain — full feature set
* COPY_TRADING domain   — strict caps; no proprietary signals
* MEMECOIN domain       — burner wallet only, isolated process,
                          per-trade cap, daily cap

Returns a :class:`ComplianceReport`; the policy + risk gates have
already run by the time this is called, so the validator's role is
the last-mile domain-aware check.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from core.contracts.governance import ComplianceReport, SystemMode

# Per-domain hard caps. Spec is explicit (manifest.md §0.6 / addons §INV-20):
# memecoin trades are tightly bounded; everything else inherits global
# limits enforced by the RiskEvaluator.
_DEFAULT_DOMAIN_CAPS: Mapping[str, Mapping[str, float]] = {
    "MEMECOIN": {"max_per_trade_usd": 250.0, "max_daily_usd": 1000.0},
    "COPY_TRADING": {"max_per_trade_usd": 5_000.0},
    "NORMAL_TRADING": {},
}


class ComplianceValidator:
    name: str = "compliance_validator"
    spec_id: str = "GOV-CP-06"

    def __init__(
        self,
        *,
        domain_caps: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self._domain_caps: dict[str, dict[str, float]] = {
            domain: dict(caps)
            for domain, caps in (domain_caps or _DEFAULT_DOMAIN_CAPS).items()
        }
        self._daily_spent: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def domain_caps(self) -> Mapping[str, Mapping[str, float]]:
        return {d: dict(c) for d, c in self._domain_caps.items()}

    def reset_daily(self) -> None:
        self._daily_spent.clear()

    def validate_order(
        self,
        *,
        domain: str,
        notional_usd: float,
        mode: SystemMode,
    ) -> ComplianceReport:
        """Validate an outbound order against the active domain caps."""

        violations: list[str] = []

        if mode is SystemMode.SAFE:
            violations.append("COMPLIANCE_NO_TRADE_IN_SAFE")
        if mode is SystemMode.LOCKED:
            violations.append("COMPLIANCE_LOCKED")
        if mode is SystemMode.PAPER and domain == "MEMECOIN":
            violations.append("COMPLIANCE_MEMECOIN_REQUIRES_LIVE_DOMAIN")

        if domain not in self._domain_caps:
            violations.append(f"COMPLIANCE_UNKNOWN_DOMAIN:{domain}")
            return ComplianceReport(
                passed=False, violations=tuple(violations)
            )

        caps = self._domain_caps[domain]
        per_trade_cap = caps.get("max_per_trade_usd")
        if per_trade_cap is not None and notional_usd > per_trade_cap:
            violations.append(
                f"COMPLIANCE_PER_TRADE_CAP:{domain}:{per_trade_cap:g}"
            )

        daily_cap = caps.get("max_daily_usd")
        if daily_cap is not None:
            spent = self._daily_spent.get(domain, 0.0)
            if spent + notional_usd > daily_cap:
                violations.append(
                    f"COMPLIANCE_DAILY_CAP:{domain}:{daily_cap:g}"
                )

        if not violations:
            self._daily_spent[domain] = (
                self._daily_spent.get(domain, 0.0) + notional_usd
            )
            return ComplianceReport(passed=True)

        return ComplianceReport(passed=False, violations=tuple(violations))

    def validate_plugin_lifecycle(
        self,
        *,
        plugin_path: str,
        target_status: str,
        forbidden_in_safe: Iterable[str] = (),
        mode: SystemMode,
    ) -> ComplianceReport:
        """Lifecycle check for a plugin transition."""

        violations: list[str] = []
        if (
            target_status == "ACTIVE"
            and mode is SystemMode.SAFE
            and plugin_path in tuple(forbidden_in_safe)
        ):
            violations.append(
                f"COMPLIANCE_LIFECYCLE_NOT_IN_SAFE:{plugin_path}"
            )
        if not violations:
            return ComplianceReport(passed=True)
        return ComplianceReport(passed=False, violations=tuple(violations))


__all__ = ["ComplianceValidator"]
