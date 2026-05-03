import { useEffect, useState } from "react";

/**
 * Tier-6 risk widget — Portfolio Greeks panel.
 *
 * Aggregate Δ / Γ / Θ / Vega / Rho across all open option
 * positions plus net DV01 for the rates leg. Mock seeds to
 * a flat-vol portfolio; live wiring drives this off the
 * position aggregator (filed Tier-6 backend).
 */
interface Greeks {
  net_delta: number;
  net_gamma: number;
  net_theta_per_day: number;
  net_vega_per_iv_point: number;
  net_rho: number;
  notional_usd: number;
  net_dv01: number;
}

const SEED: Greeks = {
  net_delta: 1.84,
  net_gamma: 0.024,
  net_theta_per_day: -842,
  net_vega_per_iv_point: 1_240,
  net_rho: 320,
  notional_usd: 248_000,
  net_dv01: -84,
};

function GreekRow({
  label,
  value,
  unit,
  tone,
  hint,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "pos" | "neg" | "neutral";
  hint?: string;
}) {
  const t =
    tone === "pos"
      ? "text-emerald-300"
      : tone === "neg"
        ? "text-rose-300"
        : "text-slate-200";
  return (
    <div className="flex items-baseline justify-between border-b border-border/40 px-3 py-1.5">
      <div>
        <div className="text-[11px] font-semibold text-slate-300">{label}</div>
        {hint && <div className="text-[10px] text-slate-500">{hint}</div>}
      </div>
      <div className="text-right font-mono">
        <span className={`text-sm ${t}`}>{value}</span>
        {unit && (
          <span className="ml-1 text-[10px] text-slate-500">{unit}</span>
        )}
      </div>
    </div>
  );
}

export function GreeksPanel() {
  const [g, setG] = useState<Greeks>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setG((prev) => {
        const drift = Math.sin(Date.now() / 8_000) * 0.005;
        return {
          ...prev,
          net_delta: prev.net_delta + drift,
          net_gamma: prev.net_gamma * (1 + drift * 0.4),
          net_theta_per_day: prev.net_theta_per_day * (1 - drift * 0.2),
          net_vega_per_iv_point: prev.net_vega_per_iv_point * (1 + drift * 0.3),
        };
      });
    }, 4_000);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Portfolio Greeks
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          aggregate exposures across open option positions
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <GreekRow
          label="Net Δ (delta)"
          value={g.net_delta.toFixed(2)}
          tone={g.net_delta >= 0 ? "pos" : "neg"}
          hint="$Δ per 1.00 spot move"
        />
        <GreekRow
          label="Net Γ (gamma)"
          value={g.net_gamma.toFixed(4)}
          hint="Δ change per spot move"
        />
        <GreekRow
          label="Θ / day"
          value={g.net_theta_per_day.toFixed(0)}
          unit="USD"
          tone={g.net_theta_per_day >= 0 ? "pos" : "neg"}
          hint="time-decay carry"
        />
        <GreekRow
          label="Vega / IV pt"
          value={g.net_vega_per_iv_point.toFixed(0)}
          unit="USD"
          hint="USD per 1 IV point"
        />
        <GreekRow
          label="ρ (rho)"
          value={g.net_rho.toFixed(0)}
          unit="USD"
          hint="USD per 1bp rate"
        />
        <GreekRow
          label="Net DV01"
          value={g.net_dv01.toFixed(0)}
          unit="USD"
          tone={g.net_dv01 >= 0 ? "pos" : "neg"}
          hint="rates leg sensitivity"
        />
        <GreekRow
          label="Gross notional"
          value={`$${(g.notional_usd / 1_000).toFixed(0)}k`}
          hint="abs sum across legs"
        />
      </div>
    </section>
  );
}
