import { useEffect, useState } from "react";

/**
 * Tier-3 / E-track AI widget — Causal risk attribution.
 *
 * Decomposes today's portfolio PnL drivers into causal factors
 * (exposure × shock) instead of correlation-only beta. Mirrors the
 * Causify Optima reference: each factor row shows the *causal*
 * contribution and the counterfactual ("PnL if shock had been zero").
 *
 * Backend hook: ``GET /api/risk/causal_attribution`` reads from a
 * future ``risk_engine.causal_attribution`` projector. Mock here so
 * the cockpit shows the surface today; activation is a backend swap.
 */
interface FactorRow {
  factor: string;
  exposure: number;
  shock_pct: number;
  causal_pnl: number;
  correlation_pnl: number;
  counterfactual: number;
}

const SEED: FactorRow[] = [
  {
    factor: "USD index",
    exposure: 1_200_000,
    shock_pct: 0.45,
    causal_pnl: 5_400,
    correlation_pnl: 6_100,
    counterfactual: -2_300,
  },
  {
    factor: "Crude oil (WTI)",
    exposure: -800_000,
    shock_pct: -1.2,
    causal_pnl: 9_600,
    correlation_pnl: 7_400,
    counterfactual: -800,
  },
  {
    factor: "BTC realized vol",
    exposure: 2_400_000,
    shock_pct: 2.1,
    causal_pnl: 50_400,
    correlation_pnl: 38_200,
    counterfactual: 4_100,
  },
  {
    factor: "10y rate",
    exposure: -3_500_000,
    shock_pct: -0.08,
    causal_pnl: 2_800,
    correlation_pnl: 4_900,
    counterfactual: 0,
  },
  {
    factor: "AI sector beta",
    exposure: 5_100_000,
    shock_pct: 0.7,
    causal_pnl: 35_700,
    correlation_pnl: 41_200,
    counterfactual: 6_500,
  },
];

export function CausalRiskAttribution() {
  const [rows, setRows] = useState<FactorRow[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((r) => {
          const drift =
            (Math.sin(Date.now() / 8_000 + r.factor.length) - 0.5) * 1_200;
          return {
            ...r,
            causal_pnl: r.causal_pnl + drift,
            correlation_pnl: r.correlation_pnl + drift * 1.1,
          };
        }),
      );
    }, 5_500);
    return () => clearInterval(id);
  }, []);

  const totalCausal = rows.reduce((acc, r) => acc + r.causal_pnl, 0);
  const totalCorrelation = rows.reduce((acc, r) => acc + r.correlation_pnl, 0);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Causal risk attribution
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          today's PnL drivers · causal vs correlation · counterfactual = PnL
          had shock been zero
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">factor</th>
              <th className="px-3 py-1.5 text-right">expo</th>
              <th className="px-3 py-1.5 text-right">shock %</th>
              <th className="px-3 py-1.5 text-right">causal PnL</th>
              <th className="px-3 py-1.5 text-right">corr PnL</th>
              <th className="px-3 py-1.5 text-right">counterfactual</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r) => (
              <tr key={r.factor}>
                <td className="px-3 py-1.5 text-slate-200">{r.factor}</td>
                <td className="px-3 py-1.5 text-right">
                  {r.exposure.toLocaleString()}
                </td>
                <td className="px-3 py-1.5 text-right">
                  <span
                    className={
                      r.shock_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                    }
                  >
                    {r.shock_pct >= 0 ? "+" : ""}
                    {r.shock_pct.toFixed(2)}%
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right">
                  <span
                    className={
                      r.causal_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                    }
                  >
                    {r.causal_pnl >= 0 ? "+" : ""}
                    {Math.round(r.causal_pnl).toLocaleString()}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right text-slate-400">
                  {r.correlation_pnl >= 0 ? "+" : ""}
                  {Math.round(r.correlation_pnl).toLocaleString()}
                </td>
                <td className="px-3 py-1.5 text-right text-slate-400">
                  {r.counterfactual >= 0 ? "+" : ""}
                  {Math.round(r.counterfactual).toLocaleString()}
                </td>
              </tr>
            ))}
            <tr className="bg-bg/40 font-semibold">
              <td className="px-3 py-1.5 text-slate-200">TOTAL</td>
              <td className="px-3 py-1.5 text-right text-slate-500">—</td>
              <td className="px-3 py-1.5 text-right text-slate-500">—</td>
              <td className="px-3 py-1.5 text-right">
                <span
                  className={
                    totalCausal >= 0 ? "text-emerald-400" : "text-rose-400"
                  }
                >
                  {totalCausal >= 0 ? "+" : ""}
                  {Math.round(totalCausal).toLocaleString()}
                </span>
              </td>
              <td className="px-3 py-1.5 text-right text-slate-400">
                {totalCorrelation >= 0 ? "+" : ""}
                {Math.round(totalCorrelation).toLocaleString()}
              </td>
              <td className="px-3 py-1.5 text-right text-slate-500">—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
