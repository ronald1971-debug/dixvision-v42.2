import { useMemo, useState } from "react";

/**
 * Tier-6 risk widget — Scenario book.
 *
 * What-if PnL across a price-shock × IV-shock grid, plus a
 * row of canned regime fixtures (COVID-2020-03, LUNA-2021-05,
 * FTX-2022-11, SVB-2023-03). Pure function over the seeded
 * portfolio Greeks below; live wiring drives this off the
 * actual position aggregator.
 */
interface PortfolioApprox {
  net_delta: number;
  net_gamma: number;
  net_vega_per_iv_point: number;
  spot: number;
  pnl_at_spot: number;
}

const PORTFOLIO: PortfolioApprox = {
  net_delta: 1.84,
  net_gamma: 0.024,
  net_vega_per_iv_point: 1_240,
  spot: 67_400,
  pnl_at_spot: 0,
};

interface Scenario {
  name: string;
  spot_shock_pct: number;
  iv_shock_pts: number;
}

const SCENARIOS: Scenario[] = [
  { name: "COVID-20-03", spot_shock_pct: -0.4, iv_shock_pts: 60 },
  { name: "LUNA-21-05", spot_shock_pct: -0.55, iv_shock_pts: 35 },
  { name: "FTX-22-11", spot_shock_pct: -0.25, iv_shock_pts: 45 },
  { name: "SVB-23-03", spot_shock_pct: -0.18, iv_shock_pts: 25 },
  { name: "Flash-down", spot_shock_pct: -0.12, iv_shock_pts: 18 },
  { name: "Squeeze-up", spot_shock_pct: 0.18, iv_shock_pts: 8 },
];

function pnlFor(p: PortfolioApprox, spot_shock: number, iv_shock_pts: number) {
  // Δ⋅dS + ½⋅Γ⋅dS² + Vega⋅dIV — second-order approximation.
  const dS = p.spot * spot_shock;
  return p.net_delta * dS + 0.5 * p.net_gamma * dS * dS + p.net_vega_per_iv_point * iv_shock_pts;
}

const SPOT_GRID = [-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2];
const IV_GRID = [-20, -10, 0, 10, 25];

function tone(v: number): string {
  if (v >= 5_000) return "bg-emerald-500/30 text-emerald-100";
  if (v >= 1_000) return "bg-emerald-500/15 text-emerald-300";
  if (v >= 0) return "text-emerald-300";
  if (v >= -1_000) return "text-rose-300";
  if (v >= -5_000) return "bg-rose-500/15 text-rose-300";
  return "bg-rose-500/30 text-rose-100";
}

export function ScenarioBook() {
  const [tab, setTab] = useState<"grid" | "fixtures">("grid");

  const grid = useMemo(
    () =>
      SPOT_GRID.map((s) =>
        IV_GRID.map((iv) => pnlFor(PORTFOLIO, s, iv)),
      ),
    [],
  );
  const fixtures = useMemo(
    () => SCENARIOS.map((s) => ({ ...s, pnl: pnlFor(PORTFOLIO, s.spot_shock_pct, s.iv_shock_pts) })),
    [],
  );

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Scenario book
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            second-order PnL · Δ + ½Γ + Vega
          </p>
        </div>
        <div className="flex gap-1">
          {(["grid", "fixtures"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded border px-2 py-0.5 text-[10px] uppercase ${
                tab === t
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border bg-bg/40 text-slate-400 hover:text-slate-200"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </header>
      {tab === "grid" ? (
        <div className="flex-1 overflow-auto p-2">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="text-[9px] uppercase tracking-wider text-slate-500">
                <th className="px-2 py-1 text-left">spot \ ΔIV</th>
                {IV_GRID.map((iv) => (
                  <th key={iv} className="px-2 py-1 text-right">
                    {iv > 0 ? "+" : ""}
                    {iv}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono">
              {SPOT_GRID.map((s, i) => (
                <tr key={s} className="border-t border-border/40">
                  <td className="px-2 py-1 text-slate-400">
                    {s > 0 ? "+" : ""}
                    {(s * 100).toFixed(0)}%
                  </td>
                  {IV_GRID.map((_, j) => (
                    <td
                      key={j}
                      className={`px-2 py-1 text-right ${tone(grid[i][j])}`}
                    >
                      {grid[i][j] >= 0 ? "+" : ""}
                      {grid[i][j].toFixed(0)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ul className="flex-1 divide-y divide-border/40 overflow-auto">
          {fixtures.map((s) => (
            <li
              key={s.name}
              className="grid grid-cols-[1fr_auto_auto_auto] items-baseline gap-3 px-3 py-2 font-mono text-[11px]"
            >
              <span className="text-slate-200">{s.name}</span>
              <span className="text-slate-500">
                spot {(s.spot_shock_pct * 100).toFixed(0)}%
              </span>
              <span className="text-slate-500">
                ΔIV {s.iv_shock_pts > 0 ? "+" : ""}
                {s.iv_shock_pts}
              </span>
              <span className={`text-right ${tone(s.pnl)}`}>
                {s.pnl >= 0 ? "+" : ""}
                {s.pnl.toFixed(0)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
