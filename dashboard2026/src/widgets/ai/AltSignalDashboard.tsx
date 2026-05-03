import { useEffect, useState } from "react";

/**
 * Tier-3 / E-track AI widget — Alternative signal dashboard.
 *
 * Surfaces 8 alt-data signal categories (satellite cargo, shipping
 * lane volume, transit ridership, port traffic, retail foot traffic,
 * ATM withdrawals, credit card spend, web search trend) with a
 * lead-time-vs-official-data label. Mirrors the AltSignal
 * Intelligence reference covered in the gap report.
 *
 * Backend hook: ``GET /api/altsignal/feeds`` reads from a future
 * ``intelligence_engine.alt_signal`` projector that turns each feed
 * into a ``SignalEvent``. Mock here so the cockpit shows the surface
 * today; activation is a backend swap, no UI change.
 */
interface AltSignal {
  key: string;
  label: string;
  category: "satellite" | "shipping" | "transit" | "spending";
  current: number;
  delta_7d_pct: number;
  lead_weeks: number;
  baseline: string;
}

const SEED: AltSignal[] = [
  {
    key: "sat-china-cargo",
    label: "China port satellite cargo",
    category: "satellite",
    current: 8420,
    delta_7d_pct: 4.1,
    lead_weeks: 4,
    baseline: "PMI export new orders",
  },
  {
    key: "sat-us-parking",
    label: "US big-box retail parking fill",
    category: "satellite",
    current: 0.62,
    delta_7d_pct: -2.3,
    lead_weeks: 5,
    baseline: "US retail sales report",
  },
  {
    key: "ship-suez-tx",
    label: "Suez Canal weekly transits",
    category: "shipping",
    current: 312,
    delta_7d_pct: 1.8,
    lead_weeks: 3,
    baseline: "Global goods PMI",
  },
  {
    key: "ship-shanghai-fv",
    label: "Shanghai container freight rate",
    category: "shipping",
    current: 2980,
    delta_7d_pct: -0.4,
    lead_weeks: 4,
    baseline: "Inflation goods component",
  },
  {
    key: "transit-mta",
    label: "NYC MTA daily ridership",
    category: "transit",
    current: 3_910_000,
    delta_7d_pct: 0.6,
    lead_weeks: 2,
    baseline: "BLS employment situation",
  },
  {
    key: "transit-tubelnd",
    label: "London Tube daily entries",
    category: "transit",
    current: 3_240_000,
    delta_7d_pct: 1.1,
    lead_weeks: 2,
    baseline: "UK retail sales",
  },
  {
    key: "spend-cc-restaurants",
    label: "US restaurant card spend",
    category: "spending",
    current: 9.2,
    delta_7d_pct: -0.7,
    lead_weeks: 3,
    baseline: "Personal consumption expenditures",
  },
  {
    key: "spend-search-recession",
    label: "'recession' search trend",
    category: "spending",
    current: 38,
    delta_7d_pct: 12.5,
    lead_weeks: 5,
    baseline: "Consumer confidence",
  },
];

const CAT_TINT: Record<AltSignal["category"], string> = {
  satellite: "border-indigo-500/40 text-indigo-300",
  shipping: "border-amber-500/40 text-amber-300",
  transit: "border-emerald-500/40 text-emerald-300",
  spending: "border-rose-500/40 text-rose-300",
};

export function AltSignalDashboard() {
  const [signals, setSignals] = useState<AltSignal[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setSignals((prev) =>
        prev.map((s) => {
          const drift = (Math.sin(Date.now() / 7_000 + s.key.length) - 0.5) * 0.4;
          return { ...s, delta_7d_pct: s.delta_7d_pct + drift };
        }),
      );
    }, 6_000);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Alt-signal dashboard
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          satellite · shipping · transit · spending — lead-weeks vs baseline
        </p>
      </header>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {signals.map((s) => (
          <li
            key={s.key}
            className="grid grid-cols-[1fr_auto] items-baseline gap-2 px-3 py-2 font-mono text-[11px] text-slate-300"
          >
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span
                  className={`shrink-0 rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${CAT_TINT[s.category]}`}
                >
                  {s.category}
                </span>
                <span className="truncate font-semibold text-slate-200">
                  {s.label}
                </span>
              </div>
              <div className="mt-0.5 flex flex-wrap items-baseline gap-3 text-[10px] text-slate-500">
                <span>
                  cur{" "}
                  <span className="text-slate-300">
                    {s.current.toLocaleString()}
                  </span>
                </span>
                <span>
                  Δ7d{" "}
                  <span
                    className={
                      s.delta_7d_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                    }
                  >
                    {s.delta_7d_pct >= 0 ? "+" : ""}
                    {s.delta_7d_pct.toFixed(1)}%
                  </span>
                </span>
                <span>
                  leads <span className="text-slate-300">{s.lead_weeks}w</span> vs{" "}
                  {s.baseline}
                </span>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
