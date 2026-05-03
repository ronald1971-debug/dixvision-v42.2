interface Sector {
  id: string;
  name: string;
  pct: number;
  weight: number; // 0..1 weight in S&P
}

const SECTORS: Sector[] = [
  { id: "tech", name: "Technology", pct: +1.42, weight: 0.30 },
  { id: "fin", name: "Financials", pct: +0.31, weight: 0.13 },
  { id: "hc", name: "Healthcare", pct: -0.18, weight: 0.13 },
  { id: "cd", name: "Cons. Disc.", pct: +0.84, weight: 0.10 },
  { id: "comm", name: "Comm. Svc.", pct: +1.07, weight: 0.09 },
  { id: "ind", name: "Industrials", pct: +0.22, weight: 0.08 },
  { id: "cs", name: "Cons. Staples", pct: -0.41, weight: 0.06 },
  { id: "en", name: "Energy", pct: -1.62, weight: 0.04 },
  { id: "ut", name: "Utilities", pct: -0.27, weight: 0.025 },
  { id: "re", name: "Real Estate", pct: -0.56, weight: 0.025 },
  { id: "mat", name: "Materials", pct: +0.18, weight: 0.025 },
];

function tone(pct: number): string {
  if (pct >= 1.0) return "bg-emerald-500/70 text-emerald-50";
  if (pct >= 0.3) return "bg-emerald-500/40 text-emerald-100";
  if (pct >= 0) return "bg-emerald-500/20 text-emerald-200";
  if (pct >= -0.3) return "bg-rose-500/20 text-rose-200";
  if (pct >= -1.0) return "bg-rose-500/40 text-rose-100";
  return "bg-rose-500/70 text-rose-50";
}

export function SectorHeatmap() {
  // largest weight first, but we lay out as a treemap-ish flex grid
  const total = SECTORS.reduce((a, s) => a + s.weight, 0);
  const sorted = [...SECTORS].sort((a, b) => b.weight - a.weight);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Sector Heatmap
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            S&P 500 sector performance · cap-weighted today
          </p>
        </div>
      </header>
      <div className="flex-1 p-2">
        <div className="flex h-full flex-wrap gap-1">
          {sorted.map((s) => {
            const flexBasis = `${(s.weight / total) * 100}%`;
            return (
              <div
                key={s.id}
                className={`flex flex-col justify-between rounded p-1.5 ${tone(s.pct)}`}
                style={{ flex: `1 1 ${flexBasis}`, minWidth: "70px", minHeight: "60px" }}
              >
                <div className="text-[10px] font-semibold uppercase tracking-wider opacity-80">
                  {s.name}
                </div>
                <div className="font-mono text-[14px] tabular-nums">
                  {s.pct >= 0 ? "+" : ""}
                  {s.pct.toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
