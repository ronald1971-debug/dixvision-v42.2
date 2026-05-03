interface RatingDist {
  strongBuy: number;
  buy: number;
  hold: number;
  sell: number;
  strongSell: number;
}

const DIST: RatingDist = {
  strongBuy: 18,
  buy: 12,
  hold: 8,
  sell: 2,
  strongSell: 1,
};

const TARGETS = {
  current: 198.4,
  low: 165.0,
  avg: 215.6,
  median: 220.0,
  high: 260.0,
};

const RECENT: Array<{
  ts: string;
  firm: string;
  action: "init" | "upgrade" | "downgrade" | "maintain";
  rating: string;
  pt: number;
}> = [
  { ts: "2026-04-21", firm: "Goldman", action: "upgrade", rating: "Buy", pt: 235 },
  { ts: "2026-04-19", firm: "Morgan Stanley", action: "maintain", rating: "Overweight", pt: 230 },
  { ts: "2026-04-18", firm: "JPMorgan", action: "maintain", rating: "Overweight", pt: 245 },
  { ts: "2026-04-15", firm: "Wedbush", action: "maintain", rating: "Outperform", pt: 260 },
  { ts: "2026-04-12", firm: "Barclays", action: "downgrade", rating: "Equal-Weight", pt: 184 },
  { ts: "2026-04-10", firm: "BofA", action: "init", rating: "Buy", pt: 225 },
];

const ACTION_TONE: Record<string, string> = {
  upgrade: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  downgrade: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  maintain: "border-slate-600/40 bg-slate-700/30 text-slate-400",
  init: "border-sky-500/40 bg-sky-500/10 text-sky-300",
};

export function AnalystRatings({ symbol = "AAPL" }: { symbol?: string }) {
  const total =
    DIST.strongBuy + DIST.buy + DIST.hold + DIST.sell + DIST.strongSell;
  const upsidePct = ((TARGETS.avg - TARGETS.current) / TARGETS.current) * 100;
  const bars = [
    { label: "Strong Buy", n: DIST.strongBuy, tone: "bg-emerald-500/70" },
    { label: "Buy", n: DIST.buy, tone: "bg-emerald-500/40" },
    { label: "Hold", n: DIST.hold, tone: "bg-slate-500/50" },
    { label: "Sell", n: DIST.sell, tone: "bg-rose-500/40" },
    { label: "Strong Sell", n: DIST.strongSell, tone: "bg-rose-500/70" },
  ];

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Analyst Ratings · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            {total} analysts · 12-mo price target consensus
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${
            upsidePct >= 0
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              : "border-rose-500/40 bg-rose-500/10 text-rose-300"
          }`}
        >
          {upsidePct >= 0 ? "+" : ""}
          {upsidePct.toFixed(1)}% upside
        </span>
      </header>
      <div className="flex-1 space-y-3 overflow-auto px-3 py-2 text-[11px]">
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
            distribution
          </div>
          <div className="space-y-1">
            {bars.map((b) => (
              <div key={b.label} className="flex items-center gap-2">
                <span className="w-20 text-slate-400">{b.label}</span>
                <div className="relative h-2 flex-1 overflow-hidden rounded bg-slate-800/40">
                  <div
                    className={`absolute inset-y-0 left-0 ${b.tone}`}
                    style={{ width: `${(b.n / total) * 100}%` }}
                  />
                </div>
                <span className="w-6 text-right font-mono text-slate-300">{b.n}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded border border-border bg-slate-900/40 px-2 py-2 font-mono">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
            12-mo price target
          </div>
          <div className="grid grid-cols-4 gap-1 text-center">
            <Cell label="low" v={TARGETS.low} />
            <Cell label="avg" v={TARGETS.avg} tone="emerald" />
            <Cell label="median" v={TARGETS.median} />
            <Cell label="high" v={TARGETS.high} />
          </div>
          <div className="mt-1 text-[10px] text-slate-500">
            current ${TARGETS.current.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
            recent actions
          </div>
          <table className="w-full font-mono">
            <tbody>
              {RECENT.map((r, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="px-1 py-0.5 text-slate-500">{r.ts.slice(5)}</td>
                  <td className="px-1 py-0.5 text-slate-300">{r.firm}</td>
                  <td className="px-1 py-0.5">
                    <span
                      className={`rounded border px-1 py-px text-[9px] uppercase ${ACTION_TONE[r.action]}`}
                    >
                      {r.action}
                    </span>
                  </td>
                  <td className="px-1 py-0.5 text-slate-300">{r.rating}</td>
                  <td className="px-1 py-0.5 text-right text-slate-200">${r.pt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Cell({ label, v, tone }: { label: string; v: number; tone?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-[12px] ${tone === "emerald" ? "text-emerald-300" : "text-slate-200"}`}>
        ${v.toFixed(2)}
      </div>
    </div>
  );
}
