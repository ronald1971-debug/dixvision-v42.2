interface CBRow {
  ccy: string;
  bank: string;
  rate: number; // current %
  next: string; // next meeting date
  cutP: number; // probability of a cut
  holdP: number;
  hikeP: number;
}

const MOCK: CBRow[] = [
  { ccy: "USD", bank: "Fed",  rate: 5.25, next: "2026-04-30", cutP: 0.18, holdP: 0.78, hikeP: 0.04 },
  { ccy: "EUR", bank: "ECB",  rate: 3.75, next: "2026-04-25", cutP: 0.34, holdP: 0.64, hikeP: 0.02 },
  { ccy: "GBP", bank: "BoE",  rate: 4.50, next: "2026-05-08", cutP: 0.22, holdP: 0.74, hikeP: 0.04 },
  { ccy: "JPY", bank: "BoJ",  rate: 0.25, next: "2026-04-28", cutP: 0.05, holdP: 0.65, hikeP: 0.30 },
  { ccy: "CHF", bank: "SNB",  rate: 1.50, next: "2026-06-19", cutP: 0.42, holdP: 0.55, hikeP: 0.03 },
  { ccy: "CAD", bank: "BoC",  rate: 4.75, next: "2026-06-04", cutP: 0.55, holdP: 0.42, hikeP: 0.03 },
  { ccy: "AUD", bank: "RBA",  rate: 4.10, next: "2026-05-06", cutP: 0.18, holdP: 0.78, hikeP: 0.04 },
  { ccy: "NZD", bank: "RBNZ", rate: 5.25, next: "2026-05-21", cutP: 0.40, holdP: 0.58, hikeP: 0.02 },
];

function bar(p: number, tone: string) {
  const w = Math.round(p * 100);
  return (
    <div className="relative h-1.5 w-16 overflow-hidden rounded bg-slate-800/60">
      <div className={`absolute inset-y-0 left-0 ${tone}`} style={{ width: `${w}%` }} />
    </div>
  );
}

export function CentralBankRates() {
  const rows = MOCK;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Central Bank Rates
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            current policy rate · next meeting · cut/hold/hike probabilities
          </p>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">ccy</th>
              <th className="px-2 py-1 text-left">bank</th>
              <th className="px-2 py-1 text-right">rate</th>
              <th className="px-2 py-1 text-left">next</th>
              <th className="px-2 py-1 text-left">cut</th>
              <th className="px-2 py-1 text-left">hold</th>
              <th className="px-2 py-1 text-left">hike</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.ccy} className="border-t border-border">
                <td className="px-2 py-1 text-slate-200">{r.ccy}</td>
                <td className="px-2 py-1 text-slate-400">{r.bank}</td>
                <td className="px-2 py-1 text-right text-slate-200">{r.rate.toFixed(2)}%</td>
                <td className="px-2 py-1 text-slate-500">{r.next.slice(5)}</td>
                <td className="px-2 py-1">
                  <div className="flex items-center gap-1">
                    {bar(r.cutP, "bg-rose-500/60")}
                    <span className="text-[10px] text-slate-400">{Math.round(r.cutP * 100)}</span>
                  </div>
                </td>
                <td className="px-2 py-1">
                  <div className="flex items-center gap-1">
                    {bar(r.holdP, "bg-slate-500/60")}
                    <span className="text-[10px] text-slate-400">{Math.round(r.holdP * 100)}</span>
                  </div>
                </td>
                <td className="px-2 py-1">
                  <div className="flex items-center gap-1">
                    {bar(r.hikeP, "bg-emerald-500/60")}
                    <span className="text-[10px] text-slate-400">{Math.round(r.hikeP * 100)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
