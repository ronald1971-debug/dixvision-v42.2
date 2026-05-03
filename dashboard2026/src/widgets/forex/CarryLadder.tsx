interface CarryRow {
  pair: string;
  longRate: number;
  shortRate: number;
}

const MOCK: CarryRow[] = [
  { pair: "USD/JPY", longRate: 5.25, shortRate: 0.25 },
  { pair: "AUD/JPY", longRate: 4.10, shortRate: 0.25 },
  { pair: "NZD/JPY", longRate: 5.25, shortRate: 0.25 },
  { pair: "GBP/JPY", longRate: 4.50, shortRate: 0.25 },
  { pair: "EUR/CHF", longRate: 3.75, shortRate: 1.50 },
  { pair: "USD/CHF", longRate: 5.25, shortRate: 1.50 },
  { pair: "AUD/USD", longRate: 4.10, shortRate: 5.25 },
  { pair: "EUR/USD", longRate: 3.75, shortRate: 5.25 },
  { pair: "GBP/USD", longRate: 4.50, shortRate: 5.25 },
];

export function CarryLadder() {
  const rows = [...MOCK]
    .map((r) => ({ ...r, diff: r.longRate - r.shortRate }))
    .sort((a, b) => b.diff - a.diff);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Carry Ladder
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            interest-rate differential · positive = long carry, negative = funding cost
          </p>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">pair</th>
              <th className="px-2 py-1 text-right">long%</th>
              <th className="px-2 py-1 text-right">short%</th>
              <th className="px-2 py-1 text-right">carry</th>
              <th className="px-2 py-1 text-left"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const positive = r.diff >= 0;
              const w = Math.min(Math.abs(r.diff) * 12, 100);
              return (
                <tr key={r.pair} className="border-t border-border">
                  <td className="px-2 py-1 text-slate-200">{r.pair}</td>
                  <td className="px-2 py-1 text-right text-slate-300">
                    {r.longRate.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-right text-slate-500">
                    {r.shortRate.toFixed(2)}
                  </td>
                  <td
                    className={`px-2 py-1 text-right ${
                      positive ? "text-emerald-300" : "text-rose-300"
                    }`}
                  >
                    {positive ? "+" : ""}
                    {r.diff.toFixed(2)}
                  </td>
                  <td className="px-2 py-1">
                    <div className="relative h-1.5 w-24 overflow-hidden rounded bg-slate-800/60">
                      <div
                        className={`absolute inset-y-0 left-0 ${
                          positive ? "bg-emerald-500/60" : "bg-rose-500/60"
                        }`}
                        style={{ width: `${w}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
