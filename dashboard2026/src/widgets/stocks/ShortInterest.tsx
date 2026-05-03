interface ShortRow {
  symbol: string;
  pct: number; // % of float
  daysToCover: number;
  borrowFee: number; // %
  changePct: number; // 14d change in pct
}

const MOCK: ShortRow[] = [
  { symbol: "AAPL", pct: 0.71, daysToCover: 1.2, borrowFee: 0.5, changePct: -0.04 },
  { symbol: "TSLA", pct: 3.42, daysToCover: 1.8, borrowFee: 1.1, changePct: +0.31 },
  { symbol: "GME", pct: 16.8, daysToCover: 4.2, borrowFee: 18.4, changePct: -1.2 },
  { symbol: "BBBY", pct: 32.1, daysToCover: 7.8, borrowFee: 110.0, changePct: +2.6 },
  { symbol: "AMC", pct: 21.5, daysToCover: 5.1, borrowFee: 36.5, changePct: +0.8 },
  { symbol: "BYND", pct: 41.7, daysToCover: 9.3, borrowFee: 84.2, changePct: -3.1 },
  { symbol: "UPST", pct: 28.3, daysToCover: 6.0, borrowFee: 22.7, changePct: +1.5 },
];

function severity(pct: number) {
  if (pct >= 30) return { label: "extreme", tone: "border-rose-500/40 bg-rose-500/10 text-rose-300" };
  if (pct >= 15) return { label: "elevated", tone: "border-amber-500/40 bg-amber-500/10 text-amber-300" };
  return { label: "normal", tone: "border-slate-600/40 bg-slate-700/30 text-slate-400" };
}

export function ShortInterest({ symbol = "AAPL" }: { symbol?: string }) {
  const rows = MOCK;
  const focus = rows.find((r) => r.symbol === symbol) ?? rows[0];
  const others = rows.filter((r) => r.symbol !== focus.symbol);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Short Interest · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            % of float · days to cover · borrow fee · 14d change
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${severity(focus.pct).tone}`}
        >
          {severity(focus.pct).label}
        </span>
      </header>
      <div className="border-b border-border bg-slate-900/40 px-3 py-2 font-mono text-[11px]">
        <div className="grid grid-cols-4 gap-2 text-center">
          <Cell label="short %" v={`${focus.pct.toFixed(1)}%`} tone={focus.pct >= 15 ? "rose" : "slate"} />
          <Cell label="days to cover" v={focus.daysToCover.toFixed(1)} />
          <Cell label="borrow fee" v={`${focus.borrowFee.toFixed(1)}%`} tone={focus.borrowFee >= 20 ? "rose" : "slate"} />
          <Cell
            label="14d Δ"
            v={`${focus.changePct >= 0 ? "+" : ""}${focus.changePct.toFixed(1)}pp`}
            tone={focus.changePct >= 0 ? "rose" : "emerald"}
          />
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <div className="px-3 pt-2 text-[10px] uppercase tracking-wider text-slate-500">
          peers
        </div>
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">sym</th>
              <th className="px-2 py-1 text-right">short%</th>
              <th className="px-2 py-1 text-right">DTC</th>
              <th className="px-2 py-1 text-right">borrow%</th>
              <th className="px-2 py-1 text-right">14d Δ</th>
            </tr>
          </thead>
          <tbody>
            {others.map((r) => (
              <tr key={r.symbol} className="border-t border-border">
                <td className="px-2 py-1 text-slate-200">{r.symbol}</td>
                <td className={`px-2 py-1 text-right ${r.pct >= 15 ? "text-rose-300" : "text-slate-300"}`}>
                  {r.pct.toFixed(1)}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">{r.daysToCover.toFixed(1)}</td>
                <td className={`px-2 py-1 text-right ${r.borrowFee >= 20 ? "text-rose-300" : "text-slate-300"}`}>
                  {r.borrowFee.toFixed(1)}
                </td>
                <td
                  className={`px-2 py-1 text-right ${
                    r.changePct >= 0 ? "text-rose-300" : "text-emerald-300"
                  }`}
                >
                  {r.changePct >= 0 ? "+" : ""}
                  {r.changePct.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Cell({ label, v, tone }: { label: string; v: string; tone?: string }) {
  const cls =
    tone === "rose" ? "text-rose-300" : tone === "emerald" ? "text-emerald-300" : "text-slate-200";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-[12px] ${cls}`}>{v}</div>
    </div>
  );
}
