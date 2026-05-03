interface MetricRow {
  label: string;
  value: string;
  context?: string;
  tone?: "good" | "warn" | "bad" | "neutral";
}

const SECTIONS: Array<{ heading: string; rows: MetricRow[] }> = [
  {
    heading: "Valuation",
    rows: [
      { label: "P/E (TTM)", value: "32.4", context: "vs sector 28.1", tone: "warn" },
      { label: "P/E (Fwd)", value: "28.7", context: "vs sector 24.6", tone: "warn" },
      { label: "P/B", value: "47.1", context: "high", tone: "warn" },
      { label: "P/S", value: "8.6", context: "elevated", tone: "neutral" },
      { label: "EV/EBITDA", value: "23.9", context: "—" },
      { label: "PEG", value: "2.1", context: "growth not free", tone: "warn" },
    ],
  },
  {
    heading: "Profitability",
    rows: [
      { label: "Gross margin", value: "44.2%", tone: "good" },
      { label: "Op margin", value: "30.7%", tone: "good" },
      { label: "Net margin", value: "26.4%", tone: "good" },
      { label: "ROE", value: "147.3%", tone: "good" },
      { label: "ROA", value: "30.1%", tone: "good" },
      { label: "ROIC", value: "57.2%", tone: "good" },
    ],
  },
  {
    heading: "Cash flow",
    rows: [
      { label: "FCF (TTM)", value: "$104.1B", tone: "good" },
      { label: "FCF margin", value: "26.7%", tone: "good" },
      { label: "Capex / rev", value: "2.7%", tone: "good" },
      { label: "Dividend yield", value: "0.43%", context: "5y CAGR 6%", tone: "neutral" },
      { label: "Buyback yield", value: "2.9%", context: "$90B/yr", tone: "good" },
    ],
  },
  {
    heading: "Balance sheet",
    rows: [
      { label: "Debt / equity", value: "1.97", context: "elevated, but covered", tone: "warn" },
      { label: "Net debt", value: "$53B", context: "—" },
      { label: "Current ratio", value: "0.93", context: "watch", tone: "warn" },
      { label: "Interest cov", value: "29.1×", tone: "good" },
    ],
  },
];

const TONE: Record<NonNullable<MetricRow["tone"]>, string> = {
  good: "text-emerald-300",
  warn: "text-amber-300",
  bad: "text-rose-300",
  neutral: "text-slate-300",
};

export function Fundamentals({ symbol = "AAPL" }: { symbol?: string }) {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Fundamentals · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            valuation · profitability · cash flow · balance sheet (TTM)
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
          quality A
        </span>
      </header>
      <div className="flex-1 overflow-auto px-3 py-2">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          {SECTIONS.map((s) => (
            <div key={s.heading}>
              <h4 className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
                {s.heading}
              </h4>
              <table className="w-full text-[11px] font-mono">
                <tbody>
                  {s.rows.map((r) => (
                    <tr key={r.label}>
                      <td className="py-0.5 text-slate-400">{r.label}</td>
                      <td className={`py-0.5 text-right ${TONE[r.tone ?? "neutral"]}`}>
                        {r.value}
                      </td>
                      <td className="py-0.5 pl-2 text-[10px] text-slate-500">
                        {r.context ?? ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
