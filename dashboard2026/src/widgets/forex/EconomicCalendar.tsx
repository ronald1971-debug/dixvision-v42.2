interface CalendarEvent {
  ts: string; // ISO
  ccy: string;
  title: string;
  importance: "low" | "med" | "high";
  forecast: string;
  previous: string;
}

const MOCK: CalendarEvent[] = [
  {
    ts: "2026-04-22T12:30Z",
    ccy: "USD",
    title: "Initial Jobless Claims",
    importance: "med",
    forecast: "215K",
    previous: "212K",
  },
  {
    ts: "2026-04-22T14:00Z",
    ccy: "EUR",
    title: "ECB Lagarde Speech",
    importance: "high",
    forecast: "—",
    previous: "—",
  },
  {
    ts: "2026-04-23T08:30Z",
    ccy: "GBP",
    title: "BoE MPC Vote Split",
    importance: "high",
    forecast: "5-3 hold",
    previous: "6-3 hold",
  },
  {
    ts: "2026-04-23T12:30Z",
    ccy: "USD",
    title: "Core PCE (YoY)",
    importance: "high",
    forecast: "2.8%",
    previous: "2.9%",
  },
  {
    ts: "2026-04-23T23:30Z",
    ccy: "JPY",
    title: "Tokyo CPI ex-Food",
    importance: "med",
    forecast: "1.9%",
    previous: "1.8%",
  },
  {
    ts: "2026-04-24T18:00Z",
    ccy: "USD",
    title: "FOMC Statement",
    importance: "high",
    forecast: "—",
    previous: "5.25-5.50%",
  },
  {
    ts: "2026-04-25T09:00Z",
    ccy: "EUR",
    title: "ifo Business Climate",
    importance: "low",
    forecast: "87.4",
    previous: "87.0",
  },
  {
    ts: "2026-04-25T12:30Z",
    ccy: "CAD",
    title: "Retail Sales (MoM)",
    importance: "med",
    forecast: "0.3%",
    previous: "-0.1%",
  },
];

const TONE: Record<CalendarEvent["importance"], string> = {
  low: "border-slate-600/40 bg-slate-700/30 text-slate-400",
  med: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  high: "border-rose-500/40 bg-rose-500/10 text-rose-300",
};

export function EconomicCalendar() {
  const events = MOCK;
  const highCount = events.filter((e) => e.importance === "high").length;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Economic Calendar
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            ForexFactory · TradingEconomics · auto-pause for high-impact prints
          </p>
        </div>
        <span className="rounded border border-rose-500/40 bg-rose-500/10 px-1.5 py-0.5 font-mono text-[11px] text-rose-300">
          {highCount} red
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">when (UTC)</th>
              <th className="px-2 py-1 text-left">ccy</th>
              <th className="px-2 py-1 text-left">event</th>
              <th className="px-2 py-1 text-left">imp</th>
              <th className="px-2 py-1 text-right">forecast</th>
              <th className="px-2 py-1 text-right">prev</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={`${e.ts}-${i}`} className="border-t border-border">
                <td className="px-2 py-1 text-slate-300">
                  {e.ts.slice(5, 10)} {e.ts.slice(11, 16)}
                </td>
                <td className="px-2 py-1 text-slate-200">{e.ccy}</td>
                <td className="px-2 py-1 text-slate-200">{e.title}</td>
                <td className="px-2 py-1">
                  <span
                    className={`rounded border px-1 py-px text-[9px] uppercase ${TONE[e.importance]}`}
                  >
                    {e.importance}
                  </span>
                </td>
                <td className="px-2 py-1 text-right text-slate-300">{e.forecast}</td>
                <td className="px-2 py-1 text-right text-slate-500">{e.previous}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
