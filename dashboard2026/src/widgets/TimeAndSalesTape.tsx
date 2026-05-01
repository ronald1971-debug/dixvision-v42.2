import { useEventStream } from "@/state/realtime";

interface Trade {
  side: string;
  price: number;
  size: number;
  venue: string;
}

export function TimeAndSalesTape({ symbol }: { symbol: string }) {
  const trades = useEventStream<Trade>("ticks", [], 80);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Tape · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            time &amp; sales · per-venue stream
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="flex-1 overflow-auto font-mono text-[11px]">
        <table className="w-full">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">side</th>
              <th className="px-2 py-1 text-left">price</th>
              <th className="px-2 py-1 text-left">size</th>
              <th className="px-2 py-1 text-left">venue</th>
            </tr>
          </thead>
          <tbody>
            {[...trades].reverse().map((t, i) => (
              <tr
                key={i}
                className={
                  t.side === "BUY" ? "text-emerald-300" : "text-red-300"
                }
              >
                <td className="px-2 py-0.5">{t.side}</td>
                <td className="px-2 py-0.5">{t.price.toFixed(4)}</td>
                <td className="px-2 py-0.5">{t.size}</td>
                <td className="px-2 py-0.5 text-slate-400">{t.venue}</td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={4} className="px-2 py-3 text-center text-slate-600">
                  waiting for tape (SSE / mock bridge)
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
