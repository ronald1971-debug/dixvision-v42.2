interface Position {
  symbol: string;
  venue: string;
  side: "long" | "short";
  size: number;
  entry: number;
  mark: number;
  realized: number;
  unrealized: number;
}

const MOCK: Position[] = [
  {
    symbol: "BTC-PERP",
    venue: "hyperliquid",
    side: "long",
    size: 0.42,
    entry: 64210.5,
    mark: 65120.0,
    realized: 0,
    unrealized: 381.99,
  },
  {
    symbol: "SOL/USDC",
    venue: "binance-spot",
    side: "long",
    size: 38,
    entry: 142.3,
    mark: 145.8,
    realized: 12.5,
    unrealized: 133.0,
  },
  {
    symbol: "PEPE",
    venue: "raydium",
    side: "long",
    size: 1_800_000,
    entry: 0.0000091,
    mark: 0.0000119,
    realized: 0,
    unrealized: 5.04,
  },
];

export function PositionsPanel() {
  const positions = MOCK;
  const total = positions.reduce((acc, p) => acc + p.realized + p.unrealized, 0);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Positions &amp; PnL
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            per-venue rollup · realized + unrealized
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${
            total >= 0
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              : "border-red-500/40 bg-red-500/10 text-red-300"
          }`}
        >
          {total >= 0 ? "+" : ""}
          {total.toFixed(2)} USDC
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">symbol</th>
              <th className="px-2 py-1 text-left">venue</th>
              <th className="px-2 py-1 text-left">side</th>
              <th className="px-2 py-1 text-right">size</th>
              <th className="px-2 py-1 text-right">entry</th>
              <th className="px-2 py-1 text-right">mark</th>
              <th className="px-2 py-1 text-right">unrealized</th>
              <th className="px-2 py-1 text-right">realized</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={`${p.venue}-${p.symbol}`} className="border-t border-border">
                <td className="px-2 py-1 text-slate-200">{p.symbol}</td>
                <td className="px-2 py-1 text-slate-400">{p.venue}</td>
                <td
                  className={`px-2 py-1 ${
                    p.side === "long" ? "text-emerald-300" : "text-red-300"
                  }`}
                >
                  {p.side}
                </td>
                <td className="px-2 py-1 text-right text-slate-200">
                  {p.size}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {p.entry}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {p.mark}
                </td>
                <td
                  className={`px-2 py-1 text-right ${
                    p.unrealized >= 0 ? "text-emerald-300" : "text-red-300"
                  }`}
                >
                  {p.unrealized >= 0 ? "+" : ""}
                  {p.unrealized.toFixed(2)}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {p.realized.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
