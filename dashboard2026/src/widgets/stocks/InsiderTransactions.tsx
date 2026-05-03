interface InsiderTx {
  ts: string;
  insider: string;
  role: string;
  side: "buy" | "sell";
  shares: number;
  price: number;
  remaining: number;
}

const MOCK: InsiderTx[] = [
  { ts: "2026-04-19", insider: "T. Cook", role: "CEO", side: "sell", shares: 31200, price: 198.4, remaining: 3_280_000 },
  { ts: "2026-04-15", insider: "L. Maestri", role: "CFO", side: "sell", shares: 15400, price: 196.1, remaining: 540_000 },
  { ts: "2026-04-12", insider: "K. Adams", role: "Dir.", side: "buy", shares: 5_000, price: 192.8, remaining: 28_000 },
  { ts: "2026-04-08", insider: "D. Bowman", role: "EVP", side: "sell", shares: 8_500, price: 195.5, remaining: 196_000 },
  { ts: "2026-04-04", insider: "S. Henson", role: "Dir.", side: "buy", shares: 2_500, price: 188.9, remaining: 14_000 },
  { ts: "2026-04-02", insider: "T. Cook", role: "CEO", side: "sell", shares: 25000, price: 184.7, remaining: 3_311_200 },
  { ts: "2026-03-28", insider: "K. Adams", role: "Dir.", side: "buy", shares: 1_000, price: 175.0, remaining: 23_000 },
];

export function InsiderTransactions({ symbol = "AAPL" }: { symbol?: string }) {
  const txs = MOCK;
  const buys = txs.filter((t) => t.side === "buy");
  const sells = txs.filter((t) => t.side === "sell");
  const buyValue = buys.reduce((a, t) => a + t.shares * t.price, 0);
  const sellValue = sells.reduce((a, t) => a + t.shares * t.price, 0);
  const net = buyValue - sellValue;

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Insider Transactions · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Form 4 filings · last 30 days · officer + director
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${
            net >= 0
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              : "border-rose-500/40 bg-rose-500/10 text-rose-300"
          }`}
        >
          net {net >= 0 ? "+" : ""}
          {(net / 1_000_000).toFixed(1)}M
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">date</th>
              <th className="px-2 py-1 text-left">insider</th>
              <th className="px-2 py-1 text-left">role</th>
              <th className="px-2 py-1 text-left">side</th>
              <th className="px-2 py-1 text-right">shares</th>
              <th className="px-2 py-1 text-right">price</th>
              <th className="px-2 py-1 text-right">value</th>
              <th className="px-2 py-1 text-right">remaining</th>
            </tr>
          </thead>
          <tbody>
            {txs.map((t, i) => (
              <tr key={i} className="border-t border-border">
                <td className="px-2 py-1 text-slate-500">{t.ts.slice(5)}</td>
                <td className="px-2 py-1 text-slate-200">{t.insider}</td>
                <td className="px-2 py-1 text-slate-400">{t.role}</td>
                <td
                  className={`px-2 py-1 ${
                    t.side === "buy" ? "text-emerald-300" : "text-rose-300"
                  }`}
                >
                  {t.side}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {t.shares.toLocaleString()}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {t.price.toFixed(2)}
                </td>
                <td className="px-2 py-1 text-right text-slate-200">
                  ${(t.shares * t.price / 1_000_000).toFixed(2)}M
                </td>
                <td className="px-2 py-1 text-right text-slate-500">
                  {t.remaining.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
