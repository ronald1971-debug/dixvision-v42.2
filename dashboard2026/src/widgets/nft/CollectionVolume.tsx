interface Row {
  collection: string;
  floor: number;
  volume24h: number; // ETH
  volume7d: number;  // ETH
  delta24h: number;  // %
  sales24h: number;
}

const MOCK: Row[] = [
  { collection: "Pudgy",      floor: 7.40,  volume24h: 612, volume7d: 4128, delta24h: -1.4, sales24h: 84 },
  { collection: "BAYC",       floor: 18.2,  volume24h: 421, volume7d: 2867, delta24h: +2.3, sales24h: 23 },
  { collection: "Azuki",      floor: 4.10,  volume24h: 286, volume7d: 1980, delta24h: +5.1, sales24h: 71 },
  { collection: "Doodles",    floor: 1.85,  volume24h: 102, volume7d:  734, delta24h: -3.6, sales24h: 56 },
  { collection: "Mfers",      floor: 0.92,  volume24h:  82, volume7d:  601, delta24h: +0.8, sales24h: 88 },
  { collection: "DeGods",     floor: 3.21,  volume24h:  74, volume7d:  445, delta24h: -2.1, sales24h: 24 },
  { collection: "MAYC",       floor: 2.84,  volume24h:  67, volume7d:  390, delta24h: +1.2, sales24h: 25 },
  { collection: "CloneX",     floor: 0.78,  volume24h:  43, volume7d:  287, delta24h: -0.4, sales24h: 56 },
];

export function CollectionVolume() {
  const rows = [...MOCK].sort((a, b) => b.volume24h - a.volume24h);
  const total24h = rows.reduce((a, r) => a + r.volume24h, 0);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Collection Volume
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            top blue-chips · 24h volume · sales · floor delta
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
          Ξ {total24h.toLocaleString()} 24h
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">collection</th>
              <th className="px-2 py-1 text-right">floor</th>
              <th className="px-2 py-1 text-right">vol 24h</th>
              <th className="px-2 py-1 text-right">vol 7d</th>
              <th className="px-2 py-1 text-right">sales</th>
              <th className="px-2 py-1 text-right">24h Δ</th>
              <th className="px-2 py-1 text-left"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const w = (r.volume24h / rows[0].volume24h) * 100;
              return (
                <tr key={r.collection} className="border-t border-border">
                  <td className="px-2 py-1 text-slate-200">{r.collection}</td>
                  <td className="px-2 py-1 text-right text-slate-100">{r.floor.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right text-slate-200">{r.volume24h}</td>
                  <td className="px-2 py-1 text-right text-slate-500">{r.volume7d}</td>
                  <td className="px-2 py-1 text-right text-slate-300">{r.sales24h}</td>
                  <td
                    className={`px-2 py-1 text-right ${
                      r.delta24h >= 0 ? "text-emerald-300" : "text-rose-300"
                    }`}
                  >
                    {r.delta24h >= 0 ? "+" : ""}
                    {r.delta24h.toFixed(1)}%
                  </td>
                  <td className="px-2 py-1">
                    <div className="h-1.5 w-24 overflow-hidden rounded bg-slate-800/40">
                      <div className="h-full bg-emerald-500/50" style={{ width: `${w}%` }} />
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
