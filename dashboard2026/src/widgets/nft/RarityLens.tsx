interface Band {
  label: string;
  range: string;
  count: number;
  floor: number;
  median: number;
  delta24h: number;
  tone: string;
}

const BANDS: Band[] = [
  { label: "Mythic",   range: "top 0.1%",    count: 10,   floor: 81.4, median: 96.2, delta24h: +18.4, tone: "border-violet-500/40 bg-violet-500/10 text-violet-300" },
  { label: "Legendary",range: "0.1-1%",     count: 90,   floor: 28.0, median: 34.6, delta24h: +6.7,  tone: "border-amber-500/40 bg-amber-500/10 text-amber-300" },
  { label: "Epic",     range: "1-5%",       count: 400,  floor: 14.2, median: 17.0, delta24h: +1.9,  tone: "border-rose-500/40 bg-rose-500/10 text-rose-300" },
  { label: "Rare",     range: "5-15%",      count: 1000, floor: 9.30, median: 10.4, delta24h: +0.5,  tone: "border-sky-500/40 bg-sky-500/10 text-sky-300" },
  { label: "Uncommon", range: "15-40%",     count: 2500, floor: 7.55, median: 8.10, delta24h: -0.6,  tone: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" },
  { label: "Common",   range: "40-100%",    count: 6000, floor: 7.40, median: 7.62, delta24h: -0.4,  tone: "border-slate-600/40 bg-slate-700/30 text-slate-400" },
];

export function RarityLens({ collection = "Pudgy" }: { collection?: string }) {
  const total = BANDS.reduce((a, b) => a + b.count, 0);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Rarity Lens · {collection}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            floors stratified by rarity band · {total.toLocaleString()} supply
          </p>
        </div>
      </header>
      <div className="flex-1 overflow-auto px-3 py-2">
        <div className="space-y-2">
          {BANDS.map((b) => {
            const w = (b.count / total) * 100;
            return (
              <div key={b.label} className="rounded border border-border bg-slate-900/40 px-2 py-1.5">
                <div className="flex items-baseline justify-between">
                  <span className={`rounded border px-1 py-px text-[9px] uppercase ${b.tone}`}>
                    {b.label}
                  </span>
                  <span className="font-mono text-[11px] text-slate-300">{b.range}</span>
                </div>
                <div className="mt-1 grid grid-cols-3 gap-2 font-mono text-[11px]">
                  <Cell label="supply" v={b.count.toLocaleString()} />
                  <Cell label="floor" v={`Ξ ${b.floor.toFixed(2)}`} />
                  <Cell
                    label="24h"
                    v={`${b.delta24h >= 0 ? "+" : ""}${b.delta24h.toFixed(1)}%`}
                    tone={b.delta24h >= 0 ? "emerald" : "rose"}
                  />
                </div>
                <div className="mt-1 h-1 overflow-hidden rounded bg-slate-800/40">
                  <div
                    className="h-full bg-slate-500/50"
                    style={{ width: `${w.toFixed(2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function Cell({ label, v, tone }: { label: string; v: string; tone?: string }) {
  const cls = tone === "emerald" ? "text-emerald-300" : tone === "rose" ? "text-rose-300" : "text-slate-100";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={cls}>{v}</div>
    </div>
  );
}
