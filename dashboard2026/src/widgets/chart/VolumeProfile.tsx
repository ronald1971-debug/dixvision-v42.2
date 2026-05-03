import { useMemo } from "react";

/**
 * Volume Profile sub-pane (horizontal histogram). Mock distribution
 * with a Point-of-Control row highlighted; replaces classic time-series
 * volume with price-bucket volume.
 */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function seedOf(s: string): number {
  let h = 0;
  for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h;
}

export function VolumeProfile({
  symbol,
  buckets = 30,
  basePrice = 100,
}: {
  symbol: string;
  buckets?: number;
  basePrice?: number;
}) {
  const { rows, poc, vah, val } = useMemo(() => {
    const r = rng(seedOf(`vp:${symbol}:${buckets}`));
    const peak = Math.floor(buckets * (0.4 + r() * 0.3));
    const out: Array<{ price: number; vol: number }> = [];
    for (let i = 0; i < buckets; i += 1) {
      const dist = Math.abs(i - peak);
      const vol = Math.exp(-(dist * dist) / (2 * 4 * 4)) * (0.6 + r() * 0.4);
      const price = basePrice + (i - buckets / 2) * 0.5;
      out.push({ price, vol });
    }
    const total = out.reduce((a, x) => a + x.vol, 0);
    const sorted = [...out].sort((a, b) => b.vol - a.vol);
    let cum = 0;
    const valueArea: typeof out = [];
    for (const row of sorted) {
      cum += row.vol;
      valueArea.push(row);
      if (cum / total >= 0.7) break;
    }
    const vahPrice = Math.max(...valueArea.map((x) => x.price));
    const valPrice = Math.min(...valueArea.map((x) => x.price));
    return {
      rows: out,
      poc: out[peak].price,
      vah: vahPrice,
      val: valPrice,
    };
  }, [symbol, buckets, basePrice]);

  const max = Math.max(...rows.map((r) => r.vol));

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Volume Profile
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            POC · VAH · VAL · {symbol}
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
          POC {poc.toFixed(2)}
        </span>
      </header>
      <div className="flex-1 overflow-auto px-3 py-2">
        <table className="w-full font-mono text-[10px]">
          <tbody>
            {[...rows].reverse().map((row) => {
              const isPoc = Math.abs(row.price - poc) < 0.01;
              const isEdge =
                Math.abs(row.price - vah) < 0.01 || Math.abs(row.price - val) < 0.01;
              const w = (row.vol / max) * 100;
              const tone = isPoc
                ? "bg-emerald-500/60"
                : isEdge
                  ? "bg-amber-500/40"
                  : "bg-sky-500/30";
              return (
                <tr key={row.price.toFixed(2)} className="border-b border-border/40">
                  <td className="w-14 py-0.5 text-slate-400">
                    {row.price.toFixed(2)}
                  </td>
                  <td className="py-0.5">
                    <div
                      className={`h-2 rounded ${tone}`}
                      style={{ width: `${w}%` }}
                    />
                  </td>
                  <td className="w-10 py-0.5 pl-2 text-right text-slate-500">
                    {(row.vol * 100).toFixed(0)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <footer className="border-t border-border px-3 py-1 font-mono text-[10px] text-slate-500">
        VAH {vah.toFixed(2)} · POC {poc.toFixed(2)} · VAL {val.toFixed(2)}
      </footer>
    </div>
  );
}
