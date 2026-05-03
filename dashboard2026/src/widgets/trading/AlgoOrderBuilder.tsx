import { useMemo, useState } from "react";

/**
 * F-track widget — Algo order builder.
 *
 * Builds parameterized algo orders (TWAP, VWAP, Iceberg, POV).
 * The builder produces an ``ExecutionIntent``-shaped payload that
 * the operator-approval edge (INV-72) gates before
 * ``ExecutionEngine.execute(intent)`` is invoked. UI ships first;
 * the algo-routing adapter that decomposes a parent algo into child
 * slices lands in a follow-up backend PR.
 *
 * Algo defs:
 *   TWAP    — slice notional uniformly over duration_min
 *   VWAP    — slice notional weighted by historical volume curve
 *   Iceberg — show only display_pct of remaining size on book
 *   POV     — track participation_pct of realized volume
 */
type Algo = "TWAP" | "VWAP" | "Iceberg" | "POV";

interface AlgoForm {
  algo: Algo;
  symbol: string;
  side: "BUY" | "SELL";
  notional: number;
  duration_min: number;
  display_pct: number;
  participation_pct: number;
}

const DEFAULTS: AlgoForm = {
  algo: "TWAP",
  symbol: "BTC-USDT",
  side: "BUY",
  notional: 100_000,
  duration_min: 30,
  display_pct: 10,
  participation_pct: 8,
};

export function AlgoOrderBuilder() {
  const [form, setForm] = useState<AlgoForm>(DEFAULTS);
  const [staged, setStaged] = useState(false);

  const slices = useMemo(() => {
    if (form.algo === "TWAP") {
      const n = Math.max(1, Math.floor(form.duration_min / 2));
      const each = form.notional / n;
      return Array.from({ length: n }, (_, i) => ({
        ts_offset_min: i * 2,
        size_pct: 100 / n,
        notional: each,
      }));
    }
    if (form.algo === "VWAP") {
      const curve = [0.04, 0.07, 0.11, 0.14, 0.16, 0.14, 0.13, 0.10, 0.08, 0.03];
      const buckets = curve.length;
      const step = form.duration_min / buckets;
      return curve.map((w, i) => ({
        ts_offset_min: Math.round(i * step),
        size_pct: w * 100,
        notional: form.notional * w,
      }));
    }
    if (form.algo === "Iceberg") {
      const visible = (form.notional * form.display_pct) / 100;
      const n = Math.max(1, Math.ceil(form.notional / Math.max(visible, 1)));
      return Array.from({ length: n }, (_, i) => {
        const remaining = form.notional - i * visible;
        const sliceNotional = Math.min(visible, remaining);
        return {
          ts_offset_min: 0,
          size_pct: form.notional > 0 ? (sliceNotional / form.notional) * 100 : 0,
          notional: sliceNotional,
          wave: i + 1,
        };
      });
    }
    return [
      {
        ts_offset_min: 0,
        size_pct: form.participation_pct,
        notional: form.notional,
        note: `track ${form.participation_pct}% of realized volume`,
      },
    ];
  }, [form]);

  const update = <K extends keyof AlgoForm>(k: K, v: AlgoForm[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Algo order builder
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          TWAP · VWAP · Iceberg · POV — operator-approval-gated (INV-72)
        </p>
      </header>
      <div className="border-b border-border bg-bg/40 px-3 py-2">
        <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-slate-300 sm:grid-cols-4">
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              algo
            </span>
            <select
              value={form.algo}
              onChange={(e) => update("algo", e.target.value as Algo)}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            >
              <option value="TWAP">TWAP</option>
              <option value="VWAP">VWAP</option>
              <option value="Iceberg">Iceberg</option>
              <option value="POV">POV</option>
            </select>
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              symbol
            </span>
            <input
              value={form.symbol}
              onChange={(e) => update("symbol", e.target.value.toUpperCase())}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              side
            </span>
            <select
              value={form.side}
              onChange={(e) => update("side", e.target.value as "BUY" | "SELL")}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              notional USD
            </span>
            <input
              type="number"
              value={form.notional}
              onChange={(e) => update("notional", Number(e.target.value) || 0)}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
          {(form.algo === "TWAP" || form.algo === "VWAP") && (
            <label className="flex flex-col gap-0.5 sm:col-span-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">
                duration (min)
              </span>
              <input
                type="number"
                value={form.duration_min}
                onChange={(e) =>
                  update("duration_min", Math.max(1, Number(e.target.value) || 1))
                }
                className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
              />
            </label>
          )}
          {form.algo === "Iceberg" && (
            <label className="flex flex-col gap-0.5 sm:col-span-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">
                display %
              </span>
              <input
                type="number"
                value={form.display_pct}
                onChange={(e) =>
                  update(
                    "display_pct",
                    Math.max(0.1, Math.min(100, Number(e.target.value) || 1)),
                  )
                }
                className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
              />
            </label>
          )}
          {form.algo === "POV" && (
            <label className="flex flex-col gap-0.5 sm:col-span-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">
                participation %
              </span>
              <input
                type="number"
                value={form.participation_pct}
                onChange={(e) =>
                  update(
                    "participation_pct",
                    Math.max(0.1, Math.min(100, Number(e.target.value) || 1)),
                  )
                }
                className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
              />
            </label>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">slice</th>
              <th className="px-3 py-1.5 text-right">t+min</th>
              <th className="px-3 py-1.5 text-right">size %</th>
              <th className="px-3 py-1.5 text-right">notional</th>
              <th className="px-3 py-1.5 text-left">note</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {slices.map((s, i) => (
              <tr key={i}>
                <td className="px-3 py-1 text-slate-200">#{i + 1}</td>
                <td className="px-3 py-1 text-right">{s.ts_offset_min}</td>
                <td className="px-3 py-1 text-right">{s.size_pct.toFixed(1)}%</td>
                <td className="px-3 py-1 text-right">
                  {Math.round(s.notional).toLocaleString()}
                </td>
                <td className="px-3 py-1 text-[10px] text-slate-500">
                  {"wave" in s
                    ? `wave ${s.wave}`
                    : "note" in s
                      ? s.note
                      : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <footer className="flex items-center gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        <span>
          {slices.length} slice{slices.length === 1 ? "" : "s"} · {form.symbol}{" "}
          {form.side}
        </span>
        <button
          type="button"
          onClick={() => setStaged((s) => !s)}
          className={`ml-auto rounded border px-2 py-0.5 uppercase tracking-wider ${
            staged
              ? "border-accent/40 bg-accent/10 text-accent"
              : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
          }`}
        >
          {staged ? "staged" : "stage"}
        </button>
      </footer>
    </section>
  );
}
