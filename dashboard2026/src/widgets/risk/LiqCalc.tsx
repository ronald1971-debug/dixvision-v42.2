import { useMemo, useState } from "react";

/**
 * Tier-6 risk widget — Perp liquidation calculator.
 *
 * Inputs: side, entry price, position size (USD), leverage,
 * maintenance margin %. Outputs: liquidation price + distance
 * + bankruptcy price + the live margin used. Pure-function;
 * no external data needed today.
 *
 * Real wiring binds entry / margin to the active position
 * from the Tier-6 position aggregator (filed).
 */
type Side = "LONG" | "SHORT";

function liqPrice(
  side: Side,
  entry: number,
  leverage: number,
  mmr: number,
): number {
  // Standard isolated-margin liquidation formula:
  //   long  : entry * (1 - 1/lev + mmr)
  //   short : entry * (1 + 1/lev - mmr)
  if (side === "LONG") {
    return entry * (1 - 1 / leverage + mmr);
  }
  return entry * (1 + 1 / leverage - mmr);
}

function bankruptcy(side: Side, entry: number, leverage: number): number {
  if (side === "LONG") return entry * (1 - 1 / leverage);
  return entry * (1 + 1 / leverage);
}

export function LiqCalc() {
  const [side, setSide] = useState<Side>("LONG");
  const [entry, setEntry] = useState(67_400);
  const [size, setSize] = useState(50_000);
  const [leverage, setLeverage] = useState(10);
  const [mmrPct, setMmrPct] = useState(0.5);

  const out = useMemo(() => {
    const mmr = mmrPct / 100;
    const liq = liqPrice(side, entry, leverage, mmr);
    const bk = bankruptcy(side, entry, leverage);
    const distPct = ((liq - entry) / entry) * (side === "LONG" ? -1 : 1);
    const initialMargin = size / leverage;
    const maintenanceMargin = size * mmr;
    return {
      liq,
      bk,
      distPct,
      initialMargin,
      maintenanceMargin,
    };
  }, [side, entry, size, leverage, mmrPct]);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Liquidation calculator
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          isolated margin · standard maintenance formula
        </p>
      </header>
      <div className="grid grid-cols-2 gap-2 p-3 text-[11px]">
        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase text-slate-500">side</span>
          <div className="flex gap-1">
            {(["LONG", "SHORT"] as Side[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSide(s)}
                className={`flex-1 rounded border px-2 py-1 text-[10px] uppercase ${
                  side === s
                    ? s === "LONG"
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                      : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                    : "border-border bg-bg/40 text-slate-400 hover:text-slate-200"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase text-slate-500">entry</span>
          <input
            type="number"
            value={entry}
            onChange={(e) => setEntry(parseFloat(e.target.value) || 0)}
            className="rounded border border-border bg-bg/40 px-2 py-1 font-mono text-slate-200"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase text-slate-500">
            position USD
          </span>
          <input
            type="number"
            value={size}
            onChange={(e) => setSize(parseFloat(e.target.value) || 0)}
            className="rounded border border-border bg-bg/40 px-2 py-1 font-mono text-slate-200"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase text-slate-500">leverage</span>
          <input
            type="number"
            value={leverage}
            min={1}
            max={125}
            onChange={(e) =>
              setLeverage(Math.max(1, parseFloat(e.target.value) || 1))
            }
            className="rounded border border-border bg-bg/40 px-2 py-1 font-mono text-slate-200"
          />
        </label>
        <label className="col-span-2 flex flex-col gap-1">
          <span className="text-[10px] uppercase text-slate-500">
            maintenance margin %
          </span>
          <input
            type="number"
            step={0.1}
            value={mmrPct}
            onChange={(e) => setMmrPct(parseFloat(e.target.value) || 0)}
            className="rounded border border-border bg-bg/40 px-2 py-1 font-mono text-slate-200"
          />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 border-t border-border bg-bg/40 px-3 py-2 text-[11px] font-mono">
        <span className="text-slate-500">liquidation</span>
        <span className="text-right text-rose-300">${out.liq.toFixed(2)}</span>
        <span className="text-slate-500">distance</span>
        <span
          className={`text-right ${
            out.distPct < 0.05 ? "text-rose-300" : "text-slate-200"
          }`}
        >
          {(out.distPct * 100).toFixed(2)}%
        </span>
        <span className="text-slate-500">bankruptcy</span>
        <span className="text-right text-slate-300">
          ${out.bk.toFixed(2)}
        </span>
        <span className="text-slate-500">initial margin</span>
        <span className="text-right text-slate-300">
          ${out.initialMargin.toFixed(0)}
        </span>
        <span className="text-slate-500">maint. margin</span>
        <span className="text-right text-slate-300">
          ${out.maintenanceMargin.toFixed(0)}
        </span>
      </div>
    </section>
  );
}
