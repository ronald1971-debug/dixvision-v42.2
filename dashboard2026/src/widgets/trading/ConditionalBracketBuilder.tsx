import { useState } from "react";

/**
 * F-track widget — Conditional + bracket builder.
 *
 * Builds an entry order plus optional take-profit / stop-loss /
 * trailing-stop legs (a 'bracket'), and a parent if-then trigger so
 * the bracket only arms when a price-condition fires.
 *
 * Operator-approval edge (INV-72) gates activation; the staged
 * payload mirrors the `ExecutionIntent` contract from PR #78.
 */
type Side = "BUY" | "SELL";

interface BracketForm {
  symbol: string;
  side: Side;
  trigger_op: ">" | "<" | "≥" | "≤";
  trigger_px: number;
  entry_px: number;
  qty: number;
  tp_px: number;
  sl_px: number;
  trail_pct: number;
  oco: boolean;
}

const DEFAULTS: BracketForm = {
  symbol: "ETH-USDT",
  side: "BUY",
  trigger_op: ">",
  trigger_px: 3_500,
  entry_px: 3_510,
  qty: 5,
  tp_px: 3_750,
  sl_px: 3_390,
  trail_pct: 0.6,
  oco: true,
};

export function ConditionalBracketBuilder() {
  const [form, setForm] = useState<BracketForm>(DEFAULTS);
  const [staged, setStaged] = useState(false);

  const update = <K extends keyof BracketForm>(k: K, v: BracketForm[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  const tpPct =
    form.side === "BUY"
      ? ((form.tp_px - form.entry_px) / form.entry_px) * 100
      : ((form.entry_px - form.tp_px) / form.entry_px) * 100;
  const slPct =
    form.side === "BUY"
      ? ((form.entry_px - form.sl_px) / form.entry_px) * 100
      : ((form.sl_px - form.entry_px) / form.entry_px) * 100;
  const rr = slPct > 0 ? tpPct / slPct : 0;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Conditional + bracket builder
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          if-then trigger · entry · TP · SL · trailing · OCO bracket
        </p>
      </header>
      <div className="flex-1 overflow-auto p-3">
        <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-slate-300 sm:grid-cols-3">
          <Field label="symbol">
            <input
              value={form.symbol}
              onChange={(e) => update("symbol", e.target.value.toUpperCase())}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="side">
            <select
              value={form.side}
              onChange={(e) => update("side", e.target.value as Side)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </Field>
          <Field label="qty">
            <input
              type="number"
              value={form.qty}
              onChange={(e) => update("qty", Number(e.target.value) || 0)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="trigger when last">
            <div className="flex gap-1">
              <select
                value={form.trigger_op}
                onChange={(e) =>
                  update("trigger_op", e.target.value as BracketForm["trigger_op"])
                }
                className="w-14 rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
              >
                <option value=">">{">"}</option>
                <option value="<">{"<"}</option>
                <option value="≥">{"≥"}</option>
                <option value="≤">{"≤"}</option>
              </select>
              <input
                type="number"
                value={form.trigger_px}
                onChange={(e) =>
                  update("trigger_px", Number(e.target.value) || 0)
                }
                className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
              />
            </div>
          </Field>
          <Field label="entry px">
            <input
              type="number"
              value={form.entry_px}
              onChange={(e) => update("entry_px", Number(e.target.value) || 0)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="trail %">
            <input
              type="number"
              value={form.trail_pct}
              step="0.1"
              onChange={(e) => update("trail_pct", Number(e.target.value) || 0)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="TP px">
            <input
              type="number"
              value={form.tp_px}
              onChange={(e) => update("tp_px", Number(e.target.value) || 0)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="SL px">
            <input
              type="number"
              value={form.sl_px}
              onChange={(e) => update("sl_px", Number(e.target.value) || 0)}
              className="w-full rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </Field>
          <Field label="OCO">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={form.oco}
                onChange={(e) => update("oco", e.target.checked)}
                className="h-3 w-3 accent-accent"
              />
              <span className="text-[10px] text-slate-400">
                cancel sibling on fill
              </span>
            </label>
          </Field>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 rounded border border-border/60 bg-bg/30 p-2 font-mono text-[11px] text-slate-300">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500">
              TP move
            </div>
            <div className={tpPct >= 0 ? "text-emerald-400" : "text-rose-400"}>
              {tpPct >= 0 ? "+" : ""}
              {tpPct.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500">
              SL move
            </div>
            <div className={slPct >= 0 ? "text-rose-400" : "text-emerald-400"}>
              {slPct >= 0 ? "-" : "+"}
              {Math.abs(slPct).toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500">
              R:R
            </div>
            <div
              className={
                rr >= 1.5
                  ? "text-emerald-400"
                  : rr >= 1
                    ? "text-amber-400"
                    : "text-rose-400"
              }
            >
              {rr.toFixed(2)}
            </div>
          </div>
        </div>
      </div>
      <footer className="flex items-center gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        <span>
          arms when last {form.trigger_op} {form.trigger_px} · {form.symbol}{" "}
          {form.side} {form.qty}
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

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}
