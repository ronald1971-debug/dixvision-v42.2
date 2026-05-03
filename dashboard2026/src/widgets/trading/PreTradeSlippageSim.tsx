import { useMemo, useState } from "react";

/**
 * F-track widget — Pre-trade slippage simulator.
 *
 * Approximates expected slippage for a parent order against a
 * synthetic L2 book using a square-root impact model:
 *   impact_bps ≈ k * sqrt(notional / ADV) * sigma_bps
 * Defaults match Almgren-Chriss conventions used in PR #61.
 *
 * Backend hook: ``GET /api/risk/slippage_sim?symbol&notional&urgency``
 * will read from ``risk_engine.slippage_model`` once that adapter
 * lands. Today it computes deterministically client-side so the
 * surface is usable.
 */
type Urgency = "passive" | "balanced" | "aggressive";

interface Form {
  symbol: string;
  notional: number;
  side: "BUY" | "SELL";
  urgency: Urgency;
}

const URGENCY_K: Record<Urgency, number> = {
  passive: 0.3,
  balanced: 0.6,
  aggressive: 1.1,
};

interface SymRef {
  symbol: string;
  adv_usd: number;
  sigma_bps: number;
  spread_bps: number;
}

const SYM_DEFAULTS: SymRef[] = [
  { symbol: "BTC-USDT", adv_usd: 32_000_000_000, sigma_bps: 220, spread_bps: 1.2 },
  { symbol: "ETH-USDT", adv_usd: 18_000_000_000, sigma_bps: 280, spread_bps: 1.4 },
  { symbol: "SOL-USDT", adv_usd: 4_500_000_000, sigma_bps: 410, spread_bps: 2.6 },
  { symbol: "WIF-USDT", adv_usd: 320_000_000, sigma_bps: 980, spread_bps: 12.0 },
];

function lookup(symbol: string): SymRef {
  return (
    SYM_DEFAULTS.find((r) => r.symbol === symbol) ?? {
      symbol,
      adv_usd: 100_000_000,
      sigma_bps: 600,
      spread_bps: 8,
    }
  );
}

export function PreTradeSlippageSim() {
  const [form, setForm] = useState<Form>({
    symbol: "ETH-USDT",
    notional: 250_000,
    side: "BUY",
    urgency: "balanced",
  });

  const sim = useMemo(() => {
    const ref = lookup(form.symbol);
    const k = URGENCY_K[form.urgency];
    const ratio = form.notional / ref.adv_usd;
    const sqrt = Math.sqrt(Math.max(ratio, 1e-9));
    const impact_bps = k * sqrt * ref.sigma_bps + ref.spread_bps / 2;
    const cost_usd = (form.notional * impact_bps) / 10_000;
    const adv_pct = ratio * 100;
    return { impact_bps, cost_usd, adv_pct, ref };
  }, [form]);

  const update = <K extends keyof Form>(k: K, v: Form[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Pre-trade slippage sim
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          square-root impact · spread + Almgren-Chriss urgency
        </p>
      </header>
      <div className="border-b border-border bg-bg/40 px-3 py-2">
        <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-slate-300 sm:grid-cols-4">
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
              onChange={(e) =>
                update("notional", Math.max(0, Number(e.target.value) || 0))
              }
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              urgency
            </span>
            <select
              value={form.urgency}
              onChange={(e) => update("urgency", e.target.value as Urgency)}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            >
              <option value="passive">passive</option>
              <option value="balanced">balanced</option>
              <option value="aggressive">aggressive</option>
            </select>
          </label>
        </div>
      </div>
      <div className="grid flex-1 grid-cols-2 gap-3 overflow-auto p-3 font-mono text-[11px] text-slate-300 sm:grid-cols-4">
        <Stat label="est. slippage" value={`${sim.impact_bps.toFixed(1)} bps`} />
        <Stat
          label="est. cost"
          value={`${Math.round(sim.cost_usd).toLocaleString()} USD`}
        />
        <Stat
          label="% of ADV"
          value={`${sim.adv_pct.toFixed(3)}%`}
          tone={
            sim.adv_pct > 1
              ? "text-rose-400"
              : sim.adv_pct > 0.1
                ? "text-amber-400"
                : "text-emerald-400"
          }
        />
        <Stat
          label="spread"
          value={`${sim.ref.spread_bps.toFixed(1)} bps`}
        />
        <Stat
          label="ADV USD"
          value={sim.ref.adv_usd.toLocaleString()}
        />
        <Stat label="vol (σ)" value={`${sim.ref.sigma_bps.toFixed(0)} bps`} />
        <Stat
          label="urgency k"
          value={URGENCY_K[form.urgency].toFixed(2)}
        />
        <Stat
          label="model"
          value="impact = k·√(N/ADV)·σ"
          tone="text-slate-500"
        />
      </div>
      <footer className="border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        deterministic client-side estimate · /api/risk/slippage_sim wires
        ``risk_engine.slippage_model`` when the backend adapter lands
      </footer>
    </section>
  );
}

function Stat({
  label,
  value,
  tone = "text-slate-200",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="rounded border border-border/60 bg-bg/30 p-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`mt-0.5 ${tone}`}>{value}</div>
    </div>
  );
}
