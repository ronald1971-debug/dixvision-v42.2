import { useMemo, useState } from "react";

import { getAutonomyMode } from "@/state/autonomy";

/**
 * SL/TP Builder (PR-#2 spec §4).
 *
 * Same engine on every form. Primitives:
 *   - Hard SL (% or absolute)
 *   - Trailing SL (% from high/low-water, with ratchet)
 *   - Timed SL (auto-exit after N minutes if price hasn't moved ±X%)
 *   - TP ladder (up to 5 legs, per-leg size % + trigger price/%)
 *   - Breakeven-after-first-TP toggle
 *   - OCO + Bracket (one-cancels-other / linked stop+target)
 *   - Form-specific extensions: rug-trip SL, dev-dump SL, in-bundle
 *     SL/TP (DEX/memecoin); margin-aware SL, funding-flip SL,
 *     reduce-only TP (perps).
 *
 * Per-form preset table from §4.2 is rendered at the top so the
 * operator can stamp Conservative / Standard / Aggressive defaults
 * with one click. Every override emits OPERATOR/SETTINGS_CHANGED to
 * the audit ledger.
 *
 * AI-propose hook: a button asks Indira to propose an SL/TP given
 * the current BeliefState + position; the proposal arrives through
 * the operator-approval edge with a DecisionTrace "Why" panel.
 */
type FormKey =
  | "spot"
  | "perps"
  | "dex"
  | "memecoin-copy"
  | "memecoin-normal"
  | "memecoin-sniper"
  | "forex"
  | "stocks"
  | "nft";

type PresetKey = "conservative" | "standard" | "aggressive";

interface TpLeg {
  size_pct: number;
  trigger_pct: number;
}

interface Preset {
  hard_sl_pct: number;
  trailing_pct: number | null;
  timed_minutes: number | null;
  tp_ladder: TpLeg[];
  breakeven_after_first_tp: boolean;
  bracket: boolean;
  oco: boolean;
  notes?: string;
}

const PRESETS: Record<FormKey, Record<PresetKey, Preset>> = {
  spot: {
    conservative: {
      hard_sl_pct: 3,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 5 },
        { size_pct: 33, trigger_pct: 10 },
        { size_pct: 34, trigger_pct: 20 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    standard: {
      hard_sl_pct: 5,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 25, trigger_pct: 20 },
        { size_pct: 25, trigger_pct: 50 },
        { size_pct: 50, trigger_pct: 100 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 8,
      trailing_pct: 15,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 50 },
        { size_pct: 33, trigger_pct: 100 },
        { size_pct: 34, trigger_pct: 200 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "trailing runner",
    },
  },
  perps: {
    conservative: {
      hard_sl_pct: 1,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 3 },
        { size_pct: 33, trigger_pct: 6 },
        { size_pct: 34, trigger_pct: 12 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "margin 50% buffer · liq-aware · funding-flip exit",
    },
    standard: {
      hard_sl_pct: 3,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 10 },
        { size_pct: 33, trigger_pct: 25 },
        { size_pct: 34, trigger_pct: 50 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "reduce-only TP legs",
    },
    aggressive: {
      hard_sl_pct: 5,
      trailing_pct: 20,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 25 },
        { size_pct: 33, trigger_pct: 50 },
        { size_pct: 34, trigger_pct: 100 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
  },
  dex: {
    conservative: {
      hard_sl_pct: 5,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 20 },
        { size_pct: 33, trigger_pct: 50 },
        { size_pct: 34, trigger_pct: 100 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "limit-order loop · rug-trip SL · dev-dump SL",
    },
    standard: {
      hard_sl_pct: 10,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 50 },
        { size_pct: 33, trigger_pct: 100 },
        { size_pct: 34, trigger_pct: 200 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 15,
      trailing_pct: 25,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 100 },
        { size_pct: 33, trigger_pct: 400 },
        { size_pct: 34, trigger_pct: 900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "2x/5x/10x · trailing runner",
    },
  },
  "memecoin-copy": {
    conservative: {
      hard_sl_pct: 30,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 100 },
        { size_pct: 50, trigger_pct: 400 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "mirrors leader · auto-exit on leader exit",
    },
    standard: {
      hard_sl_pct: 40,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 200 },
        { size_pct: 50, trigger_pct: 600 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 50,
      trailing_pct: 25,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 400 },
        { size_pct: 33, trigger_pct: 900 },
        { size_pct: 34, trigger_pct: 1900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "trailing runner",
    },
  },
  "memecoin-normal": {
    conservative: {
      hard_sl_pct: 30,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 100 },
        { size_pct: 50, trigger_pct: 400 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "honeypot-check · dev-dump watchdog",
    },
    standard: {
      hard_sl_pct: 40,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 100 },
        { size_pct: 33, trigger_pct: 400 },
        { size_pct: 34, trigger_pct: 900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 50,
      trailing_pct: 30,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 400 },
        { size_pct: 33, trigger_pct: 900 },
        { size_pct: 34, trigger_pct: 4900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "30% trailing runner",
    },
  },
  "memecoin-sniper": {
    conservative: {
      hard_sl_pct: 40,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 100 },
        { size_pct: 50, trigger_pct: 400 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "in-bundle SL+TP — pre-signed in Jito/Flashbots bundle",
    },
    standard: {
      hard_sl_pct: 50,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 200 },
        { size_pct: 50, trigger_pct: 900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 60,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 400 },
        { size_pct: 33, trigger_pct: 2400 },
        { size_pct: 34, trigger_pct: 9900 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
  },
  forex: {
    // Forex naturally trades in pips, but the SL/TP engine is unit-agnostic
    // and stores hard_sl_pct / trigger_pct as percentages of mid price. The
    // values below are the percentage equivalent of the pip targets in `notes`,
    // computed against a EUR/USD-style ~1.1000 reference (1 pip ≈ 0.009%).
    conservative: {
      hard_sl_pct: 0.09,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 0.18 },
        { size_pct: 33, trigger_pct: 0.36 },
        { size_pct: 34, trigger_pct: 0.72 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "SL 10 pips · TP 20/40/80 pips (≈0.09% / 0.18-0.72%)",
    },
    standard: {
      hard_sl_pct: 0.18,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 0.36 },
        { size_pct: 33, trigger_pct: 0.72 },
        { size_pct: 34, trigger_pct: 1.45 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "SL 20 pips · TP 40/80/160 pips (≈0.18% / 0.36-1.45%)",
    },
    aggressive: {
      hard_sl_pct: 0.27,
      trailing_pct: 0.5,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 0.55 },
        { size_pct: 33, trigger_pct: 1.1 },
        { size_pct: 34, trigger_pct: 2.18 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "SL 30 pips · TP 60/120/240 pips (≈0.27% / 0.55-2.18%)",
    },
  },
  stocks: {
    conservative: {
      hard_sl_pct: 2,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 4 },
        { size_pct: 33, trigger_pct: 8 },
        { size_pct: 34, trigger_pct: 16 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    standard: {
      hard_sl_pct: 4,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 10 },
        { size_pct: 33, trigger_pct: 20 },
        { size_pct: 34, trigger_pct: 40 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 6,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 33, trigger_pct: 20 },
        { size_pct: 33, trigger_pct: 40 },
        { size_pct: 34, trigger_pct: 80 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
  },
  nft: {
    conservative: {
      hard_sl_pct: 15,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 25 },
        { size_pct: 50, trigger_pct: 50 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
      notes: "SL/TP relative to floor",
    },
    standard: {
      hard_sl_pct: 20,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 50 },
        { size_pct: 50, trigger_pct: 100 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
    aggressive: {
      hard_sl_pct: 25,
      trailing_pct: null,
      timed_minutes: null,
      tp_ladder: [
        { size_pct: 50, trigger_pct: 100 },
        { size_pct: 50, trigger_pct: 300 },
      ],
      breakeven_after_first_tp: true,
      bracket: true,
      oco: true,
    },
  },
};

const FORM_LABEL: Record<FormKey, string> = {
  spot: "Spot",
  perps: "Perps",
  dex: "DEX",
  "memecoin-copy": "Memecoin · Copy",
  "memecoin-normal": "Memecoin · Normal",
  "memecoin-sniper": "Memecoin · Sniper",
  forex: "Forex",
  stocks: "Stocks",
  nft: "NFT",
};

export interface SLTPBuilderProps {
  /** Form-specific defaults. */
  form: FormKey;
}

export function SLTPBuilder({ form }: SLTPBuilderProps) {
  const [presetKey, setPresetKey] = useState<PresetKey>("standard");
  const [draft, setDraft] = useState<Preset>(() => PRESETS[form][presetKey]);

  const presetMatrix = useMemo(() => PRESETS[form], [form]);

  function applyPreset(key: PresetKey) {
    setPresetKey(key);
    setDraft(PRESETS[form][key]);
  }

  function patchDraft(patch: Partial<Preset>) {
    setDraft((d) => ({ ...d, ...patch }));
  }

  function patchTpLeg(idx: number, patch: Partial<TpLeg>) {
    setDraft((d) => ({
      ...d,
      tp_ladder: d.tp_ladder.map((leg, i) =>
        i === idx ? { ...leg, ...patch } : leg,
      ),
    }));
  }

  function addTpLeg() {
    setDraft((d) => ({
      ...d,
      tp_ladder: [...d.tp_ladder, { size_pct: 0, trigger_pct: 0 }].slice(0, 5),
    }));
  }

  function removeTpLeg(idx: number) {
    setDraft((d) => ({
      ...d,
      tp_ladder: d.tp_ladder.filter((_, i) => i !== idx),
    }));
  }

  function requestAIPropose() {
    // AI-propose hook (PR-#2 §4 + cognitive chat surface). Routed
    // through the operator-approval edge so the UI sees the proposal
    // only after Governance accepts it.
    void fetch("/api/cognitive/sl_tp/propose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ form, current: draft }),
    });
  }

  function commit() {
    void fetch("/api/operator/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: "OPERATOR/SETTINGS_CHANGED",
        setting: `sl_tp/${form}`,
        next: draft,
        autonomy_mode: getAutonomyMode(),
        timestamp_iso: new Date().toISOString(),
      }),
    });
  }

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface text-sm">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            SL / TP Builder · {FORM_LABEL[form]}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            same engine on every form · primitives: hard / trailing / timed /
            ladder / breakeven / OCO / bracket
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          PR-#2 §4
        </span>
      </header>
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <span className="font-mono text-[10px] uppercase text-slate-500">
          Preset
        </span>
        {(Object.keys(presetMatrix) as PresetKey[]).map((key) => (
          <button
            key={key}
            type="button"
            className={`rounded border px-2 py-1 font-mono text-[11px] uppercase tracking-wider ${
              presetKey === key
                ? "border-accent bg-accent text-bg"
                : "border-border bg-bg text-slate-400 hover:text-accent"
            }`}
            onClick={() => applyPreset(key)}
          >
            {key}
          </button>
        ))}
        <button
          type="button"
          onClick={requestAIPropose}
          className="ml-auto rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 font-mono text-[11px] uppercase tracking-wider text-emerald-300 hover:bg-emerald-500/20"
          title="Indira proposes an SL/TP given the current BeliefState + position; arrives via the operator-approval edge with DecisionTrace."
        >
          AI propose
        </button>
        <button
          type="button"
          onClick={commit}
          className="rounded border border-accent/60 bg-accent/15 px-2 py-1 font-mono text-[11px] uppercase tracking-wider text-accent hover:bg-accent/25"
        >
          Commit
        </button>
      </div>
      <div className="grid flex-1 grid-cols-2 gap-3 overflow-auto p-3">
        <Field label="Hard SL %">
          <NumberInput
            value={draft.hard_sl_pct}
            onChange={(v) => patchDraft({ hard_sl_pct: v })}
            step={0.1}
          />
        </Field>
        <Field label="Trailing SL %">
          <NumberInput
            value={draft.trailing_pct ?? 0}
            onChange={(v) =>
              patchDraft({ trailing_pct: v > 0 ? v : null })
            }
            step={0.5}
          />
        </Field>
        <Field label="Timed SL (min)">
          <NumberInput
            value={draft.timed_minutes ?? 0}
            onChange={(v) =>
              patchDraft({ timed_minutes: v > 0 ? v : null })
            }
            step={1}
          />
        </Field>
        <Field label="Breakeven after first TP">
          <Toggle
            value={draft.breakeven_after_first_tp}
            onChange={(v) => patchDraft({ breakeven_after_first_tp: v })}
          />
        </Field>
        <Field label="OCO">
          <Toggle
            value={draft.oco}
            onChange={(v) => patchDraft({ oco: v })}
          />
        </Field>
        <Field label="Bracket">
          <Toggle
            value={draft.bracket}
            onChange={(v) => patchDraft({ bracket: v })}
          />
        </Field>
        <div className="col-span-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase text-slate-500">
              TP ladder · max 5 legs
            </span>
            <button
              type="button"
              onClick={addTpLeg}
              className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] uppercase text-slate-400 hover:text-accent"
            >
              + leg
            </button>
          </div>
          <table className="w-full text-[12px]">
            <thead className="text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-1 py-0.5 text-left">#</th>
                <th className="px-1 py-0.5 text-left">size %</th>
                <th className="px-1 py-0.5 text-left">trigger %</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {draft.tp_ladder.map((leg, idx) => (
                <tr key={idx} className="border-t border-border">
                  <td className="px-1 py-1 font-mono text-slate-500">
                    TP{idx + 1}
                  </td>
                  <td className="px-1 py-1">
                    <NumberInput
                      value={leg.size_pct}
                      onChange={(v) => patchTpLeg(idx, { size_pct: v })}
                      step={1}
                    />
                  </td>
                  <td className="px-1 py-1">
                    <NumberInput
                      value={leg.trigger_pct}
                      onChange={(v) => patchTpLeg(idx, { trigger_pct: v })}
                      step={1}
                    />
                  </td>
                  <td className="px-1 py-1 text-right">
                    <button
                      type="button"
                      onClick={() => removeTpLeg(idx)}
                      className="rounded border border-border px-1 py-0.5 font-mono text-[10px] uppercase text-slate-500 hover:text-danger"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {draft.notes && (
          <div className="col-span-2 rounded border border-border bg-bg/40 px-2 py-1 font-mono text-[11px] text-slate-400">
            {draft.notes}
          </div>
        )}
      </div>
    </div>
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
    <label className="flex flex-col gap-0.5 text-[11px] text-slate-400">
      <span className="font-mono uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}

function NumberInput({
  value,
  onChange,
  step = 1,
}: {
  value: number;
  onChange: (next: number) => void;
  step?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className="rounded border border-border bg-bg px-2 py-1 font-mono text-[12px] text-slate-100 focus:border-accent focus:outline-none"
    />
  );
}

function Toggle({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      role="switch"
      aria-checked={value}
      className={`flex h-6 w-11 items-center rounded-full border px-0.5 transition-colors ${
        value ? "border-accent bg-accent/40" : "border-border bg-bg"
      }`}
    >
      <span
        className={`h-4 w-4 rounded-full bg-slate-200 transition-transform ${
          value ? "translate-x-5" : "translate-x-0"
        }`}
      />
    </button>
  );
}
