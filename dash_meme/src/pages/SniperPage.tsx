import { useEffect, useState } from "react";

import { submitIntent } from "@/api/intent";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";
import { useAutonomy } from "@/state/autonomy";
import { pushToast } from "@/state/toast";

const STORAGE_KEY = "dixmeme.sniper.rules";

type SniperRule = {
  enabled: boolean;
  source: "pumpfun" | "raydium-migration" | "any";
  minLiqUsd: number;
  maxDevPct: number;
  maxBuyTaxPct: number;
  maxSellTaxPct: number;
  maxGasGwei: number;
  buyNotionalUsd: number;
  takeProfitPct: number;
  stopLossPct: number;
};

const DEFAULT: SniperRule = {
  enabled: false,
  source: "pumpfun",
  minLiqUsd: 5_000,
  maxDevPct: 5,
  maxBuyTaxPct: 5,
  maxSellTaxPct: 5,
  maxGasGwei: 50,
  buyNotionalUsd: 25,
  takeProfitPct: 100,
  stopLossPct: 30,
};

function read(): SniperRule {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT, ...(JSON.parse(raw) as Partial<SniperRule>) };
  } catch {
    // ignore
  }
  return DEFAULT;
}

function write(r: SniperRule) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(r));
  } catch {
    // ignore
  }
}

export function SniperPage() {
  const [rule, setRule] = useState<SniperRule>(read);
  const [autonomy] = useAutonomy();
  useEffect(() => write(rule), [rule]);

  const arm = async () => {
    if (!rule.enabled) {
      pushToast("Sniper rule disabled — toggle it on first", { tone: "warn" });
      return;
    }
    try {
      const res = await submitIntent({
        objective: "snipe",
        risk_mode: autonomy,
        horizon: "intra-second",
        focus: [
          `source:${rule.source}`,
          `min_liq:$${rule.minLiqUsd}`,
          `max_dev:${rule.maxDevPct}%`,
          `max_buytax:${rule.maxBuyTaxPct}%`,
          `max_selltax:${rule.maxSellTaxPct}%`,
          `max_gas:${rule.maxGasGwei}gwei`,
          `buy:$${rule.buyNotionalUsd}`,
          `tp:${rule.takeProfitPct}%`,
          `sl:${rule.stopLossPct}%`,
        ],
        reason: "arm sniper",
        requestor: "operator",
      });
      pushToast(res.approved ? `Sniper armed — ${res.summary}` : `Rejected — ${res.summary}`, {
        tone: res.approved ? "ok" : "warn",
      });
    } catch (e) {
      pushToast(`Arm failed: ${(e as Error).message}`, { tone: "danger" });
    }
  };

  const num = (k: keyof SniperRule) => (v: string) =>
    setRule({ ...rule, [k]: Math.max(0, parseFloat(v) || 0) });

  return (
    <div className="grid h-full grid-cols-12 gap-2 p-2">
      <div className="col-span-5 min-h-0">
        <Panel
          title="Sniper rule"
          right={
            <div className="flex items-center gap-2">
              <StatusPill tone="info">{autonomy}</StatusPill>
              <StatusPill tone={rule.enabled ? "ok" : "neutral"}>
                {rule.enabled ? "ARMED" : "OFF"}
              </StatusPill>
            </div>
          }
          bodyClassName="p-3"
        >
          <div className="space-y-2 text-xs">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={rule.enabled}
                onChange={(e) => setRule({ ...rule, enabled: e.target.checked })}
              />
              Enable sniper
            </label>
            <label className="block">
              <span className="text-text-secondary">Source</span>
              <select
                value={rule.source}
                onChange={(e) =>
                  setRule({ ...rule, source: e.target.value as SniperRule["source"] })
                }
                className="mt-0.5 h-7 w-full rounded border border-border bg-surface-raised px-2 text-text-primary"
              >
                <option value="pumpfun">pump.fun launches</option>
                <option value="raydium-migration">
                  raydium migrations
                </option>
                <option value="any">any new pair</option>
              </select>
            </label>
            <NumField
              label="Min liquidity (USD)"
              value={rule.minLiqUsd}
              onChange={num("minLiqUsd")}
            />
            <NumField
              label="Max dev supply %"
              value={rule.maxDevPct}
              onChange={num("maxDevPct")}
            />
            <NumField
              label="Max buy tax %"
              value={rule.maxBuyTaxPct}
              onChange={num("maxBuyTaxPct")}
            />
            <NumField
              label="Max sell tax %"
              value={rule.maxSellTaxPct}
              onChange={num("maxSellTaxPct")}
            />
            <NumField
              label="Max gas (gwei)"
              value={rule.maxGasGwei}
              onChange={num("maxGasGwei")}
            />
            <NumField
              label="Buy notional (USD)"
              value={rule.buyNotionalUsd}
              onChange={num("buyNotionalUsd")}
            />
            <NumField
              label="Take profit %"
              value={rule.takeProfitPct}
              onChange={num("takeProfitPct")}
            />
            <NumField
              label="Stop loss %"
              value={rule.stopLossPct}
              onChange={num("stopLossPct")}
            />
            <button
              type="button"
              onClick={arm}
              className="w-full rounded bg-accent px-3 py-2 text-sm font-semibold text-bg disabled:opacity-50"
              disabled={!rule.enabled}
            >
              ARM SNIPER
            </button>
          </div>
        </Panel>
      </div>
      <div className="col-span-7 min-h-0">
        <Panel title="Pre-launch / migration queue" bodyClassName="p-3">
          <p className="text-xs text-text-secondary">
            Launches matching the rule above are streamed by the harness via{" "}
            <span className="font-mono">/api/feeds/pumpfun/recent</span> and{" "}
            <span className="font-mono">/api/feeds/raydium/recent</span>. Arming
            submits a sniper intent — Governance evaluates each candidate
            launch against the current SystemMode and autonomy band before any
            order leaves the chokepoint.
          </p>
          <p className="mt-3 text-[11px] text-text-disabled">
            Manual band: every snipe candidate prompts you. Semi-auto: snipes
            within caps go automatically; out-of-cap snipes prompt you.
            Full-auto: requires AUTO mode + drift-clean. Otherwise Governance
            rejects with{" "}
            <span className="font-mono">UNAUTHORIZED_DIRECTIVE</span>.
          </p>
        </Panel>
      </div>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-text-secondary">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 h-7 w-full rounded border border-border bg-surface-raised px-2 font-mono text-text-primary focus:border-accent focus:outline-none"
      />
    </label>
  );
}
