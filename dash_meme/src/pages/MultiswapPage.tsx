import { useState } from "react";

import { submitIntent } from "@/api/intent";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";
import { useAutonomy } from "@/state/autonomy";
import { pushToast } from "@/state/toast";

type Leg = {
  side: "buy" | "sell";
  symbol: string;
  size: string;
};

const DEFAULT_LEGS: Leg[] = [
  { side: "buy", symbol: "BONK/SOL", size: "0.05" },
  { side: "buy", symbol: "WIF/SOL", size: "0.05" },
  { side: "buy", symbol: "POPCAT/SOL", size: "0.05" },
];

export function MultiswapPage() {
  const [legs, setLegs] = useState<Leg[]>(DEFAULT_LEGS);
  const [autonomy] = useAutonomy();

  const setLeg = (i: number, patch: Partial<Leg>) =>
    setLegs(legs.map((l, idx) => (idx === i ? { ...l, ...patch } : l)));
  const add = () =>
    setLegs([...legs, { side: "buy", symbol: "", size: "0.05" }]);
  const remove = (i: number) => setLegs(legs.filter((_, idx) => idx !== i));

  const submit = async () => {
    const cleaned = legs.filter((l) => l.symbol.trim() && l.size.trim());
    if (cleaned.length < 2) {
      pushToast("Multiswap needs at least 2 legs", { tone: "warn" });
      return;
    }
    try {
      const res = await submitIntent({
        objective: "rebalance",
        risk_mode: autonomy,
        horizon: "intra-minute",
        focus: cleaned.map(
          (l) => `${l.side}:${l.symbol}:${l.size}`,
        ),
        reason: `multiswap ${cleaned.length} legs`,
        requestor: "operator",
      });
      pushToast(
        res.approved
          ? `Multiswap approved — ${res.summary}`
          : `Multiswap rejected — ${res.summary}`,
        { tone: res.approved ? "ok" : "warn" },
      );
    } catch (e) {
      pushToast(`Multiswap failed: ${(e as Error).message}`, { tone: "danger" });
    }
  };

  return (
    <div className="h-full p-2">
      <Panel
        title="Multiswap"
        right={
          <div className="flex items-center gap-2">
            <StatusPill tone="info">{autonomy}</StatusPill>
            <button
              type="button"
              onClick={submit}
              className="rounded bg-accent px-3 py-0.5 text-xs font-semibold text-bg"
            >
              Submit batch
            </button>
          </div>
        }
        bodyClassName="p-3"
      >
        <table className="w-full font-mono text-xs tabular-nums">
          <thead>
            <tr className="border-b border-hairline text-text-secondary">
              <th className="px-2 py-1 text-left">#</th>
              <th className="px-2 py-1 text-left">Side</th>
              <th className="px-2 py-1 text-left">Symbol</th>
              <th className="px-2 py-1 text-right">Size</th>
              <th className="px-2 py-1 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {legs.map((l, i) => (
              <tr key={i} className="dex-row">
                <td className="px-2 py-1 text-text-disabled">{i + 1}</td>
                <td className="px-2 py-1">
                  <select
                    value={l.side}
                    onChange={(e) =>
                      setLeg(i, { side: e.target.value as "buy" | "sell" })
                    }
                    className="h-6 rounded border border-border bg-surface-raised px-1"
                  >
                    <option value="buy">BUY</option>
                    <option value="sell">SELL</option>
                  </select>
                </td>
                <td className="px-2 py-1">
                  <input
                    value={l.symbol}
                    onChange={(e) => setLeg(i, { symbol: e.target.value })}
                    className="h-6 w-32 rounded border border-border bg-surface-raised px-1"
                  />
                </td>
                <td className="px-2 py-1 text-right">
                  <input
                    value={l.size}
                    onChange={(e) => setLeg(i, { size: e.target.value })}
                    className="h-6 w-24 rounded border border-border bg-surface-raised px-1 text-right"
                  />
                </td>
                <td className="px-2 py-1 text-right">
                  <button
                    type="button"
                    onClick={() => remove(i)}
                    className="text-danger hover:underline"
                  >
                    remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          type="button"
          onClick={add}
          className="mt-2 rounded border border-border px-2 py-0.5 text-xs"
        >
          + Add leg
        </button>
        <p className="mt-3 text-[11px] text-text-disabled">
          Multiswap is allowed in PAPER, AdForward, SHADOW, CANARY and LIVE
          modes. Governance enforces the per-mode caps; in CANARY each leg is
          clamped to 1% notional.
        </p>
      </Panel>
    </div>
  );
}
