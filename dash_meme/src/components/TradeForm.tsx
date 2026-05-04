import { useState } from "react";

import { submitIntent } from "@/api/intent";
import { useAutonomy } from "@/state/autonomy";
import { pushToast } from "@/state/toast";

import { StatusPill } from "./StatusPill";

type Side = "buy" | "sell";
type OrderType = "market" | "limit";

export function TradeForm({
  symbol,
  chain,
  defaultSize = "0.1",
}: {
  symbol: string;
  chain: string;
  defaultSize?: string;
}) {
  const [autonomy] = useAutonomy();
  const [side, setSide] = useState<Side>("buy");
  const [type, setType] = useState<OrderType>("market");
  const [size, setSize] = useState(defaultSize);
  const [price, setPrice] = useState("");
  const [slippage, setSlippage] = useState("1.0");
  const [tp, setTp] = useState("");
  const [sl, setSl] = useState("");
  const [mev, setMev] = useState(true);
  const [busy, setBusy] = useState(false);
  const [lastDecision, setLastDecision] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    try {
      const focus = [
        `pair:${symbol}`,
        `chain:${chain}`,
        `side:${side}`,
        `type:${type}`,
        `size:${size}`,
        ...(price ? [`limit:${price}`] : []),
        `slippage:${slippage}%`,
        ...(tp ? [`tp:${tp}`] : []),
        ...(sl ? [`sl:${sl}`] : []),
        `mev:${mev ? "on" : "off"}`,
      ];
      const res = await submitIntent({
        objective: "trade",
        risk_mode: autonomy,
        horizon: "intra-minute",
        focus,
        reason: `manual ${side} ${size} ${symbol}`,
        requestor: "operator",
      });
      const verb = res.approved ? "Approved" : "Rejected";
      pushToast(`${verb}: ${res.summary}`, {
        tone: res.approved ? "ok" : "warn",
      });
      setLastDecision(`${verb} — ${res.summary}`);
    } catch (e) {
      pushToast(`Submit failed: ${(e as Error).message}`, { tone: "danger" });
      setLastDecision(`error — ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2 p-3 text-xs">
      <div className="flex items-center justify-between">
        <span className="text-text-secondary">Pair</span>
        <span className="font-mono text-text-primary">
          {symbol} <span className="text-text-disabled">· {chain}</span>
        </span>
      </div>

      <div className="grid grid-cols-2 gap-1">
        <button
          type="button"
          onClick={() => setSide("buy")}
          className={`rounded border px-2 py-1 font-semibold ${
            side === "buy"
              ? "border-ok bg-[var(--ok-soft)] text-ok"
              : "border-border text-text-secondary"
          }`}
        >
          BUY
        </button>
        <button
          type="button"
          onClick={() => setSide("sell")}
          className={`rounded border px-2 py-1 font-semibold ${
            side === "sell"
              ? "border-danger bg-[var(--danger-soft)] text-danger"
              : "border-border text-text-secondary"
          }`}
        >
          SELL
        </button>
      </div>

      <div className="grid grid-cols-2 gap-1">
        <button
          type="button"
          onClick={() => setType("market")}
          className={`rounded border px-2 py-1 ${
            type === "market"
              ? "border-accent text-accent"
              : "border-border text-text-secondary"
          }`}
        >
          Market
        </button>
        <button
          type="button"
          onClick={() => setType("limit")}
          className={`rounded border px-2 py-1 ${
            type === "limit"
              ? "border-accent text-accent"
              : "border-border text-text-secondary"
          }`}
        >
          Limit
        </button>
      </div>

      <Field label="Size" value={size} onChange={setSize} mono />
      {type === "limit" && (
        <Field label="Limit price" value={price} onChange={setPrice} mono />
      )}
      <Field
        label="Max slippage %"
        value={slippage}
        onChange={setSlippage}
        mono
      />
      <Field label="Take profit" value={tp} onChange={setTp} mono optional />
      <Field label="Stop loss" value={sl} onChange={setSl} mono optional />

      <label className="flex items-center gap-2 text-text-secondary">
        <input
          type="checkbox"
          checked={mev}
          onChange={(e) => setMev(e.target.checked)}
        />
        MEV protection (private mempool)
      </label>

      <div className="flex items-center justify-between border-t border-hairline pt-2">
        <span className="text-text-secondary">Autonomy</span>
        <StatusPill tone="info">{autonomy}</StatusPill>
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={busy}
        className={`w-full rounded px-3 py-2 text-sm font-semibold transition-colors ${
          side === "buy"
            ? "bg-ok text-bg hover:opacity-90"
            : "bg-danger text-bg hover:opacity-90"
        } disabled:opacity-50`}
      >
        {busy ? "Submitting…" : `${side.toUpperCase()} ${size} ${symbol}`}
      </button>

      <p className="text-[10px] text-text-disabled">
        Routes via <span className="font-mono">/api/dashboard/action/intent</span>{" "}
        → Governance. Operator-approval requirement is determined by autonomy
        band &amp; current SystemMode.
      </p>

      {lastDecision && (
        <div className="rounded border border-hairline bg-surface-raised px-2 py-1 text-[11px] text-text-secondary">
          {lastDecision}
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  mono,
  optional,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  mono?: boolean;
  optional?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-[11px] text-text-secondary">
        {label}
        {optional && (
          <span className="ml-1 text-text-disabled">(optional)</span>
        )}
      </span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`mt-0.5 h-7 w-full rounded border border-border bg-surface-raised px-2 ${
          mono ? "font-mono" : ""
        } text-text-primary focus:border-accent focus:outline-none`}
      />
    </label>
  );
}
