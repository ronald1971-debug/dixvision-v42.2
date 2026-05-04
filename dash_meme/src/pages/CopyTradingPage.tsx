import { useEffect, useState } from "react";

import { submitIntent } from "@/api/intent";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";
import { useAutonomy } from "@/state/autonomy";
import { pushToast } from "@/state/toast";

const STORAGE_KEY = "dixmeme.copy.wallets";

type CopyWallet = {
  address: string;
  label: string;
  enabled: boolean;
  ratio: number; // 0..1, fraction of source size
  maxNotional: number; // USD per mirrored trade
  chain: string;
};

const DEFAULT: CopyWallet[] = [];

function read(): CopyWallet[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed as CopyWallet[];
    }
  } catch {
    // ignore
  }
  return DEFAULT;
}

function write(rows: CopyWallet[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  } catch {
    // ignore
  }
}

export function CopyTradingPage() {
  const [wallets, setWallets] = useState<CopyWallet[]>(read);
  const [autonomy] = useAutonomy();
  const [draft, setDraft] = useState<CopyWallet>({
    address: "",
    label: "",
    enabled: true,
    ratio: 0.1,
    maxNotional: 100,
    chain: "solana",
  });

  useEffect(() => write(wallets), [wallets]);

  const add = () => {
    if (!draft.address.trim()) {
      pushToast("Wallet address required", { tone: "warn" });
      return;
    }
    setWallets([...wallets, { ...draft }]);
    setDraft({ ...draft, address: "", label: "" });
  };
  const remove = (i: number) =>
    setWallets(wallets.filter((_, idx) => idx !== i));
  const toggle = (i: number) =>
    setWallets(
      wallets.map((w, idx) => (idx === i ? { ...w, enabled: !w.enabled } : w)),
    );

  const arm = async () => {
    const enabled = wallets.filter((w) => w.enabled);
    if (enabled.length === 0) {
      pushToast("No copy wallets enabled", { tone: "warn" });
      return;
    }
    try {
      const res = await submitIntent({
        objective: "copy",
        risk_mode: autonomy,
        horizon: "intra-minute",
        focus: enabled.flatMap((w) => [
          `wallet:${w.address}`,
          `chain:${w.chain}`,
          `ratio:${w.ratio}`,
          `cap:$${w.maxNotional}`,
        ]),
        reason: `arm copy-trading on ${enabled.length} wallets`,
        requestor: "operator",
      });
      pushToast(
        res.approved
          ? `Copy armed — ${res.summary}`
          : `Copy rejected — ${res.summary}`,
        { tone: res.approved ? "ok" : "warn" },
      );
    } catch (e) {
      pushToast(`Arm failed: ${(e as Error).message}`, { tone: "danger" });
    }
  };

  return (
    <div className="grid h-full grid-cols-12 gap-2 p-2">
      <div className="col-span-8 min-h-0">
        <Panel
          title="Mirrored wallets"
          right={
            <div className="flex items-center gap-2 text-[10px]">
              <StatusPill tone="info">{autonomy}</StatusPill>
              <button
                type="button"
                onClick={arm}
                className="rounded bg-accent px-2 py-0.5 text-bg"
              >
                ARM
              </button>
            </div>
          }
        >
          <table className="w-full font-mono text-[11px] tabular-nums">
            <thead className="sticky top-0 bg-surface text-text-secondary">
              <tr className="border-b border-hairline">
                <th className="px-2 py-1 text-left">On</th>
                <th className="px-2 py-1 text-left">Label</th>
                <th className="px-2 py-1 text-left">Wallet</th>
                <th className="px-2 py-1 text-left">Chain</th>
                <th className="px-2 py-1 text-right">Ratio</th>
                <th className="px-2 py-1 text-right">Cap $</th>
                <th className="px-2 py-1 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {wallets.length === 0 && (
                <tr>
                  <td
                    colSpan={7}
                    className="px-2 py-3 text-center text-text-disabled"
                  >
                    No mirrored wallets configured.
                  </td>
                </tr>
              )}
              {wallets.map((w, i) => (
                <tr key={i} className="dex-row">
                  <td className="px-2 py-0.5">
                    <input
                      type="checkbox"
                      checked={w.enabled}
                      onChange={() => toggle(i)}
                    />
                  </td>
                  <td className="px-2 py-0.5">{w.label || "—"}</td>
                  <td className="truncate px-2 py-0.5">
                    <span className="text-text-secondary">
                      {w.address.slice(0, 6)}…{w.address.slice(-4)}
                    </span>
                  </td>
                  <td className="px-2 py-0.5 text-text-disabled">{w.chain}</td>
                  <td className="px-2 py-0.5 text-right">{w.ratio.toFixed(2)}</td>
                  <td className="px-2 py-0.5 text-right">
                    ${w.maxNotional.toLocaleString()}
                  </td>
                  <td className="px-2 py-0.5 text-right">
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
        </Panel>
      </div>
      <div className="col-span-4 min-h-0">
        <Panel title="Add wallet" bodyClassName="p-3">
          <div className="space-y-2 text-xs">
            <Field
              label="Address"
              value={draft.address}
              onChange={(v) => setDraft({ ...draft, address: v })}
            />
            <Field
              label="Label"
              value={draft.label}
              onChange={(v) => setDraft({ ...draft, label: v })}
            />
            <label className="block">
              <span className="text-text-secondary">Chain</span>
              <select
                value={draft.chain}
                onChange={(e) => setDraft({ ...draft, chain: e.target.value })}
                className="mt-0.5 h-7 w-full rounded border border-border bg-surface-raised px-2 text-text-primary"
              >
                <option value="solana">solana</option>
                <option value="ethereum">ethereum</option>
                <option value="base">base</option>
                <option value="bsc">bsc</option>
              </select>
            </label>
            <Field
              label="Mirror ratio (0..1)"
              value={String(draft.ratio)}
              onChange={(v) =>
                setDraft({
                  ...draft,
                  ratio: Math.max(0, Math.min(1, parseFloat(v) || 0)),
                })
              }
            />
            <Field
              label="Per-trade cap (USD)"
              value={String(draft.maxNotional)}
              onChange={(v) =>
                setDraft({
                  ...draft,
                  maxNotional: Math.max(0, parseFloat(v) || 0),
                })
              }
            />
            <button
              type="button"
              onClick={add}
              className="w-full rounded bg-accent px-3 py-1.5 text-sm font-semibold text-bg"
            >
              Add wallet
            </button>
            <p className="text-[10px] text-text-disabled">
              Mirror configuration is stored locally. Arming submits a copy
              intent to <span className="font-mono">/api/dashboard/action/intent</span>{" "}
              → Governance, which decides actual execution per autonomy band.
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-text-secondary">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 h-7 w-full rounded border border-border bg-surface-raised px-2 font-mono text-text-primary focus:border-accent focus:outline-none"
      />
    </label>
  );
}
