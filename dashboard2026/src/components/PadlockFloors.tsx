import { Lock, ShieldAlert, ShieldCheck, Timer, KeyRound, Layers, FileLock } from "lucide-react";

/**
 * Manifest-pinned safety floors (PR-#2 spec §7).
 *
 * Read-only padlock row pinned to the top of every per-asset surface.
 * Each floor cites the manifest clause that locks it; the operator
 * can *see* and click for audit history but cannot disable any.
 */
interface FloorSpec {
  key: string;
  label: string;
  cite: string;
  icon: typeof Lock;
}

const FLOORS: readonly FloorSpec[] = [
  {
    key: "max-dd",
    label: "Max DD 4.00%",
    cite: "§22 axiom · immutable_core.foundation · constraint_compiler.py",
    icon: ShieldAlert,
  },
  {
    key: "kill-switch",
    label: "Kill-switch",
    cite: "Manifest §1 + §3 — operator/governance one-click halt",
    icon: ShieldCheck,
  },
  {
    key: "dead-man",
    label: "Dead-man",
    cite: "Manifest §3 — heartbeat-gated halt on operator absence",
    icon: Timer,
  },
  {
    key: "wallet-clock",
    label: "WARMUP 30d → SUPERVISED 30d/$100/d",
    cite: "Manifest §8 — wallet-policy progression",
    icon: KeyRound,
  },
  {
    key: "sandbox-gate",
    label: "Sandbox gate",
    cite: "Manifest §15 — sandbox patch pipeline gates every code change",
    icon: Layers,
  },
  {
    key: "fast-path",
    label: "Fast-path frozen",
    cite: "fast_execute_trade / fast_risk_cache — two-person hardware-key amend only",
    icon: Lock,
  },
  {
    key: "manifest-ro",
    label: "Manifest read-only",
    cite: "Addenda only via sandbox pipeline",
    icon: FileLock,
  },
];

export function PadlockFloors() {
  return (
    <div
      className="flex flex-wrap items-center gap-1 rounded border border-border bg-surface px-2 py-1 font-mono text-[11px] text-slate-300"
      role="list"
      aria-label="manifest-pinned safety floors"
      data-testid="padlock-floors"
    >
      <span className="pr-1 text-[10px] uppercase tracking-wider text-slate-500">
        Floors
      </span>
      {FLOORS.map((f) => {
        const Icon = f.icon;
        return (
          <span
            key={f.key}
            role="listitem"
            className="inline-flex items-center gap-1 rounded border border-amber-500/30 bg-amber-500/5 px-1.5 py-0.5 text-amber-300/90"
            title={`🔒 LOCKED — ${f.label} — ${f.cite}`}
            data-testid={`floor-${f.key}`}
          >
            <Icon className="h-3 w-3" />
            <span className="leading-none">{f.label}</span>
          </span>
        );
      })}
    </div>
  );
}
