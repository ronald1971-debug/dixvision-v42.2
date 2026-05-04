import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ChevronLeft, ChevronRight, Power } from "lucide-react";

import { fetchMode } from "@/api/feeds";
import { submitKill } from "@/api/intent";
import { useAutonomy, type AutonomyMode } from "@/state/autonomy";
import { useSelectedPair } from "@/state/pair";
import { pushToast } from "@/state/toast";

import { StatusPill } from "./StatusPill";

const CHAINS = ["solana", "ethereum", "base", "bsc"] as const;

const AUTONOMY_BANDS: ReadonlyArray<{ id: AutonomyMode; label: string }> = [
  { id: "manual", label: "Manual" },
  { id: "semi-auto", label: "Semi-Auto" },
  { id: "full-auto", label: "Full-Auto" },
];

function modeTone(mode: string): "ok" | "warn" | "danger" | "info" | "neutral" {
  const m = mode.toUpperCase();
  if (m === "LIVE" || m === "AUTO") return "ok";
  if (m === "CANARY") return "warn";
  if (m === "LOCKED" || m === "SAFE") return "danger";
  if (m === "SHADOW" || m === "PAPER") return "info";
  return "neutral";
}

export function TopBar({
  onToggleSidebar,
  sidebarCollapsed,
}: {
  onToggleSidebar: () => void;
  sidebarCollapsed: boolean;
}) {
  const [pair, setPair] = useSelectedPair();
  const [autonomy, setAutonomy] = useAutonomy();
  const modeQ = useQuery({
    queryKey: ["mode"],
    queryFn: fetchMode,
    refetchInterval: 5_000,
  });

  const modeStr =
    typeof modeQ.data?.mode === "object" && modeQ.data?.mode
      ? String(
          (modeQ.data.mode as Record<string, unknown>).mode ??
            (modeQ.data.mode as Record<string, unknown>).current ??
            "?",
        )
      : modeQ.isLoading
        ? "…"
        : "?";

  const handleKill = async () => {
    const ok = window.confirm(
      "ARM KILL SWITCH?\nThis transitions the system to LOCKED and aborts in-flight orders.",
    );
    if (!ok) return;
    try {
      const res = await submitKill({
        reason: "operator kill (DIX MEME top bar)",
        requestor: "operator",
      });
      pushToast(
        res.approved ? "Kill switch armed" : `Kill rejected: ${res.summary}`,
        { tone: res.approved ? "danger" : "warn" },
      );
    } catch (e) {
      pushToast(`Kill failed: ${(e as Error).message}`, { tone: "danger" });
    }
  };

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-surface px-2">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="rounded p-1 text-text-secondary hover:bg-surface-raised hover:text-text-primary"
        title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      {/* Chain selector */}
      <select
        value={pair.chain}
        onChange={(e) => setPair({ ...pair, chain: e.target.value })}
        className="h-7 rounded border border-border bg-surface-raised px-2 text-xs text-text-primary"
      >
        {CHAINS.map((c) => (
          <option key={c} value={c}>
            {c.toUpperCase()}
          </option>
        ))}
      </select>

      {/* Pair search / display */}
      <input
        value={pair.symbol}
        onChange={(e) => setPair({ ...pair, symbol: e.target.value })}
        placeholder="Search pair / mint / pool…"
        className="h-7 w-64 rounded border border-border bg-surface-raised px-2 font-mono text-xs text-text-primary placeholder:text-text-disabled focus:border-accent focus:outline-none"
      />

      <div className="flex-1" />

      {/* Autonomy band */}
      <div className="flex items-center gap-0.5 rounded border border-border bg-surface-raised p-0.5">
        {AUTONOMY_BANDS.map((b) => (
          <button
            key={b.id}
            type="button"
            onClick={() => setAutonomy(b.id)}
            className={`rounded px-2 py-0.5 text-xs transition-colors ${
              autonomy === b.id
                ? "bg-[var(--accent-soft)] text-accent"
                : "text-text-secondary hover:text-text-primary"
            }`}
            title={`Autonomy: ${b.label}`}
          >
            {b.label}
          </button>
        ))}
      </div>

      {/* System mode pill (live) */}
      <StatusPill tone={modeTone(modeStr)} title="System mode (live)">
        {modeStr}
      </StatusPill>

      {modeQ.isError && (
        <StatusPill tone="danger" title={(modeQ.error as Error).message}>
          <AlertTriangle size={12} />
          OFFLINE
        </StatusPill>
      )}

      {/* Kill switch */}
      <button
        type="button"
        onClick={handleKill}
        className="flex h-7 items-center gap-1 rounded border border-danger px-2 text-xs font-medium text-danger hover:bg-[var(--danger-soft)]"
        title="Arm kill switch (LOCK system)"
      >
        <Power size={12} /> KILL
      </button>
    </header>
  );
}
