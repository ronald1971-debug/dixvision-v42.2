import {
  AUTONOMY_MODES,
  type AutonomyMode,
  useAutonomyMode,
} from "@/state/autonomy";

const TONE: Record<AutonomyMode, string> = {
  USER_CONTROLLED: "bg-slate-700/40 border-slate-500/60 text-slate-300",
  SEMI_AUTO: "bg-accent/15 border-accent/60 text-accent",
  FULL_AUTO: "bg-emerald-500/15 border-emerald-500/60 text-emerald-300",
};

const TONE_ACTIVE: Record<AutonomyMode, string> = {
  USER_CONTROLLED: "bg-slate-300 border-slate-100 text-bg",
  SEMI_AUTO: "bg-accent border-accent text-bg",
  FULL_AUTO: "bg-emerald-400 border-emerald-200 text-bg",
};

const LABEL: Record<AutonomyMode, string> = {
  USER_CONTROLLED: "Manual",
  SEMI_AUTO: "Semi-auto",
  FULL_AUTO: "Full-auto",
};

const TOOLTIP: Record<AutonomyMode, string> = {
  USER_CONTROLLED:
    "Manual — every intent waits for an operator click. Strategies still emit signals, but execution is gated on you.",
  SEMI_AUTO:
    "Semi-auto — auto-trading inside the operator-set envelope. Settings dials apply on the next tick. One-click fallback to manual.",
  FULL_AUTO:
    "Full-auto — auto-trading without asking. Manifest floors (4% DD, kill-switch, dead-man, WARMUP, sandbox-gate, fast-path-frozen) still apply. One-click fall-back to semi/manual while trades are running.",
};

export function AutonomyRibbon() {
  const [mode, setMode] = useAutonomyMode();
  return (
    <div
      className="flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider"
      role="radiogroup"
      aria-label="autonomy mode"
      data-testid="autonomy-ribbon"
    >
      <span className="px-1 text-[10px] text-slate-500">Autonomy</span>
      {AUTONOMY_MODES.map((m) => {
        const isActive = mode === m;
        const cls = `rounded border px-2 py-1 leading-none transition-colors duration-150 ${
          isActive ? TONE_ACTIVE[m] : TONE[m]
        }`;
        return (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={isActive}
            className={cls}
            title={TOOLTIP[m]}
            onClick={() => setMode(m)}
            data-active={isActive ? "true" : "false"}
            data-testid={`autonomy-${m}`}
          >
            {LABEL[m]}
          </button>
        );
      })}
    </div>
  );
}
