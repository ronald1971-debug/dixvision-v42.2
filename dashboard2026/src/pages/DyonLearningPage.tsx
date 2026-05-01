import { DyonLearningMode } from "@/widgets/DyonLearningMode";

/**
 * Dyon Learning Mode page (`#/dyon`).
 *
 * Full-surface route that exposes Dyon's learning loop: the hazard
 * journal, patch-proposal queue, sandbox runs and promotion ledger.
 */
export function DyonLearningPage() {
  return (
    <div className="flex h-full flex-col gap-3">
      <header className="flex items-baseline gap-3">
        <h1 className="text-base font-semibold tracking-tight">
          Dyon · Learning Mode
        </h1>
        <p className="text-[12px] text-slate-500">
          hazard journal / patch proposals / sandbox runs / promotion ledger —
          every merge gated
        </p>
      </header>
      <section className="flex-1">
        <DyonLearningMode />
      </section>
    </div>
  );
}
