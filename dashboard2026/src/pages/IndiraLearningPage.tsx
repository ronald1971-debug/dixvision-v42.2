import { IndiraLearningMode } from "@/widgets/IndiraLearningMode";

/**
 * Indira Learning Mode page (`#/indira`).
 *
 * Full-surface route that exposes Indira's learning loop: the
 * philosophy library, trader-feed inbox, strategy-proposal queue,
 * shadow-evaluation results and the active fine-tune corpus.
 */
export function IndiraLearningPage() {
  return (
    <div className="flex h-full flex-col gap-3">
      <header className="flex items-baseline gap-3">
        <h1 className="text-base font-semibold tracking-tight">
          Indira · Learning Mode
        </h1>
        <p className="text-[12px] text-slate-500">
          philosophy library / trader feed / proposals / shadow eval / corpus —
          governance-gated
        </p>
      </header>
      <section className="flex-1">
        <IndiraLearningMode />
      </section>
    </div>
  );
}
