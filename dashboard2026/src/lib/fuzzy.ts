/**
 * J-track tiny subsequence-fuzzy matcher for the command palette.
 *
 * Returns ``null`` if every character of ``query`` cannot be found
 * in order inside ``target``; otherwise returns a score where lower
 * is better. Bonuses are applied for word-boundary hits and a
 * contiguous prefix match. The implementation is intentionally
 * simple — no external dependency, no allocation in the hot loop.
 */
export interface FuzzyResult {
  score: number;
  matched: readonly number[];
}

export function fuzzyScore(query: string, target: string): FuzzyResult | null {
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  if (q === "") return { score: 0, matched: [] };
  if (q.length > t.length) return null;
  const matched: number[] = [];
  let qi = 0;
  let lastIdx = -1;
  let score = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti += 1) {
    if (t[ti] !== q[qi]) continue;
    matched.push(ti);
    if (lastIdx >= 0) {
      score += ti - lastIdx - 1;
    } else {
      score += ti;
    }
    const prevChar = ti === 0 ? " " : t[ti - 1];
    if (prevChar === " " || prevChar === "-" || prevChar === "/") {
      score -= 4;
    }
    lastIdx = ti;
    qi += 1;
  }
  if (qi !== q.length) return null;
  return { score, matched };
}
