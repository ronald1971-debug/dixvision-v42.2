/**
 * Read-only fetchers for the Tier-1 governance widgets:
 *
 *   /api/governance/promotion_gates
 *   /api/governance/drift
 *   /api/governance/sources
 *   /api/governance/hazards
 *
 * Schemas are derived from the Pydantic response models in
 * `core/contracts/api/governance.py` and rendered into
 * `src/types/generated/api.ts` by `tools/codegen/pydantic_to_ts.py`.
 * The drift guard at `tests/test_codegen_pydantic_to_ts.py` fails CI
 * if the generated file goes out of sync with the Pydantic source —
 * editing either side without regenerating will break the build.
 */
import { apiUrl } from "@/api/base";
import type {
  DriftComponent as GeneratedDriftComponent,
  DriftResponse as GeneratedDriftResponse,
  HazardEventRow,
  HazardTaxonomyRow,
  HazardsResponse,
  PromotionGatesResponse,
  SourceRow as GeneratedSourceRow,
  SourcesResponse,
} from "@/types/generated/api";

// Backwards-compatible aliases. Earlier consumers imported these
// names directly from `@/api/governance`; keep them stable so the
// PR diff stays scoped to the codegen wiring.
export type PromotionGatesPayload = PromotionGatesResponse;
export type DriftComponent = GeneratedDriftComponent;
export type DriftPayload = GeneratedDriftResponse;
export type SourceRow = GeneratedSourceRow;
export type SourcesPayload = SourcesResponse;
export type { HazardEventRow, HazardTaxonomyRow };
export type HazardEvent = HazardEventRow;
export type HazardsPayload = HazardsResponse;

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(apiUrl(path), {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function fetchPromotionGates(
  signal?: AbortSignal,
): Promise<PromotionGatesPayload> {
  return getJSON("/api/governance/promotion_gates", signal);
}

export function fetchDrift(signal?: AbortSignal): Promise<DriftPayload> {
  return getJSON("/api/governance/drift", signal);
}

export function fetchSources(signal?: AbortSignal): Promise<SourcesPayload> {
  return getJSON("/api/governance/sources", signal);
}

export function fetchHazards(signal?: AbortSignal): Promise<HazardsPayload> {
  return getJSON("/api/governance/hazards", signal);
}

// ---------------------------------------------------------------------------
// /api/dashboard/{strategies,decisions} — Tier-1 reuses these
// ---------------------------------------------------------------------------

export interface StrategyRow {
  strategy_id: string;
  state: string;
  // Other fields are pass-through; surface them as raw record so the
  // panel can show diagnostic fields without churning the type.
  [key: string]: unknown;
}

export interface StrategiesByState {
  [state: string]: StrategyRow[];
}

export async function fetchStrategies(
  signal?: AbortSignal,
): Promise<StrategiesByState> {
  const body = await getJSON<{ strategies: StrategiesByState }>(
    "/api/dashboard/strategies",
    signal,
  );
  return body.strategies;
}

export interface DecisionChainStep {
  ts_ns: number;
  kind: string;
  payload: Record<string, unknown>;
}

export interface DecisionChain {
  trace_id: string;
  steps: DecisionChainStep[];
  // Keep the rest as record so panel can read pass-through fields.
  [key: string]: unknown;
}

export async function fetchDecisionChains(
  limit = 50,
  signal?: AbortSignal,
): Promise<DecisionChain[]> {
  const body = await getJSON<{ chains: DecisionChain[] }>(
    `/api/dashboard/decisions?limit=${limit}`,
    signal,
  );
  return body.chains;
}
