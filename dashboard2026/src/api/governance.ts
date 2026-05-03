/**
 * Read-only fetchers for the Tier-1 governance widgets:
 *
 *   /api/governance/promotion_gates
 *   /api/governance/drift
 *   /api/governance/sources
 *   /api/governance/hazards
 *
 * Schemas mirror `ui/governance_routes.py`. They are typed locally
 * here rather than going through the Pydantic→TS codegen because the
 * governance router returns plain JSON dictionaries (no Pydantic
 * response models), and the shapes are stable across the four
 * endpoints.
 */
import { apiUrl } from "@/api/base";

export interface PromotionGatesPayload {
  path: string;
  file_present: boolean;
  file_hash: string | null;
  bound_hash: string | null;
  matches: boolean | null;
  backend_wired: boolean;
  gated_targets: string[];
  doc_url: string;
}

export interface DriftComponent {
  id: string;
  label: string;
  threshold: number;
  description: string;
  value?: number;
}

export interface DriftPayload {
  backend_wired: boolean;
  composite: number | null;
  expected_components: DriftComponent[];
  components: DriftComponent[];
  downgrade_threshold: number;
}

export interface SourceRow {
  source_id: string;
  name: string;
  category: string;
  provider: string;
  auth: string;
  enabled: boolean;
  critical: boolean;
  liveness_threshold_ms: number;
  status: string;
  last_heartbeat_ns: number;
  last_data_ns: number;
  gap_ns: number;
}

export interface SourcesPayload {
  backend_wired: boolean;
  registry_loaded: boolean;
  rows: SourceRow[];
}

export interface HazardTaxonomyRow {
  code: string;
  label: string;
  description: string;
}

export interface HazardEvent {
  code: string;
  severity: string;
  ts_ns: number;
  source: string;
  summary: string;
}

export interface HazardsPayload {
  backend_wired: boolean;
  taxonomy: HazardTaxonomyRow[];
  recent: HazardEvent[];
}

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
