import { apiGet } from "./base";

/**
 * Backend memecoin feed projections. These read SAME endpoints as
 * `/dash2/` — `dash_meme` is a different *view*, not a different
 * data plane.
 *
 * The harness uses different array keys per endpoint
 * (``launches`` for pumpfun, ``snapshots`` for raydium); we normalize
 * to a single ``items`` field at the fetch boundary so every consumer
 * has one shape to reason about (Devin Review BUG_0001 on PR #181).
 */

export type FeedItems = {
  items: ReadonlyArray<Record<string, unknown>>;
  count: number;
  feed: Record<string, unknown>;
};

type PumpFunRecentRaw = {
  launches?: ReadonlyArray<Record<string, unknown>>;
  count?: number;
  feed?: Record<string, unknown>;
};

type RaydiumRecentRaw = {
  snapshots?: ReadonlyArray<Record<string, unknown>>;
  count?: number;
  feed?: Record<string, unknown>;
};

export type MemecoinSummary = {
  memecoin: Record<string, unknown>;
};

export type ModeSnapshot = {
  mode: Record<string, unknown>;
};

export type FeedStatus = {
  status: "running" | "stopped" | "error";
  detail?: string;
  feed: Record<string, unknown>;
};

type FeedStatusRaw = {
  feed?: Record<string, unknown> & { running?: unknown };
  detail?: string;
  error?: string;
};

function asStatus(raw: FeedStatusRaw): FeedStatus {
  const feed = raw.feed ?? {};
  const running = feed.running;
  let status: FeedStatus["status"];
  if (raw.error) {
    status = "error";
  } else if (running === true) {
    status = "running";
  } else {
    status = "stopped";
  }
  return {
    status,
    detail:
      raw.detail ??
      raw.error ??
      (typeof feed.url === "string" ? feed.url : undefined),
    feed,
  };
}

function asItems(
  arr: ReadonlyArray<Record<string, unknown>> | undefined,
  count: number | undefined,
  feed: Record<string, unknown> | undefined,
): FeedItems {
  return {
    items: arr ?? [],
    count: typeof count === "number" ? count : (arr?.length ?? 0),
    feed: feed ?? {},
  };
}

export const fetchPumpFunRecent = async (): Promise<FeedItems> => {
  const raw = await apiGet<PumpFunRecentRaw>("/api/feeds/pumpfun/recent");
  return asItems(raw.launches, raw.count, raw.feed);
};

export const fetchRaydiumRecent = async (): Promise<FeedItems> => {
  const raw = await apiGet<RaydiumRecentRaw>("/api/feeds/raydium/recent");
  return asItems(raw.snapshots, raw.count, raw.feed);
};

export const fetchMemecoinSummary = () =>
  apiGet<MemecoinSummary>("/api/dashboard/memecoin");

export const fetchMode = () => apiGet<ModeSnapshot>("/api/dashboard/mode");

export const fetchPumpFunStatus = async (): Promise<FeedStatus> => {
  const raw = await apiGet<FeedStatusRaw>("/api/feeds/pumpfun/status");
  return asStatus(raw);
};

export const fetchRaydiumStatus = async (): Promise<FeedStatus> => {
  const raw = await apiGet<FeedStatusRaw>("/api/feeds/raydium/status");
  return asStatus(raw);
};
