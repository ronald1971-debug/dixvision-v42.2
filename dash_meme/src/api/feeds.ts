import { apiGet } from "./base";

/**
 * Backend memecoin feed projections. These read SAME endpoints as
 * `/dash2/` — `dash_meme` is a different *view*, not a different
 * data plane.
 */

export type PumpFunRecent = {
  status: "running" | "stopped" | "error";
  count: number;
  recent: ReadonlyArray<Record<string, unknown>>;
};

export type RaydiumRecent = {
  status: "running" | "stopped" | "error";
  count: number;
  recent: ReadonlyArray<Record<string, unknown>>;
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
};

export const fetchPumpFunRecent = () =>
  apiGet<PumpFunRecent>("/api/feeds/pumpfun/recent");

export const fetchRaydiumRecent = () =>
  apiGet<RaydiumRecent>("/api/feeds/raydium/recent");

export const fetchMemecoinSummary = () =>
  apiGet<MemecoinSummary>("/api/dashboard/memecoin");

export const fetchMode = () => apiGet<ModeSnapshot>("/api/dashboard/mode");

export const fetchPumpFunStatus = () =>
  apiGet<FeedStatus>("/api/feeds/pumpfun/status");

export const fetchRaydiumStatus = () =>
  apiGet<FeedStatus>("/api/feeds/raydium/status");
