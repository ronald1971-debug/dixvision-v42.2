/**
 * Testing-harness API client (AUDIT-P2.1).
 *
 * Server-side deterministic backtest endpoint. The widget previously
 * ran the same algorithm entirely in-browser; pulling it server-side
 * makes the seed canonical and gives the audit trail one source of
 * truth (see ``ui/server.py:/api/testing/backtest`` and
 * ``system_engine/backtest_ingest/internal/deterministic.py``).
 */
import { apiUrl } from "./base";

export type BacktestStrategy =
  | "ema_cross_20_50"
  | "rsi_2_meanrev"
  | "vwap_reversion"
  | "breakout_channel"
  | "microstructure_v1"
  | "news_event_drift"
  | "memecoin_copy"
  | "memecoin_sniper";

export type BacktestFillModel =
  | "next_tick"
  | "vwap_5min"
  | "mid_price"
  | "tob_aggress";

export interface BacktestRunRequest {
  strategy: BacktestStrategy;
  symbol: string;
  start_iso: string;
  end_iso: string;
  fill_model: BacktestFillModel;
  slippage_bps: number;
}

export interface BacktestTradeRow {
  ts_iso: string;
  side: "BUY" | "SELL";
  pnl_pct: number;
  bars_held: number;
}

export interface BacktestMetricsBlock {
  final_equity_pct: number;
  cagr: number;
  sharpe: number;
  sortino: number;
  max_dd_pct: number;
  win_rate: number;
  /** ``null`` when ``gross_loss == 0`` (no losses; ratio undefined). */
  profit_factor: number | null;
  avg_trade_pct: number;
  longest_loss_streak: number;
  n_trades: number;
}

export interface BacktestRunResponse {
  seed: string;
  request: BacktestRunRequest;
  equity: number[];
  drawdown: number[];
  trades: BacktestTradeRow[];
  metrics: BacktestMetricsBlock;
  notes: string[];
}

export class BacktestEndpointUnavailableError extends Error {
  constructor(cause: unknown) {
    super(
      `Backtest endpoint /api/testing/backtest unreachable: ${String(cause)}`,
    );
    this.name = "BacktestEndpointUnavailableError";
  }
}

export async function runBacktest(
  req: BacktestRunRequest,
): Promise<BacktestRunResponse> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/testing/backtest"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch (err) {
    throw new BacktestEndpointUnavailableError(err);
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(
      `POST /api/testing/backtest -> ${res.status}: ${detail || res.statusText}`,
    );
  }
  return (await res.json()) as BacktestRunResponse;
}
