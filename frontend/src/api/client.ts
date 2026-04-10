/**
 * REST API Client — Connects to backend production server
 * Primary: /api/v1/dashboard (single aggregated call)
 * Secondary: /api/v1/prices (unauthenticated ticker)
 */

const BASE = '/api/v1'

// ── Types matching ACTUAL backend response shapes ──

export interface DashboardData {
  portfolio: {
    equity: number
    today_pnl: number
    total_pnl: number
    total_return_pct: number
    equity_curve: Array<{ t: string; v: number }>
    sod_balance: number
  }
  positions: Array<{
    asset: string
    direction: string
    entry_price: number
    current_price: number
    quantity: number
    unrealized_pnl: number
    stop_loss: number
    entry_time: number
    confidence: number
    trade_timeframe: string
  }>
  trades: Array<{
    asset: string
    direction: string
    status: string
    entry_price: number
    exit_price: number
    pnl: number
    pnl_pct: number
    timestamp: string
    reason: string
    trade_timeframe: string
  }>
  trade_stats: {
    total: number
    wins: number
    losses: number
    win_rate: number
    profit_factor: number
    avg_win: number
    avg_loss: number
  }
  agents: {
    list: Array<{
      id: string
      name: string
      direction: number
      confidence: number
      reasoning: string
      weight: number
    }>
    consensus: string
    data_quality: number
    daily_pnl_mode: string
    enabled: boolean
    last_decision: Record<string, unknown>
    cycle_count: number
  }
  risk: {
    current_drawdown: number
    max_drawdown: number
    risk_score: number
    vpin: number
  }
  models: Record<string, {
    predictions: number[]
    actuals: number[]
    correct: number
    total: number
  }>
  status: string
  sources: Record<string, string>
  layers: Record<string, unknown>
  layer_logs: Record<string, Array<{ timestamp: string; message: string; level: string }>>
  sentiment: Record<string, unknown>
  last_update: string
}

export interface PriceData {
  [asset: string]: {
    price: number
    change_pct: number
    bid: number
    ask: number
    spread_pct: number
  }
}

// ── Fetcher ──

async function fetchJSON<T>(path: string, auth = true): Promise<T | null> {
  try {
    const headers: Record<string, string> = { Accept: 'application/json' }
    if (auth) {
      const key = import.meta.env.VITE_API_KEY || ''
      if (key) headers['X-API-Key'] = key
    }
    const res = await fetch(`${BASE}${path}`, { headers })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

// ── Public API ──

/** Single aggregated call for the entire dashboard */
export async function fetchDashboard(): Promise<DashboardData | null> {
  return fetchJSON<DashboardData>('/dashboard')
}

/** Live prices (no auth, fast ticker) */
export async function fetchPrices(): Promise<PriceData | null> {
  return fetchJSON<PriceData>('/prices', false)
}

/** System status */
export async function fetchSystemStatus() {
  return fetchJSON<{
    trading_status: string
    last_update: string
    model_version: string
    sources: Record<string, string>
  }>('/system/status')
}

/** Risk layer pipeline */
export async function fetchRiskLayers() {
  return fetchJSON<{
    layers: Record<string, unknown>
    layer_logs: Array<{ timestamp: string; message: string; level: string }>
  }>('/risk/layers')
}
