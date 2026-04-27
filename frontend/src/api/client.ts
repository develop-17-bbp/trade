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

export interface BrainAssetState {
  tick_age_s: number | null
  evidence_block: string
  evidence_lines: string[]
  open_positions_same_asset: number
  exposure_pct: number
  today_pct_total: number
  gap_to_1pct: number
  ratchet_label: string
  ratchet_sl_price: number
  conviction_tier: string
  sniper_status: string
  pattern_label: string
  pattern_score: number
  latest_decision: null | {
    ts: number
    direction: string
    tier: string
    size_pct: number
    thesis: string
    verdict: string
    plan_id: string
  }
  trace_history: Array<{
    ts: number
    direction: string
    tier: string
    verdict: string
    thesis: string
  }>
}

export interface BrainState {
  assets: Record<string, BrainAssetState>
  snapshot_ts: string
  error?: string
}

/** LLM brain state (qwen analyst per-tick decisions + tick_state evidence) */
export async function fetchBrain(): Promise<BrainState | null> {
  return fetchJSON<BrainState>('/brain')
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

// ── Live intelligence (news sentiment + Fear&Greed + derivatives) ──

export interface LiveIntelligenceEntry {
  sentiment: {
    score: number
    label: string
    confidence: number
    headline_count: number
    sources: string[]
    recent_headlines: string[]
  }
  fear_greed: { value: number | null; signal: string | null }
  funding_rate: number | null
  open_interest_usd: number | null
  put_call_ratio: number | null
  macro_composite: string | null
  macro_risk: number | null
  timestamp: string
}

export async function fetchLiveIntelligence(): Promise<Record<string, LiveIntelligenceEntry> | null> {
  return fetchJSON<Record<string, LiveIntelligenceEntry>>('/signals/live_intelligence')
}

// ── Decision audit log ──

export interface DecisionRecord {
  ts: string
  asset: string
  raw_signal: number
  decision: {
    direction: number
    confidence: number
    position_scale: number
    consensus: string
    veto: boolean
    violations: string[]
  }
  sentiment: {
    score: number
    label: string | null
    headline_count: number
    recent_headlines: string[]
    sources: string[]
  }
  macro: {
    fear_greed: number | null
    funding_rate: number | null
    open_interest_usd: number | null
    put_call_ratio: number | null
    composite: string | null
    macro_risk: number | null
  }
  agents: Record<string, { direction: number; confidence: number; veto: boolean; reasoning: string }>
}

export async function fetchRecentDecisions(limit = 50): Promise<{ decisions: DecisionRecord[]; total: number } | null> {
  return fetchJSON<{ decisions: DecisionRecord[]; total: number }>(`/decisions/recent?limit=${limit}`)
}
