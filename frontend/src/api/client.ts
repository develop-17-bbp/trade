// ── Types ────────────────────────────────────────────────────────────────────

export interface PortfolioData {
  total_value: number
  pnl_today: number
  pnl_today_pct: number
  total_pnl: number
  total_pnl_pct: number
  available_balance: number
  margin_used: number
  buying_power: number
}

export interface Position {
  symbol: string
  side: 'long' | 'short'
  size: number
  entry_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  leverage: number
  liquidation_price: number | null
  opened_at: string
}

export interface Trade {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  price: number
  size: number
  pnl: number | null
  fee: number
  timestamp: string
  strategy: string
}

export interface AgentOverlay {
  id: string
  name: string
  status: 'active' | 'idle' | 'error'
  strategy: string
  current_signal: string | null
  confidence: number
  last_action: string
  last_action_time: string
  pnl_contribution: number
}

export interface RiskMetrics {
  portfolio_var: number
  portfolio_cvar: number
  max_drawdown: number
  current_drawdown: number
  sharpe_ratio: number
  sortino_ratio: number
  win_rate: number
  profit_factor: number
  correlation_btc: number
  risk_score: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
}

export interface SystemStatus {
  status: 'online' | 'offline' | 'degraded'
  exchange: string
  exchange_connected: boolean
  websocket_connected: boolean
  agents_running: number
  agents_total: number
  uptime_seconds: number
  last_heartbeat: string
  cpu_usage: number
  memory_usage: number
}

export interface Signal {
  id: string
  symbol: string
  direction: 'long' | 'short' | 'neutral'
  strength: number
  source: string
  timestamp: string
  metadata: Record<string, unknown>
}

// ── Cache ────────────────────────────────────────────────────────────────────

interface CacheEntry<T> {
  data: T
  timestamp: number
}

const CACHE_TTL_MS = 5000
const cache = new Map<string, CacheEntry<unknown>>()

function getCached<T>(key: string): T | null {
  const entry = cache.get(key) as CacheEntry<T> | undefined
  if (!entry) return null
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
    cache.delete(key)
    return null
  }
  return entry.data
}

function setCache<T>(key: string, data: T): void {
  cache.set(key, { data, timestamp: Date.now() })
}

// ── Fetch helper ─────────────────────────────────────────────────────────────

const BASE_URL = '/api/v1'

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    message?: string,
  ) {
    super(message ?? `API error ${status}: ${statusText}`)
    this.name = 'ApiError'
  }
}

async function fetchJson<T>(endpoint: string): Promise<T> {
  const cacheKey = endpoint
  const cached = getCached<T>(cacheKey)
  if (cached !== null) return cached

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    headers: { 'Accept': 'application/json' },
  })

  if (!response.ok) {
    throw new ApiError(response.status, response.statusText)
  }

  const data: T = await response.json()
  setCache(cacheKey, data)
  return data
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function fetchPortfolio(): Promise<PortfolioData> {
  return fetchJson<PortfolioData>('/portfolio')
}

export async function fetchPositions(): Promise<Position[]> {
  return fetchJson<Position[]>('/positions')
}

export async function fetchTrades(): Promise<Trade[]> {
  return fetchJson<Trade[]>('/trades')
}

export async function fetchAgentOverlay(): Promise<AgentOverlay[]> {
  return fetchJson<AgentOverlay[]>('/agents')
}

export async function fetchRiskMetrics(): Promise<RiskMetrics> {
  return fetchJson<RiskMetrics>('/risk')
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  return fetchJson<SystemStatus>('/status')
}

export async function fetchSignals(): Promise<Signal[]> {
  return fetchJson<Signal[]>('/signals')
}

export { ApiError }
