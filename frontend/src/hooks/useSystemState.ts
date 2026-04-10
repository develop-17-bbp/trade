import { useState, useEffect, useCallback, useRef } from 'react'
import {
  fetchPortfolio,
  fetchPositions,
  fetchTrades,
  fetchAgentOverlay,
  fetchRiskMetrics,
  fetchSystemStatus,
  type PortfolioData,
  type Position,
  type Trade,
  type AgentOverlay,
  type RiskMetrics,
  type SystemStatus,
} from '../api/client'
import { useWebSocket } from '../api/websocket'

// ── Types ────────────────────────────────────────────────────────────────────

export interface SystemState {
  portfolio: PortfolioData | null
  positions: Position[]
  trades: Trade[]
  agents: AgentOverlay[]
  risk: RiskMetrics | null
  status: SystemStatus | null
  loading: boolean
  error: string | null
  wsConnected: boolean
}

interface WebSocketMessage {
  type: string
  data: unknown
}

const POLL_INTERVAL_MS = 3000

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useSystemState(): SystemState {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [trades, setTrades] = useState<Trade[]>([])
  const [agents, setAgents] = useState<AgentOverlay[]>([])
  const [risk, setRisk] = useState<RiskMetrics | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const mountedRef = useRef(true)

  // WebSocket for real-time updates
  const { data: wsMessage, connected: wsConnected } = useWebSocket<WebSocketMessage>()

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!wsMessage || !wsMessage.type) return

    switch (wsMessage.type) {
      case 'portfolio':
        setPortfolio(wsMessage.data as PortfolioData)
        break
      case 'positions':
        setPositions(wsMessage.data as Position[])
        break
      case 'trade':
        setTrades(prev => [wsMessage.data as Trade, ...prev].slice(0, 100))
        break
      case 'agents':
        setAgents(wsMessage.data as AgentOverlay[])
        break
      case 'risk':
        setRisk(wsMessage.data as RiskMetrics)
        break
      case 'status':
        setStatus(wsMessage.data as SystemStatus)
        break
    }
  }, [wsMessage])

  // REST polling for baseline state
  const pollData = useCallback(async () => {
    if (!mountedRef.current) return

    try {
      const [
        portfolioRes,
        positionsRes,
        tradesRes,
        agentsRes,
        riskRes,
        statusRes,
      ] = await Promise.allSettled([
        fetchPortfolio(),
        fetchPositions(),
        fetchTrades(),
        fetchAgentOverlay(),
        fetchRiskMetrics(),
        fetchSystemStatus(),
      ])

      if (!mountedRef.current) return

      if (portfolioRes.status === 'fulfilled') setPortfolio(portfolioRes.value)
      if (positionsRes.status === 'fulfilled') setPositions(positionsRes.value)
      if (tradesRes.status === 'fulfilled') setTrades(tradesRes.value)
      if (agentsRes.status === 'fulfilled') setAgents(agentsRes.value)
      if (riskRes.status === 'fulfilled') setRisk(riskRes.value)
      if (statusRes.status === 'fulfilled') setStatus(statusRes.value)

      // Count how many succeeded
      const results = [portfolioRes, positionsRes, tradesRes, agentsRes, riskRes, statusRes]
      const failedCount = results.filter(r => r.status === 'rejected').length
      if (failedCount === results.length) {
        setError('All API requests failed. Is the backend running?')
      } else if (failedCount > 0) {
        setError(`${failedCount} API request(s) failed`)
      } else {
        setError(null)
      }

      setLoading(false)
    } catch (err) {
      if (!mountedRef.current) return
      setError(err instanceof Error ? err.message : 'Unknown error')
      setLoading(false)
    }
  }, [])

  // Initial fetch + polling interval
  useEffect(() => {
    mountedRef.current = true
    pollData()

    const interval = setInterval(pollData, POLL_INTERVAL_MS)

    return () => {
      mountedRef.current = false
      clearInterval(interval)
    }
  }, [pollData])

  return {
    portfolio,
    positions,
    trades,
    agents,
    risk,
    status,
    loading,
    error,
    wsConnected,
  }
}
