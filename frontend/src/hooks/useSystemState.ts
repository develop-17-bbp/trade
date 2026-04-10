import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchDashboard, type DashboardData } from '../api/client'

const POLL_INTERVAL_MS = 3000

/**
 * Main state hook — fetches aggregated dashboard data from backend.
 * Single API call returns everything the frontend needs.
 */
export function useSystemState() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)

  const poll = useCallback(async () => {
    if (!mountedRef.current) return
    try {
      const result = await fetchDashboard()
      if (!mountedRef.current) return
      if (result) {
        setData(result)
        setError(null)
      } else {
        setError('API returned null — is the backend running on port 11000?')
      }
      setLoading(false)
    } catch (err) {
      if (!mountedRef.current) return
      setError(err instanceof Error ? err.message : 'Unknown error')
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    poll()
    const interval = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      mountedRef.current = false
      clearInterval(interval)
    }
  }, [poll])

  return {
    // Structured accessors matching what pages expect
    portfolio: data?.portfolio ?? null,
    positions: data?.positions ?? [],
    trades: data?.trades ?? [],
    tradeStats: data?.trade_stats ?? null,
    agents: data?.agents ?? null,
    risk: data?.risk ?? null,
    models: data?.models ?? {},
    status: data?.status ?? 'UNKNOWN',
    sources: data?.sources ?? {},
    layers: data?.layers ?? {},
    layerLogs: data?.layer_logs ?? {},
    sentiment: data?.sentiment ?? {},
    lastUpdate: data?.last_update ?? '',
    loading,
    error,
    raw: data,
  }
}
