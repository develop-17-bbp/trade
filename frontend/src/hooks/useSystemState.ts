import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchDashboard, type DashboardData } from '../api/client'

const POLL_INTERVAL_MS = 5000

/**
 * Main state hook — fetches aggregated dashboard data from backend.
 * Single API call returns everything the frontend needs.
 */
export function useSystemState() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Surfaced to the UI for the LIVE badge — operator needs to confirm
  // paper trades are auto-refreshing without manual reload.
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null)
  const mountedRef = useRef(true)

  const poll = useCallback(async () => {
    if (!mountedRef.current) return
    try {
      const result = await fetchDashboard()
      if (!mountedRef.current) return
      if (result) {
        setData(result)
        setError(null)
        setLastFetchedAt(Date.now())
      } else {
        setError('API returned null — is the backend running on port 11007?')
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
    lastFetchedAt,
    pollIntervalMs: POLL_INTERVAL_MS,
    loading,
    error,
    raw: data,
  }
}
