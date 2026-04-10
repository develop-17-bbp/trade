import { useState, useEffect, useRef, useCallback } from 'react'

interface WebSocketState<T = unknown> {
  data: T | null
  connected: boolean
  error: string | null
}

const DEFAULT_URL = 'ws://localhost:8000/ws'
const RECONNECT_BASE_DELAY = 1000
const RECONNECT_MAX_DELAY = 30000
const HEARTBEAT_INTERVAL = 30000

export function useWebSocket<T = unknown>(url: string = DEFAULT_URL): WebSocketState<T> {
  const [data, setData] = useState<T | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const unmountedRef = useRef(false)

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (heartbeatTimerRef.current !== null) {
      clearInterval(heartbeatTimerRef.current)
      heartbeatTimerRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (unmountedRef.current) return
        setConnected(true)
        setError(null)
        reconnectAttemptRef.current = 0

        // Start heartbeat pings
        heartbeatTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, HEARTBEAT_INTERVAL)
      }

      ws.onmessage = (event: MessageEvent) => {
        if (unmountedRef.current) return
        try {
          const parsed = JSON.parse(event.data as string) as T
          setData(parsed)
        } catch {
          // If not valid JSON, store raw string wrapped as unknown
          setData(event.data as T)
        }
      }

      ws.onerror = () => {
        if (unmountedRef.current) return
        setError('WebSocket connection error')
      }

      ws.onclose = (event: CloseEvent) => {
        if (unmountedRef.current) return
        setConnected(false)

        if (heartbeatTimerRef.current !== null) {
          clearInterval(heartbeatTimerRef.current)
          heartbeatTimerRef.current = null
        }

        // Don't reconnect on clean close (code 1000) or if unmounted
        if (event.code === 1000) return

        // Exponential backoff reconnect
        const attempt = reconnectAttemptRef.current
        const delay = Math.min(
          RECONNECT_BASE_DELAY * Math.pow(2, attempt),
          RECONNECT_MAX_DELAY,
        )
        reconnectAttemptRef.current = attempt + 1

        setError(`Disconnected. Reconnecting in ${Math.round(delay / 1000)}s...`)
        reconnectTimerRef.current = setTimeout(connect, delay)
      }
    } catch (err) {
      if (unmountedRef.current) return
      setError(err instanceof Error ? err.message : 'Failed to connect')
      setConnected(false)

      // Schedule reconnect on connection failure
      const attempt = reconnectAttemptRef.current
      const delay = Math.min(
        RECONNECT_BASE_DELAY * Math.pow(2, attempt),
        RECONNECT_MAX_DELAY,
      )
      reconnectAttemptRef.current = attempt + 1
      reconnectTimerRef.current = setTimeout(connect, delay)
    }
  }, [url, clearTimers])

  useEffect(() => {
    unmountedRef.current = false
    connect()

    return () => {
      unmountedRef.current = true
      clearTimers()
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted')
        wsRef.current = null
      }
    }
  }, [connect, clearTimers])

  return { data, connected, error }
}
