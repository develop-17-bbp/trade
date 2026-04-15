import { useEffect, useRef, useState, useMemo } from 'react'
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
} from 'lightweight-charts'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface OHLCVBar {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

/** ACT trade event (ENTRY or EXIT) from /api/v1/dashboard trades[] */
export interface ChartTrade {
  asset: string
  direction: string          // "LONG" | "SHORT" | "long" | "short"
  status: string             // "OPEN" | "CLOSED"
  event?: string             // "ENTRY" | "EXIT"
  entry_price: number
  exit_price: number
  pnl: number
  pnl_pct: number
  timestamp: string          // ISO datetime
  reason: string
}

/** Open position */
export interface ChartPosition {
  asset: string
  direction: string
  entry_price: number
  current_price: number
  quantity: number
  unrealized_pnl: number
  stop_loss: number
  entry_time?: number | string
}

interface Props {
  asset?: string
  height?: number
  timeframe?: string
  trades?: ChartTrade[]
  positions?: ChartPosition[]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compute EMA for given period */
function computeEMA(bars: OHLCVBar[], period: number): { time: number; value: number }[] {
  if (bars.length === 0) return []
  let emaVal = bars[0].close
  const mult = 2 / (period + 1)
  return bars.map((b) => {
    emaVal = b.close * mult + emaVal * (1 - mult)
    return { time: b.time, value: emaVal }
  })
}

/** Parse ISO timestamp to unix seconds */
function isoToUnix(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000)
}

/** Snap a timestamp to the nearest bar time in the visible range */
function snapToBar(ts: number, bars: OHLCVBar[]): number {
  if (bars.length === 0) return ts
  // Find the closest bar (earlier-or-equal preferred)
  let best = bars[0].time
  for (const b of bars) {
    if (b.time <= ts) best = b.time
    else break
  }
  return best
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CandlestickChart({
  asset = 'BTC',
  height = 360,
  timeframe = '1h',
  trades = [],
  positions = [],
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const ema8Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const ema21Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const priceLinesRef = useRef<any[]>([])

  const [bars, setBars] = useState<OHLCVBar[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // ── Fetch OHLCV data ──
  useEffect(() => {
    let cancelled = false
    async function fetchData() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(
          `/api/v1/ohlcv/${asset}?timeframe=${timeframe}&limit=300`
        )
        if (!res.ok) throw new Error(`OHLCV API error ${res.status}`)
        const data = await res.json()
        if (cancelled) return
        if (data?.bars?.length > 0) {
          setBars(data.bars)
          setError(null)
        } else {
          setError('No OHLCV data available')
          setBars([])
        }
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : 'Fetch failed')
        setBars([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchData()
    const interval = setInterval(fetchData, 60_000) // refresh every minute
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [asset, timeframe])

  // ── Computed indicators ──
  const emaFast = useMemo(() => computeEMA(bars, 8), [bars])
  const emaSlow = useMemo(() => computeEMA(bars, 21), [bars])

  // ── Filter trades for this asset and convert to markers ──
  const assetTrades = useMemo(
    () => trades.filter((t) => t.asset === asset),
    [trades, asset]
  )

  const assetPositions = useMemo(
    () => positions.filter((p) => p.asset === asset),
    [positions, asset]
  )

  // ── Create chart ──
  useEffect(() => {
    if (!containerRef.current) return
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
      candleRef.current = null
      ema8Ref.current = null
      ema21Ref.current = null
      volumeRef.current = null
      markersRef.current = null
      priceLinesRef.current = []
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#a0a0a0',
        fontSize: 11,
        fontFamily: "'Inter', system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        vertLine: { color: 'rgba(34,197,94,0.35)', width: 1, style: 2 },
        horzLine: { color: 'rgba(34,197,94,0.35)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        scaleMargins: { top: 0.05, bottom: 0.22 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Candlestick
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e88',
      wickDownColor: '#ef444488',
    })

    // EMA 8 (fast)
    const ema8 = chart.addSeries(LineSeries, {
      color: '#00ccff',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // EMA 21 (slow)
    const ema21 = chart.addSeries(LineSeries, {
      color: '#ff00aa',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      lineStyle: 2,
    })

    // Volume histogram
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })

    chartRef.current = chart
    candleRef.current = candle
    ema8Ref.current = ema8
    ema21Ref.current = ema21
    volumeRef.current = volume

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      candleRef.current = null
      ema8Ref.current = null
      ema21Ref.current = null
      volumeRef.current = null
      markersRef.current = null
      priceLinesRef.current = []
    }
  }, [height])

  // ── Update candle/EMA/volume data ──
  useEffect(() => {
    const candle = candleRef.current
    const ema8 = ema8Ref.current
    const ema21 = ema21Ref.current
    const volume = volumeRef.current
    if (!candle || bars.length === 0) return

    candle.setData(
      bars.map((b) => ({
        time: b.time as Time,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }))
    )

    if (ema8 && emaFast.length > 0) {
      ema8.setData(emaFast.map((e) => ({ time: e.time as Time, value: e.value })))
    }
    if (ema21 && emaSlow.length > 0) {
      ema21.setData(emaSlow.map((e) => ({ time: e.time as Time, value: e.value })))
    }

    if (volume) {
      volume.setData(
        bars.map((b) => ({
          time: b.time as Time,
          value: b.volume ?? 0,
          color: b.close >= b.open ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)',
        }))
      )
    }

    chartRef.current?.timeScale().fitContent()
  }, [bars, emaFast, emaSlow])

  // ── Apply trade markers from REAL ACT trades ──
  useEffect(() => {
    const candle = candleRef.current
    if (!candle || bars.length === 0) return

    // Build markers from assetTrades
    const markers: SeriesMarker<Time>[] = []
    const earliestBarTime = bars[0]?.time ?? 0
    const latestBarTime = bars[bars.length - 1]?.time ?? 0

    for (const t of assetTrades) {
      const ts = isoToUnix(t.timestamp)
      // Skip trades outside the visible chart range
      if (ts < earliestBarTime - 3600 || ts > latestBarTime + 3600) continue

      const snapped = snapToBar(ts, bars)
      const event = (t.event || t.status || '').toUpperCase()
      const dir = (t.direction || '').toUpperCase()
      const isExit = event === 'EXIT' || event === 'CLOSED'

      if (isExit) {
        const profitable = t.pnl >= 0
        markers.push({
          time: snapped as Time,
          position: 'aboveBar',
          color: profitable ? '#22c55e' : '#ef4444',
          shape: 'arrowDown',
          text: `${profitable ? 'WIN' : 'LOSS'} ${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)} (${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%)`,
          size: 2,
        })
      } else {
        // ENTRY
        const isLong = dir === 'LONG'
        markers.push({
          time: snapped as Time,
          position: 'belowBar',
          color: isLong ? '#22c55e' : '#ef4444',
          shape: isLong ? 'arrowUp' : 'arrowDown',
          text: `${isLong ? 'LONG' : 'SHORT'} $${t.entry_price.toFixed(2)}`,
          size: 2,
        })
      }
    }

    // Sort markers by time (required by lightweight-charts)
    markers.sort((a, b) => (a.time as number) - (b.time as number))

    // Create or update the markers plugin
    if (!markersRef.current) {
      markersRef.current = createSeriesMarkers(candle, markers)
    } else {
      markersRef.current.setMarkers(markers)
    }
  }, [assetTrades, bars])

  // ── Draw price lines for open positions (entry + SL) ──
  useEffect(() => {
    const candle = candleRef.current
    if (!candle) return

    // Clean up existing price lines
    for (const line of priceLinesRef.current) {
      try {
        candle.removePriceLine(line)
      } catch {
        // ignore
      }
    }
    priceLinesRef.current = []

    for (const p of assetPositions) {
      const isLong = (p.direction || '').toLowerCase() === 'long'
      const pnlColor = p.unrealized_pnl >= 0 ? '#22c55e' : '#ef4444'

      // Entry line
      const entryLine = candle.createPriceLine({
        price: p.entry_price,
        color: isLong ? '#22c55e' : '#ef4444',
        lineWidth: 2,
        lineStyle: 0, // solid
        axisLabelVisible: true,
        title: `ENTRY ${isLong ? 'LONG' : 'SHORT'}`,
      })
      priceLinesRef.current.push(entryLine)

      // Current/PnL line
      if (p.current_price > 0) {
        const currentLine = candle.createPriceLine({
          price: p.current_price,
          color: pnlColor,
          lineWidth: 1,
          lineStyle: 2, // dashed
          axisLabelVisible: true,
          title: `PnL ${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)}`,
        })
        priceLinesRef.current.push(currentLine)
      }

      // Stop loss line
      if (p.stop_loss > 0) {
        const slLine = candle.createPriceLine({
          price: p.stop_loss,
          color: '#ef4444',
          lineWidth: 1,
          lineStyle: 3, // dotted
          axisLabelVisible: true,
          title: 'STOP LOSS',
        })
        priceLinesRef.current.push(slLine)
      }
    }
  }, [assetPositions])

  // ── Count stats for legend ──
  const stats = useMemo(() => {
    let wins = 0
    let losses = 0
    let totalPnl = 0
    for (const t of assetTrades) {
      const event = (t.event || t.status || '').toUpperCase()
      if (event === 'EXIT' || event === 'CLOSED') {
        if (t.pnl > 0) wins++
        else if (t.pnl < 0) losses++
        totalPnl += t.pnl
      }
    }
    return { wins, losses, totalPnl, total: assetTrades.length }
  }, [assetTrades])

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#000]/60 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-2 border-[#22c55e] border-t-transparent rounded-full animate-spin" />
            <span className="text-[10px] text-[#a0a0a0] font-mono uppercase tracking-wider">
              Loading {asset} {timeframe}
            </span>
          </div>
        </div>
      )}
      {error && !loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#000]/70">
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="text-[11px] text-[#ef4444] font-mono">{error}</span>
            <span className="text-[9px] text-[#666] font-mono">
              Ensure /api/v1/ohlcv/{asset} is reachable
            </span>
          </div>
        </div>
      )}

      {/* Top-left legend */}
      <div className="absolute top-2 left-3 z-10 flex items-center gap-3 text-[9px] font-mono text-[#a0a0a0]">
        <span className="flex items-center gap-1">
          <span className="w-3 h-px bg-[#00ccff] inline-block" /> EMA(8)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-px bg-[#ff00aa] inline-block" /> EMA(21)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-0 h-0 border-l-[4px] border-r-[4px] border-b-[6px] border-l-transparent border-r-transparent border-b-[#22c55e]" />
          ENTRY
        </span>
        <span className="flex items-center gap-1">
          <span className="w-0 h-0 border-l-[4px] border-r-[4px] border-t-[6px] border-l-transparent border-r-transparent border-t-[#22c55e]" />
          EXIT
        </span>
      </div>

      {/* Top-right stats badge */}
      {stats.total > 0 && (
        <div className="absolute top-2 right-3 z-10 flex items-center gap-2 text-[9px] font-mono">
          <span className="text-[#a0a0a0]">ACT TRADES:</span>
          <span className="text-[#22c55e]">{stats.wins}W</span>
          <span className="text-[#666]">/</span>
          <span className="text-[#ef4444]">{stats.losses}L</span>
          <span
            className={
              stats.totalPnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'
            }
          >
            {stats.totalPnl >= 0 ? '+' : ''}${stats.totalPnl.toFixed(2)}
          </span>
        </div>
      )}

      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  )
}
