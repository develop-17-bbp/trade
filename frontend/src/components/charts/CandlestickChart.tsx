import { useEffect, useRef, useState, useMemo } from 'react'
import {
  createChart,
  type IChartApi,
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

interface Props {
  asset?: string
  height?: number
  timeframe?: string
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

/** Find simple support and resistance from recent swing highs/lows */
function computeSupportResistance(bars: OHLCVBar[], lookback = 20): { support: number; resistance: number } {
  if (bars.length < lookback) {
    const allHighs = bars.map((b) => b.high)
    const allLows = bars.map((b) => b.low)
    return {
      support: Math.min(...allLows),
      resistance: Math.max(...allHighs),
    }
  }
  const recent = bars.slice(-lookback)
  const highs = recent.map((b) => b.high)
  const lows = recent.map((b) => b.low)

  // Use second-highest and second-lowest for more stable levels
  highs.sort((a, b) => b - a)
  lows.sort((a, b) => a - b)

  return {
    support: lows[Math.min(2, lows.length - 1)],
    resistance: highs[Math.min(2, highs.length - 1)],
  }
}

/** Generate synthetic buy/sell signals from EMA crossover */
function computeSignals(
  bars: OHLCVBar[],
  emaFast: { time: number; value: number }[],
  emaSlow: { time: number; value: number }[]
): { buys: { time: number; value: number }[]; sells: { time: number; value: number }[] } {
  const buys: { time: number; value: number }[] = []
  const sells: { time: number; value: number }[] = []

  if (emaFast.length < 2 || emaSlow.length < 2) return { buys, sells }

  for (let i = 1; i < Math.min(emaFast.length, emaSlow.length); i++) {
    const prevFast = emaFast[i - 1].value
    const prevSlow = emaSlow[i - 1].value
    const currFast = emaFast[i].value
    const currSlow = emaSlow[i].value

    // Golden cross — buy
    if (prevFast <= prevSlow && currFast > currSlow) {
      buys.push({ time: bars[i].time, value: bars[i].low * 0.998 })
    }
    // Death cross — sell
    if (prevFast >= prevSlow && currFast < currSlow) {
      sells.push({ time: bars[i].time, value: bars[i].high * 1.002 })
    }
  }

  return { buys, sells }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CandlestickChart({ asset = 'BTC', height = 360, timeframe = '1h' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const seriesRefs = useRef<Record<string, any>>({})
  const [bars, setBars] = useState<OHLCVBar[]>([])
  const [loading, setLoading] = useState(true)

  // ── Fetch OHLCV data ──
  useEffect(() => {
    let cancelled = false
    async function fetchData() {
      setLoading(true)
      try {
        const res = await fetch(`/api/v1/ohlcv/${asset}?timeframe=${timeframe}&limit=200`)
        if (!res.ok) throw new Error('API error')
        const data = await res.json()
        if (!cancelled && data?.bars?.length > 0) {
          setBars(data.bars)
        } else {
          throw new Error('No bars')
        }
      } catch {
        // Generate realistic sample data when API is unavailable
        const now = Math.floor(Date.now() / 1000)
        const tfSeconds =
          timeframe === '1d' ? 86400 : timeframe === '4h' ? 14400 : 3600
        const sample: OHLCVBar[] = []
        let price = asset === 'BTC' ? 73000 : 2240
        for (let i = 200; i >= 0; i--) {
          const t = now - i * tfSeconds
          const volatility = asset === 'BTC' ? 0.008 : 0.012
          const change = (Math.random() - 0.48) * price * volatility
          const o = price
          const c = price + change
          const h = Math.max(o, c) + Math.random() * price * 0.003
          const l = Math.min(o, c) - Math.random() * price * 0.003
          const vol = (500 + Math.random() * 2000) * (asset === 'BTC' ? 1 : 100)
          sample.push({ time: t, open: o, high: h, low: l, close: c, volume: vol })
          price = c
        }
        if (!cancelled) setBars(sample)
      }
      if (!cancelled) setLoading(false)
    }
    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [asset, timeframe])

  // ── Computed indicators ──
  const emaFast = useMemo(() => computeEMA(bars, 8), [bars])
  const emaSlow = useMemo(() => computeEMA(bars, 21), [bars])
  const sr = useMemo(() => computeSupportResistance(bars, 30), [bars])
  const signals = useMemo(() => computeSignals(bars, emaFast, emaSlow), [bars, emaFast, emaSlow])

  // ── Create chart ──
  useEffect(() => {
    if (!containerRef.current) return
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
      seriesRefs.current = {}
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#5a6080',
        fontSize: 11,
        fontFamily: "'Inter', system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.025)' },
        horzLines: { color: 'rgba(255,255,255,0.025)' },
      },
      crosshair: {
        vertLine: { color: 'rgba(0,255,240,0.25)', width: 1, style: 2 },
        horzLine: { color: 'rgba(0,255,240,0.25)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        scaleMargins: { top: 0.05, bottom: 0.2 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // ── Candlestick series ──
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#00ffaa',
      downColor: '#ff2266',
      borderUpColor: '#00ffaa',
      borderDownColor: '#ff2266',
      wickUpColor: '#00ffaa66',
      wickDownColor: '#ff226666',
    })

    // ── EMA 8 (fast) ──
    const ema8 = chart.addSeries(LineSeries, {
      color: '#00ccff',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // ── EMA 21 (slow) ──
    const ema21 = chart.addSeries(LineSeries, {
      color: '#ff00aa',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      lineStyle: 2, // dashed
    })

    // ── Support line ──
    const supportLine = chart.addSeries(LineSeries, {
      color: '#00ffaa44',
      lineWidth: 1,
      lineStyle: 1, // dotted
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // ── Resistance line ──
    const resistanceLine = chart.addSeries(LineSeries, {
      color: '#ff226644',
      lineWidth: 1,
      lineStyle: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // ── Buy signal markers ──
    const buyMarkers = chart.addSeries(LineSeries, {
      color: '#00ffaa',
      lineWidth: 1 as const,
      lineVisible: false,
      pointMarkersVisible: true,
      pointMarkersRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // ── Sell signal markers ──
    const sellMarkers = chart.addSeries(LineSeries, {
      color: '#ff2266',
      lineWidth: 1 as const,
      lineVisible: false,
      pointMarkersVisible: true,
      pointMarkersRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // ── Volume histogram ──
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
    seriesRefs.current = {
      candle,
      ema8,
      ema21,
      supportLine,
      resistanceLine,
      buyMarkers,
      sellMarkers,
      volume,
    }

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
      seriesRefs.current = {}
    }
  }, [height])

  // ── Update series data ──
  useEffect(() => {
    const s = seriesRefs.current
    if (!s.candle || bars.length === 0) return

    // Candlestick data
    s.candle.setData(
      bars.map((b) => ({
        time: b.time as unknown,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }))
    )

    // EMA lines
    if (s.ema8 && emaFast.length > 0) {
      s.ema8.setData(emaFast.map((e) => ({ time: e.time as unknown, value: e.value })))
    }
    if (s.ema21 && emaSlow.length > 0) {
      s.ema21.setData(emaSlow.map((e) => ({ time: e.time as unknown, value: e.value })))
    }

    // Support/Resistance as horizontal lines spanning the full time range
    if (s.supportLine && bars.length >= 2) {
      s.supportLine.setData(
        bars.map((b) => ({ time: b.time as unknown, value: sr.support }))
      )
    }
    if (s.resistanceLine && bars.length >= 2) {
      s.resistanceLine.setData(
        bars.map((b) => ({ time: b.time as unknown, value: sr.resistance }))
      )
    }

    // Buy/Sell signal markers
    if (s.buyMarkers && signals.buys.length > 0) {
      s.buyMarkers.setData(
        signals.buys.map((sig) => ({ time: sig.time as unknown, value: sig.value }))
      )
    } else if (s.buyMarkers) {
      s.buyMarkers.setData([])
    }
    if (s.sellMarkers && signals.sells.length > 0) {
      s.sellMarkers.setData(
        signals.sells.map((sig) => ({ time: sig.time as unknown, value: sig.value }))
      )
    } else if (s.sellMarkers) {
      s.sellMarkers.setData([])
    }

    // Volume histogram
    if (s.volume) {
      s.volume.setData(
        bars.map((b) => ({
          time: b.time as unknown,
          value: b.volume ?? 0,
          color: b.close >= b.open ? 'rgba(0,255,170,0.15)' : 'rgba(255,34,102,0.15)',
        }))
      )
    }

    chartRef.current?.timeScale().fitContent()
  }, [bars, emaFast, emaSlow, sr, signals])

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#05050f]/60 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-2 border-[#00fff0] border-t-transparent rounded-full animate-spin" />
            <span className="text-[10px] text-[#5a6080] font-mono uppercase tracking-wider">
              Loading {asset} {timeframe}
            </span>
          </div>
        </div>
      )}
      {/* Legend */}
      <div className="absolute top-2 left-3 z-10 flex items-center gap-3 text-[9px] font-mono text-[#5a6080]">
        <span className="flex items-center gap-1">
          <span className="w-3 h-px bg-[#00ccff] inline-block" /> EMA(8)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-px bg-[#ff00aa] inline-block" style={{ borderTop: '1px dashed #ff00aa' }} /> EMA(21)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#00ffaa] inline-block" /> Buy
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#ff2266] inline-block" /> Sell
        </span>
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  )
}
