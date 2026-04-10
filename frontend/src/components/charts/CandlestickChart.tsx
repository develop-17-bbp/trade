import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, type ISeriesApi, type CandlestickData, ColorType } from 'lightweight-charts'

interface OHLCVBar {
  time: number  // unix seconds
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface Props {
  asset?: string
  height?: number
}

export default function CandlestickChart({ asset = 'BTC', height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const emaRef = useRef<ISeriesApi<'Line'> | null>(null)
  const [bars, setBars] = useState<OHLCVBar[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch OHLCV data
  useEffect(() => {
    let cancelled = false
    async function fetchData() {
      try {
        const res = await fetch(`/api/v1/ohlcv/${asset}?timeframe=1h&limit=200`)
        if (!res.ok) throw new Error('API error')
        const data = await res.json()
        if (!cancelled && data?.bars?.length > 0) {
          setBars(data.bars)
        }
      } catch {
        // Generate sample data if API unavailable
        const now = Math.floor(Date.now() / 1000)
        const sample: OHLCVBar[] = []
        let price = asset === 'BTC' ? 73000 : 2240
        for (let i = 200; i >= 0; i--) {
          const t = now - i * 3600
          const change = (Math.random() - 0.48) * price * 0.008
          const open = price
          const close = price + change
          const high = Math.max(open, close) + Math.random() * price * 0.003
          const low = Math.min(open, close) - Math.random() * price * 0.003
          sample.push({ time: t, open, high, low, close })
          price = close
        }
        if (!cancelled) setBars(sample)
      }
      if (!cancelled) setLoading(false)
    }
    fetchData()
    const interval = setInterval(fetchData, 60000) // refresh every minute
    return () => { cancelled = true; clearInterval(interval) }
  }, [asset])

  // Create chart
  useEffect(() => {
    if (!containerRef.current) return
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#6b7a99',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: {
        vertLine: { color: 'rgba(0,170,255,0.3)', width: 1, style: 2 },
        horzLine: { color: 'rgba(0,170,255,0.3)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff3366',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff3366',
      wickUpColor: '#00ff88',
      wickDownColor: '#ff3366',
    })

    const emaSeries = chart.addLineSeries({
      color: '#00aaff',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    seriesRef.current = candleSeries
    emaRef.current = emaSeries

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
    }
  }, [height])

  // Update data
  useEffect(() => {
    if (!seriesRef.current || bars.length === 0) return

    const candleData: CandlestickData[] = bars.map(b => ({
      time: b.time as any,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    seriesRef.current.setData(candleData)

    // Compute EMA(8)
    const emaData: { time: any; value: number }[] = []
    const period = 8
    let ema = bars[0]?.close ?? 0
    const mult = 2 / (period + 1)
    for (const b of bars) {
      ema = b.close * mult + ema * (1 - mult)
      emaData.push({ time: b.time as any, value: ema })
    }
    if (emaRef.current) {
      emaRef.current.setData(emaData)
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent()
    }
  }, [bars])

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="w-8 h-8 border-2 border-[#00aaff] border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  )
}
