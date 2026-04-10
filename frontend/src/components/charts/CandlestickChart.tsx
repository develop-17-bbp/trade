import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, ColorType, CandlestickSeries, LineSeries } from 'lightweight-charts'

interface OHLCVBar {
  time: number
  open: number
  high: number
  low: number
  close: number
}

interface Props {
  asset?: string
  height?: number
}

export default function CandlestickChart({ asset = 'BTC', height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleRef = useRef<any>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const emaRef = useRef<any>(null)
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
        } else {
          throw new Error('No bars')
        }
      } catch {
        // Generate sample data if API unavailable
        const now = Math.floor(Date.now() / 1000)
        const sample: OHLCVBar[] = []
        let price = asset === 'BTC' ? 73000 : 2240
        for (let i = 200; i >= 0; i--) {
          const t = now - i * 3600
          const change = (Math.random() - 0.48) * price * 0.008
          const o = price
          const c = price + change
          const h = Math.max(o, c) + Math.random() * price * 0.003
          const l = Math.min(o, c) - Math.random() * price * 0.003
          sample.push({ time: t, open: o, high: h, low: l, close: c })
          price = c
        }
        if (!cancelled) setBars(sample)
      }
      if (!cancelled) setLoading(false)
    }
    fetchData()
    const interval = setInterval(fetchData, 60000)
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

    // lightweight-charts v5 API
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff88',
      downColor: '#ff3366',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff3366',
      wickUpColor: '#00ff88',
      wickDownColor: '#ff3366',
    })

    const ema = chart.addSeries(LineSeries, {
      color: '#00aaff',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    candleRef.current = candle
    emaRef.current = ema

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

  // Update data when bars change
  useEffect(() => {
    if (!candleRef.current || bars.length === 0) return

    // Set candle data
    candleRef.current.setData(
      bars.map(b => ({ time: b.time as unknown, open: b.open, high: b.high, low: b.low, close: b.close }))
    )

    // Compute and set EMA(8)
    const period = 8
    let emaVal = bars[0]?.close ?? 0
    const mult = 2 / (period + 1)
    const emaData = bars.map(b => {
      emaVal = b.close * mult + emaVal * (1 - mult)
      return { time: b.time as unknown, value: emaVal }
    })
    if (emaRef.current) {
      emaRef.current.setData(emaData)
    }

    chartRef.current?.timeScale().fitContent()
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
