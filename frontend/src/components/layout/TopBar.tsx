import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUpRight, ArrowDownRight, Wifi, WifiOff } from 'lucide-react'
import StatusDot from '../shared/StatusDot'

interface PriceData {
  symbol: string
  price: number
  change24h: number
}

function useLivePrices(): PriceData[] {
  const [prices, setPrices] = useState<PriceData[]>([
    { symbol: 'BTC', price: 0, change24h: 0 },
    { symbol: 'ETH', price: 0, change24h: 0 },
  ])

  useEffect(() => {
    let cancelled = false

    async function fetchPrices() {
      try {
        const res = await fetch('/api/v1/prices')
        if (!res.ok) return
        const data = (await res.json()) as PriceData[]
        if (!cancelled) setPrices(data)
      } catch {
        // Prices will show as 0 until backend is available
      }
    }

    fetchPrices()
    const interval = setInterval(fetchPrices, 5000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return prices
}

function useClock(): string {
  const [time, setTime] = useState(() => formatTime(new Date()))

  useEffect(() => {
    const timer = setInterval(() => setTime(formatTime(new Date())), 1000)
    return () => clearInterval(timer)
  }, [])

  return time
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatPrice(price: number): string {
  if (price === 0) return '---.--'
  return price.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

interface PriceTickerProps {
  data: PriceData
}

function PriceTicker({ data }: PriceTickerProps) {
  const isUp = data.change24h >= 0
  const Arrow = isUp ? ArrowUpRight : ArrowDownRight
  const color = isUp ? 'text-accent-green' : 'text-accent-red'

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-border-glass">
      <span className="text-xs font-medium text-text-muted">{data.symbol}</span>
      <AnimatePresence mode="popLayout">
        <motion.span
          key={data.price}
          initial={{ y: isUp ? 8 : -8, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: isUp ? -8 : 8, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="text-sm font-semibold text-text-primary tabular-nums"
        >
          ${formatPrice(data.price)}
        </motion.span>
      </AnimatePresence>
      <div className={`flex items-center gap-0.5 ${color}`}>
        <Arrow size={12} />
        <span className="text-xs font-medium tabular-nums">
          {Math.abs(data.change24h).toFixed(2)}%
        </span>
      </div>
    </div>
  )
}

export default function TopBar() {
  const prices = useLivePrices()
  const time = useClock()
  const [systemOnline, setSystemOnline] = useState(true)

  // Poll system status
  useEffect(() => {
    let cancelled = false

    async function checkStatus() {
      try {
        const res = await fetch('/api/v1/status')
        if (!cancelled) setSystemOnline(res.ok)
      } catch {
        if (!cancelled) setSystemOnline(false)
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 10000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between h-14 px-6 bg-bg-secondary/60 backdrop-blur-xl border-b border-border-glass">
      {/* Left: brand */}
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-bold tracking-wider">
          <span className="gradient-text">NEXUS</span>
          <span className="text-text-muted ml-1.5 font-normal text-xs">AI TRADING</span>
        </h1>
      </div>

      {/* Center: live prices */}
      <div className="flex items-center gap-3">
        {prices.map(p => (
          <PriceTicker key={p.symbol} data={p} />
        ))}
      </div>

      {/* Right: status, exchange, time */}
      <div className="flex items-center gap-4">
        {/* System status badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-border-glass">
          {systemOnline ? (
            <Wifi size={14} className="text-accent-green" />
          ) : (
            <WifiOff size={14} className="text-accent-red" />
          )}
          <StatusDot status={systemOnline ? 'online' : 'offline'} size={6} />
          <span className={`text-xs font-medium ${systemOnline ? 'text-accent-green' : 'text-accent-red'}`}>
            {systemOnline ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>

        {/* Exchange */}
        <span className="text-xs text-text-muted font-medium">Binance</span>

        {/* Time */}
        <span className="text-xs text-text-muted tabular-nums font-mono">{time}</span>
      </div>
    </header>
  )
}
