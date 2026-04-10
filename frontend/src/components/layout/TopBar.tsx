import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUpRight, ArrowDownRight, Wifi, WifiOff } from 'lucide-react'
import StatusDot from '../shared/StatusDot'
import { fetchPrices, type PriceData } from '../../api/client'
import { useSystemState } from '../../hooks/useSystemState'

// -- Types --

interface TickerItem {
  symbol: string
  price: number
  changePct: number
}

// -- Hooks --

function useLivePrices(): TickerItem[] {
  const [tickers, setTickers] = useState<TickerItem[]>([
    { symbol: 'BTC', price: 0, changePct: 0 },
    { symbol: 'ETH', price: 0, changePct: 0 },
  ])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchPrices()
        if (cancelled || !data) return

        const items: TickerItem[] = []
        for (const [symbol, info] of Object.entries(data)) {
          items.push({
            symbol,
            price: info?.price ?? 0,
            changePct: info?.change_pct ?? 0,
          })
        }
        if (items.length > 0) setTickers(items)
      } catch {
        // Prices will show as 0 until backend is available
      }
    }

    poll()
    const interval = setInterval(poll, 5000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return tickers
}

function useClock(): string {
  const [time, setTime] = useState(() => formatTime(new Date()))

  useEffect(() => {
    const timer = setInterval(() => setTime(formatTime(new Date())), 1000)
    return () => clearInterval(timer)
  }, [])

  return time
}

// -- Formatters --

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

// -- Sub-components --

function PriceTicker({ data }: { data: TickerItem }) {
  const isUp = data.changePct >= 0
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
          {Math.abs(data.changePct).toFixed(2)}%
        </span>
      </div>
    </div>
  )
}

// -- Main component --

export default function TopBar() {
  const prices = useLivePrices()
  const time = useClock()
  const { status } = useSystemState()

  const isOnline = status === 'ONLINE' || status === 'TRADING'
  const statusLabel = status || 'UNKNOWN'
  const statusDotState: 'online' | 'offline' | 'degraded' =
    isOnline ? 'online' :
    status === 'INITIALIZING' ? 'degraded' : 'offline'

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
          {isOnline ? (
            <Wifi size={14} className="text-accent-green" />
          ) : (
            <WifiOff size={14} className="text-accent-red" />
          )}
          <StatusDot status={statusDotState} size={6} />
          <span className={`text-xs font-medium ${isOnline ? 'text-accent-green' : status === 'INITIALIZING' ? 'text-yellow-500' : 'text-accent-red'}`}>
            {statusLabel}
          </span>
        </div>

        {/* Exchange */}
        <span className="text-xs text-text-muted font-medium">Robinhood</span>

        {/* Time */}
        <span className="text-xs text-text-muted tabular-nums font-mono">{time}</span>
      </div>
    </header>
  )
}
