import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUpRight, ArrowDownRight, Wifi, WifiOff, Zap } from 'lucide-react'
import StatusDot from '../shared/StatusDot'
import { fetchPrices, type PriceData } from '../../api/client'
import { useSystemState } from '../../hooks/useSystemState'

interface TickerItem { symbol: string; price: number; changePct: number }

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
          items.push({ symbol, price: info?.price ?? 0, changePct: info?.change_pct ?? 0 })
        }
        if (items.length > 0) setTickers(items)
      } catch { /* */ }
    }
    poll()
    const interval = setInterval(poll, 5000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])
  return tickers
}

function useClock(): string {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }))
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })), 1000)
    return () => clearInterval(timer)
  }, [])
  return time
}

function formatPrice(price: number): string {
  if (price === 0) return '---.--'
  return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function PriceTicker({ data }: { data: TickerItem }) {
  const isUp = data.changePct >= 0
  const Arrow = isUp ? ArrowUpRight : ArrowDownRight

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all"
      style={{
        background: 'rgba(10, 8, 25, 0.6)',
        borderColor: isUp ? 'rgba(0,255,170,0.2)' : 'rgba(255,34,102,0.2)',
        boxShadow: isUp
          ? '0 0 10px rgba(0,255,170,0.06), inset 0 0 10px rgba(0,255,170,0.02)'
          : '0 0 10px rgba(255,34,102,0.06), inset 0 0 10px rgba(255,34,102,0.02)',
      }}
    >
      <span className="text-xs font-bold text-[#5a6080] font-mono">{data.symbol}</span>
      <AnimatePresence mode="popLayout">
        <motion.span
          key={data.price}
          initial={{ y: isUp ? 8 : -8, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: isUp ? -8 : 8, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="text-sm font-bold tabular-nums font-mono"
          style={{ color: isUp ? '#00ffaa' : '#ff2266' }}
        >
          ${formatPrice(data.price)}
        </motion.span>
      </AnimatePresence>
      <Arrow size={12} style={{ color: isUp ? '#00ffaa' : '#ff2266' }} />
      <span className="text-[10px] font-mono tabular-nums" style={{ color: isUp ? '#00ffaa' : '#ff2266' }}>
        {Math.abs(data.changePct).toFixed(2)}%
      </span>
    </div>
  )
}

export default function TopBar() {
  const prices = useLivePrices()
  const time = useClock()
  const { status } = useSystemState()

  const isOnline = status === 'ONLINE' || status === 'TRADING'
  const statusLabel = status || 'UNKNOWN'

  return (
    <header
      className="sticky top-0 z-20 flex items-center justify-between h-14 px-6 border-b"
      style={{
        background: 'linear-gradient(90deg, rgba(10,8,25,0.9), rgba(5,5,15,0.95))',
        borderColor: 'rgba(100, 80, 255, 0.1)',
        backdropFilter: 'blur(20px)',
        boxShadow: '0 2px 20px rgba(0,0,0,0.3)',
      }}
    >
      {/* Left: brand with 3D holographic logo */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center relative"
          style={{
            background: 'linear-gradient(135deg, #00fff0, #bf5fff, #ff00aa)',
            backgroundSize: '200% 200%',
            animation: 'gradient-shift 3s ease infinite',
            boxShadow: '0 0 12px rgba(0,255,240,0.35), 0 0 25px rgba(191,95,255,0.15)',
            border: '1px solid rgba(0,255,240,0.25)',
            transform: 'perspective(200px) rotateY(-5deg)',
          }}
        >
          <span className="text-[#05050f] font-black text-xs" style={{ textShadow: '0 0 3px rgba(0,255,240,0.5)' }}>A</span>
        </div>
        <h1 className="text-sm font-bold tracking-[0.15em]">
          <span className="gradient-text glitch-text" data-text="ACT's">ACT's</span>
          <span className="text-[#5a6080] ml-2 font-normal text-[10px] tracking-[0.25em]">AI TRADING</span>
        </h1>
      </div>

      {/* Center: live prices */}
      <div className="flex items-center gap-3">
        {prices.map(p => <PriceTicker key={p.symbol} data={p} />)}
      </div>

      {/* Right: status, exchange, time */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border"
          style={{
            background: 'rgba(10, 8, 25, 0.6)',
            borderColor: isOnline ? 'rgba(0,255,170,0.2)' : 'rgba(255,34,102,0.2)',
          }}
        >
          {isOnline ? (
            <Wifi size={13} style={{ color: '#00ffaa', filter: 'drop-shadow(0 0 4px rgba(0,255,170,0.5))' }} />
          ) : (
            <WifiOff size={13} style={{ color: '#ff2266' }} />
          )}
          <StatusDot status={isOnline ? 'online' : 'offline'} size={6} />
          <span className="text-[10px] font-bold tracking-wider font-mono"
            style={{ color: isOnline ? '#00ffaa' : '#ff2266', textShadow: isOnline ? '0 0 8px rgba(0,255,170,0.4)' : '0 0 8px rgba(255,34,102,0.4)' }}
          >
            {statusLabel}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-[#5a6080] font-mono tracking-wider">
          <Zap size={10} className="text-[#bf5fff]" style={{ filter: 'drop-shadow(0 0 3px rgba(191,95,255,0.5))' }} />
          <span>Robinhood</span>
        </div>

        <span className="text-[10px] text-[#5a6080] tabular-nums font-mono tracking-wider"
          style={{ textShadow: '0 0 6px rgba(0,255,240,0.2)' }}
        >
          {time}
        </span>
      </div>
    </header>
  )
}
