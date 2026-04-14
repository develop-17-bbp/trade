import { useState, useEffect } from 'react'
import { ArrowUpRight, ArrowDownRight } from 'lucide-react'
import { fetchPrices } from '../../api/client'
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

export default function TopBar() {
  const prices = useLivePrices()
  const time = useClock()
  const { status } = useSystemState()

  const isOnline = status === 'ONLINE' || status === 'TRADING'

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between h-12 px-4 border-b border-[#222] bg-[#0a0a0a]">
      {/* Left: brand */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-bold tracking-wider text-white">ACT's</span>
        <span className="text-[10px] text-[#666] tracking-wider">AI TRADING</span>
      </div>

      {/* Center: live prices */}
      <div className="flex items-center gap-4">
        {prices.map(p => {
          const isUp = p.changePct >= 0
          const Arrow = isUp ? ArrowUpRight : ArrowDownRight
          return (
            <div key={p.symbol} className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-[#666] font-mono">{p.symbol}</span>
              <span className={`text-sm font-bold tabular-nums font-mono ${isUp ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                ${formatPrice(p.price)}
              </span>
              <Arrow size={12} className={isUp ? 'text-[#22c55e]' : 'text-[#ef4444]'} />
              <span className={`text-[10px] font-mono tabular-nums ${isUp ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                {isUp ? '+' : ''}{p.changePct.toFixed(2)}%
              </span>
            </div>
          )
        })}
      </div>

      {/* Right: status + time */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`} />
          <span className={`text-[10px] font-bold font-mono tracking-wider ${isOnline ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
            {status || 'OFFLINE'}
          </span>
        </div>
        <span className="text-[10px] text-[#666] font-mono">Robinhood</span>
        <span className="text-[10px] text-[#666] tabular-nums font-mono">{time}</span>
      </div>
    </header>
  )
}
