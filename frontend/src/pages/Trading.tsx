import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { CandlestickChart, BookOpen, Layers, Radio, Clock } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import PositionCard from '../components/cards/PositionCard'
import { useSystemState } from '../hooks/useSystemState'

// -- Animation --

const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
}

const child = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

// -- Skeleton --

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
}

function SkeletonTrading() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-96" />
      <div className="grid grid-cols-2 gap-6">
        <Skeleton className="h-72" />
        <Skeleton className="h-72" />
      </div>
      <Skeleton className="h-32" />
    </div>
  )
}

function formatPrice(n: number): string {
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function formatTime(ts: string): string {
  const d = new Date(ts)
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function directionLabel(dir: string): string {
  if (dir === 'long' || dir === 'LONG') return 'LONG'
  if (dir === 'short' || dir === 'SHORT') return 'SHORT'
  return dir.toUpperCase()
}

// -- Component --

export default function Trading() {
  const { positions, trades, portfolio, loading, error } = useSystemState()

  const recentTrades = useMemo(() => {
    return [...trades]
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 20)
  }, [trades])

  if (loading) return <SkeletonTrading />

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-accent-red text-sm">{error}</p>
      </div>
    )
  }

  return (
    <motion.div
      className="space-y-6"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* Chart area */}
      <motion.div variants={child}>
        <GlassCard className="relative overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <CandlestickChart size={18} className="text-accent-blue" />
              <h2 className="text-sm font-semibold text-text-primary">BTC / USD</h2>
              <span className="text-xs text-text-muted">1H</span>
            </div>
            <div className="flex items-center gap-2">
              {['1m', '5m', '15m', '1H', '4H', '1D'].map((tf) => (
                <button
                  key={tf}
                  className={`text-[10px] px-2 py-1 rounded ${
                    tf === '1H'
                      ? 'bg-accent-blue/10 text-accent-blue'
                      : 'text-text-muted hover:text-text-primary'
                  } transition-colors`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          {/* Placeholder for TradingView / lightweight-charts */}
          <div className="flex items-center justify-center h-80 rounded-lg bg-bg-primary/60 border border-border-glass">
            <div className="text-center space-y-2">
              <CandlestickChart size={40} className="text-text-muted/30 mx-auto" />
              <p className="text-text-muted text-sm">TradingView Chart -- BTC/USD</p>
              <p className="text-text-muted/60 text-xs">lightweight-charts integration pending</p>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Positions + Trade Log */}
      <div className="grid grid-cols-2 gap-6">
        {/* Open Positions */}
        <motion.div variants={child}>
          <GlassCard padding={false}>
            <div className="flex items-center gap-2 px-5 pt-5 pb-3">
              <Layers size={16} className="text-accent-green" />
              <h2 className="text-sm font-semibold text-text-primary">Open Positions</h2>
              <span className="text-xs text-text-muted ml-auto">{positions.length}</span>
            </div>

            <div className="px-3 pb-4 space-y-2 max-h-96 overflow-y-auto">
              {positions.length > 0 ? (
                positions.map((p, idx) => (
                  <PositionCard
                    key={`${p.asset}-${idx}`}
                    symbol={p.asset}
                    direction={p.direction}
                    entryPrice={p.entry_price}
                    currentPrice={p.current_price}
                    quantity={p.quantity}
                    unrealizedPnl={p.unrealized_pnl}
                  />
                ))
              ) : (
                <div className="text-center py-12 text-text-muted text-sm">
                  No open positions
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>

        {/* Trade Log */}
        <motion.div variants={child}>
          <GlassCard padding={false}>
            <div className="flex items-center gap-2 px-5 pt-5 pb-3">
              <BookOpen size={16} className="text-accent-purple" />
              <h2 className="text-sm font-semibold text-text-primary">Trade Log</h2>
              <span className="text-xs text-text-muted ml-auto">{trades.length} total</span>
            </div>

            <div className="px-3 pb-4 max-h-96 overflow-y-auto">
              {recentTrades.length > 0 ? (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-text-muted text-[10px] uppercase tracking-wider border-b border-border-glass">
                      <th className="text-left pb-2 font-medium">Asset</th>
                      <th className="text-left pb-2 font-medium">Dir</th>
                      <th className="text-right pb-2 font-medium">Entry</th>
                      <th className="text-right pb-2 font-medium">Exit</th>
                      <th className="text-right pb-2 font-medium">PnL</th>
                      <th className="text-right pb-2 font-medium">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTrades.map((t, idx) => {
                      const isProfit = (t.pnl ?? 0) >= 0
                      return (
                        <tr key={`${t.asset}-${t.timestamp}-${idx}`} className="border-b border-border-glass/50">
                          <td className="py-2 text-text-primary font-medium">{t.asset}</td>
                          <td className={`py-2 ${t.direction === 'long' || t.direction === 'LONG' ? 'text-accent-green' : 'text-accent-red'}`}>
                            {directionLabel(t.direction)}
                          </td>
                          <td className="py-2 text-right text-text-primary tabular-nums">
                            ${formatPrice(t.entry_price)}
                          </td>
                          <td className="py-2 text-right text-text-primary tabular-nums">
                            {t.exit_price ? `$${formatPrice(t.exit_price)}` : '--'}
                          </td>
                          <td className={`py-2 text-right tabular-nums font-medium ${isProfit ? 'text-accent-green' : 'text-accent-red'}`}>
                            {isProfit ? '+' : ''}{t.pnl?.toFixed(2) ?? '--'}
                            {t.pnl_pct != null && (
                              <span className="text-text-muted ml-1">({t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%)</span>
                            )}
                          </td>
                          <td className="py-2 text-right text-text-muted tabular-nums">
                            {formatTime(t.timestamp)}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              ) : (
                <div className="text-center py-12 text-text-muted text-sm">
                  No trades yet
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* Trade execution panel */}
      <motion.div variants={child}>
        <GlassCard>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Radio size={16} className="text-accent-cyan" />
              <h2 className="text-sm font-semibold text-text-primary">Trade Execution</h2>
              <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-accent-purple/10 text-accent-purple">
                Paper Mode
              </span>
            </div>
            <span className="text-xs text-text-muted">
              Equity: ${formatPrice(portfolio?.equity ?? 0)}
            </span>
          </div>

          <div className="mt-4 grid grid-cols-4 gap-4">
            <div>
              <label className="block text-[10px] text-text-muted uppercase tracking-wider mb-1">Asset</label>
              <div className="glass-card !rounded-lg px-3 py-2 text-sm text-text-primary">
                BTC/USD
              </div>
            </div>

            <div>
              <label className="block text-[10px] text-text-muted uppercase tracking-wider mb-1">Direction</label>
              <div className="flex gap-2">
                <div className="flex-1 glass-card !rounded-lg px-3 py-2 text-xs text-center text-accent-green bg-accent-green/5 border border-accent-green/20">
                  LONG
                </div>
                <div className="flex-1 glass-card !rounded-lg px-3 py-2 text-xs text-center text-text-muted">
                  SHORT
                </div>
              </div>
            </div>

            <div>
              <label className="block text-[10px] text-text-muted uppercase tracking-wider mb-1">Size</label>
              <div className="glass-card !rounded-lg px-3 py-2 text-sm text-text-muted">
                AI Managed
              </div>
            </div>

            <div>
              <label className="block text-[10px] text-text-muted uppercase tracking-wider mb-1">Status</label>
              <div className="glass-card !rounded-lg px-3 py-2 text-xs text-text-muted flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-accent-green animate-pulse-glow" />
                Autonomous -- Read Only
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
