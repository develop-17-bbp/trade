import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { CandlestickChart, BookOpen, Layers, Radio } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import PositionCard from '../components/cards/PositionCard'
import { useSystemState } from '../hooks/useSystemState'

// ── Animation ───────────────────────────────────────────────────────────────

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

// ── Skeleton ────────────────────────────────────────────────────────────────

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

// ── Mock order book data ────────────────────────────────────────────────────

interface OrderLevel {
  price: number
  size: number
  total: number
}

function generateOrderBook(midPrice: number): { bids: OrderLevel[]; asks: OrderLevel[] } {
  const bids: OrderLevel[] = []
  const asks: OrderLevel[] = []
  let bidTotal = 0
  let askTotal = 0

  for (let i = 0; i < 10; i++) {
    const bidSize = +(Math.random() * 2 + 0.1).toFixed(4)
    bidTotal += bidSize
    bids.push({
      price: +(midPrice - (i + 1) * 12.5).toFixed(2),
      size: bidSize,
      total: +bidTotal.toFixed(4),
    })

    const askSize = +(Math.random() * 2 + 0.1).toFixed(4)
    askTotal += askSize
    asks.push({
      price: +(midPrice + (i + 1) * 12.5).toFixed(2),
      size: askSize,
      total: +askTotal.toFixed(4),
    })
  }

  return { bids, asks }
}

function formatPrice(n: number): string {
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

// ── Component ───────────────────────────────────────────────────────────────

export default function Trading() {
  const { positions, portfolio, loading } = useSystemState()

  const orderBook = useMemo(
    () => generateOrderBook(68_000),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const maxBidTotal = orderBook.bids[orderBook.bids.length - 1]?.total ?? 1
  const maxAskTotal = orderBook.asks[orderBook.asks.length - 1]?.total ?? 1

  if (loading) return <SkeletonTrading />

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

      {/* Order book + Positions */}
      <div className="grid grid-cols-2 gap-6">
        {/* Order Book */}
        <motion.div variants={child}>
          <GlassCard padding={false}>
            <div className="flex items-center gap-2 px-5 pt-5 pb-3">
              <BookOpen size={16} className="text-accent-purple" />
              <h2 className="text-sm font-semibold text-text-primary">Order Book</h2>
            </div>

            <div className="grid grid-cols-3 gap-2 px-5 py-2 text-[10px] font-medium text-text-muted uppercase tracking-wider border-b border-border-glass mx-2">
              <span>Price</span>
              <span className="text-right">Size</span>
              <span className="text-right">Total</span>
            </div>

            <div className="px-3 py-1">
              {/* Asks (reversed so lowest ask is near spread) */}
              {[...orderBook.asks].reverse().map((level, i) => (
                <div key={`ask-${i}`} className="relative grid grid-cols-3 gap-2 py-1 px-2 text-xs">
                  <div
                    className="absolute inset-y-0 right-0 bg-accent-red/[0.06] rounded-sm"
                    style={{ width: `${(level.total / maxAskTotal) * 100}%` }}
                  />
                  <span className="text-accent-red tabular-nums relative z-10">
                    ${formatPrice(level.price)}
                  </span>
                  <span className="text-text-primary tabular-nums text-right relative z-10">
                    {level.size.toFixed(4)}
                  </span>
                  <span className="text-text-muted tabular-nums text-right relative z-10">
                    {level.total.toFixed(4)}
                  </span>
                </div>
              ))}

              {/* Spread */}
              <div className="text-center py-2 text-xs">
                <span className="text-accent-blue font-semibold tabular-nums">
                  $68,000.00
                </span>
                <span className="text-text-muted ml-2 text-[10px]">Spread $25.00</span>
              </div>

              {/* Bids */}
              {orderBook.bids.map((level, i) => (
                <div key={`bid-${i}`} className="relative grid grid-cols-3 gap-2 py-1 px-2 text-xs">
                  <div
                    className="absolute inset-y-0 right-0 bg-accent-green/[0.06] rounded-sm"
                    style={{ width: `${(level.total / maxBidTotal) * 100}%` }}
                  />
                  <span className="text-accent-green tabular-nums relative z-10">
                    ${formatPrice(level.price)}
                  </span>
                  <span className="text-text-primary tabular-nums text-right relative z-10">
                    {level.size.toFixed(4)}
                  </span>
                  <span className="text-text-muted tabular-nums text-right relative z-10">
                    {level.total.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </GlassCard>
        </motion.div>

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
                positions.map((p) => (
                  <PositionCard
                    key={p.symbol}
                    symbol={p.symbol}
                    direction={p.side}
                    entryPrice={p.entry_price}
                    currentPrice={p.current_price}
                    quantity={p.size}
                    unrealizedPnl={p.unrealized_pnl}
                    unrealizedPnlPct={p.unrealized_pnl_pct}
                    leverage={p.leverage}
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
              Balance: ${formatPrice(portfolio?.available_balance ?? 0)}
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
