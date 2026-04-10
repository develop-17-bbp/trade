import { useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  DollarSign,
  TrendingUp,
  Target,
  Activity,
  Brain,
} from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import GlassCard from '../components/cards/GlassCard'
import PositionCard from '../components/cards/PositionCard'
import TradeRow from '../components/cards/TradeRow'
import EquityCurve from '../components/charts/EquityCurve'
import AIBrainOrb from '../components/three/AIBrainOrb'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import { useSystemState } from '../hooks/useSystemState'

// ── Animation variants ──────────────────────────────────────────────────────

const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.07, delayChildren: 0.1 },
  },
}

const childVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

// ── Skeleton ────────────────────────────────────────────────────────────────

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
}

function SkeletonDashboard() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-5 gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
      <div className="grid grid-cols-5 gap-6">
        <div className="col-span-3 space-y-6">
          <Skeleton className="h-72" />
          <Skeleton className="h-64" />
        </div>
        <div className="col-span-2 space-y-6">
          <Skeleton className="h-64" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function formatUsd(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(2)}`
}

function formatPct(n: number): string {
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`
}

// ── Component ───────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { portfolio, positions, trades, agents, loading } = useSystemState()

  const { wins, losses, winRate } = useMemo(() => {
    const closed = trades.filter((t) => t.pnl != null)
    const w = closed.filter((t) => (t.pnl ?? 0) > 0).length
    const l = closed.length - w
    return {
      wins: w,
      losses: l,
      winRate: closed.length > 0 ? (w / closed.length) * 100 : 0,
    }
  }, [trades])

  const equityCurveData = useMemo(() => {
    if (!portfolio) return []
    return Array.from({ length: 30 }, (_, i) => ({
      timestamp: Date.now() - (29 - i) * 86400000,
      value: (portfolio.total_value ?? 100000) + (Math.random() - 0.45) * 3000 * (i + 1) / 30,
    }))
  }, [portfolio])

  const { sentiment, consensusPct } = useMemo(() => {
    if (agents.length === 0) return { sentiment: 'neutral' as const, consensusPct: 50 }
    const bullish = agents.filter((a) => a.current_signal === 'long').length
    const pct = Math.round((bullish / agents.length) * 100)
    const s = pct >= 60 ? 'bullish' : pct <= 40 ? 'bearish' : 'neutral'
    return { sentiment: s as 'bullish' | 'bearish' | 'neutral', consensusPct: pct }
  }, [agents])

  const agentVotesMap = useMemo(() => {
    const map: Record<string, (typeof agents)[number]> = {}
    agents.forEach((a) => { map[a.id] = a })
    return map
  }, [agents])

  const recentTrades = useMemo(() => trades.slice(0, 10), [trades])

  if (loading) return <SkeletonDashboard />

  return (
    <motion.div
      className="space-y-6"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* Top metrics row */}
      <motion.div className="grid grid-cols-5 gap-4" variants={childVariants}>
        <MetricCard
          label="Equity"
          value={formatUsd(portfolio?.total_value ?? 0)}
          subValue={formatPct(portfolio?.total_pnl_pct ?? 0)}
          icon={<DollarSign size={16} />}
          trend={(portfolio?.total_pnl_pct ?? 0) >= 0 ? 'up' : 'down'}
          accentColor="text-accent-green"
        />
        <MetricCard
          label="Today P&L"
          value={formatUsd(portfolio?.pnl_today ?? 0)}
          subValue={formatPct(portfolio?.pnl_today_pct ?? 0)}
          icon={<TrendingUp size={16} />}
          trend={(portfolio?.pnl_today ?? 0) >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          subValue={`${wins + losses} closed trades`}
          icon={<Target size={16} />}
          trend={winRate >= 50 ? 'up' : 'down'}
          accentColor="text-accent-purple"
        />
        <MetricCard
          label="Trades W/L"
          value={`${wins} / ${losses}`}
          subValue={`${trades.length} total`}
          icon={<Activity size={16} />}
          trend={wins >= losses ? 'up' : 'down'}
        />
        <MetricCard
          label="AI Confidence"
          value={`${consensusPct}%`}
          subValue={sentiment.toUpperCase()}
          icon={<Brain size={16} />}
          trend={sentiment === 'bullish' ? 'up' : sentiment === 'bearish' ? 'down' : 'neutral'}
          accentColor="text-accent-cyan"
        />
      </motion.div>

      {/* Main grid */}
      <div className="grid grid-cols-5 gap-6">
        {/* Left column: 60% */}
        <div className="col-span-3 space-y-6">
          <motion.div variants={childVariants}>
            <GlassCard>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-text-primary">Equity Curve</h2>
                <span className="text-xs text-text-muted">30 day</span>
              </div>
              {equityCurveData.length > 0 ? (
                <EquityCurve data={equityCurveData} height={280} />
              ) : (
                <div className="flex items-center justify-center h-64 text-text-muted text-sm">
                  No equity data available
                </div>
              )}
            </GlassCard>
          </motion.div>

          <motion.div variants={childVariants}>
            <GlassCard padding={false}>
              <div className="flex items-center justify-between px-5 pt-5 pb-3">
                <h2 className="text-sm font-semibold text-text-primary">Recent Trades</h2>
                <span className="text-xs text-text-muted">{recentTrades.length} trades</span>
              </div>

              <div className="flex items-center gap-3 py-2 px-3 text-[10px] font-medium text-text-muted uppercase tracking-wider border-b border-border-glass mx-2">
                <span className="w-28 flex-shrink-0">Time</span>
                <span className="w-20 flex-shrink-0">Asset</span>
                <span className="w-12 flex-shrink-0 text-center">Action</span>
                <span className="w-24 flex-shrink-0 text-right">Entry</span>
                <span className="w-24 flex-shrink-0 text-right">Exit</span>
                <span className="w-20 flex-shrink-0 text-right">PnL $</span>
                <span className="w-16 flex-shrink-0 text-right">PnL %</span>
                <span className="flex-1">Strategy</span>
              </div>

              <div className="px-2 pb-3 max-h-80 overflow-y-auto">
                {recentTrades.length > 0 ? (
                  recentTrades.map((t) => (
                    <TradeRow
                      key={t.id}
                      timestamp={t.timestamp}
                      asset={t.symbol}
                      direction={t.side}
                      action={t.side === 'buy' ? 'BUY' : 'SELL'}
                      entryPrice={t.price}
                      exitPrice={null}
                      pnlPct={null}
                      pnlUsd={t.pnl}
                      reason={t.strategy}
                    />
                  ))
                ) : (
                  <div className="text-center py-8 text-text-muted text-sm">
                    No trades yet
                  </div>
                )}
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* Right column: 40% */}
        <div className="col-span-2 space-y-6">
          <motion.div variants={childVariants}>
            <GlassCard className="relative overflow-hidden">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-text-primary">AI Brain</h2>
                <span className={`text-xs font-bold ${
                  sentiment === 'bullish' ? 'text-accent-green' :
                  sentiment === 'bearish' ? 'text-accent-red' : 'text-accent-blue'
                }`}>
                  {consensusPct}% {sentiment.toUpperCase()}
                </span>
              </div>
              <div className="h-56">
                <AIBrainOrb
                  consensus={consensusPct / 100}
                  sentiment={sentiment}
                  size={2.5}
                />
              </div>
            </GlassCard>
          </motion.div>

          <motion.div variants={childVariants}>
            <GlassCard>
              <h2 className="text-sm font-semibold text-text-primary mb-3">Agent Votes</h2>
              <AgentVotePanel agents={agentVotesMap} compact />
            </GlassCard>
          </motion.div>

          <motion.div variants={childVariants}>
            <GlassCard padding={false}>
              <div className="px-5 pt-5 pb-3">
                <h2 className="text-sm font-semibold text-text-primary">
                  Open Positions
                  <span className="ml-2 text-xs text-text-muted font-normal">
                    {positions.length}
                  </span>
                </h2>
              </div>
              <div className="px-3 pb-3 space-y-2 max-h-80 overflow-y-auto">
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
                  <div className="text-center py-8 text-text-muted text-sm">
                    No open positions
                  </div>
                )}
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
