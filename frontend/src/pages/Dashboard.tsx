import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { DollarSign, TrendingUp, Target, Activity, Brain } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import GlassCard from '../components/cards/GlassCard'
import PositionCard from '../components/cards/PositionCard'
import TradeRow from '../components/cards/TradeRow'
import EquityCurve from '../components/charts/EquityCurve'
import TradingViewWidget from '../components/charts/TradingViewWidget'
import AIBrainOrb from '../components/three/AIBrainOrb'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import { useSystemState } from '../hooks/useSystemState'

const pageV = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.07, delayChildren: 0.1 } } }
const childV = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.4 } } }

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
}
function SkeletonDash() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-5 gap-4">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-24" />)}</div>
      <div className="grid grid-cols-5 gap-6">
        <div className="col-span-3 space-y-6"><Skeleton className="h-72" /><Skeleton className="h-64" /></div>
        <div className="col-span-2 space-y-6"><Skeleton className="h-64" /><Skeleton className="h-48" /></div>
      </div>
    </div>
  )
}

function fmtUsd(n: number) { const a = Math.abs(n); if (a >= 1e6) return `$${(n/1e6).toFixed(2)}M`; if (a >= 1e3) return `$${(n/1e3).toFixed(1)}K`; return `$${n.toFixed(2)}` }
function fmtPct(n: number) { return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%` }

export default function Dashboard() {
  const { portfolio, positions, trades, tradeStats, agents, loading, error } = useSystemState()

  // Trade stats from backend (pre-computed)
  const wins = tradeStats?.wins ?? 0
  const losses = tradeStats?.losses ?? 0
  const winRate = (tradeStats?.win_rate ?? 0) * 100

  // Equity curve from backend
  const equityCurveData = useMemo(() => {
    const curve = portfolio?.equity_curve ?? []
    return curve.map((pt) => ({ timestamp: new Date(pt.t).getTime(), value: pt.v }))
  }, [portfolio])

  // Agent consensus
  const { sentiment, consensusPct } = useMemo(() => {
    const agentList = agents?.list ?? []
    if (agentList.length === 0) return { sentiment: 'neutral' as const, consensusPct: 50 }
    const bullish = agentList.filter(a => a.direction > 0).length
    const pct = Math.round((bullish / agentList.length) * 100)
    const s = pct >= 60 ? 'bullish' : pct <= 40 ? 'bearish' : 'neutral'
    return { sentiment: s as 'bullish' | 'bearish' | 'neutral', consensusPct: pct }
  }, [agents])

  // Agent votes for panel — transform list to Record
  const agentVotesMap = useMemo(() => {
    const map: Record<string, { direction: number; confidence: number; reasoning?: string }> = {}
    for (const a of agents?.list ?? []) {
      map[a.name] = { direction: a.direction, confidence: a.confidence, reasoning: a.reasoning }
    }
    return map
  }, [agents])

  const recentTrades = useMemo(() => trades.slice(-10).reverse(), [trades])

  if (loading) return <SkeletonDash />

  return (
    <motion.div className="space-y-6" variants={pageV} initial="hidden" animate="show">
      {/* Error banner */}
      {error && (
        <div className="glass-card px-4 py-2 border-l-2 border-amber-500 text-amber-400 text-xs">
          {error}
        </div>
      )}

      {/* Top metrics */}
      <motion.div className="grid grid-cols-2 md:grid-cols-5 gap-4" variants={childV}>
        <MetricCard label="Equity" value={fmtUsd(portfolio?.equity ?? 0)} subValue={fmtPct(portfolio?.total_return_pct ?? 0)} icon={<DollarSign size={16} />} trend={(portfolio?.total_return_pct ?? 0) >= 0 ? 'up' : 'down'} accentColor="text-[#00ff88]" />
        <MetricCard label="Today P&L" value={fmtUsd(portfolio?.today_pnl ?? 0)} subValue="" icon={<TrendingUp size={16} />} trend={(portfolio?.today_pnl ?? 0) >= 0 ? 'up' : 'down'} />
        <MetricCard label="Win Rate" value={`${winRate.toFixed(1)}%`} subValue={`${wins + losses} closed`} icon={<Target size={16} />} trend={winRate >= 50 ? 'up' : 'down'} accentColor="text-[#aa55ff]" />
        <MetricCard label="Trades W/L" value={`${wins} / ${losses}`} subValue={`PF ${tradeStats?.profit_factor?.toFixed(2) ?? '—'}`} icon={<Activity size={16} />} trend={wins >= losses ? 'up' : 'down'} />
        <MetricCard label="AI Consensus" value={agents?.consensus ?? 'N/A'} subValue={`${agents?.cycle_count ?? 0} cycles`} icon={<Brain size={16} />} trend={sentiment === 'bullish' ? 'up' : sentiment === 'bearish' ? 'down' : 'neutral'} accentColor="text-[#00ffcc]" />
      </motion.div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left 60% */}
        <div className="lg:col-span-3 space-y-6">
          {/* TradingView Live Chart */}
          <motion.div variants={childV}>
            <GlassCard>
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-[#e8ecf4]">Live Market — BTC/USD</h2>
                <span className="text-[10px] text-[#5a6080] font-mono">TradingView Real-Time</span>
              </div>
              <TradingViewWidget symbol="KRAKEN:BTCUSD" interval="60" height={350} />
            </GlassCard>
          </motion.div>

          <motion.div variants={childV}>
            <GlassCard>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[#e8ecf4]">Equity Curve</h2>
                <span className="text-xs text-[#6b7a99]">{equityCurveData.length} points</span>
              </div>
              {equityCurveData.length > 0 ? (
                <EquityCurve data={equityCurveData} height={280} />
              ) : (
                <div className="flex items-center justify-center h-64 text-[#6b7a99] text-sm">No equity data yet — waiting for trades</div>
              )}
            </GlassCard>
          </motion.div>

          <motion.div variants={childV}>
            <GlassCard>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-[#e8ecf4]">Recent Trades</h2>
                <span className="text-xs text-[#6b7a99]">{recentTrades.length} trades</span>
              </div>
              <div className="space-y-1 max-h-80 overflow-y-auto">
                {recentTrades.length > 0 ? recentTrades.map((t, i) => (
                  <TradeRow
                    key={i}
                    timestamp={t.timestamp}
                    asset={t.asset}
                    direction={t.direction}
                    action={t.status === 'CLOSED' ? 'CLOSE' : 'OPEN'}
                    entryPrice={t.entry_price}
                    exitPrice={t.exit_price || null}
                    pnlPct={t.pnl_pct || null}
                    pnlUsd={t.pnl || null}
                    reason={t.reason || t.trade_timeframe || ''}
                  />
                )) : (
                  <div className="text-center py-8 text-[#6b7a99] text-sm">No trades yet</div>
                )}
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* Right 40% */}
        <div className="lg:col-span-2 space-y-6">
          <motion.div variants={childV}>
            <GlassCard className="relative overflow-hidden">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-[#e8ecf4]">AI Brain</h2>
                <span className={`text-xs font-bold ${sentiment === 'bullish' ? 'text-[#00ff88]' : sentiment === 'bearish' ? 'text-[#ff3366]' : 'text-[#00aaff]'}`}>
                  {consensusPct}% {sentiment.toUpperCase()}
                </span>
              </div>
              <div className="h-56">
                <AIBrainOrb consensus={consensusPct / 100} sentiment={sentiment} size={2.5} />
              </div>
            </GlassCard>
          </motion.div>

          <motion.div variants={childV}>
            <GlassCard>
              <h2 className="text-sm font-semibold text-[#e8ecf4] mb-3">Agent Votes ({(agents?.list ?? []).length})</h2>
              <AgentVotePanel agents={agentVotesMap} compact />
            </GlassCard>
          </motion.div>

          <motion.div variants={childV}>
            <GlassCard>
              <h2 className="text-sm font-semibold text-[#e8ecf4] mb-3">
                Open Positions <span className="ml-2 text-xs text-[#6b7a99] font-normal">{positions.length}</span>
              </h2>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {positions.length > 0 ? positions.map((p, i) => (
                  <PositionCard
                    key={i}
                    symbol={p.asset}
                    direction={p.direction?.toLowerCase() === 'long' ? 'long' : 'short'}
                    entryPrice={p.entry_price}
                    currentPrice={p.current_price}
                    quantity={p.quantity}
                    unrealizedPnl={p.unrealized_pnl ?? 0}
                    unrealizedPnlPct={p.entry_price > 0 ? ((p.current_price - p.entry_price) / p.entry_price) * 100 : 0}
                    leverage={1}
                  />
                )) : (
                  <div className="text-center py-8 text-[#6b7a99] text-sm">No open positions</div>
                )}
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
