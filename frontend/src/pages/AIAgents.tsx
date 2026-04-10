import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Brain, Zap, Clock } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import AIBrainOrb from '../components/three/AIBrainOrb'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import ConsensusGauge from '../components/ai/ConsensusGauge'
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

function SkeletonAI() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-6">
        <Skeleton className="col-span-2 h-96" />
        <div className="space-y-6">
          <Skeleton className="h-48" />
          <Skeleton className="h-44" />
        </div>
      </div>
      <Skeleton className="h-64" />
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function formatTime(ts: string): string {
  const d = new Date(ts)
  return d.toLocaleString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

// ── Component ───────────────────────────────────────────────────────────────

export default function AIAgents() {
  const { agents, loading } = useSystemState()

  const { sentiment, consensusPct, bullCount, bearCount, neutralCount } = useMemo(() => {
    if (agents.length === 0) {
      return { sentiment: 'neutral' as const, consensusPct: 50, bullCount: 0, bearCount: 0, neutralCount: 0 }
    }
    const bull = agents.filter((a) => a.current_signal === 'long').length
    const bear = agents.filter((a) => a.current_signal === 'short').length
    const neut = agents.length - bull - bear
    const pct = Math.round((bull / agents.length) * 100)
    const s = pct >= 60 ? 'bullish' : pct <= 40 ? 'bearish' : 'neutral'
    return {
      sentiment: s as 'bullish' | 'bearish' | 'neutral',
      consensusPct: pct,
      bullCount: bull,
      bearCount: bear,
      neutralCount: neut,
    }
  }, [agents])

  const consensusLevel = useMemo(() => {
    if (consensusPct >= 70) return 'STRONG_BULLISH'
    if (consensusPct >= 55) return 'BULLISH'
    if (consensusPct <= 30) return 'STRONG_BEARISH'
    if (consensusPct <= 45) return 'BEARISH'
    return 'NEUTRAL'
  }, [consensusPct])

  const agentVotesMap = useMemo(() => {
    const map: Record<string, (typeof agents)[number]> = {}
    agents.forEach((a) => { map[a.id] = a })
    return map
  }, [agents])

  const recentDecisions = useMemo(() => {
    return [...agents]
      .filter((a) => a.last_action_time)
      .sort((a, b) => new Date(b.last_action_time).getTime() - new Date(a.last_action_time).getTime())
      .slice(0, 5)
  }, [agents])

  if (loading) return <SkeletonAI />

  return (
    <motion.div
      className="space-y-6"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* Orb + Consensus row */}
      <div className="grid grid-cols-3 gap-6">
        {/* AI Brain Orb - large */}
        <motion.div className="col-span-2" variants={child}>
          <GlassCard className="relative overflow-hidden">
            <div className="flex items-center gap-2 mb-3">
              <Brain size={18} className="text-accent-purple" />
              <h2 className="text-sm font-semibold text-text-primary">AI Neural Core</h2>
              <span className="ml-auto text-xs text-text-muted">
                {agents.filter((a) => a.status === 'active').length}/{agents.length} agents active
              </span>
            </div>

            <div className="relative h-80">
              <AIBrainOrb
                consensus={consensusPct / 100}
                sentiment={sentiment}
                size={3.5}
              />

              {/* Consensus overlay */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <motion.div
                    className={`text-6xl font-bold tabular-nums ${
                      sentiment === 'bullish' ? 'text-accent-green' :
                      sentiment === 'bearish' ? 'text-accent-red' : 'text-accent-blue'
                    }`}
                    key={consensusPct}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 0.9 }}
                    transition={{ type: 'spring', stiffness: 200 }}
                    style={{ textShadow: '0 0 30px rgba(0,0,0,0.8)' }}
                  >
                    {consensusPct}%
                  </motion.div>
                  <p className="text-xs text-text-muted mt-1" style={{ textShadow: '0 0 10px rgba(0,0,0,0.8)' }}>
                    {consensusLevel.replace('_', ' ')}
                  </p>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* Side: Consensus gauge + Decision feed */}
        <motion.div variants={child} className="space-y-6">
          <ConsensusGauge
            level={consensusLevel}
            percentage={consensusPct}
            bullish={bullCount}
            bearish={bearCount}
            neutral={neutralCount}
          />

          <GlassCard>
            <div className="flex items-center gap-2 mb-3">
              <Clock size={14} className="text-accent-cyan" />
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
                Decision Feed
              </h3>
            </div>

            <div className="space-y-2 max-h-52 overflow-y-auto">
              {recentDecisions.length > 0 ? (
                recentDecisions.map((agent) => {
                  const signalColor =
                    agent.current_signal === 'long' ? 'text-accent-green' :
                    agent.current_signal === 'short' ? 'text-accent-red' : 'text-accent-blue'

                  return (
                    <div
                      key={agent.id}
                      className="flex items-start gap-2 p-2 rounded-lg bg-white/[0.02] border border-border-glass"
                    >
                      <Zap size={12} className={`${signalColor} mt-0.5 flex-shrink-0`} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-text-primary truncate">
                            {agent.name}
                          </span>
                          <span className="text-[10px] text-text-muted tabular-nums flex-shrink-0 ml-2">
                            {formatTime(agent.last_action_time)}
                          </span>
                        </div>
                        <p className="text-[10px] text-text-muted mt-0.5 truncate">
                          {agent.last_action} --{' '}
                          <span className={signalColor}>
                            {(agent.current_signal ?? 'neutral').toUpperCase()}
                          </span>{' '}
                          ({Math.round(agent.confidence * 100)}%)
                        </p>
                      </div>
                    </div>
                  )
                })
              ) : (
                <p className="text-text-muted text-xs text-center py-4">
                  No recent decisions
                </p>
              )}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* All agents vote panel */}
      <motion.div variants={child}>
        <GlassCard>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-text-primary">All Agents ({agents.length})</h2>
            <div className="flex items-center gap-3 text-xs text-text-muted">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-accent-green" /> {bullCount} bullish
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-accent-blue" /> {neutralCount} neutral
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-accent-red" /> {bearCount} bearish
              </span>
            </div>
          </div>
          <AgentVotePanel agents={agentVotesMap} />
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
