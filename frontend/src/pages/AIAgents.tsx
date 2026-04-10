import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Brain, Zap, Clock } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import AIBrainOrb from '../components/three/AIBrainOrb'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import ConsensusGauge from '../components/ai/ConsensusGauge'
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

// -- Helpers --

/** Map numeric direction (-1/0/1) to a signal string */
function dirToSignal(dir: number): 'long' | 'short' | 'neutral' {
  if (dir > 0) return 'long'
  if (dir < 0) return 'short'
  return 'neutral'
}

// -- Component --

export default function AIAgents() {
  const { agents, loading, error } = useSystemState()

  const agentList = agents?.list ?? []

  const { sentiment, consensusPct, bullCount, bearCount, neutralCount } = useMemo(() => {
    if (agentList.length === 0) {
      return { sentiment: 'neutral' as const, consensusPct: 50, bullCount: 0, bearCount: 0, neutralCount: 0 }
    }
    const bull = agentList.filter((a) => a.direction > 0).length
    const bear = agentList.filter((a) => a.direction < 0).length
    const neut = agentList.length - bull - bear
    const pct = Math.round((bull / agentList.length) * 100)
    const s = pct >= 60 ? 'bullish' : pct <= 40 ? 'bearish' : 'neutral'
    return {
      sentiment: s as 'bullish' | 'bearish' | 'neutral',
      consensusPct: pct,
      bullCount: bull,
      bearCount: bear,
      neutralCount: neut,
    }
  }, [agentList])

  const consensusLabel = agents?.consensus ?? 'N/A'

  // Build vote map as Record<string, {direction, confidence, reasoning}> for AgentVotePanel
  const agentVotesMap = useMemo(() => {
    const map: Record<string, { id: string; name: string; direction: number; confidence: number; reasoning: string; weight: number }> = {}
    agentList.forEach((a) => { map[a.id] = a })
    return map
  }, [agentList])

  if (loading) return <SkeletonAI />

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
      {/* Orb + Consensus row */}
      <div className="grid grid-cols-3 gap-6">
        {/* AI Brain Orb - large */}
        <motion.div className="col-span-2" variants={child}>
          <GlassCard className="relative overflow-hidden">
            <div className="flex items-center gap-2 mb-3">
              <Brain size={18} className="text-accent-purple" />
              <h2 className="text-sm font-semibold text-text-primary">AI Neural Core</h2>
              <span className="ml-auto text-xs text-text-muted">
                {agentList.length} agents
                {agents?.enabled ? ' -- ENABLED' : ' -- DISABLED'}
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
                    {consensusLabel}
                  </p>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* Side: Consensus gauge + Info */}
        <motion.div variants={child} className="space-y-6">
          <ConsensusGauge
            level={consensusLabel}
            percentage={consensusPct}
            bullish={bullCount}
            bearish={bearCount}
            neutral={neutralCount}
          />

          <GlassCard>
            <div className="flex items-center gap-2 mb-3">
              <Clock size={14} className="text-accent-cyan" />
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
                Agent Reasoning
              </h3>
            </div>

            <div className="space-y-2 max-h-52 overflow-y-auto">
              {agentList.length > 0 ? (
                agentList.map((agent) => {
                  const signal = dirToSignal(agent.direction)
                  const signalColor =
                    signal === 'long' ? 'text-accent-green' :
                    signal === 'short' ? 'text-accent-red' : 'text-accent-blue'

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
                            w={agent.weight.toFixed(1)}
                          </span>
                        </div>
                        <p className="text-[10px] text-text-muted mt-0.5 line-clamp-2">
                          <span className={signalColor}>
                            {signal.toUpperCase()}
                          </span>{' '}
                          ({Math.round(agent.confidence * 100)}%) -- {agent.reasoning || 'No reasoning'}
                        </p>
                      </div>
                    </div>
                  )
                })
              ) : (
                <p className="text-text-muted text-xs text-center py-4">
                  No agent data available
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
            <h2 className="text-sm font-semibold text-text-primary">All Agents ({agentList.length})</h2>
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
              {agents?.data_quality != null && (
                <span className="text-text-muted">
                  Data quality: {Math.round(agents.data_quality * 100)}%
                </span>
              )}
            </div>
          </div>
          <AgentVotePanel agents={agentVotesMap} />
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
