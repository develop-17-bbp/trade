import { motion } from 'framer-motion'
import type { AgentOverlay } from '../../api/client'
import GlassCard from '../cards/GlassCard'

interface AgentVotePanelProps {
  agents: Record<string, AgentOverlay>
  compact?: boolean
}

const SIGNAL_LABEL: Record<string, { text: string; color: string; bg: string }> = {
  long: { text: 'LONG', color: 'text-accent-green', bg: 'bg-accent-green/10' },
  short: { text: 'SHORT', color: 'text-accent-red', bg: 'bg-accent-red/10' },
  neutral: { text: 'NEUTRAL', color: 'text-accent-blue', bg: 'bg-accent-blue/10' },
}

const STATUS_DOT: Record<string, string> = {
  active: 'bg-accent-green',
  idle: 'bg-text-muted',
  error: 'bg-accent-red',
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.04 },
  },
}

const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
}

export default function AgentVotePanel({ agents, compact = false }: AgentVotePanelProps) {
  const entries = Object.values(agents)

  if (entries.length === 0) {
    return (
      <GlassCard>
        <p className="text-text-muted text-sm text-center py-6">No agent data available</p>
      </GlassCard>
    )
  }

  return (
    <motion.div
      className={`grid gap-2 ${compact ? 'grid-cols-2' : 'grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'}`}
      variants={container}
      initial="hidden"
      animate="show"
    >
      {entries.map((agent) => {
        const signal = SIGNAL_LABEL[agent.current_signal ?? 'neutral'] ?? SIGNAL_LABEL.neutral
        const confidence = Math.round(agent.confidence * 100)
        const statusColor = STATUS_DOT[agent.status] ?? STATUS_DOT.idle

        return (
          <motion.div key={agent.id} variants={item}>
            <GlassCard className="!p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${statusColor}`} />
                  <span className="text-xs font-medium text-text-primary truncate">
                    {agent.name}
                  </span>
                </div>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${signal.color} ${signal.bg}`}>
                  {signal.text}
                </span>
              </div>

              {/* Confidence bar */}
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      agent.current_signal === 'long'
                        ? 'bg-accent-green'
                        : agent.current_signal === 'short'
                          ? 'bg-accent-red'
                          : 'bg-accent-blue'
                    }`}
                    style={{ width: `${confidence}%` }}
                  />
                </div>
                <span className="text-[10px] text-text-muted tabular-nums w-8 text-right">
                  {confidence}%
                </span>
              </div>

              {!compact && (
                <div className="mt-2 text-[10px] text-text-muted truncate">
                  {agent.last_action}
                </div>
              )}
            </GlassCard>
          </motion.div>
        )
      })}
    </motion.div>
  )
}
