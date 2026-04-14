interface AgentEntry {
  direction: number
  confidence: number
  reasoning?: string
  [key: string]: unknown
}

interface AgentVotePanelProps {
  agents: Record<string, AgentEntry>
  compact?: boolean
}

function dirToSignal(dir: number): 'long' | 'short' | 'neutral' {
  if (dir > 0) return 'long'
  if (dir < 0) return 'short'
  return 'neutral'
}

const SIGNAL_CFG = {
  long: { text: 'LONG', color: 'text-[#22c55e]', bg: 'bg-[#22c55e]/10', dot: 'bg-[#22c55e]', bar: 'bg-[#22c55e]' },
  short: { text: 'SHORT', color: 'text-[#ef4444]', bg: 'bg-[#ef4444]/10', dot: 'bg-[#ef4444]', bar: 'bg-[#ef4444]' },
  neutral: { text: 'FLAT', color: 'text-[#666]', bg: 'bg-[#666]/10', dot: 'bg-[#666]', bar: 'bg-[#666]' },
}

export default function AgentVotePanel({ agents, compact = false }: AgentVotePanelProps) {
  const entries = Object.entries(agents)

  if (entries.length === 0) {
    return <p className="text-[#666] text-sm text-center py-6">No agent data available</p>
  }

  return (
    <div className={`space-y-1.5 max-h-64 overflow-y-auto ${compact ? '' : ''}`}>
      {entries.map(([name, agent]) => {
        const signal = dirToSignal(agent.direction)
        const cfg = SIGNAL_CFG[signal]
        const confidence = Math.round(agent.confidence * 100)

        return (
          <div key={name} className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-[#0a0a0a] transition-colors">
            <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
            <span className="text-xs text-white truncate flex-1 min-w-0">{name}</span>
            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded font-mono ${cfg.color} ${cfg.bg}`}>
              {cfg.text}
            </span>
            <div className="w-16 h-1 rounded-full bg-[#1a1a1a] overflow-hidden flex-shrink-0">
              <div className={`h-full rounded-full ${cfg.bar}`} style={{ width: `${confidence}%` }} />
            </div>
            <span className="text-[10px] text-[#666] tabular-nums font-mono w-7 text-right flex-shrink-0">
              {confidence}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
