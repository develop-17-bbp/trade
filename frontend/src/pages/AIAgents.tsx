import { useMemo } from 'react'
import { Brain, Zap, Clock } from 'lucide-react'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import { useSystemState } from '../hooks/useSystemState'

// -- Skeleton --

function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-xl ${className}`}
      style={{ backgroundColor: '#111111' }}
    />
  )
}

function SkeletonAI() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
        <Skeleton className="h-96" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <Skeleton className="h-48" />
          <Skeleton className="h-44" />
        </div>
      </div>
      <Skeleton className="h-64" />
    </div>
  )
}

// -- Helpers --

function dirToSignal(dir: number): 'long' | 'short' | 'neutral' {
  if (dir > 0) return 'long'
  if (dir < 0) return 'short'
  return 'neutral'
}

// -- Consensus Bar --

function ConsensusBar({ percentage, sentiment }: { percentage: number; sentiment: string }) {
  const barColor = sentiment === 'bullish' ? '#22c55e' : sentiment === 'bearish' ? '#ef4444' : '#666666'

  return (
    <div style={{ width: '100%', height: 8, backgroundColor: '#222222', borderRadius: 4, overflow: 'hidden' }}>
      <div
        style={{
          width: `${percentage}%`,
          height: '100%',
          backgroundColor: barColor,
          borderRadius: 4,
          transition: 'width 0.3s ease',
        }}
      />
    </div>
  )
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

  const agentVotesMap = useMemo(() => {
    const map: Record<string, { id: string; name: string; direction: number; confidence: number; reasoning: string; weight: number }> = {}
    agentList.forEach((a) => { map[a.id] = a })
    return map
  }, [agentList])

  if (loading) return <SkeletonAI />

  if (error) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}>
        <p style={{ color: '#ef4444', fontSize: 14 }}>{error}</p>
      </div>
    )
  }

  const pctColor = sentiment === 'bullish' ? '#22c55e' : sentiment === 'bearish' ? '#ef4444' : '#a0a0a0'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, backgroundColor: '#000000', minHeight: '100%' }}>

      {/* Consensus Gauge + Breakdown row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>

        {/* Consensus Gauge - large */}
        <div
          className="card"
          style={{
            backgroundColor: '#111111',
            border: '1px solid #222222',
            borderRadius: 12,
            padding: 24,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <Brain size={18} style={{ color: '#a0a0a0' }} />
            <h2 style={{ fontSize: 14, fontWeight: 600, color: '#ffffff', margin: 0 }}>AI Consensus</h2>
            <span style={{ marginLeft: 'auto', fontSize: 12, color: '#666666' }}>
              {agentList.length} agents
              {agents?.enabled ? ' -- ENABLED' : ' -- DISABLED'}
            </span>
          </div>

          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '48px 0',
          }}>
            <div
              style={{
                fontSize: 80,
                fontWeight: 700,
                color: pctColor,
                lineHeight: 1,
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              {consensusPct}%
            </div>
            <p style={{ fontSize: 14, color: '#666666', marginTop: 8 }}>
              {consensusLabel}
            </p>

            <div style={{ width: '80%', maxWidth: 400, marginTop: 24 }}>
              <ConsensusBar percentage={consensusPct} sentiment={sentiment} />
            </div>

            <div style={{ display: 'flex', gap: 24, marginTop: 16 }}>
              <span style={{ fontSize: 12, color: '#22c55e' }}>{bullCount} Bullish</span>
              <span style={{ fontSize: 12, color: '#a0a0a0' }}>{neutralCount} Neutral</span>
              <span style={{ fontSize: 12, color: '#ef4444' }}>{bearCount} Bearish</span>
            </div>
          </div>
        </div>

        {/* Side: Agent Reasoning */}
        <div
          className="card"
          style={{
            backgroundColor: '#111111',
            border: '1px solid #222222',
            borderRadius: 12,
            padding: 24,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <Clock size={14} style={{ color: '#a0a0a0' }} />
            <h3 style={{
              fontSize: 11,
              fontWeight: 600,
              color: '#ffffff',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              margin: 0,
            }}>
              Agent Reasoning
            </h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 380, overflowY: 'auto' }}>
            {agentList.length > 0 ? (
              agentList.map((agent) => {
                const signal = dirToSignal(agent.direction)
                const signalColor =
                  signal === 'long' ? '#22c55e' :
                  signal === 'short' ? '#ef4444' : '#a0a0a0'

                return (
                  <div
                    key={agent.id}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 8,
                      padding: 8,
                      borderRadius: 8,
                      backgroundColor: '#0a0a0a',
                      border: '1px solid #222222',
                    }}
                  >
                    <Zap size={12} style={{ color: signalColor, marginTop: 2, flexShrink: 0 }} />
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{
                          fontSize: 12,
                          fontWeight: 500,
                          color: '#ffffff',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {agent.name}
                        </span>
                        <span style={{
                          fontSize: 10,
                          color: '#666666',
                          fontVariantNumeric: 'tabular-nums',
                          flexShrink: 0,
                          marginLeft: 8,
                        }}>
                          w={agent.weight.toFixed(1)}
                        </span>
                      </div>
                      <p style={{
                        fontSize: 10,
                        color: '#666666',
                        marginTop: 2,
                        margin: 0,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}>
                        <span style={{ color: signalColor }}>
                          {signal.toUpperCase()}
                        </span>{' '}
                        ({Math.round(agent.confidence * 100)}%) -- {agent.reasoning || 'No reasoning'}
                      </p>
                    </div>
                  </div>
                )
              })
            ) : (
              <p style={{ color: '#666666', fontSize: 12, textAlign: 'center', padding: '16px 0' }}>
                No agent data available
              </p>
            )}
          </div>
        </div>
      </div>

      {/* All agents vote panel */}
      <div
        className="card"
        style={{
          backgroundColor: '#111111',
          border: '1px solid #222222',
          borderRadius: 12,
          padding: 24,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <h2 style={{ fontSize: 14, fontWeight: 600, color: '#ffffff', margin: 0 }}>
            All Agents ({agentList.length})
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 12, color: '#666666' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#22c55e', display: 'inline-block' }} /> {bullCount} bullish
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#a0a0a0', display: 'inline-block' }} /> {neutralCount} neutral
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} /> {bearCount} bearish
            </span>
            {agents?.data_quality != null && (
              <span style={{ color: '#666666' }}>
                Data quality: {Math.round(agents.data_quality * 100)}%
              </span>
            )}
          </div>
        </div>
        <AgentVotePanel agents={agentVotesMap} />
      </div>
    </div>
  )
}
