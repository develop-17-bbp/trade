import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import { useMemo } from 'react'

interface AgentVote {
  direction: number // -1 (short) to +1 (long)
  confidence: number // 0 to 1
}

interface AgentRadarProps {
  votes: Record<string, AgentVote>
}

interface RadarDataPoint {
  agent: string
  strength: number
  direction: number
  confidence: number
}

interface TooltipPayloadItem {
  payload: RadarDataPoint
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: TooltipPayloadItem[]
}) {
  if (!active || !payload?.[0]) return null

  const item = payload[0].payload
  const dirLabel =
    item.direction > 0.1 ? 'LONG' : item.direction < -0.1 ? 'SHORT' : 'NEUTRAL'
  const dirColor =
    item.direction > 0.1
      ? '#00ff88'
      : item.direction < -0.1
        ? '#ff3366'
        : '#00aaff'

  return (
    <div className="glass-card rounded-lg px-3 py-2 border border-[var(--color-border-glass)] shadow-lg">
      <p className="text-xs font-mono text-[var(--color-text-primary)] mb-1">
        {item.agent}
      </p>
      <p className="text-xs" style={{ color: dirColor }}>
        {dirLabel} &middot; {(item.confidence * 100).toFixed(0)}% conf
      </p>
    </div>
  )
}

export default function AgentRadar({ votes }: AgentRadarProps) {
  const { chartData, fillColor, strokeColor } = useMemo(() => {
    const entries = Object.entries(votes)
    const chartData: RadarDataPoint[] = entries.map(([agent, vote]) => ({
      agent,
      strength: vote.confidence * 100,
      direction: vote.direction,
      confidence: vote.confidence,
    }))

    // Determine consensus color
    const avgDirection =
      entries.reduce((sum, [, v]) => sum + v.direction * v.confidence, 0) /
      Math.max(entries.reduce((sum, [, v]) => sum + v.confidence, 0), 0.01)

    // Check how mixed the signals are
    const hasLong = entries.some(([, v]) => v.direction > 0.2)
    const hasShort = entries.some(([, v]) => v.direction < -0.2)
    const isMixed = hasLong && hasShort

    let fillColor: string
    let strokeColor: string

    if (isMixed && Math.abs(avgDirection) < 0.3) {
      fillColor = 'rgba(170, 85, 255, 0.25)'
      strokeColor = '#aa55ff'
    } else if (avgDirection > 0) {
      fillColor = 'rgba(0, 255, 136, 0.25)'
      strokeColor = '#00ff88'
    } else {
      fillColor = 'rgba(255, 51, 102, 0.25)'
      strokeColor = '#ff3366'
    }

    return { chartData, fillColor, strokeColor }
  }, [votes])

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart
        cx="50%"
        cy="50%"
        outerRadius="75%"
        data={chartData}
      >
        <PolarGrid
          stroke="rgba(255,255,255,0.06)"
          radialLines={false}
        />
        <PolarAngleAxis
          dataKey="agent"
          tick={{
            fill: '#6b7a99',
            fontSize: 10,
            fontFamily: 'monospace',
          }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Radar
          name="Vote Strength"
          dataKey="strength"
          stroke={strokeColor}
          fill={fillColor}
          strokeWidth={2}
          animationDuration={800}
          animationEasing="ease-out"
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
