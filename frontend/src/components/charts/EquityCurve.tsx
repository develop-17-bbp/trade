import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useMemo } from 'react'

interface EquityDataPoint {
  timestamp: number
  value: number
}

interface EquityCurveProps {
  data: EquityDataPoint[]
  height?: number
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatValue(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(1)}K`
  return `$${v.toFixed(2)}`
}

interface TooltipPayloadItem {
  value: number
  payload: EquityDataPoint
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string
}) {
  if (!active || !payload?.[0]) return null

  const point = payload[0]
  const value = point.value
  const ts = point.payload.timestamp
  const isPositive = value >= 0
  const date = new Date(ts)

  return (
    <div className="glass-card rounded-lg px-3 py-2 border border-[var(--color-border-glass)] shadow-lg">
      <p className="text-[var(--color-text-muted)] text-xs mb-1">
        {date.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        })}
      </p>
      <p
        className="text-sm font-semibold"
        style={{ color: isPositive ? '#00ff88' : '#ff3366' }}
      >
        {formatValue(value)}
      </p>
    </div>
  )
}

export default function EquityCurve({ data, height = 250 }: EquityCurveProps) {
  const baseline = 0

  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity
    let max = -Infinity
    for (const d of data) {
      if (d.value < min) min = d.value
      if (d.value > max) max = d.value
    }
    const padding = (max - min) * 0.1 || 1
    return { minVal: min - padding, maxVal: max + padding }
  }, [data])

  // Split data into positive and negative segments for dual-color gradient
  const gradientOffset = useMemo(() => {
    if (maxVal <= 0) return 0
    if (minVal >= 0) return 1
    return maxVal / (maxVal - minVal)
  }, [minVal, maxVal])

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
      >
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00ff88" stopOpacity={0.4} />
            <stop
              offset={`${gradientOffset * 100}%`}
              stopColor="#00ff88"
              stopOpacity={0.05}
            />
            <stop
              offset={`${gradientOffset * 100}%`}
              stopColor="#ff3366"
              stopOpacity={0.05}
            />
            <stop offset="100%" stopColor="#ff3366" stopOpacity={0.4} />
          </linearGradient>
          <linearGradient id="equityStroke" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00ff88" />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#00ff88" />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#ff3366" />
            <stop offset="100%" stopColor="#ff3366" />
          </linearGradient>
        </defs>

        <XAxis
          dataKey="timestamp"
          tickFormatter={formatTimestamp}
          stroke="#3a4566"
          tick={{ fill: '#6b7a99', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          minTickGap={40}
        />
        <YAxis
          tickFormatter={formatValue}
          stroke="#3a4566"
          tick={{ fill: '#6b7a99', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={60}
          domain={[minVal, maxVal]}
        />

        <Tooltip
          content={<CustomTooltip />}
          cursor={{
            stroke: 'rgba(255,255,255,0.1)',
            strokeWidth: 1,
            strokeDasharray: '4 4',
          }}
        />

        <ReferenceLine
          y={baseline}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="3 3"
        />

        <Area
          type="monotone"
          dataKey="value"
          stroke="url(#equityStroke)"
          strokeWidth={2}
          fill="url(#equityGradient)"
          animationDuration={1200}
          animationEasing="ease-out"
          baseLine={baseline}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
