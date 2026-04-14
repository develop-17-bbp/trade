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

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[]; label?: string }) {
  if (!active || !payload?.[0]) return null
  const point = payload[0]
  const value = point.value
  const ts = point.payload.timestamp
  const isPositive = value >= 0

  return (
    <div className="bg-[#1a1a1a] border border-[#333] rounded px-3 py-2 shadow-lg">
      <p className="text-[#666] text-xs mb-1">
        {new Date(ts).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
      </p>
      <p className={`text-sm font-bold font-mono ${isPositive ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
        {formatValue(value)}
      </p>
    </div>
  )
}

export default function EquityCurve({ data, height = 250 }: EquityCurveProps) {
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity, max = -Infinity
    for (const d of data) {
      if (d.value < min) min = d.value
      if (d.value > max) max = d.value
    }
    const padding = (max - min) * 0.1 || 1
    return { minVal: min - padding, maxVal: max + padding }
  }, [data])

  const gradientOffset = useMemo(() => {
    if (maxVal <= 0) return 0
    if (minVal >= 0) return 1
    return maxVal / (maxVal - minVal)
  }, [minVal, maxVal])

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#22c55e" stopOpacity={0.02} />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#ef4444" stopOpacity={0.02} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.3} />
          </linearGradient>
          <linearGradient id="equityStroke" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#22c55e" />
            <stop offset={`${gradientOffset * 100}%`} stopColor="#ef4444" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} stroke="#333" tick={{ fill: '#666', fontSize: 10 }} axisLine={false} tickLine={false} minTickGap={40} />
        <YAxis tickFormatter={formatValue} stroke="#333" tick={{ fill: '#666', fontSize: 10 }} axisLine={false} tickLine={false} width={55} domain={[minVal, maxVal]} />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.06)', strokeWidth: 1, strokeDasharray: '4 4' }} />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="value" stroke="url(#equityStroke)" strokeWidth={1.5} fill="url(#equityGradient)" animationDuration={800} baseLine={0} />
      </AreaChart>
    </ResponsiveContainer>
  )
}
