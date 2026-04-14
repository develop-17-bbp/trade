import type { ReactNode } from 'react'

interface MetricCardProps {
  label: string
  value: string | number
  subValue?: string
  icon?: ReactNode
  trend?: 'up' | 'down' | 'neutral'
  accentColor?: string
}

export default function MetricCard({
  label,
  value,
  subValue,
  icon,
  trend,
}: MetricCardProps) {
  const trendColor = trend === 'up' ? 'text-[#22c55e]' : trend === 'down' ? 'text-[#ef4444]' : 'text-white'

  return (
    <div className="card p-4 flex flex-col gap-1.5 min-w-0">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-[#666] uppercase tracking-wider">
          {label}
        </span>
        {icon && <span className="text-[#444]">{icon}</span>}
      </div>
      <div className={`text-xl font-bold tabular-nums metric-value ${trendColor}`}>
        {value}
      </div>
      {subValue && (
        <span className="text-[11px] tabular-nums text-[#666]">
          {subValue}
        </span>
      )}
    </div>
  )
}
