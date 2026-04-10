import { motion } from 'framer-motion'
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
  accentColor = 'text-accent-blue',
}: MetricCardProps) {
  const trendColor =
    trend === 'up'
      ? 'text-accent-green'
      : trend === 'down'
        ? 'text-accent-red'
        : 'text-text-muted'

  return (
    <motion.div
      className="glass-card p-4 flex flex-col gap-2 min-w-0"
      whileHover={{ scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-muted uppercase tracking-wider truncate">
          {label}
        </span>
        {icon && (
          <span className={`flex-shrink-0 ${accentColor}`}>{icon}</span>
        )}
      </div>
      <div className={`text-2xl font-bold tabular-nums ${trendColor}`}>
        {value}
      </div>
      {subValue && (
        <span className={`text-xs tabular-nums ${trendColor}`}>
          {subValue}
        </span>
      )}
    </motion.div>
  )
}
