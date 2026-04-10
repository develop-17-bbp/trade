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
  accentColor,
}: MetricCardProps) {
  const isUp = trend === 'up'
  const isDown = trend === 'down'
  const neonColor = isUp ? '#00ffaa' : isDown ? '#ff2266' : '#00ccff'
  const glowColor = isUp ? 'rgba(0,255,170,0.15)' : isDown ? 'rgba(255,34,102,0.15)' : 'rgba(0,204,255,0.1)'

  return (
    <motion.div
      className="glass-card holo-shimmer p-4 flex flex-col gap-2 min-w-0"
      whileHover={{ scale: 1.03, y: -2 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      style={{
        borderColor: `${neonColor}22`,
        boxShadow: `0 0 1px ${neonColor}44, 0 4px 20px rgba(0,0,0,0.4), inset 0 0 30px ${glowColor}`,
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-bold text-[#5a6080] uppercase tracking-[0.15em] font-mono truncate">
          {label}
        </span>
        {icon && (
          <span className={accentColor || ''} style={{ color: accentColor ? undefined : neonColor, filter: `drop-shadow(0 0 4px ${glowColor})` }}>
            {icon}
          </span>
        )}
      </div>
      <div
        className="text-2xl font-black tabular-nums font-mono"
        style={{ color: neonColor, textShadow: `0 0 12px ${glowColor}` }}
      >
        {value}
      </div>
      {subValue && (
        <span
          className="text-xs tabular-nums font-mono"
          style={{ color: `${neonColor}aa` }}
        >
          {subValue}
        </span>
      )}
    </motion.div>
  )
}
