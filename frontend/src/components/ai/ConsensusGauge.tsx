import { motion } from 'framer-motion'

interface ConsensusGaugeProps {
  level: string
  percentage: number
  bullish: number
  bearish: number
  neutral: number
}

export default function ConsensusGauge({
  level,
  percentage,
  bullish,
  bearish,
  neutral,
}: ConsensusGaugeProps) {
  const total = bullish + bearish + neutral || 1
  const bullPct = (bullish / total) * 100
  const bearPct = (bearish / total) * 100
  const neutPct = (neutral / total) * 100

  const levelColor =
    level.includes('BULLISH')
      ? 'text-accent-green'
      : level.includes('BEARISH')
        ? 'text-accent-red'
        : 'text-accent-blue'

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
          AI Consensus
        </span>
        <span className={`text-sm font-bold ${levelColor}`}>
          {level.replace('_', ' ')}
        </span>
      </div>

      {/* Large percentage display */}
      <div className="text-center">
        <motion.span
          className={`text-5xl font-bold tabular-nums ${levelColor}`}
          key={percentage}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300 }}
        >
          {percentage}%
        </motion.span>
        <p className="text-xs text-text-muted mt-1">Bullish Consensus</p>
      </div>

      {/* Stacked bar */}
      <div className="flex h-2 rounded-full overflow-hidden bg-white/5">
        <motion.div
          className="bg-accent-green"
          initial={{ width: 0 }}
          animate={{ width: `${bullPct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
        <motion.div
          className="bg-accent-blue"
          initial={{ width: 0 }}
          animate={{ width: `${neutPct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.1 }}
        />
        <motion.div
          className="bg-accent-red"
          initial={{ width: 0 }}
          animate={{ width: `${bearPct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
        />
      </div>

      {/* Legend */}
      <div className="flex justify-between text-xs">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent-green" />
          <span className="text-text-muted">Bull {bullish}</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent-blue" />
          <span className="text-text-muted">Neutral {neutral}</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent-red" />
          <span className="text-text-muted">Bear {bearish}</span>
        </span>
      </div>
    </div>
  )
}
