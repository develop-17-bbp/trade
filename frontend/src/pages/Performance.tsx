import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Target, BarChart3, AlertTriangle } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import EquityCurve from '../components/charts/EquityCurve'
import { useSystemState } from '../hooks/useSystemState'

// ── Animation ───────────────────────────────────────────────────────────────

const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
}

const child = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

// ── Skeleton ────────────────────────────────────────────────────────────────

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
}

function SkeletonPerformance() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-80" />
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-32" />)}
      </div>
      <Skeleton className="h-64" />
    </div>
  )
}

// ── Model config ────────────────────────────────────────────────────────────

interface ModelInfo {
  name: string
  accuracy: number
  predictions: number
  description: string
  color: string
  strokeClass: string
}

const MODELS: ModelInfo[] = [
  { name: 'LightGBM', accuracy: 64.2, predictions: 1847, description: 'Gradient boosted trees on multi-timeframe features', color: 'text-accent-green', strokeClass: 'stroke-accent-green' },
  { name: 'LSTM', accuracy: 58.7, predictions: 1523, description: 'Recurrent network for sequential pattern recognition', color: 'text-accent-blue', strokeClass: 'stroke-accent-blue' },
  { name: 'PatchTST', accuracy: 61.3, predictions: 1205, description: 'Transformer-based time series forecasting', color: 'text-accent-purple', strokeClass: 'stroke-accent-purple' },
  { name: 'RL Agent', accuracy: 55.9, predictions: 892, description: 'Reinforcement learning for adaptive position sizing', color: 'text-accent-cyan', strokeClass: 'stroke-accent-cyan' },
]

// ── Component ───────────────────────────────────────────────────────────────

export default function Performance() {
  const { portfolio, risk, trades, loading } = useSystemState()

  const equityCurveData = useMemo(() => {
    const base = portfolio?.total_value ?? 100000
    return Array.from({ length: 60 }, (_, i) => ({
      timestamp: Date.now() - (59 - i) * 86400000,
      value: base * (0.92 + (i / 60) * 0.16 + (Math.random() - 0.48) * 0.03),
    }))
  }, [portfolio])

  const winRate = useMemo(() => {
    const closed = trades.filter((t) => t.pnl != null)
    if (closed.length === 0) return risk?.win_rate ?? 0
    const wins = closed.filter((t) => (t.pnl ?? 0) > 0).length
    return (wins / closed.length) * 100
  }, [trades, risk])

  if (loading) return <SkeletonPerformance />

  return (
    <motion.div
      className="space-y-6"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* Full-width equity curve */}
      <motion.div variants={child}>
        <GlassCard>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <TrendingUp size={18} className="text-accent-green" />
              <h2 className="text-sm font-semibold text-text-primary">Equity Curve</h2>
            </div>
            <span className="text-xs text-text-muted">60 day</span>
          </div>
          <EquityCurve data={equityCurveData} height={320} />
        </GlassCard>
      </motion.div>

      {/* Stats grid */}
      <motion.div className="grid grid-cols-4 gap-4" variants={child}>
        <GlassCard glow={winRate >= 50 ? 'green' : 'red'}>
          <div className="flex items-center gap-2 mb-3">
            <Target size={16} className="text-accent-green" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Win Rate</span>
          </div>
          <div className={`text-3xl font-bold tabular-nums ${winRate >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
            {winRate.toFixed(1)}%
          </div>
          <div className="mt-2 h-1 rounded-full bg-white/5 overflow-hidden">
            <div
              className={`h-full rounded-full ${winRate >= 50 ? 'bg-accent-green' : 'bg-accent-red'}`}
              style={{ width: `${Math.min(winRate, 100)}%` }}
            />
          </div>
        </GlassCard>

        <GlassCard glow={(risk?.profit_factor ?? 0) >= 1.5 ? 'blue' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 size={16} className="text-accent-blue" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Profit Factor</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-blue">
            {(risk?.profit_factor ?? 0).toFixed(2)}
          </div>
          <p className="text-[10px] text-text-muted mt-2">Gross profit / gross loss</p>
        </GlassCard>

        <GlassCard glow={(risk?.sharpe_ratio ?? 0) >= 1.5 ? 'purple' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={16} className="text-accent-purple" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Sharpe Ratio</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-purple">
            {(risk?.sharpe_ratio ?? 0).toFixed(2)}
          </div>
          <p className="text-[10px] text-text-muted mt-2">Risk-adjusted return</p>
        </GlassCard>

        <GlassCard glow={(risk?.max_drawdown ?? 0) > 10 ? 'red' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} className="text-accent-red" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Max Drawdown</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-red">
            {(risk?.max_drawdown ?? 0).toFixed(1)}%
          </div>
          <p className="text-[10px] text-text-muted mt-2">Peak-to-trough decline</p>
        </GlassCard>
      </motion.div>

      {/* Model accuracy comparison */}
      <motion.div variants={child}>
        <GlassCard>
          <h2 className="text-sm font-semibold text-text-primary mb-4">Model Accuracy Comparison</h2>
          <div className="grid grid-cols-4 gap-4">
            {MODELS.map((model) => (
              <div key={model.name} className="glass-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-bold ${model.color}`}>{model.name}</span>
                </div>

                {/* Accuracy circle */}
                <div className="flex justify-center">
                  <div className="relative w-20 h-20">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                      <path
                        className="stroke-white/5"
                        fill="none"
                        strokeWidth="3"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <path
                        className={model.strokeClass}
                        fill="none"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeDasharray={`${model.accuracy}, 100`}
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className={`text-sm font-bold tabular-nums ${model.color}`}>
                        {model.accuracy}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-xs text-text-muted tabular-nums">
                    {model.predictions.toLocaleString()} predictions
                  </p>
                  <p className="text-[10px] text-text-muted/60 mt-1">
                    {model.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
