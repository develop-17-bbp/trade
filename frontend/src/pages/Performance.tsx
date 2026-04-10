import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Target, BarChart3, AlertTriangle } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import EquityCurve from '../components/charts/EquityCurve'
import { useSystemState } from '../hooks/useSystemState'

// -- Animation --

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

// -- Skeleton --

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

// -- Model display config --

const MODEL_CONFIG: Record<string, { label: string; color: string; strokeClass: string; description: string }> = {
  lightgbm: { label: 'LightGBM', color: 'text-accent-green', strokeClass: 'stroke-accent-green', description: 'Gradient boosted trees on multi-timeframe features' },
  patchtst: { label: 'PatchTST', color: 'text-accent-purple', strokeClass: 'stroke-accent-purple', description: 'Transformer-based time series forecasting' },
  rl_agent: { label: 'RL Agent', color: 'text-accent-cyan', strokeClass: 'stroke-accent-cyan', description: 'Reinforcement learning for adaptive position sizing' },
  strategist: { label: 'Strategist', color: 'text-accent-blue', strokeClass: 'stroke-accent-blue', description: 'High-level strategy and regime detection' },
}

// -- Component --

export default function Performance() {
  const { portfolio, risk, tradeStats, models, loading, error } = useSystemState()

  // Convert equity_curve [{t, v}] to [{timestamp, value}] for EquityCurve component
  const equityCurveData = useMemo(() => {
    const curve = portfolio?.equity_curve
    if (!curve || curve.length === 0) return []
    return curve.map((pt) => ({
      timestamp: new Date(pt.t).getTime(),
      value: pt.v,
    }))
  }, [portfolio])

  const winRate = tradeStats?.win_rate ?? 0
  const profitFactor = tradeStats?.profit_factor ?? 0
  const avgWin = tradeStats?.avg_win ?? 0
  const avgLoss = tradeStats?.avg_loss ?? 0
  const maxDrawdown = risk?.max_drawdown ?? 0
  const currentDrawdown = risk?.current_drawdown ?? 0

  // Build model cards from live data
  const modelCards = useMemo(() => {
    return Object.entries(models).map(([key, data]) => {
      const cfg = MODEL_CONFIG[key] ?? {
        label: key,
        color: 'text-accent-blue',
        strokeClass: 'stroke-accent-blue',
        description: key,
      }
      const accuracy = data.total > 0 ? (data.correct / data.total) * 100 : 0
      return { key, accuracy, total: data.total, ...cfg }
    })
  }, [models])

  if (loading) return <SkeletonPerformance />

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-accent-red text-sm">{error}</p>
      </div>
    )
  }

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
            <span className="text-xs text-text-muted">
              {equityCurveData.length} data points
            </span>
          </div>
          {equityCurveData.length > 0 ? (
            <EquityCurve data={equityCurveData} height={320} />
          ) : (
            <div className="flex items-center justify-center h-80 text-text-muted text-sm">
              No equity curve data available
            </div>
          )}
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
          <p className="text-[10px] text-text-muted mt-2">
            {tradeStats?.wins ?? 0}W / {tradeStats?.losses ?? 0}L of {tradeStats?.total ?? 0}
          </p>
        </GlassCard>

        <GlassCard glow={profitFactor >= 1.5 ? 'blue' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 size={16} className="text-accent-blue" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Profit Factor</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-blue">
            {profitFactor.toFixed(2)}
          </div>
          <p className="text-[10px] text-text-muted mt-2">
            Avg win: ${avgWin.toFixed(2)} / Avg loss: ${avgLoss.toFixed(2)}
          </p>
        </GlassCard>

        <GlassCard glow={currentDrawdown > 5 ? 'red' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} className="text-accent-red" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Current DD</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-red">
            {currentDrawdown.toFixed(1)}%
          </div>
          <p className="text-[10px] text-text-muted mt-2">Current drawdown from peak</p>
        </GlassCard>

        <GlassCard glow={maxDrawdown > 10 ? 'red' : 'none'}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} className="text-accent-purple" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Max Drawdown</span>
          </div>
          <div className="text-3xl font-bold tabular-nums text-accent-purple">
            {maxDrawdown.toFixed(1)}%
          </div>
          <p className="text-[10px] text-text-muted mt-2">Peak-to-trough decline</p>
        </GlassCard>
      </motion.div>

      {/* Model accuracy comparison */}
      <motion.div variants={child}>
        <GlassCard>
          <h2 className="text-sm font-semibold text-text-primary mb-4">Model Accuracy Comparison</h2>
          {modelCards.length > 0 ? (
            <div className="grid grid-cols-4 gap-4">
              {modelCards.map((model) => (
                <div key={model.key} className="glass-card p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className={`text-sm font-bold ${model.color}`}>{model.label}</span>
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
                          {model.accuracy.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="text-center">
                    <p className="text-xs text-text-muted tabular-nums">
                      {model.total.toLocaleString()} predictions
                    </p>
                    <p className="text-[10px] text-text-muted/60 mt-1">
                      {model.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-text-muted text-sm">
              No model data available
            </div>
          )}
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
