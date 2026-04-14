import { useMemo } from 'react'
import { TrendingUp, Target, BarChart3, AlertTriangle } from 'lucide-react'
import EquityCurve from '../components/charts/EquityCurve'
import { useSystemState } from '../hooks/useSystemState'

// -- Skeleton --

function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded ${className}`}
      style={{ backgroundColor: '#111111' }}
    />
  )
}

function SkeletonPerformance() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-80" />
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
      <Skeleton className="h-64" />
    </div>
  )
}

// -- Model display config --

const MODEL_CONFIG: Record<
  string,
  { label: string; description: string }
> = {
  lightgbm: {
    label: 'LightGBM',
    description: 'Gradient boosted trees on multi-timeframe features',
  },
  patchtst: {
    label: 'PatchTST',
    description: 'Transformer-based time series forecasting',
  },
  rl_agent: {
    label: 'RL Agent',
    description: 'Reinforcement learning for adaptive position sizing',
  },
  strategist: {
    label: 'Strategist',
    description: 'High-level strategy and regime detection',
  },
}

// -- Inline styles --

const styles = {
  card: {
    backgroundColor: '#111111',
    border: '1px solid #222222',
    borderRadius: '8px',
    padding: '20px',
  } as React.CSSProperties,
  textPrimary: { color: '#ffffff' },
  textSecondary: { color: '#a0a0a0' },
  textMuted: { color: '#666666' },
  green: { color: '#22c55e' },
  red: { color: '#ef4444' },
  bgGreen: { backgroundColor: '#22c55e' },
  bgRed: { backgroundColor: '#ef4444' },
}

// -- Component --

export default function Performance() {
  const { portfolio, risk, tradeStats, models, loading, error } = useSystemState()

  const equityCurveData = useMemo(() => {
    const curve = portfolio?.equity_curve
    if (!curve || curve.length === 0) return []
    return curve.map((pt) => ({
      timestamp: new Date(pt.t).getTime(),
      value: pt.v,
    }))
  }, [portfolio])

  const winRate = (tradeStats?.win_rate ?? 0) * 100
  const profitFactor = tradeStats?.profit_factor ?? 0
  const avgWin = tradeStats?.avg_win ?? 0
  const avgLoss = tradeStats?.avg_loss ?? 0
  const maxDrawdown = risk?.max_drawdown ?? 0
  const currentDrawdown = risk?.current_drawdown ?? 0

  const modelCards = useMemo(() => {
    return Object.entries(models).map(([key, data]) => {
      const cfg = MODEL_CONFIG[key] ?? {
        label: key,
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
        <p style={styles.red} className="text-sm">
          {error}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6" style={{ backgroundColor: '#000000', minHeight: '100%' }}>
      {/* Equity Curve */}
      <div className="card" style={styles.card}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp size={18} style={styles.green} />
            <h2 className="text-sm font-semibold" style={styles.textPrimary}>
              Equity Curve
            </h2>
          </div>
          <span className="text-xs" style={styles.textMuted}>
            {equityCurveData.length} data points
          </span>
        </div>
        {equityCurveData.length > 0 ? (
          <EquityCurve data={equityCurveData} height={320} />
        ) : (
          <div
            className="flex items-center justify-center h-80 text-sm"
            style={styles.textMuted}
          >
            No equity curve data available
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-4">
        {/* Win Rate */}
        <div className="card" style={styles.card}>
          <div className="flex items-center gap-2 mb-3">
            <Target size={16} style={styles.green} />
            <span
              className="text-xs uppercase tracking-wider"
              style={styles.textMuted}
            >
              Win Rate
            </span>
          </div>
          <div
            className="text-3xl font-bold tabular-nums"
            style={winRate >= 50 ? styles.green : styles.red}
          >
            {winRate.toFixed(1)}%
          </div>
          <div
            className="mt-2 h-1 rounded-full overflow-hidden"
            style={{ backgroundColor: '#222222' }}
          >
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(winRate, 100)}%`,
                ...(winRate >= 50 ? styles.bgGreen : styles.bgRed),
              }}
            />
          </div>
          <p className="text-[10px] mt-2" style={styles.textMuted}>
            {tradeStats?.wins ?? 0}W / {tradeStats?.losses ?? 0}L of{' '}
            {tradeStats?.total ?? 0}
          </p>
        </div>

        {/* Profit Factor */}
        <div className="card" style={styles.card}>
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 size={16} style={styles.textSecondary} />
            <span
              className="text-xs uppercase tracking-wider"
              style={styles.textMuted}
            >
              Profit Factor
            </span>
          </div>
          <div
            className="text-3xl font-bold tabular-nums"
            style={profitFactor >= 1 ? styles.green : styles.red}
          >
            {profitFactor.toFixed(2)}
          </div>
          <p className="text-[10px] mt-2" style={styles.textMuted}>
            Avg win: ${avgWin.toFixed(2)} / Avg loss: ${avgLoss.toFixed(2)}
          </p>
        </div>

        {/* Current Drawdown */}
        <div className="card" style={styles.card}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} style={styles.red} />
            <span
              className="text-xs uppercase tracking-wider"
              style={styles.textMuted}
            >
              Current DD
            </span>
          </div>
          <div className="text-3xl font-bold tabular-nums" style={styles.red}>
            {currentDrawdown.toFixed(1)}%
          </div>
          <p className="text-[10px] mt-2" style={styles.textMuted}>
            Current drawdown from peak
          </p>
        </div>

        {/* Max Drawdown */}
        <div className="card" style={styles.card}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} style={styles.red} />
            <span
              className="text-xs uppercase tracking-wider"
              style={styles.textMuted}
            >
              Max Drawdown
            </span>
          </div>
          <div className="text-3xl font-bold tabular-nums" style={styles.red}>
            {maxDrawdown.toFixed(1)}%
          </div>
          <p className="text-[10px] mt-2" style={styles.textMuted}>
            Peak-to-trough decline
          </p>
        </div>
      </div>

      {/* Model Accuracy Comparison */}
      <div className="card" style={styles.card}>
        <h2 className="text-sm font-semibold mb-4" style={styles.textPrimary}>
          Model Accuracy Comparison
        </h2>
        {modelCards.length > 0 ? (
          <div className="grid grid-cols-4 gap-4">
            {modelCards.map((model) => {
              const accColor = model.accuracy >= 50 ? '#22c55e' : '#ef4444'
              const accStroke = model.accuracy >= 50 ? '#22c55e' : '#ef4444'
              return (
                <div
                  key={model.key}
                  className="card"
                  style={{
                    backgroundColor: '#111111',
                    border: '1px solid #222222',
                    borderRadius: '8px',
                    padding: '16px',
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className="text-sm font-bold"
                      style={styles.textPrimary}
                    >
                      {model.label}
                    </span>
                  </div>

                  {/* Accuracy circle */}
                  <div className="flex justify-center mt-3">
                    <div className="relative w-20 h-20">
                      <svg
                        className="w-full h-full"
                        viewBox="0 0 36 36"
                        style={{ transform: 'rotate(-90deg)' }}
                      >
                        <path
                          fill="none"
                          stroke="#222222"
                          strokeWidth="3"
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                        <path
                          fill="none"
                          stroke={accStroke}
                          strokeWidth="3"
                          strokeLinecap="round"
                          strokeDasharray={`${model.accuracy}, 100`}
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span
                          className="text-sm font-bold tabular-nums"
                          style={{ color: accColor }}
                        >
                          {model.accuracy.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="text-center mt-3">
                    <p className="text-xs tabular-nums" style={styles.textSecondary}>
                      {model.total.toLocaleString()} predictions
                    </p>
                    <p className="text-[10px] mt-1" style={styles.textMuted}>
                      {model.description}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div
            className="text-center py-8 text-sm"
            style={styles.textMuted}
          >
            No model data available
          </div>
        )}
      </div>
    </div>
  )
}
