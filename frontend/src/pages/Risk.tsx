import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Shield, AlertTriangle, Gauge, Layers } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import GlassCard from '../components/cards/GlassCard'
import { useSystemState } from '../hooks/useSystemState'

// ── Animation ───────────────────────────────────────────────────────────────

const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
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

function SkeletonRisk() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
      </div>
      {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
      <Skeleton className="h-32" />
    </div>
  )
}

// ── Pipeline layers ─────────────────────────────────────────────────────────

interface PipelineLayer {
  id: string
  name: string
  description: string
  status: 'ACTIVE' | 'IDLE' | 'WARNING' | 'ERROR'
  lastLog: string
}

function getPipelineLayers(): PipelineLayer[] {
  return [
    { id: 'L1', name: 'Data Ingestion', description: 'Real-time OHLCV + order book feeds', status: 'ACTIVE', lastLog: 'Binance WS connected -- 50ms latency' },
    { id: 'L2', name: 'Feature Engineering', description: 'Technical indicators + cross-asset features', status: 'ACTIVE', lastLog: '147 features computed in 23ms' },
    { id: 'L3', name: 'ML Inference', description: 'LightGBM + LSTM + PatchTST ensemble', status: 'ACTIVE', lastLog: 'Ensemble prediction: 0.67 bullish confidence' },
    { id: 'L4', name: 'RL Agent', description: 'PPO policy network for position sizing', status: 'ACTIVE', lastLog: 'Action: HOLD -- reward: +0.003' },
    { id: 'L5', name: 'Signal Aggregation', description: 'Weighted consensus from all model signals', status: 'ACTIVE', lastLog: 'Consensus: BULLISH (67%) -- 8/12 agents agree' },
    { id: 'L6', name: 'Risk Gate', description: 'Position limits, drawdown checks, VaR', status: 'ACTIVE', lastLog: 'All risk checks PASSED -- drawdown 3.2%' },
    { id: 'L7', name: 'Order Generation', description: 'Limit order placement with smart routing', status: 'IDLE', lastLog: 'No new orders -- last order 4m ago' },
    { id: 'L8', name: 'Execution Engine', description: 'Order lifecycle management + fill tracking', status: 'ACTIVE', lastLog: '2 open orders -- avg fill: 99.7% of target' },
    { id: 'L9', name: 'Post-Trade Analytics', description: 'PnL attribution, slippage analysis, logging', status: 'ACTIVE', lastLog: 'Session PnL: +$1,247.83 -- slippage: 0.02%' },
  ]
}

const STATUS_CONFIG: Record<string, { color: string; bg: string; border: string; dot: string }> = {
  ACTIVE: { color: 'text-accent-green', bg: 'bg-accent-green/5', border: 'border-accent-green/20', dot: 'bg-accent-green' },
  IDLE: { color: 'text-text-muted', bg: 'bg-white/[0.02]', border: 'border-border-glass', dot: 'bg-text-muted' },
  WARNING: { color: 'text-yellow-500', bg: 'bg-yellow-500/5', border: 'border-yellow-500/20', dot: 'bg-yellow-500' },
  ERROR: { color: 'text-accent-red', bg: 'bg-accent-red/5', border: 'border-accent-red/20', dot: 'bg-accent-red' },
}

// ── Component ───────────────────────────────────────────────────────────────

export default function Risk() {
  const { risk, loading } = useSystemState()

  const layers = useMemo(() => getPipelineLayers(), [])

  const riskScoreColor = useMemo(() => {
    const score = risk?.risk_score ?? 0
    if (score <= 30) return 'text-accent-green'
    if (score <= 60) return 'text-yellow-500'
    return 'text-accent-red'
  }, [risk])

  const riskScoreGlow = useMemo((): 'green' | 'red' | 'none' => {
    const score = risk?.risk_score ?? 0
    if (score <= 30) return 'green'
    if (score <= 60) return 'none'
    return 'red'
  }, [risk])

  if (loading) return <SkeletonRisk />

  return (
    <motion.div
      className="space-y-6"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* Top risk metrics */}
      <motion.div className="grid grid-cols-3 gap-4" variants={child}>
        <MetricCard
          label="Current Drawdown"
          value={`${(risk?.current_drawdown ?? 0).toFixed(1)}%`}
          subValue={`VaR: $${((risk?.portfolio_var ?? 0) / 1000).toFixed(1)}K`}
          icon={<AlertTriangle size={16} />}
          trend={(risk?.current_drawdown ?? 0) > 5 ? 'down' : 'up'}
          accentColor="text-accent-red"
        />
        <MetricCard
          label="Max Drawdown"
          value={`${(risk?.max_drawdown ?? 0).toFixed(1)}%`}
          subValue={`Risk level: ${(risk?.risk_level ?? 'unknown').toUpperCase()}`}
          icon={<Shield size={16} />}
          trend={(risk?.max_drawdown ?? 0) > 10 ? 'down' : 'up'}
          accentColor="text-accent-purple"
        />
        <MetricCard
          label="Risk Score"
          value={`${risk?.risk_score ?? 0}/100`}
          subValue={
            (risk?.risk_score ?? 0) <= 30 ? 'Low risk' :
            (risk?.risk_score ?? 0) <= 60 ? 'Moderate risk' : 'High risk'
          }
          icon={<Gauge size={16} />}
          trend={(risk?.risk_score ?? 0) <= 50 ? 'up' : 'down'}
          accentColor={riskScoreColor}
        />
      </motion.div>

      {/* Pipeline visualization */}
      <motion.div variants={child}>
        <GlassCard>
          <div className="flex items-center gap-2 mb-5">
            <Layers size={18} className="text-accent-blue" />
            <h2 className="text-sm font-semibold text-text-primary">Trading Pipeline</h2>
            <span className="ml-auto text-xs text-text-muted">
              {layers.filter((l) => l.status === 'ACTIVE').length}/{layers.length} layers active
            </span>
          </div>

          <div className="space-y-2">
            {layers.map((layer, idx) => {
              const cfg = STATUS_CONFIG[layer.status] ?? STATUS_CONFIG.IDLE

              return (
                <motion.div
                  key={layer.id}
                  className={`relative flex items-center gap-4 p-3 rounded-xl border ${cfg.border} ${cfg.bg} transition-all`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05, duration: 0.3 }}
                >
                  {/* Layer number */}
                  <div className="w-9 h-9 rounded-lg bg-white/[0.05] flex items-center justify-center flex-shrink-0">
                    <span className="text-xs font-bold text-text-muted">{layer.id}</span>
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text-primary">{layer.name}</span>
                      <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${cfg.color} ${cfg.bg} border ${cfg.border}`}>
                        {layer.status}
                      </span>
                    </div>
                    <p className="text-[10px] text-text-muted mt-0.5">{layer.description}</p>
                  </div>

                  {/* Last log */}
                  <div className="flex items-center gap-2 flex-shrink-0 max-w-xs">
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot} ${
                      layer.status === 'ACTIVE' ? 'animate-pulse-glow' : ''
                    }`} />
                    <span className="text-[10px] text-text-muted truncate">{layer.lastLog}</span>
                  </div>

                  {/* Connector line */}
                  {idx < layers.length - 1 && (
                    <div className="absolute -bottom-2 left-[30px] w-px h-2 bg-border-glass" />
                  )}
                </motion.div>
              )
            })}
          </div>
        </GlassCard>
      </motion.div>

      {/* Spread cost visualization */}
      <motion.div variants={child}>
        <GlassCard glow={riskScoreGlow}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-text-primary">Spread Cost Impact</h2>
            <span className="text-xs text-text-muted">Last 24h</span>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-red tabular-nums">3.34%</p>
              <p className="text-[10px] text-text-muted mt-1">Avg Spread Impact</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-blue tabular-nums">0.02%</p>
              <p className="text-[10px] text-text-muted mt-1">Avg Slippage</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-purple tabular-nums">$47.20</p>
              <p className="text-[10px] text-text-muted mt-1">Total Fees (24h)</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-green tabular-nums">99.7%</p>
              <p className="text-[10px] text-text-muted mt-1">Fill Rate</p>
            </div>
          </div>

          {/* Visual bar */}
          <div className="mt-4">
            <div className="flex items-center justify-between text-[10px] text-text-muted mb-1">
              <span>Cost Breakdown</span>
              <span>3.34% total</span>
            </div>
            <div className="flex h-3 rounded-full overflow-hidden bg-white/5">
              <div className="bg-accent-red/60 flex items-center justify-center" style={{ width: '60%' }}>
                <span className="text-[8px] text-white/80">Spread</span>
              </div>
              <div className="bg-accent-purple/60 flex items-center justify-center" style={{ width: '25%' }}>
                <span className="text-[8px] text-white/80">Fees</span>
              </div>
              <div className="bg-accent-blue/60 flex items-center justify-center" style={{ width: '15%' }}>
                <span className="text-[8px] text-white/80">Slip</span>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
