import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Shield, AlertTriangle, Gauge, Layers, Activity } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import GlassCard from '../components/cards/GlassCard'
import { useSystemState } from '../hooks/useSystemState'

// -- Animation --

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

// -- Skeleton --

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
}

function SkeletonRisk() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
      </div>
      {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
    </div>
  )
}

// -- Helpers --

function computeRiskLevel(score: number): { label: string; color: string } {
  if (score <= 3) return { label: 'LOW', color: 'text-accent-green' }
  if (score <= 5) return { label: 'MEDIUM', color: 'text-yellow-500' }
  if (score <= 7) return { label: 'HIGH', color: 'text-accent-red' }
  return { label: 'CRITICAL', color: 'text-accent-red' }
}

// Pipeline layer names for display
const LAYER_NAMES: Record<string, string> = {
  L1: 'Data Ingestion',
  L2: 'Feature Engineering',
  L3: 'ML Inference',
  L4: 'RL Agent',
  L5: 'Signal Aggregation',
  L6: 'Risk Gate',
  L7: 'Order Generation',
  L8: 'Execution Engine',
  L9: 'Post-Trade Analytics',
}

const LEVEL_CONFIG: Record<string, { color: string; bg: string; border: string; dot: string }> = {
  info: { color: 'text-accent-green', bg: 'bg-accent-green/5', border: 'border-accent-green/20', dot: 'bg-accent-green' },
  warning: { color: 'text-yellow-500', bg: 'bg-yellow-500/5', border: 'border-yellow-500/20', dot: 'bg-yellow-500' },
  error: { color: 'text-accent-red', bg: 'bg-accent-red/5', border: 'border-accent-red/20', dot: 'bg-accent-red' },
  debug: { color: 'text-text-muted', bg: 'bg-white/[0.02]', border: 'border-border-glass', dot: 'bg-text-muted' },
}

const DEFAULT_LEVEL_CFG = { color: 'text-accent-green', bg: 'bg-accent-green/5', border: 'border-accent-green/20', dot: 'bg-accent-green' }

// -- Component --

export default function Risk() {
  const { risk, layerLogs, loading, error } = useSystemState()

  const riskScore = risk?.risk_score ?? 0
  const riskLevel = useMemo(() => computeRiskLevel(riskScore), [riskScore])
  const vpin = risk?.vpin ?? 0

  const riskScoreGlow = useMemo((): 'green' | 'red' | 'none' => {
    if (riskScore <= 3) return 'green'
    if (riskScore <= 5) return 'none'
    return 'red'
  }, [riskScore])

  // Build ordered pipeline layers from layerLogs
  const pipelineLayers = useMemo(() => {
    const layerKeys = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']
    return layerKeys.map((key) => {
      const logs = layerLogs[key] ?? []
      const lastLog = logs.length > 0 ? logs[logs.length - 1] : null
      const hasError = logs.some((l) => l.level === 'error')
      const hasWarning = logs.some((l) => l.level === 'warning')
      const status = hasError ? 'error' : hasWarning ? 'warning' : logs.length > 0 ? 'info' : 'debug'
      return {
        id: key,
        name: LAYER_NAMES[key] ?? key,
        status,
        lastMessage: lastLog?.message ?? 'No logs',
        lastLevel: lastLog?.level ?? 'debug',
        logCount: logs.length,
      }
    })
  }, [layerLogs])

  if (loading) return <SkeletonRisk />

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
      {/* Top risk metrics */}
      <motion.div className="grid grid-cols-4 gap-4" variants={child}>
        <MetricCard
          label="Current Drawdown"
          value={`${(risk?.current_drawdown ?? 0).toFixed(1)}%`}
          subValue="From equity peak"
          icon={<AlertTriangle size={16} />}
          trend={(risk?.current_drawdown ?? 0) > 5 ? 'down' : 'up'}
          accentColor="text-accent-red"
        />
        <MetricCard
          label="Max Drawdown"
          value={`${(risk?.max_drawdown ?? 0).toFixed(1)}%`}
          subValue={`Risk: ${riskLevel.label}`}
          icon={<Shield size={16} />}
          trend={(risk?.max_drawdown ?? 0) > 10 ? 'down' : 'up'}
          accentColor="text-accent-purple"
        />
        <MetricCard
          label="Risk Score"
          value={`${riskScore.toFixed(1)}/10`}
          subValue={riskLevel.label}
          icon={<Gauge size={16} />}
          trend={riskScore <= 5 ? 'up' : 'down'}
          accentColor={riskLevel.color}
        />
        <MetricCard
          label="VPIN"
          value={vpin.toFixed(3)}
          subValue="Volume-weighted price impact"
          icon={<Activity size={16} />}
          trend={vpin < 0.5 ? 'up' : 'down'}
          accentColor="text-accent-cyan"
        />
      </motion.div>

      {/* Pipeline visualization */}
      <motion.div variants={child}>
        <GlassCard>
          <div className="flex items-center gap-2 mb-5">
            <Layers size={18} className="text-accent-blue" />
            <h2 className="text-sm font-semibold text-text-primary">Trading Pipeline</h2>
            <span className="ml-auto text-xs text-text-muted">
              {pipelineLayers.filter((l) => l.logCount > 0).length}/{pipelineLayers.length} layers reporting
            </span>
          </div>

          <div className="space-y-2">
            {pipelineLayers.map((layer, idx) => {
              const cfg = LEVEL_CONFIG[layer.status] ?? DEFAULT_LEVEL_CFG

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
                        {layer.status === 'info' ? 'ACTIVE' : layer.status.toUpperCase()}
                      </span>
                      {layer.logCount > 0 && (
                        <span className="text-[10px] text-text-muted">{layer.logCount} logs</span>
                      )}
                    </div>
                  </div>

                  {/* Last log */}
                  <div className="flex items-center gap-2 flex-shrink-0 max-w-xs">
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot} ${
                      layer.status === 'info' ? 'animate-pulse-glow' : ''
                    }`} />
                    <span className="text-[10px] text-text-muted truncate">{layer.lastMessage}</span>
                  </div>

                  {/* Connector line */}
                  {idx < pipelineLayers.length - 1 && (
                    <div className="absolute -bottom-2 left-[30px] w-px h-2 bg-border-glass" />
                  )}
                </motion.div>
              )
            })}
          </div>
        </GlassCard>
      </motion.div>

      {/* Risk score visual */}
      <motion.div variants={child}>
        <GlassCard glow={riskScoreGlow}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-text-primary">Risk Assessment</h2>
            <span className={`text-sm font-bold ${riskLevel.color}`}>{riskLevel.label}</span>
          </div>

          {/* Risk score bar */}
          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between text-[10px] text-text-muted mb-1">
                <span>Risk Score</span>
                <span>{riskScore.toFixed(1)} / 10</span>
              </div>
              <div className="flex h-3 rounded-full overflow-hidden bg-white/5">
                <div
                  className={`rounded-full transition-all duration-500 ${
                    riskScore <= 3 ? 'bg-accent-green' :
                    riskScore <= 5 ? 'bg-yellow-500' :
                    riskScore <= 7 ? 'bg-accent-red/80' : 'bg-accent-red'
                  }`}
                  style={{ width: `${Math.min(riskScore * 10, 100)}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 gap-4 mt-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-accent-green tabular-nums">0-3</p>
                <p className="text-[10px] text-text-muted mt-1">Low</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-yellow-500 tabular-nums">3-5</p>
                <p className="text-[10px] text-text-muted mt-1">Medium</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-accent-red/80 tabular-nums">5-7</p>
                <p className="text-[10px] text-text-muted mt-1">High</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-accent-red tabular-nums">7+</p>
                <p className="text-[10px] text-text-muted mt-1">Critical</p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}
