import { useMemo } from 'react'
import { Shield, AlertTriangle, Gauge, Layers, Activity } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import GlassCard from '../components/cards/GlassCard'
import { useSystemState } from '../hooks/useSystemState'

// -- Skeleton --

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[#111] ${className}`} />
}

function SkeletonRisk() {
  return (
    <div className="space-y-6 bg-[#000] min-h-screen p-6">
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
      </div>
      {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
    </div>
  )
}

// -- Helpers --

function computeRiskLevel(score: number): { label: string; color: string } {
  if (score <= 3) return { label: 'LOW', color: 'text-[#22c55e]' }
  if (score <= 5) return { label: 'MEDIUM', color: 'text-[#eab308]' }
  if (score <= 7) return { label: 'HIGH', color: 'text-[#ef4444]' }
  return { label: 'CRITICAL', color: 'text-[#ef4444]' }
}

// Pipeline layer names
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

// Status dot colors
const STATUS_DOT: Record<string, string> = {
  info: 'bg-[#22c55e]',
  warning: 'bg-[#eab308]',
  error: 'bg-[#ef4444]',
  debug: 'bg-[#666]',
}

// Status badge styles
const STATUS_BADGE: Record<string, string> = {
  info: 'text-[#22c55e] border-[#22c55e]/30',
  warning: 'text-[#eab308] border-[#eab308]/30',
  error: 'text-[#ef4444] border-[#ef4444]/30',
  debug: 'text-[#666] border-[#666]/30',
}

// -- Component --

export default function Risk() {
  const { risk, layerLogs, loading, error } = useSystemState()

  const riskScore = risk?.risk_score ?? 0
  const riskLevel = useMemo(() => computeRiskLevel(riskScore), [riskScore])
  const vpin = risk?.vpin ?? 0

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
      <div className="flex items-center justify-center h-64 bg-[#000]">
        <p className="text-[#ef4444] text-sm font-mono">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Top risk metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Current Drawdown"
          value={`${(risk?.current_drawdown ?? 0).toFixed(1)}%`}
          subValue="From equity peak"
          icon={<AlertTriangle size={16} />}
          trend={(risk?.current_drawdown ?? 0) > 5 ? 'down' : 'up'}
        />
        <MetricCard
          label="Max Drawdown"
          value={`${(risk?.max_drawdown ?? 0).toFixed(1)}%`}
          subValue={`Risk: ${riskLevel.label}`}
          icon={<Shield size={16} />}
          trend={(risk?.max_drawdown ?? 0) > 10 ? 'down' : 'up'}
        />
        <MetricCard
          label="Risk Score"
          value={`${riskScore.toFixed(1)}/10`}
          subValue={riskLevel.label}
          icon={<Gauge size={16} />}
          trend={riskScore <= 5 ? 'up' : 'down'}
        />
        <MetricCard
          label="VPIN"
          value={vpin.toFixed(3)}
          subValue="Volume-weighted price impact"
          icon={<Activity size={16} />}
          trend={vpin < 0.5 ? 'up' : 'down'}
        />
      </div>

      {/* Pipeline visualization */}
      <GlassCard>
        <div className="flex items-center gap-2 mb-5">
          <Layers size={18} className="text-[#a0a0a0]" />
          <h2 className="text-sm font-semibold text-white font-mono">Trading Pipeline</h2>
          <span className="ml-auto text-xs text-[#666] font-mono">
            {pipelineLayers.filter((l) => l.logCount > 0).length}/{pipelineLayers.length} layers reporting
          </span>
        </div>

        <div className="space-y-1">
          {pipelineLayers.map((layer, idx) => {
            const dotColor = STATUS_DOT[layer.status] ?? STATUS_DOT.info
            const badgeStyle = STATUS_BADGE[layer.status] ?? STATUS_BADGE.info
            const statusLabel = layer.status === 'info' ? 'ACTIVE' : layer.status.toUpperCase()

            return (
              <div key={layer.id} className="relative">
                <div className="flex items-center gap-4 p-3 rounded border border-[#222] bg-[#111] hover:bg-[#161616] transition-colors">
                  {/* Layer number */}
                  <div className="w-9 h-9 rounded bg-[#1a1a1a] border border-[#222] flex items-center justify-center flex-shrink-0">
                    <span className="text-xs font-bold text-[#a0a0a0] font-mono">{layer.id}</span>
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white font-mono">{layer.name}</span>
                      <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded border font-mono ${badgeStyle}`}>
                        {statusLabel}
                      </span>
                      {layer.logCount > 0 && (
                        <span className="text-[10px] text-[#666] font-mono">{layer.logCount} logs</span>
                      )}
                    </div>
                  </div>

                  {/* Last log */}
                  <div className="flex items-center gap-2 flex-shrink-0 max-w-xs">
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotColor}`} />
                    <span className="text-[10px] text-[#666] truncate font-mono">{layer.lastMessage}</span>
                  </div>
                </div>

                {/* Connector line */}
                {idx < pipelineLayers.length - 1 && (
                  <div className="ml-[30px] w-px h-1 bg-[#222]" />
                )}
              </div>
            )
          })}
        </div>
      </GlassCard>

      {/* Risk score visual */}
      <GlassCard>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white font-mono">Risk Assessment</h2>
          <span className={`text-sm font-bold font-mono ${riskLevel.color}`}>{riskLevel.label}</span>
        </div>

        <div className="space-y-3">
          {/* Risk score bar */}
          <div>
            <div className="flex items-center justify-between text-[10px] text-[#666] mb-1 font-mono">
              <span>Risk Score</span>
              <span>{riskScore.toFixed(1)} / 10</span>
            </div>
            <div className="flex h-3 rounded overflow-hidden bg-[#1a1a1a] border border-[#222]">
              <div
                className={`rounded transition-all duration-500 ${
                  riskScore <= 3 ? 'bg-[#22c55e]' :
                  riskScore <= 5 ? 'bg-[#eab308]' :
                  riskScore <= 7 ? 'bg-[#ef4444]/80' : 'bg-[#ef4444]'
                }`}
                style={{ width: `${Math.min(riskScore * 10, 100)}%` }}
              />
            </div>
          </div>

          {/* Risk scale legend */}
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-[#22c55e] tabular-nums font-mono">0-3</p>
              <p className="text-[10px] text-[#666] mt-1 font-mono">Low</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-[#eab308] tabular-nums font-mono">3-5</p>
              <p className="text-[10px] text-[#666] mt-1 font-mono">Medium</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-[#ef4444]/80 tabular-nums font-mono">5-7</p>
              <p className="text-[10px] text-[#666] mt-1 font-mono">High</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-[#ef4444] tabular-nums font-mono">7+</p>
              <p className="text-[10px] text-[#666] mt-1 font-mono">Critical</p>
            </div>
          </div>
        </div>
      </GlassCard>
    </div>
  )
}
