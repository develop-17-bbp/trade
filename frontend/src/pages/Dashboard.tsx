import { useMemo, useState, useEffect, useCallback } from 'react'
import { DollarSign, TrendingUp, Target, Activity, Brain, ChevronDown, ChevronUp } from 'lucide-react'
import GlassCard from '../components/cards/GlassCard'
import EquityCurve from '../components/charts/EquityCurve'
import CandlestickChart from '../components/charts/CandlestickChart'
import AgentVotePanel from '../components/ai/AgentVotePanel'
import LiveIntelligencePanel from '../components/ai/LiveIntelligencePanel'
import LiveBadge from '../components/shared/LiveBadge'
import { useSystemState } from '../hooks/useSystemState'

function fmtUsd(n: number) {
  const a = Math.abs(n)
  if (a >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `$${(n / 1e3).toFixed(1)}K`
  return `$${n.toFixed(2)}`
}
function fmtPct(n: number) { return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%` }

export default function Dashboard() {
  const {
    portfolio,
    positions,
    trades,
    tradeStats,
    agents,
    lastFetchedAt,
    pollIntervalMs,
    loading,
    error,
  } = useSystemState()
  const [selectedAsset, setSelectedAsset] = useState<'BTC' | 'ETH'>('BTC')
  const [panelOpen, setPanelOpen] = useState(false)
  const [chartHeight, setChartHeight] = useState(600)

  const recalcHeight = useCallback(() => {
    // 48px topbar + 44px metrics bar + 44px toggle bar + panel if open
    const overhead = 48 + 44 + 44 + (panelOpen ? 310 : 0)
    setChartHeight(Math.max(300, window.innerHeight - overhead))
  }, [panelOpen])

  useEffect(() => {
    recalcHeight()
    window.addEventListener('resize', recalcHeight)
    return () => window.removeEventListener('resize', recalcHeight)
  }, [recalcHeight])

  const wins = tradeStats?.wins ?? 0
  const losses = tradeStats?.losses ?? 0
  const winRate = (tradeStats?.win_rate ?? 0) * 100

  const equityCurveData = useMemo(() => {
    const curve = portfolio?.equity_curve ?? []
    return curve.map((pt) => ({ timestamp: new Date(pt.t).getTime(), value: pt.v }))
  }, [portfolio])

  const { sentiment, consensusPct } = useMemo(() => {
    const agentList = agents?.list ?? []
    if (agentList.length === 0) return { sentiment: 'neutral' as const, consensusPct: 50 }
    const bullish = agentList.filter(a => a.direction > 0).length
    const pct = Math.round((bullish / agentList.length) * 100)
    const s = pct >= 60 ? 'bullish' : pct <= 40 ? 'bearish' : 'neutral'
    return { sentiment: s as 'bullish' | 'bearish' | 'neutral', consensusPct: pct }
  }, [agents])

  const agentVotesMap = useMemo(() => {
    const map: Record<string, { direction: number; confidence: number; reasoning?: string }> = {}
    for (const a of agents?.list ?? []) {
      map[a.name] = { direction: a.direction, confidence: a.confidence, reasoning: a.reasoning }
    }
    return map
  }, [agents])

  const recentTrades = useMemo(() => trades.slice(-10).reverse(), [trades])

  // Active positions for the selected asset
  const assetPositions = useMemo(() =>
    positions.filter(p => p.asset === selectedAsset),
    [positions, selectedAsset]
  )

  if (loading) return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="text-[#666] text-sm">Loading...</div>
    </div>
  )

  const todayPnl = portfolio?.today_pnl ?? 0
  const equity = portfolio?.equity ?? 0
  const totalReturnPct = portfolio?.total_return_pct ?? 0

  return (
    <div className="flex flex-col h-[calc(100vh-48px)]">
      {error && (
        <div className="px-3 py-1.5 border-b border-[#ef4444]/30 text-[#ef4444] text-xs bg-[#ef4444]/5">{error}</div>
      )}

      {/* ── Compact metrics bar ── */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-[#222] bg-[#0a0a0a] flex-shrink-0">
        {/* Asset selector */}
        <div className="flex gap-1">
          {(['BTC', 'ETH'] as const).map(asset => (
            <button
              key={asset}
              onClick={() => setSelectedAsset(asset)}
              className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                selectedAsset === asset
                  ? 'bg-white text-black font-bold'
                  : 'text-[#666] hover:text-white'
              }`}
            >
              {asset}
            </button>
          ))}
        </div>

        <span className="w-px h-5 bg-[#222]" />

        {/* Position badges on the bar */}
        {assetPositions.length > 0 && assetPositions.map((p, i) => {
          const pnlPct = p.entry_price > 0 ? ((p.current_price - p.entry_price) / p.entry_price) * 100 : 0
          const isProfit = pnlPct >= 0
          return (
            <div key={i} className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-mono ${isProfit ? 'bg-[#22c55e]/10 text-[#22c55e]' : 'bg-[#ef4444]/10 text-[#ef4444]'}`}>
              <span className="font-bold">{p.direction}</span>
              <span>@ ${p.entry_price.toLocaleString()}</span>
              <span className="font-bold">{isProfit ? '+' : ''}{pnlPct.toFixed(2)}%</span>
            </div>
          )
        })}

        {assetPositions.length === 0 && (
          <span className="text-[10px] text-[#666] font-mono">No position</span>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Compact metrics */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <DollarSign size={11} className="text-[#666]" />
            <span className="text-[#666] text-[10px]">Equity</span>
            <span className="text-white font-bold tabular-nums">{fmtUsd(equity)}</span>
            <span className={`text-[10px] tabular-nums ${totalReturnPct >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
              {fmtPct(totalReturnPct)}
            </span>
          </div>

          <span className="w-px h-4 bg-[#222]" />

          <div className="flex items-center gap-1.5">
            <TrendingUp size={11} className="text-[#666]" />
            <span className="text-[#666] text-[10px]">P&L</span>
            <span className={`font-bold tabular-nums ${todayPnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
              {fmtUsd(todayPnl)}
            </span>
          </div>

          <span className="w-px h-4 bg-[#222]" />

          <div className="flex items-center gap-1.5">
            <Target size={11} className="text-[#666]" />
            <span className="text-[#666] text-[10px]">WR</span>
            <span className={`font-bold tabular-nums ${winRate >= 50 ? 'text-[#22c55e]' : 'text-white'}`}>
              {winRate.toFixed(1)}%
            </span>
            <span className="text-[#666] text-[10px]">{wins}W/{losses}L</span>
          </div>

          <span className="w-px h-4 bg-[#222]" />

          <div className="flex items-center gap-1.5">
            <Activity size={11} className="text-[#666]" />
            <span className="text-[#666] text-[10px]">PF</span>
            <span className="text-white font-bold tabular-nums">{tradeStats?.profit_factor?.toFixed(2) ?? '--'}</span>
          </div>

          <span className="w-px h-4 bg-[#222]" />

          <div className="flex items-center gap-1.5">
            <Brain size={11} className="text-[#666]" />
            <span className="text-[#666] text-[10px]">AI</span>
            <span className={`font-bold ${
              sentiment === 'bullish' ? 'text-[#22c55e]' :
              sentiment === 'bearish' ? 'text-[#ef4444]' : 'text-[#666]'
            }`}>
              {agents?.consensus ?? 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* ── Full-page Native Chart with ACT trade markers ── */}
      <div className="flex-1 min-h-0 relative">
        <CandlestickChart
          asset={selectedAsset}
          timeframe="1h"
          height={chartHeight}
          trades={trades}
          positions={positions}
        />

        {/* ── Collapsible bottom panel ── */}
        <div className="absolute bottom-0 left-0 right-0 bg-black border-t border-[#222]">
          {/* Toggle bar */}
          <button
            onClick={() => setPanelOpen(!panelOpen)}
            className="w-full flex items-center justify-between px-4 py-2 hover:bg-[#0a0a0a] transition-colors"
          >
            <div className="flex items-center gap-4 text-[10px] font-mono text-[#666]">
              <span>{positions.length} position{positions.length !== 1 ? 's' : ''}</span>
              <span>{recentTrades.length} trades</span>
              <span>{(agents?.list ?? []).length} agents</span>
              {equityCurveData.length > 0 && <span>{equityCurveData.length} equity pts</span>}
            </div>
            <div className="flex items-center gap-1 text-[#666]">
              <span className="text-[10px] font-mono">{panelOpen ? 'HIDE' : 'DETAILS'}</span>
              {panelOpen ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
            </div>
          </button>

          {/* Expanded panel content */}
          {panelOpen && (
            <div className="grid grid-cols-5 gap-3 px-4 pb-3 max-h-[300px] overflow-y-auto">
              {/* Positions */}
              <GlassCard>
                <h3 className="text-[10px] font-bold text-[#666] uppercase tracking-wider mb-2">Open Positions</h3>
                <div className="space-y-1.5">
                  {positions.length > 0 ? positions.map((p, i) => {
                    const pnlPct = p.entry_price > 0 ? ((p.current_price - p.entry_price) / p.entry_price) * 100 : 0
                    const isProfit = pnlPct >= 0
                    const isLong = p.direction?.toLowerCase() === 'long'
                    return (
                      <div key={i} className="flex items-center justify-between text-xs font-mono py-1 border-b border-[#111]">
                        <div className="flex items-center gap-2">
                          <span className="text-white font-bold">{p.asset}</span>
                          <span className={`text-[9px] font-bold px-1 py-0.5 rounded ${isLong ? 'text-[#22c55e] bg-[#22c55e]/10' : 'text-[#ef4444] bg-[#ef4444]/10'}`}>
                            {p.direction}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[#666]">${p.entry_price.toLocaleString()}</span>
                          <span className={`font-bold ${isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                            {isProfit ? '+' : ''}{pnlPct.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    )
                  }) : (
                    <div className="text-[#666] text-xs text-center py-3">No positions</div>
                  )}
                </div>
              </GlassCard>

              {/* Trade History */}
              <GlassCard>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-[10px] font-bold text-[#666] uppercase tracking-wider">Recent Trades</h3>
                  <LiveBadge lastFetchedAt={lastFetchedAt} pollIntervalMs={pollIntervalMs} />
                </div>
                <div className="space-y-0 max-h-[220px] overflow-y-auto">
                  {recentTrades.length > 0 ? recentTrades.map((t, i) => {
                    const pnl = t.pnl || 0
                    const isProfit = pnl >= 0
                    const isClosed = t.status === 'CLOSED'
                    return (
                      <div key={i} className="flex items-center justify-between text-[10px] font-mono py-1 border-b border-[#111]">
                        <div className="flex items-center gap-2">
                          <span className="text-[#666]">{new Date(t.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}</span>
                          <span className="text-white">{t.asset}</span>
                          <span className={t.direction?.toLowerCase() === 'long' ? 'text-[#22c55e]' : 'text-[#ef4444]'}>{t.direction}</span>
                        </div>
                        <span className={`font-bold ${isClosed ? (isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]') : 'text-[#666]'}`}>
                          {isClosed ? `${isProfit ? '+' : ''}$${pnl.toFixed(2)}` : 'OPEN'}
                        </span>
                      </div>
                    )
                  }) : (
                    <div className="text-[#666] text-xs text-center py-3">No trades yet</div>
                  )}
                </div>
              </GlassCard>

              {/* AI Consensus + Agent Votes */}
              <GlassCard>
                <h3 className="text-[10px] font-bold text-[#666] uppercase tracking-wider mb-2">AI Agents</h3>
                {/* Consensus bar */}
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex-1 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${consensusPct}%`,
                        background: consensusPct >= 60 ? '#22c55e' : consensusPct <= 40 ? '#ef4444' : '#666',
                      }}
                    />
                  </div>
                  <span className={`text-xs font-bold font-mono ${
                    sentiment === 'bullish' ? 'text-[#22c55e]' :
                    sentiment === 'bearish' ? 'text-[#ef4444]' : 'text-[#666]'
                  }`}>
                    {consensusPct}%
                  </span>
                </div>
                <AgentVotePanel agents={agentVotesMap} compact />
              </GlassCard>

              {/* Equity Curve */}
              <GlassCard>
                <h3 className="text-[10px] font-bold text-[#666] uppercase tracking-wider mb-2">Equity Curve</h3>
                {equityCurveData.length > 0 ? (
                  <EquityCurve data={equityCurveData} height={200} />
                ) : (
                  <div className="flex items-center justify-center h-32 text-[#666] text-xs">Waiting for trades</div>
                )}
              </GlassCard>

              {/* Live Intelligence — RSS sentiment + Fear&Greed + derivatives */}
              <LiveIntelligencePanel />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
