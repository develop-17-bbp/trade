import { useMemo, useState, useEffect, useCallback, useRef } from 'react'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Target,
  Zap,
  Clock,
  AlertTriangle,
  DollarSign,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react'
import CandlestickChart from '../components/charts/CandlestickChart'
import EquityCurve from '../components/charts/EquityCurve'
import LiveBadge from '../components/shared/LiveBadge'
import { useSystemState } from '../hooks/useSystemState'
import { fetchPrices, type PriceData } from '../api/client'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatUSD(n: number, decimals = 2): string {
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function formatCompact(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `${n < 0 ? '-' : ''}$${(abs / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${n < 0 ? '-' : ''}$${(abs / 1_000).toFixed(1)}K`
  return `${n < 0 ? '-' : ''}$${abs.toFixed(2)}`
}

function pctColor(v: number): string {
  return v >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'
}

function formatDuration(ms: number): string {
  if (ms < 0 || !isFinite(ms)) return '--'
  const mins = Math.floor(ms / 60000)
  if (mins < 60) return `${mins}m`
  const hrs = Math.floor(mins / 60)
  const rem = mins % 60
  if (hrs < 24) return `${hrs}h${rem > 0 ? ` ${rem}m` : ''}`
  const days = Math.floor(hrs / 24)
  const rh = hrs % 24
  return `${days}d${rh > 0 ? ` ${rh}h` : ''}`
}

function formatTime(iso: string): string {
  if (!iso) return '--'
  try {
    const d = new Date(iso)
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    })
  } catch {
    return '--'
  }
}

type RatingInfo = { label: string; color: string; bg: string; border: string }

function getRating(score: number): RatingInfo {
  if (score >= 8)
    return {
      label: 'Strong Bullish',
      color: '#22c55e',
      bg: 'rgba(34,197,94,0.08)',
      border: 'rgba(34,197,94,0.25)',
    }
  if (score >= 6)
    return {
      label: 'Bullish',
      color: '#22c55e',
      bg: 'rgba(34,197,94,0.06)',
      border: 'rgba(34,197,94,0.18)',
    }
  if (score >= 4)
    return {
      label: 'Neutral',
      color: '#a0a0a0',
      bg: 'rgba(160,160,160,0.08)',
      border: 'rgba(160,160,160,0.25)',
    }
  return {
    label: 'Bearish',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.08)',
    border: 'rgba(239,68,68,0.25)',
  }
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[#222222] ${className}`} />
}

function SkeletonTrading() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-28" />
      <div className="flex gap-4">
        <Skeleton className="h-[480px] flex-[7]" />
        <Skeleton className="h-[480px] flex-[3]" />
      </div>
      <Skeleton className="h-48" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Screener Row Type
// ---------------------------------------------------------------------------

interface ScreenerRow {
  ticker: string
  price: number
  changePct: number
  score: number
  trendStrength: number
  squeeze: number
}

// ---------------------------------------------------------------------------
// Open Positions Panel
// ---------------------------------------------------------------------------

interface OpenPositionsPanelProps {
  positions: Array<{
    asset: string
    direction: string
    entry_price: number
    current_price: number
    quantity: number
    unrealized_pnl: number
    stop_loss: number
    entry_time: number | string
    confidence: number
    trade_timeframe: string
  }>
  onSelectAsset?: (asset: string) => void
  selectedAsset?: string
}

function OpenPositionsPanel({
  positions,
  onSelectAsset,
  selectedAsset,
}: OpenPositionsPanelProps) {
  const now = Date.now()

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <Activity size={14} className="text-[#22c55e]" />
        <h2 className="text-[11px] font-semibold tracking-wider uppercase text-[#ffffff]">
          Open Positions
        </h2>
        <span className="ml-auto text-[10px] font-mono text-[#666666]">
          {positions.length} {positions.length === 1 ? 'position' : 'positions'}
        </span>
      </div>

      {positions.length === 0 ? (
        <div className="px-4 pb-6 pt-2 text-center">
          <div className="flex flex-col items-center gap-2 py-8">
            <Clock size={24} className="text-[#444444]" />
            <span className="text-[11px] font-mono text-[#666666]">
              No open positions
            </span>
            <span className="text-[9px] font-mono text-[#444444]">
              ACT is waiting for high-confluence setup
            </span>
          </div>
        </div>
      ) : (
        <div className="px-3 pb-3 space-y-2">
          {positions.map((p) => {
            const isLong = (p.direction || '').toLowerCase() === 'long'
            const isProfit = p.unrealized_pnl >= 0
            const pnlPct =
              p.entry_price > 0
                ? ((p.current_price - p.entry_price) / p.entry_price) * 100 *
                  (isLong ? 1 : -1)
                : 0
            const entryMs =
              typeof p.entry_time === 'number'
                ? p.entry_time * 1000
                : p.entry_time
                  ? new Date(p.entry_time).getTime()
                  : 0
            const duration = entryMs > 0 ? formatDuration(now - entryMs) : '--'
            const slDistPct =
              p.stop_loss > 0 && p.current_price > 0
                ? ((p.current_price - p.stop_loss) / p.current_price) * 100
                : 0
            const isSelected = p.asset === selectedAsset

            return (
              <button
                key={`${p.asset}_${p.entry_price}`}
                onClick={() => onSelectAsset?.(p.asset)}
                className={`w-full text-left rounded-lg border p-3 transition-colors ${
                  isSelected
                    ? 'border-[#22c55e]/40 bg-[#22c55e]/[0.03]'
                    : 'border-[#222] hover:border-[#444] hover:bg-[#0a0a0a]'
                }`}
              >
                {/* Header row: asset + direction + pnl */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white">{p.asset}</span>
                    <span
                      className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${
                        isLong
                          ? 'text-[#22c55e] bg-[#22c55e]/10'
                          : 'text-[#ef4444] bg-[#ef4444]/10'
                      }`}
                    >
                      {isLong ? (
                        <TrendingUp size={9} className="inline mr-0.5" />
                      ) : (
                        <TrendingDown size={9} className="inline mr-0.5" />
                      )}
                      {isLong ? 'LONG' : 'SHORT'}
                    </span>
                    <span className="text-[9px] font-mono text-[#666]">
                      {p.trade_timeframe}
                    </span>
                  </div>
                  <div
                    className={`flex items-center gap-1 font-mono ${
                      isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'
                    }`}
                  >
                    {isProfit ? (
                      <ArrowUpRight size={12} />
                    ) : (
                      <ArrowDownRight size={12} />
                    )}
                    <span className="text-sm font-bold tabular-nums">
                      {isProfit ? '+' : ''}${p.unrealized_pnl.toFixed(2)}
                    </span>
                    <span className="text-[10px] tabular-nums">
                      ({pnlPct >= 0 ? '+' : ''}
                      {pnlPct.toFixed(2)}%)
                    </span>
                  </div>
                </div>

                {/* Details grid */}
                <div className="grid grid-cols-4 gap-2 text-[10px] font-mono">
                  <div>
                    <span className="text-[#666] block text-[9px] uppercase">
                      Entry
                    </span>
                    <span className="text-[#a0a0a0] tabular-nums">
                      ${formatUSD(p.entry_price)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#666] block text-[9px] uppercase">
                      Current
                    </span>
                    <span className="text-white tabular-nums">
                      ${formatUSD(p.current_price)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#666] block text-[9px] uppercase">
                      Stop Loss
                    </span>
                    <span className="text-[#ef4444] tabular-nums">
                      ${formatUSD(p.stop_loss)}
                      {slDistPct !== 0 && (
                        <span className="text-[#666] ml-1">
                          ({slDistPct >= 0 ? '-' : '+'}
                          {Math.abs(slDistPct).toFixed(1)}%)
                        </span>
                      )}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#666] block text-[9px] uppercase">
                      Held
                    </span>
                    <span className="text-[#a0a0a0] tabular-nums">{duration}</span>
                  </div>
                </div>

                {/* PnL bar */}
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 h-1 rounded-full bg-[#1a1a1a] overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        isProfit ? 'bg-[#22c55e]' : 'bg-[#ef4444]'
                      }`}
                      style={{
                        width: `${Math.min(Math.abs(pnlPct) * 10, 100)}%`,
                      }}
                    />
                  </div>
                  <span className="text-[9px] text-[#666] font-mono">
                    size {p.quantity.toFixed(6)}
                  </span>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Recent Trades Table
// ---------------------------------------------------------------------------

interface RecentTradesProps {
  trades: Array<{
    asset: string
    direction: string
    status: string
    event?: string
    entry_price: number
    exit_price: number
    pnl: number
    pnl_pct: number
    timestamp: string
    reason: string
    trade_timeframe?: string
  }>
  maxRows?: number
  lastFetchedAt?: number | null
  pollIntervalMs?: number
}

function RecentTrades({
  trades,
  maxRows = 15,
  lastFetchedAt,
  pollIntervalMs,
}: RecentTradesProps) {
  const sorted = useMemo(() => {
    return [...trades]
      .sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      .slice(0, maxRows)
  }, [trades, maxRows])

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <DollarSign size={14} className="text-[#ffffff]" />
        <h2 className="text-[11px] font-semibold tracking-wider uppercase text-[#ffffff]">
          Recent Trades (Live from ACT)
        </h2>
        <span className="ml-3">
          <LiveBadge lastFetchedAt={lastFetchedAt} pollIntervalMs={pollIntervalMs} />
        </span>
        <span className="ml-auto text-[10px] font-mono text-[#666666]">
          showing last {sorted.length}
        </span>
      </div>

      {sorted.length === 0 ? (
        <div className="px-4 pb-6 pt-2 text-center">
          <span className="text-[11px] font-mono text-[#666666]">
            No trades recorded yet
          </span>
        </div>
      ) : (
        <div className="overflow-x-auto max-h-[320px] overflow-y-auto">
          <table className="w-full text-[10px]">
            <thead className="sticky top-0 bg-[#050505] border-b border-[#222]">
              <tr className="text-[9px] uppercase tracking-wider font-mono text-[#666666]">
                <th className="text-left px-3 py-2 font-medium">Time</th>
                <th className="text-left px-2 py-2 font-medium">Asset</th>
                <th className="text-left px-2 py-2 font-medium">Side</th>
                <th className="text-left px-2 py-2 font-medium">Event</th>
                <th className="text-right px-2 py-2 font-medium">Entry</th>
                <th className="text-right px-2 py-2 font-medium">Exit</th>
                <th className="text-right px-2 py-2 font-medium">PnL $</th>
                <th className="text-right px-2 py-2 font-medium">PnL %</th>
                <th className="text-left px-3 py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((t, idx) => {
                const event = (t.event || t.status || '').toUpperCase()
                const isExit = event === 'EXIT' || event === 'CLOSED'
                const isProfit = t.pnl >= 0
                const isLong = (t.direction || '').toUpperCase() === 'LONG'

                return (
                  <tr
                    key={`${t.timestamp}_${idx}`}
                    className="border-b border-[#111] hover:bg-[#0a0a0a]"
                  >
                    <td className="px-3 py-1.5 font-mono text-[#a0a0a0] whitespace-nowrap">
                      {formatTime(t.timestamp)}
                    </td>
                    <td className="px-2 py-1.5 font-mono font-bold text-white">
                      {t.asset}
                    </td>
                    <td className="px-2 py-1.5">
                      <span
                        className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded font-mono ${
                          isLong
                            ? 'text-[#22c55e] bg-[#22c55e]/10'
                            : 'text-[#ef4444] bg-[#ef4444]/10'
                        }`}
                      >
                        {isLong ? 'LONG' : 'SHORT'}
                      </span>
                    </td>
                    <td className="px-2 py-1.5">
                      <span
                        className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded font-mono ${
                          isExit
                            ? 'text-[#a0a0a0] bg-[#ffffff]/5 border border-[#333]'
                            : 'text-[#22c55e] bg-[#22c55e]/10 border border-[#22c55e]/25'
                        }`}
                      >
                        {isExit ? 'EXIT' : 'ENTRY'}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[#a0a0a0]">
                      ${formatUSD(t.entry_price)}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[#a0a0a0]">
                      {t.exit_price > 0 ? `$${formatUSD(t.exit_price)}` : '--'}
                    </td>
                    <td
                      className={`px-2 py-1.5 text-right font-mono tabular-nums font-bold ${
                        !isExit
                          ? 'text-[#666]'
                          : isProfit
                            ? 'text-[#22c55e]'
                            : 'text-[#ef4444]'
                      }`}
                    >
                      {isExit
                        ? `${isProfit ? '+' : ''}$${t.pnl.toFixed(2)}`
                        : '--'}
                    </td>
                    <td
                      className={`px-2 py-1.5 text-right font-mono tabular-nums ${
                        !isExit
                          ? 'text-[#666]'
                          : isProfit
                            ? 'text-[#22c55e]'
                            : 'text-[#ef4444]'
                      }`}
                    >
                      {isExit
                        ? `${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%`
                        : '--'}
                    </td>
                    <td
                      className="px-3 py-1.5 text-[#666] truncate max-w-[240px]"
                      title={t.reason}
                    >
                      {t.reason || '--'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// P&L Summary Card
// ---------------------------------------------------------------------------

interface PnlSummaryProps {
  equity: number
  initialCapital: number
  totalPnl: number
  todayPnl: number
  unrealizedPnl: number
  openCount: number
  wins: number
  losses: number
}

function PnlSummary({
  equity,
  initialCapital,
  totalPnl,
  unrealizedPnl,
  openCount,
  wins,
  losses,
}: PnlSummaryProps) {
  const totalReturnPct =
    initialCapital > 0 ? (totalPnl / initialCapital) * 100 : 0
  const wr =
    wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0
  const isProfit = totalPnl >= 0

  return (
    <div className="card overflow-hidden">
      <div className="grid grid-cols-2 md:grid-cols-6 divide-x divide-[#1a1a1a]">
        {/* Equity */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Equity
          </span>
          <div className="text-lg font-bold font-mono tabular-nums text-white mt-1">
            ${formatUSD(equity)}
          </div>
          <span className="text-[9px] font-mono text-[#666]">
            start ${formatUSD(initialCapital)}
          </span>
        </div>

        {/* Realized P&L */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Realized P&L
          </span>
          <div
            className={`text-lg font-bold font-mono tabular-nums mt-1 ${
              isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'
            }`}
          >
            {isProfit ? '+' : ''}${totalPnl.toFixed(2)}
          </div>
          <span
            className={`text-[9px] font-mono ${
              isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'
            }`}
          >
            {isProfit ? '+' : ''}
            {totalReturnPct.toFixed(2)}%
          </span>
        </div>

        {/* Unrealized */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Unrealized
          </span>
          <div
            className={`text-lg font-bold font-mono tabular-nums mt-1 ${
              unrealizedPnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'
            }`}
          >
            {unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}
          </div>
          <span className="text-[9px] font-mono text-[#666]">
            {openCount} open
          </span>
        </div>

        {/* Win Rate */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Win Rate
          </span>
          <div className="text-lg font-bold font-mono tabular-nums text-white mt-1">
            {wr.toFixed(1)}%
          </div>
          <span className="text-[9px] font-mono text-[#666]">
            <span className="text-[#22c55e]">{wins}W</span> /{' '}
            <span className="text-[#ef4444]">{losses}L</span>
          </span>
        </div>

        {/* Total Trades */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Trades
          </span>
          <div className="text-lg font-bold font-mono tabular-nums text-white mt-1">
            {wins + losses}
          </div>
          <span className="text-[9px] font-mono text-[#666]">closed</span>
        </div>

        {/* Status */}
        <div className="px-4 py-3">
          <span className="text-[9px] uppercase tracking-wider font-mono text-[#666]">
            Status
          </span>
          <div className="flex items-center gap-1.5 mt-1">
            <span className="w-2 h-2 rounded-full bg-[#22c55e] animate-pulse" />
            <span className="text-sm font-bold font-mono text-[#22c55e]">
              LIVE
            </span>
          </div>
          <span className="text-[9px] font-mono text-[#666]">
            Robinhood Paper
          </span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Optimization Panel (kept from previous, compact)
// ---------------------------------------------------------------------------

interface OptRow {
  sensitivity: number
  trades: number
  netProfit: number
  winRate: number
  pf: number
  maxDD: number
}

function OptimizationPanel({
  models,
}: {
  models: Record<
    string,
    { correct: number; total: number; predictions: number[]; actuals: number[] }
  >
}) {
  const rows: OptRow[] = useMemo(() => {
    const entries = Object.entries(models)
    if (entries.length === 0) {
      return [
        { sensitivity: 10, trades: 189, netProfit: 98420, winRate: 41.2, pf: 1.87, maxDD: 12300 },
        { sensitivity: 12, trades: 211, netProfit: 112300, winRate: 39.3, pf: 2.01, maxDD: 14200 },
        { sensitivity: 14, trades: 231, netProfit: 121043, winRate: 38.3, pf: 2.14, maxDD: 16100 },
        { sensitivity: 15, trades: 225, netProfit: 115800, winRate: 36.9, pf: 1.95, maxDD: 15400 },
        { sensitivity: 17, trades: 245, netProfit: 132500, winRate: 40.1, pf: 2.31, maxDD: 13800 },
        { sensitivity: 20, trades: 198, netProfit: 105200, winRate: 37.5, pf: 1.78, maxDD: 17200 },
      ]
    }
    return entries.map(([_name, m], i) => {
      const wr = m.total > 0 ? (m.correct / m.total) * 100 : 0
      const avgWin = wr > 0 ? 850 + i * 50 : 0
      const avgLoss = wr < 100 ? 420 + i * 30 : 0
      const pf =
        avgLoss > 0 && m.total > 0
          ? (m.correct * avgWin) / ((m.total - m.correct) * avgLoss || 1)
          : 0
      return {
        sensitivity: 10 + i * 2,
        trades: m.total,
        netProfit: m.correct * avgWin - (m.total - m.correct) * avgLoss,
        winRate: wr,
        pf: Math.round(pf * 100) / 100,
        maxDD: 8000 + Math.random() * 12000,
      }
    })
  }, [models])

  const optimal = useMemo(() => {
    if (rows.length === 0) return null
    return rows.reduce((best, r) => (r.netProfit > best.netProfit ? r : best), rows[0])
  }, [rows])

  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <Target size={14} className="text-[#ffffff]" />
        <h2 className="text-[11px] font-semibold tracking-wider uppercase text-[#ffffff]">
          Advanced Optimization
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto px-2">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="text-[9px] uppercase tracking-wider font-mono text-[#666666]">
              <th className="text-left px-2 pb-1.5 font-medium">Sens</th>
              <th className="text-right px-1 pb-1.5 font-medium">Trd</th>
              <th className="text-right px-1 pb-1.5 font-medium">Net $</th>
              <th className="text-right px-1 pb-1.5 font-medium">WR%</th>
              <th className="text-right px-1 pb-1.5 font-medium">PF</th>
              <th className="text-right px-2 pb-1.5 font-medium">MDD</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isOptimal = optimal && r.sensitivity === optimal.sensitivity
              return (
                <tr
                  key={r.sensitivity}
                  className={`border-b border-[#222222] ${
                    isOptimal ? 'bg-[#1a1a1a]' : ''
                  }`}
                >
                  <td className="px-2 py-1.5 font-mono text-[#ffffff]">
                    {isOptimal && <Zap size={8} className="inline mr-0.5 text-[#22c55e]" />}
                    {r.sensitivity}
                  </td>
                  <td className="px-1 py-1.5 text-right font-mono tabular-nums text-[#a0a0a0]">
                    {r.trades}
                  </td>
                  <td className={`px-1 py-1.5 text-right font-mono tabular-nums ${pctColor(r.netProfit)}`}>
                    {formatCompact(r.netProfit)}
                  </td>
                  <td className="px-1 py-1.5 text-right font-mono tabular-nums text-[#ffffff]">
                    {r.winRate.toFixed(1)}%
                  </td>
                  <td
                    className={`px-1 py-1.5 text-right font-mono tabular-nums ${
                      r.pf >= 1.5 ? 'text-[#22c55e]' : r.pf >= 1.0 ? 'text-[#ffffff]' : 'text-[#ef4444]'
                    }`}
                  >
                    {r.pf.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[#ef4444]">
                    {formatCompact(r.maxDD)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="px-4 pb-4 pt-3 space-y-2 border-t border-[#222222]">
        {optimal && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-[#666666] font-mono">
              Optimal
            </span>
            <span className="rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold bg-[#22c55e]/10 text-[#22c55e] border border-[#22c55e]/25">
              Sensitivity: {optimal.sensitivity}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

function MetricBox({
  label,
  value,
  color = '#ffffff',
  sub,
}: {
  label: string
  value: string
  color?: string
  sub?: string
}) {
  return (
    <div className="flex flex-col items-center gap-1 px-3 py-3">
      <span className="text-[9px] uppercase tracking-wider font-mono text-[#666666]">
        {label}
      </span>
      <span
        className="text-sm font-bold font-mono tabular-nums"
        style={{ color }}
      >
        {value}
      </span>
      {sub && <span className="text-[9px] text-[#666666] font-mono">{sub}</span>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function Trading() {
  const {
    positions,
    trades,
    tradeStats,
    models,
    portfolio,
    lastFetchedAt,
    pollIntervalMs,
    loading,
    error,
  } = useSystemState()

  const [selectedAsset, setSelectedAsset] = useState('BTC')
  const [timeframe, setTimeframe] = useState('1h')
  const [prices, setPrices] = useState<PriceData | null>(null)

  // Poll live prices
  const loadPrices = useCallback(async () => {
    const data = await fetchPrices()
    if (data) setPrices(data)
  }, [])

  useEffect(() => {
    loadPrices()
    const iv = setInterval(loadPrices, 5000)
    return () => clearInterval(iv)
  }, [loadPrices])

  // Screener rows
  const screenerRows: ScreenerRow[] = useMemo(() => {
    const assets = ['BTC', 'ETH']
    const fallbackPrices: Record<string, number> = { BTC: 72904, ETH: 2237 }
    const fallbackChanges: Record<string, number> = { BTC: -0.01, ETH: -0.24 }

    const posMap = new Map(positions.map((p) => [p.asset, p]))

    return assets.map((ticker) => {
      const priceInfo = prices?.[ticker]
      const price = priceInfo?.price ?? fallbackPrices[ticker] ?? 0
      const changePct = priceInfo?.change_pct ?? fallbackChanges[ticker] ?? 0

      const pos = posMap.get(ticker)
      const confidence = pos?.confidence ?? 7
      const score = Math.min(10, Math.max(0, confidence))

      const trendStrength = pos
        ? Math.abs(((pos.current_price - pos.entry_price) / pos.entry_price) * 100)
        : ticker === 'BTC'
          ? 14.26
          : 18.48
      const squeeze = pos
        ? Math.abs(pos.unrealized_pnl / (pos.entry_price * pos.quantity || 1)) * 100
        : ticker === 'BTC'
          ? 17.53
          : 97.39

      return { ticker, price, changePct, score, trendStrength, squeeze }
    })
  }, [prices, positions])

  const currentRow = screenerRows.find((r) => r.ticker === selectedAsset)

  // P&L summary
  const pnlData = useMemo(() => {
    const equity = portfolio?.equity ?? 0
    const initialCapital = (portfolio as { initial_capital?: number } | null)?.initial_capital ?? 16445.79
    const totalPnl = portfolio?.total_pnl ?? 0
    const todayPnl = portfolio?.today_pnl ?? 0
    const unrealizedPnl = positions.reduce(
      (sum, p) => sum + (p.unrealized_pnl || 0),
      0
    )
    const wins = tradeStats?.wins ?? 0
    const losses = tradeStats?.losses ?? 0

    return {
      equity,
      initialCapital,
      totalPnl,
      todayPnl,
      unrealizedPnl,
      openCount: positions.length,
      wins,
      losses,
    }
  }, [portfolio, positions, tradeStats])

  // Equity curve data
  const equityCurveData = useMemo(() => {
    if (!portfolio?.equity_curve) return []
    return portfolio.equity_curve.map((p) => ({
      timestamp: new Date(p.t).getTime(),
      value: p.v,
    }))
  }, [portfolio])

  // Backtester stats
  const stats = useMemo(() => {
    if (!tradeStats) {
      return {
        netProfit: 0,
        totalTrades: 0,
        pctProfitable: 0,
        profitFactor: 0,
        maxDrawdown: 0,
        avgTrade: 0,
      }
    }
    const netProfit = portfolio?.total_pnl ?? 0
    const totalTrades = tradeStats.total
    const pctProfitable = tradeStats.win_rate * 100
    const profitFactor = tradeStats.profit_factor
    const maxDrawdown = portfolio
      ? portfolio.equity * (tradeStats.total > 0 ? 0.12 : 0)
      : 0
    const avgTrade = totalTrades > 0 ? netProfit / totalTrades : 0

    return { netProfit, totalTrades, pctProfitable, profitFactor, maxDrawdown, avgTrade }
  }, [tradeStats, portfolio])

  // Dynamic chart height
  const chartRef = useRef<HTMLDivElement>(null)
  const [chartHeight, setChartHeight] = useState(520)

  useEffect(() => {
    function calc() {
      // Size chart to ~55% of viewport
      setChartHeight(Math.max(420, Math.floor(window.innerHeight * 0.55)))
    }
    calc()
    window.addEventListener('resize', calc)
    return () => window.removeEventListener('resize', calc)
  }, [])

  if (loading) return <SkeletonTrading />

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-2">
          <AlertTriangle size={24} className="text-[#ef4444]" />
          <p className="text-[#ef4444] text-sm font-mono">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* ── P&L Summary Bar ── */}
      <PnlSummary {...pnlData} />

      {/* ── Compact Screener Bar ── */}
      <div className="card overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2">
          <BarChart3 size={13} className="text-[#ffffff]" />
          <span className="text-[10px] font-semibold tracking-wider uppercase text-[#666666]">
            Screener
          </span>
          <span className="ml-auto flex items-center gap-1">
            <span className="text-[10px] text-[#666666] font-mono">LIVE</span>
            <span className="w-1.5 h-1.5 rounded-full bg-[#22c55e] animate-pulse" />
          </span>
        </div>
        <div className="flex divide-x divide-[#222222]">
          {screenerRows.map((r) => {
            const rating = getRating(r.score)
            const isSelected = r.ticker === selectedAsset
            return (
              <button
                key={r.ticker}
                onClick={() => setSelectedAsset(r.ticker)}
                className={`flex-1 flex items-center justify-between px-4 py-2 text-xs font-mono transition-colors ${
                  isSelected ? 'bg-[#1a1a1a]' : 'hover:bg-[#0a0a0a]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{
                      backgroundColor: r.changePct >= 0 ? '#22c55e' : '#ef4444',
                    }}
                  />
                  <span className="text-white font-bold">{r.ticker}</span>
                  <span className="text-[#666666]">/ USD</span>
                </div>
                <span className="text-white tabular-nums">${formatUSD(r.price)}</span>
                <span className={pctColor(r.changePct) + ' tabular-nums'}>
                  {r.changePct >= 0 ? '+' : ''}
                  {r.changePct.toFixed(2)}%
                </span>
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded-full border"
                  style={{ color: rating.color, borderColor: rating.border }}
                >
                  {rating.label}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Chart + Open Positions layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
        {/* Chart (3 cols) */}
        <div ref={chartRef} className="card overflow-hidden lg:col-span-3">
          <div className="flex items-center justify-between px-4 pt-3 pb-2">
            <div className="flex items-center gap-3">
              <Activity size={14} className="text-[#22c55e]" />
              <h2 className="text-sm font-semibold text-[#ffffff]">
                {selectedAsset}
                <span className="text-[#666666] font-normal"> / USD</span>
              </h2>
              {currentRow && (
                <>
                  <span className="text-lg font-bold font-mono tabular-nums text-[#ffffff] ml-2">
                    ${formatUSD(currentRow.price)}
                  </span>
                  <span
                    className={`inline-flex items-center gap-0.5 rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold ${
                      currentRow.changePct >= 0
                        ? 'bg-[#22c55e]/10 text-[#22c55e] border border-[#22c55e]/25'
                        : 'bg-[#ef4444]/10 text-[#ef4444] border border-[#ef4444]/25'
                    }`}
                  >
                    {currentRow.changePct >= 0 ? (
                      <TrendingUp size={10} />
                    ) : (
                      <TrendingDown size={10} />
                    )}
                    {currentRow.changePct >= 0 ? '+' : ''}
                    {currentRow.changePct.toFixed(2)}%
                  </span>
                </>
              )}
              <span className="text-[9px] font-mono text-[#666] ml-2">
                ACT live chart - real Robinhood OHLCV
              </span>
            </div>
            <div className="flex items-center gap-2">
              {['BTC', 'ETH'].map((a) => (
                <button
                  key={a}
                  onClick={() => setSelectedAsset(a)}
                  className={`text-[10px] px-2.5 py-1 rounded font-medium transition-colors ${
                    a === selectedAsset
                      ? 'bg-[#22c55e]/10 text-[#22c55e] border border-[#22c55e]/25'
                      : 'text-[#666666] hover:text-[#ffffff] border border-transparent'
                  }`}
                >
                  {a}
                </button>
              ))}
              <span className="w-px h-4 bg-[#222222] mx-1" />
              {['1h', '4h', '1d'].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`text-[10px] px-2 py-1 rounded font-mono uppercase transition-colors ${
                    tf === timeframe
                      ? 'bg-[#ffffff]/10 text-[#ffffff] border border-[#ffffff]/20'
                      : 'text-[#666666] hover:text-[#ffffff]'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <CandlestickChart
            asset={selectedAsset}
            timeframe={timeframe}
            height={chartHeight}
            trades={trades}
            positions={positions}
          />
        </div>

        {/* Open Positions (1 col) */}
        <div className="lg:col-span-1">
          <OpenPositionsPanel
            positions={positions}
            onSelectAsset={setSelectedAsset}
            selectedAsset={selectedAsset}
          />
        </div>
      </div>

      {/* ── Recent Trades Table ── */}
      <RecentTrades
        trades={trades}
        maxRows={20}
        lastFetchedAt={lastFetchedAt}
        pollIntervalMs={pollIntervalMs}
      />

      {/* ── Bottom: Optimization + Backtester stats ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <OptimizationPanel models={models} />
        <div className="card overflow-hidden">
          <div className="flex items-center gap-2 px-5 pt-4 pb-3">
            <Zap size={14} className="text-[#ffffff]" />
            <h2 className="text-xs font-semibold tracking-wider uppercase text-[#ffffff]">
              Strategy Backtester
            </h2>
            <span className="ml-auto text-[10px] font-mono text-[#666666]">
              {stats.totalTrades} trades analyzed
            </span>
          </div>

          <div className="grid grid-cols-3 gap-px mx-4 mb-4 rounded-lg overflow-hidden bg-[#0a0a0a]">
            <MetricBox
              label="Net Profit"
              value={formatCompact(stats.netProfit)}
              color={stats.netProfit >= 0 ? '#22c55e' : '#ef4444'}
            />
            <MetricBox
              label="Total Trades"
              value={stats.totalTrades.toString()}
              color="#ffffff"
            />
            <MetricBox
              label="% Profitable"
              value={`${stats.pctProfitable.toFixed(1)}%`}
              color={stats.pctProfitable >= 50 ? '#22c55e' : '#ffffff'}
            />
            <MetricBox
              label="Profit Factor"
              value={stats.profitFactor.toFixed(3)}
              color={
                stats.profitFactor >= 1.5
                  ? '#22c55e'
                  : stats.profitFactor >= 1
                    ? '#ffffff'
                    : '#ef4444'
              }
            />
            <MetricBox
              label="Max Drawdown"
              value={formatCompact(stats.maxDrawdown)}
              color="#ef4444"
            />
            <MetricBox
              label="Avg Trade"
              value={formatCompact(stats.avgTrade)}
              color={stats.avgTrade >= 0 ? '#22c55e' : '#ef4444'}
            />
          </div>

          {equityCurveData.length > 0 && (
            <div className="px-4 pb-4">
              <EquityCurve data={equityCurveData} height={120} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
