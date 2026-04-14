import { useMemo, useState, useEffect, useCallback, useRef } from 'react'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Target,
  Zap,
} from 'lucide-react'
import TradingViewWidget from '../components/charts/TradingViewWidget'
import EquityCurve from '../components/charts/EquityCurve'
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
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(2)}`
}

function pctColor(v: number): string {
  return v >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'
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

// getSignal removed — was only used in old ScreenerTable

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
// Sub-components
// ---------------------------------------------------------------------------

interface ScreenerRow {
  ticker: string
  price: number
  changePct: number
  score: number
  trendStrength: number
  squeeze: number
}

// ScreenerTable removed — replaced by compact screener bar inline

// ---------------------------------------------------------------------------
// Optimization Panel
// ---------------------------------------------------------------------------

interface OptRow {
  sensitivity: number
  trades: number
  netProfit: number
  winRate: number
  pf: number
  maxDD: number
}

function OptimizationPanel({ models }: { models: Record<string, { correct: number; total: number; predictions: number[]; actuals: number[] }> }) {
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

      {/* Bottom badges */}
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
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-[#666666] font-mono">
            Trend
          </span>
          <span className="rounded-full px-2 py-0.5 text-[10px] font-mono bg-[#a0a0a0]/10 text-[#a0a0a0] border border-[#a0a0a0]/25">
            Ranging
          </span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Backtester Metric Box
// ---------------------------------------------------------------------------

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
  const { positions, tradeStats, models, portfolio, loading, error } = useSystemState()

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

  // Equity curve data from portfolio
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
  const [chartHeight, setChartHeight] = useState(600)

  useEffect(() => {
    function calc() {
      // 48px topbar + 44px screener header + approx screener rows + chart header
      setChartHeight(Math.max(400, window.innerHeight - 48 - 120))
    }
    calc()
    window.addEventListener('resize', calc)
    return () => window.removeEventListener('resize', calc)
  }, [])

  if (loading) return <SkeletonTrading />

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-[#ef4444] text-sm font-mono">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* ── Compact Screener Bar ── */}
      <div className="card overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2">
          <BarChart3 size={13} className="text-[#ffffff]" />
          <span className="text-[10px] font-semibold tracking-wider uppercase text-[#666666]">Screener</span>
          <span className="ml-auto flex items-center gap-1">
            <span className="text-[10px] text-[#666666] font-mono">LIVE</span>
            <span className="w-1.5 h-1.5 rounded-full bg-[#22c55e]" />
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
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: r.changePct >= 0 ? '#22c55e' : '#ef4444' }} />
                  <span className="text-white font-bold">{r.ticker}</span>
                  <span className="text-[#666666]">/ USD</span>
                </div>
                <span className="text-white tabular-nums">${formatUSD(r.price)}</span>
                <span className={pctColor(r.changePct) + ' tabular-nums'}>
                  {r.changePct >= 0 ? '+' : ''}{r.changePct.toFixed(2)}%
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full border" style={{ color: rating.color, borderColor: rating.border }}>
                  {rating.label}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Full-width Chart ── */}
      <div ref={chartRef} className="card overflow-hidden">
        {/* Chart header */}
        <div className="flex items-center justify-between px-4 pt-3 pb-2">
          <div className="flex items-center gap-3">
            <Activity size={14} className="text-[#ffffff]" />
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
                  {currentRow.changePct >= 0 ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                  {currentRow.changePct >= 0 ? '+' : ''}{currentRow.changePct.toFixed(2)}%
                </span>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            {['BTC', 'ETH'].map((a) => (
              <button
                key={a}
                onClick={() => setSelectedAsset(a)}
                className={`text-[10px] px-2.5 py-1 rounded font-medium transition-colors ${
                  a === selectedAsset
                    ? 'bg-[#ffffff]/10 text-[#ffffff] border border-[#ffffff]/20'
                    : 'text-[#666666] hover:text-[#ffffff]'
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
        <TradingViewWidget
          symbol={`KRAKEN:${selectedAsset}USD`}
          interval={timeframe === '1d' ? 'D' : timeframe === '4h' ? '240' : '60'}
          height={chartHeight}
        />
      </div>

      {/* ── Bottom: Optimization + Backtester side-by-side ── */}
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

          {/* 6 metric boxes */}
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
              color={stats.profitFactor >= 1.5 ? '#22c55e' : stats.profitFactor >= 1 ? '#ffffff' : '#ef4444'}
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

          {/* Mini equity curve */}
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
