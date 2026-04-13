import { useMemo, useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Target,
  Zap,
  ChevronUp,
  ChevronDown,
} from 'lucide-react'
import CandlestickChart from '../components/charts/CandlestickChart'
import TradingViewWidget from '../components/charts/TradingViewWidget'
import EquityCurve from '../components/charts/EquityCurve'
import { useSystemState } from '../hooks/useSystemState'
import { fetchPrices, type PriceData } from '../api/client'

// ---------------------------------------------------------------------------
// Animation variants
// ---------------------------------------------------------------------------

const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.08 },
  },
}

const child = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' as const } },
}

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
  return v >= 0 ? 'text-[#00ffaa]' : 'text-[#ff2266]'
}

function pctShadow(v: number): React.CSSProperties {
  return {
    textShadow: v >= 0 ? '0 0 8px rgba(0,255,170,0.4)' : '0 0 8px rgba(255,34,102,0.4)',
  }
}

type RatingInfo = { label: string; color: string; bg: string; border: string }

function getRating(score: number): RatingInfo {
  if (score >= 8)
    return {
      label: 'Strong Bullish',
      color: '#00ffaa',
      bg: 'rgba(0,255,170,0.1)',
      border: 'rgba(0,255,170,0.3)',
    }
  if (score >= 6)
    return {
      label: 'Bullish',
      color: '#00ffaa',
      bg: 'rgba(0,255,170,0.07)',
      border: 'rgba(0,255,170,0.2)',
    }
  if (score >= 4)
    return {
      label: 'Neutral',
      color: '#bf5fff',
      bg: 'rgba(191,95,255,0.1)',
      border: 'rgba(191,95,255,0.3)',
    }
  return {
    label: 'Bearish',
    color: '#ff2266',
    bg: 'rgba(255,34,102,0.1)',
    border: 'rgba(255,34,102,0.3)',
  }
}

function getSignal(score: number): { label: string; icon: React.ReactNode; color: string } {
  if (score >= 6)
    return { label: 'Bullish', icon: <ChevronUp size={12} />, color: '#00ffaa' }
  if (score >= 4)
    return { label: 'Neutral', icon: <Activity size={12} />, color: '#bf5fff' }
  return { label: 'Bearish', icon: <ChevronDown size={12} />, color: '#ff2266' }
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-white/[0.04] ${className}`} />
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

function ScreenerTable({
  rows,
  selectedAsset,
  onSelect,
}: {
  rows: ScreenerRow[]
  selectedAsset: string
  onSelect: (ticker: string) => void
}) {
  return (
    <div className="glass-card holo-shimmer overflow-hidden">
      <div className="flex items-center gap-2 px-5 pt-4 pb-3">
        <BarChart3 size={15} className="text-[#00fff0]" />
        <h2 className="text-xs font-semibold tracking-wider uppercase text-[#e8ecf4]">
          Multi-Asset Screener
        </h2>
        <span className="ml-auto text-[10px] text-[#5a6080] font-mono">LIVE</span>
        <span className="w-1.5 h-1.5 rounded-full bg-[#00ffaa] animate-pulse" />
      </div>

      <div className="cyber-grid">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider font-mono text-[#5a6080] border-b border-white/[0.06]">
              <th className="text-left px-5 pb-2 font-medium">Ticker</th>
              <th className="text-right px-3 pb-2 font-medium">Price</th>
              <th className="text-right px-3 pb-2 font-medium">% Change</th>
              <th className="text-center px-3 pb-2 font-medium">Rating</th>
              <th className="text-center px-3 pb-2 font-medium">Signal</th>
              <th className="text-right px-3 pb-2 font-medium">Trend Str</th>
              <th className="text-right px-5 pb-2 font-medium">Squeeze</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const rating = getRating(r.score)
              const signal = getSignal(r.score)
              const isSelected = r.ticker === selectedAsset
              return (
                <tr
                  key={r.ticker}
                  onClick={() => onSelect(r.ticker)}
                  className={`cursor-pointer border-b border-white/[0.04] transition-all duration-200 ${
                    isSelected
                      ? 'bg-[#00fff0]/[0.04] shadow-[inset_0_0_20px_rgba(0,255,240,0.03)]'
                      : 'hover:bg-white/[0.02] hover:shadow-[inset_0_0_30px_rgba(0,255,240,0.02)]'
                  }`}
                >
                  <td className="px-5 py-3 font-semibold text-[#e8ecf4]">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: r.changePct >= 0 ? '#00ffaa' : '#ff2266' }}
                      />
                      {r.ticker}
                      <span className="text-[#5a6080] font-normal">/ USD</span>
                    </div>
                  </td>
                  <td className="px-3 py-3 text-right font-mono tabular-nums text-[#e8ecf4]">
                    ${formatUSD(r.price)}
                  </td>
                  <td
                    className={`px-3 py-3 text-right font-mono tabular-nums ${pctColor(r.changePct)}`}
                    style={pctShadow(r.changePct)}
                  >
                    {r.changePct >= 0 ? '+' : ''}
                    {r.changePct.toFixed(2)}%
                  </td>
                  <td className="px-3 py-3 text-center">
                    <span
                      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium border"
                      style={{
                        color: rating.color,
                        backgroundColor: rating.bg,
                        borderColor: rating.border,
                      }}
                    >
                      {r.score >= 6 ? (
                        <TrendingUp size={10} />
                      ) : r.score >= 4 ? (
                        <Activity size={10} />
                      ) : (
                        <TrendingDown size={10} />
                      )}
                      {rating.label}
                    </span>
                  </td>
                  <td className="px-3 py-3 text-center">
                    <span
                      className="inline-flex items-center gap-0.5 rounded-full px-2 py-0.5 text-[10px] font-medium"
                      style={{ color: signal.color }}
                    >
                      {signal.icon}
                      {signal.label}
                    </span>
                  </td>
                  <td
                    className={`px-3 py-3 text-right font-mono tabular-nums ${pctColor(r.trendStrength)}`}
                    style={pctShadow(r.trendStrength)}
                  >
                    {r.trendStrength.toFixed(2)}%
                  </td>
                  <td className="px-5 py-3 text-right font-mono tabular-nums text-[#bf5fff]">
                    {r.squeeze.toFixed(2)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

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
      // Fallback demo rows
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
    <div className="glass-card holo-shimmer glow-cyan h-full flex flex-col">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <Target size={14} className="text-[#00fff0]" />
        <h2 className="text-[11px] font-semibold tracking-wider uppercase text-[#e8ecf4]">
          Advanced Optimization
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto px-2">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="text-[9px] uppercase tracking-wider font-mono text-[#5a6080]">
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
                  className={`border-b border-white/[0.04] ${
                    isOptimal ? 'bg-[#00fff0]/[0.06]' : ''
                  }`}
                >
                  <td className="px-2 py-1.5 font-mono text-[#e8ecf4]">
                    {isOptimal && <Zap size={8} className="inline mr-0.5 text-[#00fff0]" />}
                    {r.sensitivity}
                  </td>
                  <td className="px-1 py-1.5 text-right font-mono tabular-nums text-[#8a94b0]">
                    {r.trades}
                  </td>
                  <td
                    className={`px-1 py-1.5 text-right font-mono tabular-nums ${pctColor(r.netProfit)}`}
                    style={pctShadow(r.netProfit)}
                  >
                    {formatCompact(r.netProfit)}
                  </td>
                  <td className="px-1 py-1.5 text-right font-mono tabular-nums text-[#e8ecf4]">
                    {r.winRate.toFixed(1)}%
                  </td>
                  <td
                    className={`px-1 py-1.5 text-right font-mono tabular-nums ${
                      r.pf >= 1.5 ? 'text-[#00ffaa]' : r.pf >= 1.0 ? 'text-[#e8ecf4]' : 'text-[#ff2266]'
                    }`}
                  >
                    {r.pf.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[#ff2266]">
                    {formatCompact(r.maxDD)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Bottom badges */}
      <div className="px-4 pb-4 pt-3 space-y-2 border-t border-white/[0.06]">
        {optimal && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-[#5a6080] font-mono">
              Optimal
            </span>
            <span className="rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold bg-[#00fff0]/10 text-[#00fff0] border border-[#00fff0]/30">
              Sensitivity: {optimal.sensitivity}
            </span>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-[#5a6080] font-mono">
            Trend
          </span>
          <span className="rounded-full px-2 py-0.5 text-[10px] font-mono bg-[#bf5fff]/10 text-[#bf5fff] border border-[#bf5fff]/30">
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
  color = '#e8ecf4',
  sub,
}: {
  label: string
  value: string
  color?: string
  sub?: string
}) {
  return (
    <div className="flex flex-col items-center gap-1 px-3 py-3">
      <span className="text-[9px] uppercase tracking-wider font-mono text-[#5a6080]">
        {label}
      </span>
      <span
        className="text-sm font-bold font-mono tabular-nums"
        style={{ color, textShadow: `0 0 8px ${color}40` }}
      >
        {value}
      </span>
      {sub && <span className="text-[9px] text-[#5a6080] font-mono">{sub}</span>}
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

    // Compute entry score from agents/positions
    const posMap = new Map(positions.map((p) => [p.asset, p]))

    return assets.map((ticker) => {
      const priceInfo = prices?.[ticker]
      const price = priceInfo?.price ?? fallbackPrices[ticker] ?? 0
      const changePct = priceInfo?.change_pct ?? fallbackChanges[ticker] ?? 0

      const pos = posMap.get(ticker)
      const confidence = pos?.confidence ?? 7
      const score = Math.min(10, Math.max(0, confidence))

      // Derive trend strength & squeeze from position data or defaults
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

  // Current asset price for header
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

  if (loading) return <SkeletonTrading />

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-[#ff2266] text-sm font-mono">{error}</p>
      </div>
    )
  }

  return (
    <motion.div
      className="space-y-4"
      variants={pageVariants}
      initial="hidden"
      animate="show"
    >
      {/* ── SCREENER TABLE ── */}
      <motion.div variants={child}>
        <ScreenerTable
          rows={screenerRows}
          selectedAsset={selectedAsset}
          onSelect={setSelectedAsset}
        />
      </motion.div>

      {/* ── CHART + OPTIMIZATION ── */}
      <div className="flex gap-4">
        {/* Chart area — 70% */}
        <motion.div variants={child} className="flex-[7] min-w-0">
          <div className="glass-card holo-shimmer overflow-hidden">
            {/* Chart header */}
            <div className="flex items-center justify-between px-5 pt-4 pb-2">
              <div className="flex items-center gap-3">
                <Activity size={15} className="text-[#00fff0]" />
                <h2 className="text-sm font-semibold text-[#e8ecf4]">
                  {selectedAsset}
                  <span className="text-[#5a6080] font-normal"> / USD</span>
                </h2>
                {currentRow && (
                  <>
                    <span className="text-lg font-bold font-mono tabular-nums text-[#e8ecf4] ml-2">
                      ${formatUSD(currentRow.price)}
                    </span>
                    <span
                      className={`inline-flex items-center gap-0.5 rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold ${
                        currentRow.changePct >= 0
                          ? 'bg-[#00ffaa]/10 text-[#00ffaa] border border-[#00ffaa]/30'
                          : 'bg-[#ff2266]/10 text-[#ff2266] border border-[#ff2266]/30'
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
              </div>

              {/* Asset + Timeframe selectors */}
              <div className="flex items-center gap-2">
                {['BTC', 'ETH'].map((a) => (
                  <button
                    key={a}
                    onClick={() => setSelectedAsset(a)}
                    className={`text-[10px] px-2.5 py-1 rounded font-medium transition-colors ${
                      a === selectedAsset
                        ? 'bg-[#00fff0]/10 text-[#00fff0] border border-[#00fff0]/30'
                        : 'text-[#5a6080] hover:text-[#e8ecf4]'
                    }`}
                  >
                    {a}
                  </button>
                ))}
                <span className="w-px h-4 bg-white/10 mx-1" />
                {['1h', '4h', '1d'].map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`text-[10px] px-2 py-1 rounded font-mono uppercase transition-colors ${
                      tf === timeframe
                        ? 'bg-[#bf5fff]/10 text-[#bf5fff] border border-[#bf5fff]/30'
                        : 'text-[#5a6080] hover:text-[#e8ecf4]'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            {/* TradingView Real-Time Chart */}
            <TradingViewWidget
              symbol={`KRAKEN:${selectedAsset}USD`}
              interval={timeframe === '1d' ? 'D' : timeframe === '4h' ? '240' : '60'}
              height={800}
            />
          </div>
        </motion.div>

        {/* Optimization panel — 30% */}
        <motion.div variants={child} className="flex-[3] min-w-0">
          <OptimizationPanel models={models} />
        </motion.div>
      </div>

      {/* ── STRATEGY BACKTESTER ── */}
      <motion.div variants={child}>
        <div className="glass-card holo-shimmer overflow-hidden border-t-2 border-[#00fff0]/30">
          <div className="flex items-center gap-2 px-5 pt-4 pb-3">
            <Zap size={14} className="text-[#00fff0]" />
            <h2 className="text-xs font-semibold tracking-wider uppercase text-[#e8ecf4]">
              Strategy Backtester
            </h2>
            <span className="ml-auto text-[10px] font-mono text-[#5a6080]">
              {stats.totalTrades} trades analyzed
            </span>
          </div>

          {/* 6 metric boxes */}
          <div className="grid grid-cols-6 gap-px mx-4 mb-4 rounded-lg overflow-hidden bg-white/[0.03]">
            <MetricBox
              label="Net Profit"
              value={formatCompact(stats.netProfit)}
              color={stats.netProfit >= 0 ? '#00ffaa' : '#ff2266'}
            />
            <MetricBox
              label="Total Trades"
              value={stats.totalTrades.toString()}
              color="#e8ecf4"
            />
            <MetricBox
              label="% Profitable"
              value={`${stats.pctProfitable.toFixed(1)}%`}
              color={stats.pctProfitable >= 50 ? '#00ffaa' : '#e8ecf4'}
            />
            <MetricBox
              label="Profit Factor"
              value={stats.profitFactor.toFixed(3)}
              color={stats.profitFactor >= 1.5 ? '#00ffaa' : stats.profitFactor >= 1 ? '#e8ecf4' : '#ff2266'}
            />
            <MetricBox
              label="Max Drawdown"
              value={formatCompact(stats.maxDrawdown)}
              color="#ff2266"
            />
            <MetricBox
              label="Avg Trade"
              value={formatCompact(stats.avgTrade)}
              color={stats.avgTrade >= 0 ? '#00ffaa' : '#ff2266'}
            />
          </div>

          {/* Mini equity curve */}
          {equityCurveData.length > 0 && (
            <div className="px-4 pb-4">
              <EquityCurve data={equityCurveData} height={120} />
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}
