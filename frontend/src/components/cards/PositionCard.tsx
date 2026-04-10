import { ArrowUpRight, ArrowDownRight } from 'lucide-react'

interface PositionCardProps {
  symbol: string
  direction: string
  entryPrice: number
  currentPrice: number
  quantity: number
  unrealizedPnl: number
  unrealizedPnlPct?: number
  leverage?: number
}

function formatUsd(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(2)}`
}

function formatPrice(n: number): string {
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export default function PositionCard({
  symbol,
  direction,
  entryPrice,
  currentPrice,
  quantity,
  unrealizedPnl,
  unrealizedPnlPct,
  leverage,
}: PositionCardProps) {
  const isLong = direction === 'long'
  const isProfit = unrealizedPnl >= 0
  const Arrow = isProfit ? ArrowUpRight : ArrowDownRight
  const pnlColor = isProfit ? 'text-accent-green' : 'text-accent-red'
  const dirColor = isLong ? 'text-accent-green' : 'text-accent-red'
  const dirBg = isLong ? 'bg-accent-green/10' : 'bg-accent-red/10'

  return (
    <div className="glass-card p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-text-primary">{symbol}</span>
          <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${dirColor} ${dirBg}`}>
            {direction}
          </span>
          {leverage != null && leverage > 1 && (
            <span className="text-[10px] font-mono text-text-muted">{leverage}x</span>
          )}
        </div>
        <div className={`flex items-center gap-1 ${pnlColor}`}>
          <Arrow size={14} />
          <span className="text-sm font-semibold tabular-nums">
            {formatUsd(unrealizedPnl)}
          </span>
        </div>
      </div>

      {/* Details grid */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <span className="text-text-muted block">Entry</span>
          <span className="text-text-primary tabular-nums">${formatPrice(entryPrice)}</span>
        </div>
        <div>
          <span className="text-text-muted block">Current</span>
          <span className="text-text-primary tabular-nums">${formatPrice(currentPrice)}</span>
        </div>
        <div>
          <span className="text-text-muted block">Size</span>
          <span className="text-text-primary tabular-nums">{quantity}</span>
        </div>
      </div>

      {/* PnL bar */}
      {unrealizedPnlPct !== undefined && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
            <div
              className={`h-full rounded-full ${isProfit ? 'bg-accent-green' : 'bg-accent-red'}`}
              style={{ width: `${Math.min(Math.abs(unrealizedPnlPct) * 10, 100)}%` }}
            />
          </div>
          <span className={`text-[10px] tabular-nums font-medium ${pnlColor}`}>
            {unrealizedPnlPct >= 0 ? '+' : ''}{unrealizedPnlPct.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  )
}
