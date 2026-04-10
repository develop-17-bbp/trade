import { ArrowUpRight, ArrowDownRight } from 'lucide-react'

interface TradeRowProps {
  timestamp: string
  asset: string
  direction: string
  action: string
  entryPrice: number
  exitPrice?: number | null
  pnlPct?: number | null
  pnlUsd?: number | null
  reason: string
}

function formatTime(ts: string): string {
  const d = new Date(ts)
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function formatPrice(n: number): string {
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export default function TradeRow({
  timestamp,
  asset,
  direction,
  action,
  entryPrice,
  exitPrice,
  pnlPct,
  pnlUsd,
  reason,
}: TradeRowProps) {
  const isBuy = direction === 'buy'
  const hasPnl = pnlUsd != null
  const isProfit = (pnlUsd ?? 0) >= 0
  const Arrow = isProfit ? ArrowUpRight : ArrowDownRight
  const pnlColor = hasPnl
    ? isProfit ? 'text-accent-green' : 'text-accent-red'
    : 'text-text-muted'
  const dirColor = isBuy ? 'text-accent-green' : 'text-accent-red'
  const dirBg = isBuy ? 'bg-accent-green/10' : 'bg-accent-red/10'

  return (
    <div className="flex items-center gap-3 py-2.5 px-3 rounded-lg hover:bg-white/[0.02] transition-colors text-xs">
      {/* Time */}
      <span className="text-text-muted tabular-nums w-28 flex-shrink-0">
        {formatTime(timestamp)}
      </span>

      {/* Asset + Direction */}
      <span className="font-medium text-text-primary w-20 flex-shrink-0">{asset}</span>
      <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${dirColor} ${dirBg} w-12 text-center flex-shrink-0`}>
        {action}
      </span>

      {/* Prices */}
      <span className="text-text-primary tabular-nums w-24 flex-shrink-0 text-right">
        ${formatPrice(entryPrice)}
      </span>
      <span className="text-text-muted tabular-nums w-24 flex-shrink-0 text-right">
        {exitPrice != null ? `$${formatPrice(exitPrice)}` : '--'}
      </span>

      {/* PnL */}
      <span className={`tabular-nums w-20 flex-shrink-0 text-right flex items-center justify-end gap-1 ${pnlColor}`}>
        {hasPnl ? (
          <>
            <Arrow size={12} />
            ${Math.abs(pnlUsd!).toFixed(2)}
          </>
        ) : (
          '--'
        )}
      </span>
      <span className={`tabular-nums w-16 flex-shrink-0 text-right ${pnlColor}`}>
        {pnlPct != null ? `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%` : '--'}
      </span>

      {/* Reason */}
      <span className="text-text-muted truncate flex-1">{reason}</span>
    </div>
  )
}
