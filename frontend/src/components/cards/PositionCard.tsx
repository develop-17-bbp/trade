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
}: PositionCardProps) {
  const isLong = direction === 'long'
  const isProfit = unrealizedPnl >= 0
  const Arrow = isProfit ? ArrowUpRight : ArrowDownRight

  return (
    <div className="border border-[#222] rounded-lg p-3 space-y-2 hover:border-[#333] transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-white">{symbol}</span>
          <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${
            isLong ? 'text-[#22c55e] bg-[#22c55e]/10' : 'text-[#ef4444] bg-[#ef4444]/10'
          }`}>
            {direction}
          </span>
        </div>
        <div className={`flex items-center gap-1 ${isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
          <Arrow size={14} />
          <span className="text-sm font-semibold tabular-nums font-mono">
            ${Math.abs(unrealizedPnl).toFixed(2)}
          </span>
        </div>
      </div>

      {/* Details */}
      <div className="grid grid-cols-3 gap-2 text-xs font-mono">
        <div>
          <span className="text-[#666] block text-[10px]">Entry</span>
          <span className="text-[#a0a0a0] tabular-nums">${formatPrice(entryPrice)}</span>
        </div>
        <div>
          <span className="text-[#666] block text-[10px]">Current</span>
          <span className="text-white tabular-nums">${formatPrice(currentPrice)}</span>
        </div>
        <div>
          <span className="text-[#666] block text-[10px]">Qty</span>
          <span className="text-[#a0a0a0] tabular-nums">{quantity}</span>
        </div>
      </div>

      {/* PnL bar */}
      {unrealizedPnlPct !== undefined && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1 rounded-full bg-[#1a1a1a] overflow-hidden">
            <div
              className={`h-full rounded-full ${isProfit ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`}
              style={{ width: `${Math.min(Math.abs(unrealizedPnlPct) * 10, 100)}%` }}
            />
          </div>
          <span className={`text-[10px] tabular-nums font-bold font-mono ${isProfit ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
            {unrealizedPnlPct >= 0 ? '+' : ''}{unrealizedPnlPct.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  )
}
