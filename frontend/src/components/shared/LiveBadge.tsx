interface LiveBadgeProps {
  // Wall-clock ms of the last successful poll. Surfaced so operators
  // can confirm paper-trade rows are auto-refreshing without manual reload.
  lastFetchedAt?: number | null
  // Expected poll period; >2x without a fetch flips the indicator to STALE.
  pollIntervalMs?: number
}

function formatRefreshTime(ts: number | null | undefined): string {
  if (!ts) return '--'
  return new Date(ts).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export default function LiveBadge({ lastFetchedAt, pollIntervalMs }: LiveBadgeProps) {
  const stale =
    !!lastFetchedAt &&
    !!pollIntervalMs &&
    Date.now() - lastFetchedAt > pollIntervalMs * 2

  const dotClass = stale ? 'bg-[#666]' : 'bg-[#ef4444] animate-pulse'
  const labelClass = stale ? 'text-[#666]' : 'text-[#ef4444]'

  return (
    <span className="flex items-center gap-1">
      <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} aria-hidden="true" />
      <span className={`text-[9px] font-mono uppercase tracking-wider ${labelClass}`}>
        {stale ? 'STALE' : 'LIVE'}
      </span>
      <span className="text-[9px] font-mono text-[#666]">
        {formatRefreshTime(lastFetchedAt)}
      </span>
    </span>
  )
}
