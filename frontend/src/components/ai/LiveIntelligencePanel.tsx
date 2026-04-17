/**
 * LiveIntelligencePanel — Real-time sentiment + Fear&Greed + derivatives
 *
 * Shows what the agents actually saw on the most recent orchestrator cycle:
 *   - Aggregate news sentiment score (from RSS headlines, time-decayed)
 *   - Fear & Greed Index (alternative.me)
 *   - Funding rate + open interest (Bybit/Binance)
 *   - Put/Call ratio (Deribit)
 *   - 5 most recent headlines that fed the sentiment model
 *
 * Polls /api/v1/signals/live_intelligence every 15s.
 */

import { useEffect, useState } from 'react'
import { Newspaper, Gauge, TrendingUp, ExternalLink } from 'lucide-react'
import GlassCard from '../cards/GlassCard'
import { fetchLiveIntelligence, type LiveIntelligenceEntry } from '../../api/client'

type AssetKey = 'BTC' | 'ETH'

function sentimentColor(label: string | undefined | null): string {
  switch ((label || '').toUpperCase()) {
    case 'STRONG_POSITIVE': return '#16a34a'
    case 'POSITIVE':        return '#22c55e'
    case 'NEGATIVE':        return '#f87171'
    case 'STRONG_NEGATIVE': return '#dc2626'
    default:                return '#9ca3af'
  }
}

function fngColor(value: number | null | undefined): string {
  if (value == null) return '#9ca3af'
  if (value <= 25) return '#dc2626'
  if (value <= 45) return '#f87171'
  if (value <= 55) return '#9ca3af'
  if (value <= 75) return '#22c55e'
  return '#16a34a'
}

function fngLabel(value: number | null | undefined): string {
  if (value == null) return 'UNKNOWN'
  if (value <= 25) return 'EXTREME FEAR'
  if (value <= 45) return 'FEAR'
  if (value <= 55) return 'NEUTRAL'
  if (value <= 75) return 'GREED'
  return 'EXTREME GREED'
}

function fmtPct(n: number | null | undefined, digits = 4): string {
  if (n == null || !isFinite(n)) return '—'
  return `${(n * 100).toFixed(digits)}%`
}

function fmtBil(n: number | null | undefined): string {
  if (n == null || !isFinite(n) || n === 0) return '—'
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`
  return `$${n.toFixed(0)}`
}

function sincefmt(iso: string | undefined): string {
  if (!iso) return ''
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}

export default function LiveIntelligencePanel() {
  const [data, setData] = useState<Record<string, LiveIntelligenceEntry> | null>(null)
  const [asset, setAsset] = useState<AssetKey>('BTC')

  useEffect(() => {
    let alive = true
    const tick = async () => {
      const d = await fetchLiveIntelligence()
      if (alive) setData(d)
    }
    tick()
    const id = setInterval(tick, 15000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const entry = data?.[asset]
  const s = entry?.sentiment
  const fg = entry?.fear_greed?.value
  const sentLabel = s?.label ?? 'NO DATA'
  const sentScore = s?.score ?? 0
  const barWidth = Math.max(4, Math.min(100, Math.abs(sentScore) * 100))

  return (
    <GlassCard className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Newspaper size={18} />
          <h3 className="font-semibold">Live Intelligence</h3>
          {entry?.timestamp && (
            <span className="text-xs text-gray-400">· {sincefmt(entry.timestamp)}</span>
          )}
        </div>
        <div className="flex gap-1">
          {(['BTC', 'ETH'] as AssetKey[]).map(a => (
            <button
              key={a}
              onClick={() => setAsset(a)}
              className={`px-2 py-1 text-xs rounded ${asset === a ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            >{a}</button>
          ))}
        </div>
      </div>

      {!entry ? (
        <div className="text-sm text-gray-400">Waiting for first cycle…</div>
      ) : (
        <div className="space-y-3">
          {/* News sentiment */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-gray-400">News Sentiment ({s?.headline_count ?? 0} headlines)</span>
              <span style={{ color: sentimentColor(sentLabel) }}>{sentLabel}</span>
            </div>
            <div className="h-2 bg-gray-800 rounded relative overflow-hidden">
              <div
                className="absolute top-0 bottom-0"
                style={{
                  left: sentScore >= 0 ? '50%' : `${50 - barWidth / 2}%`,
                  width: `${barWidth / 2}%`,
                  backgroundColor: sentimentColor(sentLabel),
                }}
              />
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-600" />
            </div>
            <div className="text-xs text-gray-500 mt-1">
              score {sentScore.toFixed(2)} · sources: {(s?.sources ?? []).join(', ') || '—'}
            </div>
          </div>

          {/* Fear & Greed */}
          <div className="flex items-center gap-3">
            <Gauge size={16} className="text-gray-400" />
            <div className="flex-1">
              <div className="text-xs text-gray-400">Fear & Greed Index</div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold" style={{ color: fngColor(fg) }}>
                  {fg ?? '—'}
                </span>
                <span className="text-xs" style={{ color: fngColor(fg) }}>
                  {fngLabel(fg)}
                </span>
              </div>
            </div>
          </div>

          {/* Derivatives row */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <div className="text-gray-400">Funding</div>
              <div className={entry.funding_rate != null && entry.funding_rate > 0 ? 'text-red-400' : 'text-green-400'}>
                {fmtPct(entry.funding_rate, 4)}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Open Interest</div>
              <div>{fmtBil(entry.open_interest_usd)}</div>
            </div>
            <div>
              <div className="text-gray-400">Put/Call</div>
              <div>{entry.put_call_ratio?.toFixed(2) ?? '—'}</div>
            </div>
          </div>

          {/* Macro */}
          {entry.macro_composite && (
            <div className="flex items-center gap-2 text-xs">
              <TrendingUp size={14} className="text-gray-400" />
              <span className="text-gray-400">Macro:</span>
              <span>{entry.macro_composite}</span>
              {entry.macro_risk != null && (
                <span className="text-gray-500">· risk {entry.macro_risk}%</span>
              )}
            </div>
          )}

          {/* Recent headlines */}
          {s?.recent_headlines && s.recent_headlines.length > 0 && (
            <div>
              <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                <ExternalLink size={12} /> Recent Headlines
              </div>
              <ul className="space-y-1">
                {s.recent_headlines.slice(0, 5).map((h, i) => (
                  <li key={i} className="text-xs text-gray-300 truncate" title={h}>
                    · {h}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </GlassCard>
  )
}
