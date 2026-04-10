import type { ReactNode } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
  glow?: 'green' | 'red' | 'blue' | 'purple' | 'cyan' | 'none'
  padding?: boolean
  shimmer?: boolean
}

const GLOW_MAP: Record<string, string> = {
  green: 'glow-green',
  red: 'glow-red',
  blue: 'glow-blue',
  purple: 'glow-purple',
  cyan: 'glow-cyan',
  none: '',
}

export default function GlassCard({
  children,
  className = '',
  glow = 'none',
  padding = true,
  shimmer = true,
}: GlassCardProps) {
  return (
    <div className={`glass-card ${GLOW_MAP[glow]} ${shimmer ? 'holo-shimmer' : ''} ${padding ? 'p-5' : ''} ${className}`}>
      {children}
    </div>
  )
}
