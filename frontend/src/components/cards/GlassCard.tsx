import type { ReactNode } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
  glow?: 'green' | 'red' | 'blue' | 'purple' | 'none'
  padding?: boolean
}

const GLOW_MAP: Record<string, string> = {
  green: 'glow-green',
  red: 'glow-red',
  blue: 'glow-blue',
  purple: 'glow-purple',
  none: '',
}

export default function GlassCard({
  children,
  className = '',
  glow = 'none',
  padding = true,
}: GlassCardProps) {
  return (
    <div
      className={`glass-card ${GLOW_MAP[glow]} ${padding ? 'p-5' : ''} ${className}`}
    >
      {children}
    </div>
  )
}
