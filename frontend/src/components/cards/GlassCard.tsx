import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  glow?: 'green' | 'red' | 'blue' | 'purple' | 'cyan' | 'none'
  padding?: boolean
  shimmer?: boolean
}

export default function GlassCard({
  children,
  className = '',
  padding = true,
}: CardProps) {
  return (
    <div className={`card ${padding ? 'p-4' : ''} ${className}`}>
      {children}
    </div>
  )
}
