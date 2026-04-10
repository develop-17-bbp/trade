interface StatusDotProps {
  status: 'online' | 'offline' | 'degraded' | 'idle' | 'error'
  size?: number
}

const STATUS_COLORS: Record<StatusDotProps['status'], string> = {
  online: 'bg-accent-green',
  offline: 'bg-accent-red',
  degraded: 'bg-yellow-500',
  idle: 'bg-text-muted',
  error: 'bg-accent-red',
}

const GLOW_COLORS: Record<StatusDotProps['status'], string> = {
  online: 'shadow-[0_0_6px_rgba(0,255,136,0.6)]',
  offline: 'shadow-[0_0_6px_rgba(255,51,102,0.6)]',
  degraded: 'shadow-[0_0_6px_rgba(234,179,8,0.6)]',
  idle: '',
  error: 'shadow-[0_0_6px_rgba(255,51,102,0.6)]',
}

export default function StatusDot({ status, size = 8 }: StatusDotProps) {
  return (
    <span
      className={`inline-block rounded-full ${STATUS_COLORS[status]} ${GLOW_COLORS[status]} ${
        status === 'online' ? 'animate-pulse-glow' : ''
      }`}
      style={{ width: size, height: size }}
    />
  )
}
