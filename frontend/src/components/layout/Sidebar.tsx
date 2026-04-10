import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  CandlestickChart,
  Brain,
  TrendingUp,
  Shield,
} from 'lucide-react'
import StatusDot from '../shared/StatusDot'

interface NavItem {
  path: string
  label: string
  icon: React.ComponentType<{ size?: number; className?: string }>
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/trading', label: 'Trading', icon: CandlestickChart },
  { path: '/ai', label: 'AI Agents', icon: Brain },
  { path: '/performance', label: 'Performance', icon: TrendingUp },
  { path: '/risk', label: 'Risk', icon: Shield },
]

export default function Sidebar() {
  const [expanded, setExpanded] = useState(false)

  return (
    <aside
      className="fixed top-0 left-0 h-screen z-30 flex flex-col transition-all duration-300 ease-in-out bg-bg-secondary/80 backdrop-blur-xl border-r border-border-glass"
      style={{ width: expanded ? 240 : 70 }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      {/* Brand */}
      <div className="flex items-center h-16 px-4 border-b border-border-glass">
        <div className="w-[38px] h-[38px] rounded-lg bg-gradient-to-br from-accent-green to-accent-blue flex items-center justify-center flex-shrink-0">
          <span className="text-bg-primary font-bold text-sm">N</span>
        </div>
        <span
          className="ml-3 font-semibold text-text-primary text-sm tracking-wider whitespace-nowrap overflow-hidden transition-opacity duration-200"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          NEXUS
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 flex flex-col gap-1">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              `relative flex items-center h-12 px-4 mx-2 rounded-lg transition-all duration-200 group ${
                isActive
                  ? 'bg-accent-blue/10 text-accent-blue'
                  : 'text-text-muted hover:text-text-primary hover:bg-white/[0.03]'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {/* Active indicator bar */}
                {isActive && (
                  <div className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r-full bg-accent-blue" />
                )}
                <Icon size={20} className="flex-shrink-0" />
                <span
                  className="ml-3 text-sm font-medium whitespace-nowrap overflow-hidden transition-opacity duration-200"
                  style={{ opacity: expanded ? 1 : 0 }}
                >
                  {label}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom: system status + brand */}
      <div className="p-4 border-t border-border-glass flex items-center gap-3">
        <StatusDot status="online" size={8} />
        <span
          className="text-xs text-text-muted tracking-widest uppercase whitespace-nowrap overflow-hidden transition-opacity duration-200"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          NEXUS v1.0
        </span>
      </div>
    </aside>
  )
}
