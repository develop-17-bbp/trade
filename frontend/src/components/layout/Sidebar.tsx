import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  CandlestickChart,
  Brain,
  TrendingUp,
  Shield,
} from 'lucide-react'

const NAV_ITEMS = [
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
      className="fixed top-0 left-0 h-screen z-30 flex flex-col transition-all duration-200 border-r border-[#222]"
      style={{ width: expanded ? 200 : 56, background: '#0a0a0a' }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      {/* Brand */}
      <div className="flex items-center h-12 px-3 border-b border-[#222]">
        <div className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0 bg-white">
          <span className="text-black font-black text-sm">A</span>
        </div>
        <span
          className="ml-3 font-bold text-xs tracking-[0.2em] whitespace-nowrap overflow-hidden transition-opacity duration-150 text-white"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          ACT's
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 flex flex-col gap-0.5">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              `relative flex items-center h-10 px-4 transition-all duration-150 ${
                isActive
                  ? 'text-white bg-[#1a1a1a]'
                  : 'text-[#666] hover:text-white hover:bg-[#111]'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div className="absolute left-0 top-1 bottom-1 w-[2px] bg-white" />
                )}
                <Icon size={18} className="flex-shrink-0" />
                <span
                  className="ml-3 text-xs font-medium whitespace-nowrap overflow-hidden transition-opacity duration-150"
                  style={{ opacity: expanded ? 1 : 0 }}
                >
                  {label}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom */}
      <div className="px-4 py-3 border-t border-[#222] flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-[#22c55e]" />
        <span
          className="text-[9px] text-[#666] tracking-wider uppercase whitespace-nowrap overflow-hidden transition-opacity duration-150 font-mono"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          v1.0
        </span>
      </div>
    </aside>
  )
}
