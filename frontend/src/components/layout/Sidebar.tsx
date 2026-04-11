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
      className="fixed top-0 left-0 h-screen z-30 flex flex-col transition-all duration-300 ease-in-out border-r"
      style={{
        width: expanded ? 240 : 70,
        background: 'linear-gradient(180deg, rgba(10,8,25,0.95), rgba(5,5,15,0.98))',
        borderColor: 'rgba(100, 80, 255, 0.12)',
        backdropFilter: 'blur(24px)',
      }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      {/* Brand */}
      <div className="flex items-center h-16 px-4 border-b border-[rgba(100,80,255,0.12)]">
        <div className="w-[38px] h-[38px] rounded-lg flex items-center justify-center flex-shrink-0 relative animate-pulse-glow"
          style={{
            background: 'linear-gradient(135deg, #00fff0, #bf5fff, #ff00aa)',
            backgroundSize: '200% 200%',
            animation: 'gradient-shift 3s ease infinite, pulse-glow 4s ease-in-out infinite',
            boxShadow: '0 0 15px rgba(0,255,240,0.4), 0 0 30px rgba(191,95,255,0.2), 0 0 45px rgba(255,0,170,0.1)',
            border: '1px solid rgba(0,255,240,0.3)',
          }}
        >
          <span className="text-[#05050f] font-black text-sm" style={{ textShadow: '0 0 4px rgba(0,255,240,0.5)' }}>A</span>
        </div>
        <span
          className="ml-3 font-bold text-sm tracking-[0.2em] whitespace-nowrap overflow-hidden transition-opacity duration-200 gradient-text"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          ACT's
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
                  ? 'text-[#00fff0]'
                  : 'text-[#5a6080] hover:text-[#e0e6ff] hover:bg-[rgba(0,255,240,0.03)]'
              }`
            }
            style={({ isActive }) => isActive ? {
              background: 'linear-gradient(90deg, rgba(0,255,240,0.08), rgba(191,95,255,0.04))',
              boxShadow: 'inset 0 0 20px rgba(0,255,240,0.03)',
            } : {}}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div
                    className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r-full"
                    style={{
                      background: 'linear-gradient(180deg, #00fff0, #bf5fff)',
                      boxShadow: '0 0 8px rgba(0,255,240,0.5)',
                    }}
                  />
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

      {/* Bottom */}
      <div className="p-4 border-t border-[rgba(100,80,255,0.12)] flex items-center gap-3">
        <StatusDot status="online" size={8} />
        <span
          className="text-[10px] text-[#5a6080] tracking-[0.3em] uppercase whitespace-nowrap overflow-hidden transition-opacity duration-200 font-mono"
          style={{ opacity: expanded ? 1 : 0 }}
        >
          ACT's v1.0
        </span>
      </div>
    </aside>
  )
}
