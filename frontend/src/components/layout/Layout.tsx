import { type ReactNode, Suspense, lazy } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

const ParticleField = lazy(() => import('./ParticleField'))

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="relative min-h-screen w-full bg-bg-primary">
      {/* Three.js particle background - renders behind everything */}
      <Suspense fallback={null}>
        <div className="fixed inset-0 z-0 pointer-events-none">
          <ParticleField />
        </div>
      </Suspense>

      {/* Sidebar */}
      <Sidebar />

      {/* Main content area: offset by sidebar width */}
      <div className="ml-[70px] relative z-10 flex flex-col min-h-screen">
        <TopBar />
        <main className="flex-1 p-6 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
