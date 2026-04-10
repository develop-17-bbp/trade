import { type ReactNode, Suspense, lazy } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

const ParticleField = lazy(() => import('./ParticleField'))

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="relative min-h-screen w-full cyber-bg scan-lines">
      {/* Three.js particle background */}
      <Suspense fallback={null}>
        <div className="fixed inset-0 z-0 pointer-events-none">
          <ParticleField />
        </div>
      </Suspense>

      {/* Ambient neon glow spots */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-[#bf5fff]/[0.04] rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-[#00fff0]/[0.03] rounded-full blur-3xl" />
        <div className="absolute top-1/2 right-0 w-64 h-64 bg-[#ff00aa]/[0.02] rounded-full blur-3xl" />
      </div>

      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <div className="ml-[70px] relative z-10 flex flex-col min-h-screen">
        <TopBar />
        <main className="flex-1 p-6 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
