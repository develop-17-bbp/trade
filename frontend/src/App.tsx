import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Trading = lazy(() => import('./pages/Trading'))
const AIAgents = lazy(() => import('./pages/AIAgents'))
const Performance = lazy(() => import('./pages/Performance'))
const Risk = lazy(() => import('./pages/Risk'))

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-full w-full min-h-[60vh]">
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
        <span className="text-text-muted text-sm tracking-wider uppercase">Loading</span>
      </div>
    </div>
  )
}

function App() {
  return (
    <Layout>
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/trading" element={<Trading />} />
          <Route path="/ai" element={<AIAgents />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/risk" element={<Risk />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}

export default App
