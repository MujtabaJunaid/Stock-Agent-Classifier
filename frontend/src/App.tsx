import { useState } from 'react'
import { useStore } from './store'
import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import Monitoring from './pages/Monitoring'
import Navigation from './components/Navigation'

type Page = 'dashboard' | 'analysis' | 'monitoring'

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard')

  return (
    <div className="min-h-screen bg-dark-900">
      <Navigation currentPage={currentPage} onPageChange={setCurrentPage} />
      <main className="pt-20">
        {currentPage === 'dashboard' && <Dashboard />}
        {currentPage === 'analysis' && <Analysis />}
        {currentPage === 'monitoring' && <Monitoring />}
      </main>
    </div>
  )
}
