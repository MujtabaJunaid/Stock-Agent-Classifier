import { useState } from 'react'
import { useStore } from '../store'

interface NavigationProps {
  currentPage: 'dashboard' | 'analysis' | 'monitoring'
  onPageChange: (page: 'dashboard' | 'analysis' | 'monitoring') => void
}

export default function Navigation({ currentPage, onPageChange }: NavigationProps) {
  const { sessionId } = useStore()

  return (
    <nav className="fixed top-0 left-0 right-0 bg-dark-800 border-b border-dark-700 shadow-lg z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo & Project Name */}
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-accent-primary flex items-center justify-center text-dark-900 font-bold text-lg">
              📈
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">MLOps Stock Analyst</h1>
              <p className="text-xs text-gray-400">Powered by LangGraph & LSTM</p>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-8">
            <button
              onClick={() => onPageChange('dashboard')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                currentPage === 'dashboard'
                  ? 'bg-accent-primary text-dark-900'
                  : 'text-gray-300 hover:text-white hover:bg-dark-700'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => onPageChange('analysis')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                currentPage === 'analysis'
                  ? 'bg-accent-primary text-dark-900'
                  : 'text-gray-300 hover:text-white hover:bg-dark-700'
              }`}
            >
              Analysis
            </button>
            <button
              onClick={() => onPageChange('monitoring')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                currentPage === 'monitoring'
                  ? 'bg-accent-primary text-dark-900'
                  : 'text-gray-300 hover:text-white hover:bg-dark-700'
              }`}
            >
              Monitoring
            </button>

            {/* Session Info */}
            <div className="pl-8 border-l border-dark-700">
              <p className="text-xs text-gray-400">Session ID</p>
              <p className="text-sm font-mono text-gray-200">{sessionId.substring(0, 8)}</p>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
