import { create } from 'zustand'

interface AnalysisResult {
  ticker: string
  timestamp: string
  forecast: Array<{date: string; close: number}>
  history: Array<{date: string; close: number}>
  report?: string
  confidence?: number
}

interface Store {
  sessionId: string
  currentTicker: string
  analysisResults: AnalysisResult | null
  isLoading: boolean
  error: string | null
  setCurrentTicker: (ticker: string) => void
  setAnalysisResults: (results: AnalysisResult) => void
  setIsLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useStore = create<Store>((set) => ({
  sessionId: Math.random().toString(36).substring(2, 11),
  currentTicker: 'NVDA',
  analysisResults: null,
  isLoading: false,
  error: null,
  setCurrentTicker: (ticker: string) => set({ currentTicker: ticker }),
  setAnalysisResults: (results: AnalysisResult) => set({ analysisResults: results }),
  setIsLoading: (loading: boolean) => set({ isLoading: loading }),
  setError: (error: string | null) => set({ error }),
}))
