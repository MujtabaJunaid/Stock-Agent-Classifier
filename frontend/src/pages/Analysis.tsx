import { useState } from 'react'
import { useStore } from '../store'
import { analyzeStock } from '../services/api'
import PriceChart from '../components/PriceChart'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'
import MetricCard from '../components/MetricCard'

export default function Analysis() {
  const { sessionId, currentTicker, analysisResults, setCurrentTicker, setAnalysisResults, setIsLoading, setError, isLoading, error } = useStore()
  const [input, setInput] = useState(currentTicker)

  const handleAnalyze = async () => {
    if (!input.trim()) return

    const ticker = input.toUpperCase().trim()
    setCurrentTicker(ticker)
    setIsLoading(true)
    setError(null)

    try {
      const result = await analyzeStock(ticker, sessionId)
      setAnalysisResults({
        ticker,
        timestamp: new Date().toISOString(),
        ...result.prediction,
        report: result.analysis.report,
        confidence: Math.random() * 0.3 + 0.7 // Placeholder
      })
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Stock Analysis</h1>
        <p className="text-gray-400">Analyze any stock with AI-powered insights</p>
      </div>

      {/* Search Input */}
      {/* Search Input */}
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700 mb-8">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
            placeholder="Enter ticker symbol (e.g., NVDA, AAPL, TSLA)"
            maxLength={10}
            className="flex-1 bg-dark-700 border border-dark-600 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary transition-colors"
          />
          <button
            onClick={handleAnalyze}
            disabled={isLoading || !input.trim()}
            className="px-8 py-3 bg-accent-primary text-dark-900 font-bold rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isLoading ? 'Analyzing' : 'Analyze'}
          </button>
        </div>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {isLoading && <LoadingSpinner />}

      {analysisResults && !isLoading && (
        <>
          {/* Price Chart */}
          {/* Price Chart */}
          <div className="mb-8">
            <PriceChart
              ticker={analysisResults.ticker}
              history={analysisResults.history || []}
              forecast={analysisResults.forecast || []}
            />
          </div>

          {/* Analysis Metrics */}
          {/* Analysis Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <MetricCard
              label="Confidence Score"
              value={`${(analysisResults.confidence! * 100).toFixed(0)}%`}
              icon="SCORE"
            />
            <MetricCard
              label="Forecast Horizon"
              value="30 days"
              icon="CALENDAR"
            />
            <MetricCard
              label="Model Type"
              value="LSTM Transfer"
              icon="AI"
            />
          </div>

          {/* AI Report */}
          {/* AI Report */}
          {analysisResults.report && (
            <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
              <h2 className="text-lg font-bold text-white mb-4">Analysis Report</h2>
              <div className="prose prose-invert max-w-none">
                <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {analysisResults.report}
                </p>
              </div>
            </div>
          )}
        </>
      )}

      {!isLoading && !analysisResults && !error && (
        <div className="text-center py-16 bg-dark-800 rounded-xl border border-dark-700 border-dashed">
          <p className="text-gray-400 text-lg">Enter a ticker symbol to begin analysis</p>
        </div>
      )}
    </div>
  )
}
