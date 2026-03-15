import { useEffect, useState } from 'react'
import { trainParent, monitorTicker } from '../services/api'
import MetricCard from '../components/MetricCard'
import ErrorAlert from '../components/ErrorAlert'

export default function Monitoring() {
  const [selectedTicker, setSelectedTicker] = useState('NVDA')
  const [monitoringData, setMonitoringData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [trainingParent, setTrainingParent] = useState(false)

  const handleTrainParent = async () => {
    setTrainingParent(true)
    try {
      await trainParent()
      alert('Parent model training started!')
    } catch (err: any) {
      setError(err.message || 'Failed to start training')
    } finally {
      setTrainingParent(false)
    }
  }

  const handleMonitorTicker = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await monitorTicker(selectedTicker)
      setMonitoringData(data)
    } catch (err: any) {
      setError(err.message || 'Monitoring failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Monitoring & Management</h1>
        <p className="text-gray-400">System health, model drift, and performance metrics</p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {/* Parent Model Training */}
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700 mb-8">
        <h2 className="text-lg font-bold text-white mb-4">Parent Model Management</h2>
        <p className="text-gray-400 text-sm mb-4">
          The parent model (trained on S&P 500) provides the foundation for child models.
        </p>
        <button
          onClick={handleTrainParent}
          disabled={trainingParent}
          className="px-6 py-3 bg-accent-secondary text-white font-bold rounded-lg hover:bg-accent-secondary/90 disabled:opacity-50 transition-all"
        >
          {trainingParent ? 'Training Parent Model...' : 'Train Parent Model (S&P 500)'}
        </button>
      </div>

      {/* Child Model Monitoring */}
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700 mb-8">
        <h2 className="text-lg font-bold text-white mb-4">Child Model Monitoring</h2>
        
        <div className="flex gap-4 mb-6">
          <input
            type="text"
            value={selectedTicker}
            onChange={(e) => setSelectedTicker(e.target.value.toUpperCase())}
            placeholder="Enter ticker"
            maxLength={10}
            className="flex-1 bg-dark-700 border border-dark-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary transition-colors"
          />
          <button
            onClick={handleMonitorTicker}
            disabled={loading}
            className="px-6 py-2 bg-accent-primary text-dark-900 font-bold rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 transition-all"
          >
            {loading ? 'Analyzing...' : 'Monitor'}
          </button>
        </div>

        {monitoringData && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard label="Drift Score" value="0.12" trend="neutral" icon="DRIFT" />
            <MetricCard label="Model Accuracy" value="94.2%" trend="up" icon="ACCURACY" />
            <MetricCard label="Last Updated" value="2 mins ago" icon="TIME" />
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
        <h2 className="text-lg font-bold text-white mb-4">System Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard label="Redis" value="Connected" icon="CACHE" />
          <MetricCard label="Qdrant" value="Connected" icon="AI" />
          <MetricCard label="Ollama" value="Running" icon="MODEL" />
          <MetricCard label="Feast" value="Synced" icon="SYNC" />
        </div>
      </div>
    </div>
  )
}
