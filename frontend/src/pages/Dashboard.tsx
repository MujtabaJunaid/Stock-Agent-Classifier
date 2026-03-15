import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { checkHealth, getSystemCache, getMetrics } from '../services/api'
import MetricCard from '../components/MetricCard'
import ErrorAlert from '../components/ErrorAlert'

export default function Dashboard() {
  const { sessionId } = useStore()
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [cache, setCache] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [healthData, metricsData, cacheData] = await Promise.all([
          checkHealth(),
          getMetrics(),
          getSystemCache()
        ])
        setHealth(healthData)
        setMetrics(metricsData)
        setCache(cacheData)
      } catch (err: any) {
        setError(err.message || 'Failed to load dashboard data')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">System Dashboard</h1>
        <p className="text-gray-400">Real-time overview of your MLOps infrastructure</p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-accent-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-400 mt-4">Loading system metrics...</p>
        </div>
      ) : (
        <>
          {/* Health & System Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <MetricCard
              label="API Status"
              value={health?.status === 'healthy' ? '✓ Healthy' : '✗ Offline'}
              icon="🚀"
            />
            <MetricCard
              label="Prediction Latency"
              value={`${metrics?.prediction_latency?.toFixed(2) || 0}ms`}
              icon="⚡"
            />
            <MetricCard
              label="Cache Hit Rate"
              value={`${cache?.hit_rate?.toFixed(1) || 0}%`}
              icon="💾"
            />
            <MetricCard
              label="Active Sessions"
              value={metrics?.active_sessions || 0}
              icon="👥"
            />
          </div>

          {/* System Info */}
          <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
            <h2 className="text-lg font-bold text-white mb-4">System Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="border-l-4 border-accent-primary pl-4">
                <p className="text-gray-400 text-sm mb-1">Session ID</p>
                <p className="text-white font-mono text-sm">{sessionId}</p>
              </div>
              <div className="border-l-4 border-accent-secondary pl-4">
                <p className="text-gray-400 text-sm mb-1">API Version</p>
                <p className="text-white font-mono text-sm">v3.1</p>
              </div>
              <div className="border-l-4 border-accent-warning pl-4">
                <p className="text-gray-400 text-sm mb-1">Environment</p>
                <p className="text-white font-mono text-sm">Production</p>
              </div>
              <div className="border-l-4 border-accent-danger pl-4">
                <p className="text-gray-400 text-sm mb-1">Pipeline Status</p>
                <p className="text-white font-mono text-sm">Operational</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
