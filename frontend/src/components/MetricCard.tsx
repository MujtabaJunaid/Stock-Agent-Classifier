interface MetricCardProps {
  label: string
  value: string | number
  change?: number
  icon?: string
  trend?: 'up' | 'down' | 'neutral'
}

export default function MetricCard({ label, value, change, icon, trend = 'neutral' }: MetricCardProps) {
  const trendColor = trend === 'up' ? 'text-accent-primary' : trend === 'down' ? 'text-red-500' : 'text-gray-400'

  return (
    <div className="bg-dark-800 border border-dark-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm font-medium">{label}</span>
        {icon && <span className="text-2xl">{icon}</span>}
      </div>
      
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-white">{value}</span>
        {change !== undefined && (
          <span className={`text-sm font-semibold ${trendColor}`}>
            {change > 0 ? '+' : ''}{change.toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  )
}
