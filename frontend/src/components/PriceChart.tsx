import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { format, parseISO } from 'date-fns'

interface DataPoint {
  date: string
  close: number
}

interface PriceChartProps {
  history: DataPoint[]
  forecast: DataPoint[]
  ticker: string
}

export default function PriceChart({ history, forecast, ticker }: PriceChartProps) {
  // Combine and sort data
  const combinedData = [
    ...history.map(d => ({
      date: d.date,
      close: d.close,
      type: 'history'
    })),
    ...forecast.map(d => ({
      date: d.date,
      close: d.close,
      type: 'forecast'
    }))
  ].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

  return (
    <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
      <div className="mb-4">
        <h2 className="text-lg font-bold text-white">{ticker} Price Analysis</h2>
        <p className="text-sm text-gray-400">Historical prices + ML forecasts</p>
      </div>
      
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={combinedData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            tickFormatter={(date) => format(parseISO(date), 'MMM dd')}
          />
          <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#fff'
            }}
            formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
            labelFormatter={(label) => format(parseISO(label), 'PPP')}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="Historical Price"
            data={history.map(d => ({ date: d.date, close: d.close }))}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#10b981"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            isAnimationActive={false}
            name="Forecast"
            data={forecast.map(d => ({ date: d.date, close: d.close }))}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
