import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
})

export interface AnalyzeRequest {
  ticker: string
  thread_id: string
}

export interface AnalyzeResponse {
  task_id: string
  status: string
  prediction: {
    forecast: Array<{date: string; close: number}>
    history: Array<{date: string; close: number}>
  }
  analysis: {
    sentiment: string
    recommendation: string
    report: string
  }
}

export const analyzeStock = async (ticker: string, threadId: string) => {
  const response = await api.post<AnalyzeResponse>('/analyze', {
    ticker,
    thread_id: threadId,
  })
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

export const getTaskStatus = async (taskId: string) => {
  const response = await api.get(`/status/${taskId}`)
  return response.data
}

export const getMetrics = async () => {
  const response = await api.get('/metrics')
  return response.data
}

export const getSystemCache = async () => {
  const response = await api.get('/system/cache')
  return response.data
}

export const trainParent = async () => {
  const response = await api.post('/train-parent')
  return response.data
}

export const monitorTicker = async (ticker: string) => {
  const response = await api.post(`/monitor/${ticker}`)
  return response.data
}

export default api
