export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type RiskLevel = 'High' | 'Medium' | 'Low'

export interface Reason {
  source: 'fraud' | 'anomaly'
  feature: string
  label: string
  value: number
  contribution: number
}

export interface Transaction {
  trans_num: string
  cc_num: number
  trans_time: string
  amt: number
  category: string
  merchant: string | null
  merch_lat: number
  merch_long: number
  is_fraud: number | null
  source: 'train' | 'test' | 'api'
  fraud_score: number | null
  anomaly_score: number | null
  risk_level: RiskLevel | null
  reasons: Reason[]
}

export interface ScoreRequest {
  cc_num: number
  trans_time?: string
  amt: number
  category: string
  merch_lat: number
  merch_long: number
  merchant?: string
  trans_num?: string
  save?: boolean
}

export interface ScoreResponse {
  trans_num: string
  cc_num: number
  trans_time: string
  amt: number
  category: string
  fraud_score: number
  anomaly_score: number
  risk_level: RiskLevel
  alert: boolean
  reasons: Reason[]
  features: Record<string, number>
  history_size: number
  saved: boolean
}

export interface CardResponse {
  card: { cc_num: number; lat: number; long: number; city: string; state: string }
  transactions: Transaction[]
}

export interface BudgetRow {
  budget: number
  alerts: number
  tp: number
  recall: number
  precision: number
  f1: number
  threshold: number
}

export interface EvalBlock {
  transactions: number
  fraud: number
  pr_auc: number
  roc_auc: number
  high: { tp: number; fp: number; fn: number; precision: number; recall: number; f1: number }
  by_level: Record<RiskLevel, { transactions: number; fraud: number }>
  budgets: BudgetRow[]
}

export interface ModelReport {
  system: {
    fraud_model: string
    anomaly_model: string
    alert_budget: number
    t_high: number
    budget_reached_target_recall: boolean
    target_recall: number
    medium_x: number
    t_anomaly: number
    medium_max_rate: number
    val_start: string
  }
  splits: Record<'train' | 'val' | 'test', { rows: number; fraud: number; start: string; end: string }>
  validation: EvalBlock & {
    medium_options: {
      x: number
      medium_alerts: number
      medium_rate: number
      fraud_in_medium: number
      combined_recall: number
      fraud_only_same_alerts_recall: number
      gain_vs_fraud_only: number
      within_cap: boolean
    }[]
  }
  test: EvalBlock & { combined_alerts: number; combined_tp: number; fraud_only_same_alerts_tp: number }
  feature_importance: Record<string, number>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* giữ statusText */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  categories: () => request<string[]>('/api/categories'),
  model: () => request<ModelReport>('/api/model'),
  score: (body: ScoreRequest) => request<ScoreResponse>('/api/score', { method: 'POST', body: JSON.stringify(body) }),
  card: (ccNum: number, limit = 15) => request<CardResponse>(`/api/cards/${ccNum}?limit=${limit}`),
  samples: (kind: 'fraud' | 'legit' | 'random', n = 1) => request<Transaction[]>(`/api/samples?kind=${kind}&n=${n}`),
  alerts: (level: RiskLevel | '', limit: number, offset: number) =>
    request<{ total: number; items: Transaction[] }>(
      `/api/alerts?limit=${limit}&offset=${offset}${level ? `&level=${level}` : ''}`,
    ),
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
export const money = (x: number) => `$${x.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
