import { useEffect, useState } from 'react'
import { api, API_URL } from './api'
import type { ModelReport } from './api'
import AlertsPage from './pages/AlertsPage'
import ModelPage from './pages/ModelPage'
import ScorePage from './pages/ScorePage'

type Tab = 'score' | 'alerts' | 'model'

const TABS: [Tab, string][] = [
  ['score', 'Chấm điểm giao dịch'],
  ['alerts', 'Cảnh báo'],
  ['model', 'Mô hình & đánh giá'],
]

export default function App() {
  const [tab, setTab] = useState<Tab>('score')
  const [categories, setCategories] = useState<string[]>([])
  const [report, setReport] = useState<ModelReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.categories(), api.model()])
      .then(([c, r]) => { setCategories(c); setReport(r) })
      .catch((e: Error) => setError(`Không kết nối được API tại ${API_URL}: ${e.message}`))
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Financial Fraud Detection</h1>
          <p className="muted small">XGBoost (gian lận) + Isolation Forest (bất thường) · dữ liệu Sparkov</p>
        </div>
        <nav className="tabs">
          {TABS.map(([key, label]) => (
            <button key={key} className={tab === key ? 'tab active' : 'tab'} onClick={() => setTab(key)}>{label}</button>
          ))}
        </nav>
      </header>
      {error && <div className="error">{error}</div>}
      <main>
        {tab === 'score' && <ScorePage categories={categories} report={report} />}
        {tab === 'alerts' && <AlertsPage />}
        {tab === 'model' && <ModelPage report={report} />}
      </main>
    </div>
  )
}
