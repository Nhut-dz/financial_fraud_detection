import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { api, money, pct } from '../api'
import type { CardResponse, ModelReport, ScoreResponse, Transaction } from '../api'
import { ErrorBox, Label, ReasonList, RiskBadge, ScoreBar } from '../components'

interface FormState {
  cc_num: string
  trans_time: string
  amt: string
  category: string
  merch_lat: string
  merch_long: string
  merchant: string
  trans_num: string
  save: boolean
}

const EMPTY: FormState = {
  cc_num: '', trans_time: '', amt: '', category: '', merch_lat: '', merch_long: '', merchant: '', trans_num: '', save: false,
}

const KEY_FEATURES: [string, string][] = [
  ['amt_ratio_card_mean', 'Số tiền / trung bình của thẻ'],
  ['amt_zscore_card', 'Độ lệch số tiền (z)'],
  ['amt_sum_24h', 'Tổng tiền 24h trước'],
  ['n_tx_1h', 'Số giao dịch 1h trước'],
  ['n_tx_24h', 'Số giao dịch 24h trước'],
  ['hours_since_prev', 'Số giờ từ giao dịch trước'],
  ['is_night', 'Giao dịch ban đêm'],
  ['dist_home_merch_km', 'Khoảng cách nhà → merchant (km)'],
  ['speed_kmh', 'Tốc độ di chuyển (km/h)'],
]

export default function ScorePage({ categories, report }: { categories: string[]; report: ModelReport | null }) {
  const [form, setForm] = useState<FormState>(EMPTY)
  const [sample, setSample] = useState<Transaction | null>(null)
  const [result, setResult] = useState<ScoreResponse | null>(null)
  const [card, setCard] = useState<CardResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const set = (k: keyof FormState, v: string | boolean) => setForm((f) => ({ ...f, [k]: v }))

  useEffect(() => {
    loadSample('fraud')
  }, [])

  async function loadSample(kind: 'fraud' | 'legit' | 'random') {
    setError(null)
    try {
      const [tx] = await api.samples(kind, 1)
      setSample(tx)
      setResult(null)
      setForm({
        cc_num: String(tx.cc_num),
        trans_time: tx.trans_time.replace(' ', 'T'),
        amt: String(tx.amt),
        category: tx.category,
        merch_lat: String(tx.merch_lat),
        merch_long: String(tx.merch_long),
        merchant: tx.merchant ?? '',
        trans_num: tx.trans_num,
        save: false,
      })
      setCard(await api.card(tx.cc_num))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function newTransaction() {
    // Giữ thẻ, xóa thời gian và mã giao dịch: API sẽ đặt thời điểm ngay sau giao dịch gần nhất của thẻ
    setSample(null)
    setResult(null)
    setForm((f) => ({ ...f, trans_time: '', trans_num: '', merchant: '' }))
  }

  async function submit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await api.score({
        cc_num: Number(form.cc_num),
        trans_time: form.trans_time || undefined,
        amt: Number(form.amt),
        category: form.category,
        merch_lat: Number(form.merch_lat),
        merch_long: Number(form.merch_long),
        merchant: form.merchant || undefined,
        trans_num: form.trans_num || undefined,
        save: form.save,
      })
      setResult(res)
      setCard(await api.card(res.cc_num))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Giao dịch cần chấm điểm</h2>
        <div className="row gap">
          <button type="button" onClick={() => loadSample('fraud')}>Mẫu gian lận (test)</button>
          <button type="button" onClick={() => loadSample('legit')}>Mẫu hợp lệ (test)</button>
          <button type="button" className="ghost" onClick={newTransaction}>Giao dịch mới cho thẻ này</button>
        </div>
        {sample && (
          <p className="muted small">
            Đang chấm lại giao dịch <code>{sample.trans_num.slice(0, 12)}…</code> trong tập test. Nhãn thật: <Label isFraud={sample.is_fraud} />
          </p>
        )}
        <form onSubmit={submit} className="form">
          <label>Số thẻ (cc_num)<input required value={form.cc_num} onChange={(e) => set('cc_num', e.target.value)} /></label>
          <label>Thời điểm
            <input type="datetime-local" step="1" value={form.trans_time} onChange={(e) => set('trans_time', e.target.value)} />
            <span className="muted small">Để trống: ngay sau giao dịch gần nhất của thẻ</span>
          </label>
          <label>Số tiền (USD)<input required type="number" step="0.01" min="0.01" value={form.amt} onChange={(e) => set('amt', e.target.value)} /></label>
          <label>Loại dịch vụ
            <select required value={form.category} onChange={(e) => set('category', e.target.value)}>
              <option value="" disabled>Chọn…</option>
              {categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label>Vĩ độ merchant<input required type="number" step="any" value={form.merch_lat} onChange={(e) => set('merch_lat', e.target.value)} /></label>
          <label>Kinh độ merchant<input required type="number" step="any" value={form.merch_long} onChange={(e) => set('merch_long', e.target.value)} /></label>
          <label className="full">Merchant<input value={form.merchant} onChange={(e) => set('merchant', e.target.value)} /></label>
          <label className="check full">
            <input type="checkbox" checked={form.save} onChange={(e) => set('save', e.target.checked)} />
            Lưu giao dịch vào lịch sử sau khi chấm
          </label>
          <button type="submit" className="primary full" disabled={loading}>{loading ? 'Đang chấm…' : 'Chấm điểm'}</button>
        </form>
        <ErrorBox error={error} />
      </section>

      <section className="panel">
        <h2>Kết quả</h2>
        {!result && <p className="muted">Bấm “Chấm điểm” để xem kết quả.</p>}
        {result && (
          <>
            <div className="row between">
              <RiskBadge level={result.risk_level} long />
              <span className="muted small">{result.history_size} giao dịch trước đó của thẻ</span>
            </div>
            <ScoreBar label="Điểm nghi vấn gian lận (XGBoost)" value={result.fraud_score} threshold={report?.system.t_high}
              hint={report ? `High khi ≥ ${pct(report.system.t_high, 2)} (ngân sách cảnh báo ${pct(report.system.alert_budget)})` : undefined} />
            <ScoreBar label="Mức bất thường (Isolation Forest, phân vị so với train)" value={result.anomaly_score} threshold={report?.system.t_anomaly}
              hint={report ? `Medium khi ≥ ${pct(report.system.t_anomaly)} và không thuộc High` : undefined} />
            <h3>Lý do</h3>
            <ReasonList reasons={result.reasons} />
            <h3>Đặc trưng chính</h3>
            <table className="table compact">
              <tbody>
                {KEY_FEATURES.map(([k, label]) => (
                  <tr key={k}><td>{label}</td><td className="num">{result.features[k]?.toLocaleString('en-US', { maximumFractionDigits: 2 })}</td></tr>
                ))}
              </tbody>
            </table>
            {result.saved && <p className="muted small">Đã lưu vào lịch sử (mã {result.trans_num.slice(0, 12)}…).</p>}
          </>
        )}
      </section>

      {card && (
        <section className="panel span-2">
          <h2>Lịch sử gần nhất của thẻ {card.card.cc_num} <span className="muted small">({card.card.city}, {card.card.state})</span></h2>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Thời điểm</th><th>Số tiền</th><th>Loại dịch vụ</th><th>Nguồn</th><th>Mức rủi ro</th><th>Nhãn thật</th></tr>
              </thead>
              <tbody>
                {card.transactions.map((t) => (
                  <tr key={t.trans_num}>
                    <td>{t.trans_time}</td>
                    <td className="num">{money(t.amt)}</td>
                    <td>{t.category}</td>
                    <td>{t.source}</td>
                    <td><RiskBadge level={t.risk_level} /></td>
                    <td><Label isFraud={t.is_fraud} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
