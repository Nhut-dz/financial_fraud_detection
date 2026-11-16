import { Fragment, useEffect, useState } from 'react'
import { api, categoryLabel, money, pct } from '../api'
import type { RiskLevel, Transaction } from '../api'
import { ErrorBox, Label, ReasonList, RiskBadge } from '../components'

const PAGE = 25

export default function AlertsPage() {
  const [level, setLevel] = useState<RiskLevel | ''>('High')
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<{ total: number; items: Transaction[] } | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    api.alerts(level, PAGE, offset).then(setData).catch((e: Error) => setError(e.message))
  }, [level, offset])

  return (
    <section className="panel">
      <div className="row between wrap">
        <h2>Cảnh báo trên tập test và giao dịch đã lưu</h2>
        <div className="row gap">
          {(['High', 'Medium', ''] as const).map((l) => (
            <button key={l || 'all'} className={level === l ? 'primary' : ''} onClick={() => { setLevel(l); setOffset(0) }}>
              {l || 'Tất cả cảnh báo'}
            </button>
          ))}
        </div>
      </div>
      <ErrorBox error={error} />
      {data && (
        <>
          <p className="muted small">{data.total.toLocaleString()} cảnh báo. Bấm vào một dòng để xem lý do.</p>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Thời điểm</th><th>Thẻ</th><th>Số tiền</th><th>Loại dịch vụ</th><th>Điểm fraud</th><th>Bất thường</th><th>Mức</th><th>Nhãn thật</th></tr>
              </thead>
              <tbody>
                {data.items.map((t) => (
                  <Fragment key={t.trans_num}>
                    <tr className="clickable" onClick={() => setOpen(open === t.trans_num ? null : t.trans_num)}>
                      <td>{t.trans_time}</td>
                      <td>{t.cc_num}</td>
                      <td className="num">{money(t.amt)}</td>
                      <td>{categoryLabel(t.category)}</td>
                      <td className="num">{t.fraud_score !== null ? pct(t.fraud_score, 2) : '—'}</td>
                      <td className="num">{t.anomaly_score !== null ? pct(t.anomaly_score) : '—'}</td>
                      <td><RiskBadge level={t.risk_level} /></td>
                      <td><Label isFraud={t.is_fraud} /></td>
                    </tr>
                    {open === t.trans_num && (
                      <tr className="detail"><td colSpan={8}><ReasonList reasons={t.reasons} /></td></tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row gap pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Trước</button>
            <span className="muted small">{offset + 1}–{Math.min(offset + PAGE, data.total)} / {data.total.toLocaleString()}</span>
            <button disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>Sau →</button>
          </div>
        </>
      )}
    </section>
  )
}
