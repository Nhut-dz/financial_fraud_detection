import { pct } from '../api'
import type { BudgetRow, ModelReport } from '../api'

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="metric">
      <div className="muted small">{label}</div>
      <div className="metric-value">{value}</div>
      {note && <div className="muted small">{note}</div>}
    </div>
  )
}

function BudgetTable({ rows, chosen }: { rows: BudgetRow[]; chosen: number }) {
  return (
    <table className="table compact">
      <thead><tr><th>Ngân sách</th><th>Cảnh báo</th><th>Recall</th><th>Precision</th><th>F1</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.budget} className={r.budget === chosen ? 'highlight' : ''}>
            <td>{pct(r.budget)}</td><td className="num">{r.alerts.toLocaleString()}</td>
            <td className="num">{pct(r.recall)}</td><td className="num">{pct(r.precision)}</td><td className="num">{r.f1.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function ModelPage({ report }: { report: ModelReport | null }) {
  if (!report) return <section className="panel"><p className="muted">Đang tải…</p></section>
  const { system: s, test: t, validation: v } = report
  const maxImp = Math.max(...Object.values(report.feature_importance))
  return (
    <div className="grid-2">
      <section className="panel span-2">
        <h2>Kết quả trên tập test (đánh giá một lần)</h2>
        <div className="metrics">
          <Metric label="PR-AUC" value={t.pr_auc.toFixed(3)} note="chỉ số chính" />
          <Metric label="ROC-AUC" value={t.roc_auc.toFixed(4)} />
          <Metric label="Recall mức High" value={pct(t.high.recall)} note={`mục tiêu ≥ ${pct(s.target_recall, 0)}`} />
          <Metric label="Precision mức High" value={pct(t.high.precision)} />
          <Metric label="F1 mức High" value={t.high.f1.toFixed(3)} />
        </div>
        <p className="muted small">
          Test: {report.splits.test.rows.toLocaleString()} giao dịch, {report.splits.test.fraud.toLocaleString()} gian lận
          ({report.splits.test.start.slice(0, 10)} → {report.splits.test.end.slice(0, 10)}).
        </p>
      </section>

      <section className="panel">
        <h2>Cấu hình đã chốt</h2>
        <table className="table compact">
          <tbody>
            <tr><td>Mô hình Fraud</td><td>{s.fraud_model}</td></tr>
            <tr><td>Mô hình Anomaly</td><td>{s.anomaly_model}</td></tr>
            <tr><td>Ngân sách cảnh báo (High)</td><td>{pct(s.alert_budget)} → ngưỡng {s.t_high.toFixed(4)}</td></tr>
            <tr><td>Đạt Recall mục tiêu trên validation</td><td>{s.budget_reached_target_recall ? 'Có' : 'Không'}</td></tr>
            <tr><td>Medium: top X% bất thường</td><td>{pct(s.medium_x)} (trần {pct(s.medium_max_rate)} số giao dịch)</td></tr>
            <tr><td>Validation bắt đầu từ</td><td>{s.val_start}</td></tr>
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Theo mức rủi ro (test)</h2>
        <table className="table compact">
          <thead><tr><th>Mức</th><th>Giao dịch</th><th>Gian lận</th><th>Precision</th></tr></thead>
          <tbody>
            {(['High', 'Medium', 'Low'] as const).map((l) => {
              const x = t.by_level[l]
              return (
                <tr key={l}><td>{l}</td><td className="num">{x.transactions.toLocaleString()}</td>
                  <td className="num">{x.fraud.toLocaleString()}</td>
                  <td className="num">{x.transactions ? pct(x.fraud / x.transactions) : '—'}</td></tr>
              )
            })}
          </tbody>
        </table>
        <p className="small">
          Hệ thống kết hợp (High + Medium) bắt <strong>{t.combined_tp.toLocaleString()}</strong> gian lận với {t.combined_alerts.toLocaleString()} cảnh báo;
          chỉ dùng mô hình Fraud với cùng số cảnh báo bắt <strong>{t.fraud_only_same_alerts_tp.toLocaleString()}</strong>.
          Mức Medium là lớp “bất thường, cần xem xét”, không phải công cụ tăng Recall.
        </p>
      </section>

      <section className="panel">
        <h2>Ngân sách cảnh báo – validation</h2>
        <BudgetTable rows={v.budgets} chosen={s.alert_budget} />
      </section>
      <section className="panel">
        <h2>Ngân sách cảnh báo – test</h2>
        <BudgetTable rows={t.budgets} chosen={s.alert_budget} />
      </section>

      <section className="panel span-2">
        <h2>Top đặc trưng quan trọng (XGBoost)</h2>
        <div className="bars">
          {Object.entries(report.feature_importance).map(([k, val]) => (
            <div key={k} className="bar-row">
              <span className="bar-label">{k}</span>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${(val / maxImp) * 100}%` }} /></div>
              <span className="num small">{val.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
