import type { Reason, RiskLevel } from './api'

const RISK_TEXT: Record<RiskLevel, string> = {
  High: 'Cao – nghi vấn gian lận',
  Medium: 'Trung bình – bất thường, cần xem xét',
  Low: 'Thấp',
}

export function RiskBadge({ level, long = false }: { level: RiskLevel | null; long?: boolean }) {
  if (!level) return <span className="badge">—</span>
  return <span className={`badge badge-${level.toLowerCase()}`}>{long ? RISK_TEXT[level] : level}</span>
}

export function ScoreBar({ label, value, threshold, hint }: { label: string; value: number; threshold?: number; hint?: string }) {
  return (
    <div className="scorebar">
      <div className="scorebar-head">
        <span>{label}</span>
        <strong>{(value * 100).toFixed(2)}%</strong>
      </div>
      <div className="scorebar-track">
        <div className="scorebar-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
        {threshold !== undefined && <div className="scorebar-threshold" style={{ left: `${threshold * 100}%` }} title="ngưỡng" />}
      </div>
      {hint && <div className="muted small">{hint}</div>}
    </div>
  )
}

export function ReasonList({ reasons }: { reasons: Reason[] }) {
  if (!reasons.length) return <p className="muted">Không có cảnh báo, nên không cần giải thích.</p>
  const source = reasons[0].source === 'fraud' ? 'Mô hình Fraud (SHAP)' : 'Mô hình Anomaly (đặc trưng lệch nhất so với train)'
  return (
    <div>
      <div className="muted small">Nguồn: {source}</div>
      <ol className="reasons">
        {reasons.map((r) => (
          <li key={r.feature}>
            <strong>{r.label}</strong> = {r.value.toLocaleString('en-US', { maximumFractionDigits: 2 })}
            <span className="muted small"> (đóng góp {r.contribution.toFixed(2)})</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

export function ErrorBox({ error }: { error: string | null }) {
  return error ? <div className="error">{error}</div> : null
}

export function Label({ isFraud }: { isFraud: number | null }) {
  if (isFraud === null || isFraud === undefined) return <span className="muted">—</span>
  return isFraud ? <span className="tag tag-fraud">Gian lận</span> : <span className="tag">Hợp lệ</span>
}
