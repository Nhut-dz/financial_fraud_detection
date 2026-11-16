"""Ngân sách cảnh báo, mức rủi ro và các chỉ số đánh giá."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HIGH, MEDIUM, LOW = "High", "Medium", "Low"


def metrics_at_budget(y_true: np.ndarray, scores: np.ndarray, budget: float) -> dict:
    k = max(1, int(round(budget * len(scores))))
    top = np.argsort(-scores, kind="stable")[:k]
    tp = int(y_true[top].sum())
    recall = tp / max(1, int(y_true.sum()))
    precision = tp / k
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"budget": budget, "alerts": k, "tp": tp, "recall": recall, "precision": precision, "f1": f1,
            "threshold": float(np.sort(scores)[::-1][k - 1])}


def choose_budget(y_true, scores, budgets, target_recall) -> tuple[dict, bool]:
    """Ngân sách nhỏ nhất đạt Recall mục tiêu; không đạt thì lấy ngân sách lớn nhất (quyết định 7)."""
    rows = [metrics_at_budget(y_true, scores, b) for b in budgets]
    for r in rows:
        if r["recall"] >= target_recall:
            return r, True
    return rows[-1], False


def assign_risk(fraud_score, anomaly_score, t_high: float, t_anomaly: float) -> np.ndarray:
    fraud_score, anomaly_score = np.asarray(fraud_score), np.asarray(anomaly_score)
    high = fraud_score >= t_high
    medium = (~high) & (anomaly_score >= t_anomaly)
    return np.where(high, HIGH, np.where(medium, MEDIUM, LOW))


def choose_medium_x(y_true, fraud_score, anomaly_pct, t_high, x_options, max_rate) -> tuple[dict, list[dict]]:
    """X lớn nhất mà số cảnh báo Medium không vượt trần (quyết định 8). Trả về lựa chọn và bảng so sánh."""
    high = fraud_score >= t_high
    high_tp = int(y_true[high].sum())
    rows = []
    for x in x_options:
        medium = (~high) & (anomaly_pct >= 1 - x)
        total = int(high.sum() + medium.sum())
        fraud_only_tp = metrics_at_budget(y_true, fraud_score, total / len(fraud_score))["tp"]
        rows.append({
            "x": x, "medium_alerts": int(medium.sum()), "medium_rate": float(medium.mean()),
            "fraud_in_medium": int(y_true[medium].sum()),
            "combined_recall": (high_tp + int(y_true[medium].sum())) / y_true.sum(),
            "fraud_only_same_alerts_recall": fraud_only_tp / y_true.sum(),
            "gain_vs_fraud_only": high_tp + int(y_true[medium].sum()) - fraud_only_tp,
            "within_cap": bool(medium.mean() <= max_rate),
        })
    ok = [r for r in rows if r["within_cap"]] or rows[:1]
    return ok[-1], rows


def evaluate(y_true, fraud_score, risk_level) -> dict:
    y_true, risk_level = np.asarray(y_true), np.asarray(risk_level)
    pred_high = risk_level == HIGH
    tp = int((pred_high & (y_true == 1)).sum())
    fp = int((pred_high & (y_true == 0)).sum())
    fn = int((~pred_high & (y_true == 1)).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    levels = pd.DataFrame({"risk_level": risk_level, "fraud": y_true}).groupby("risk_level")["fraud"].agg(["size", "sum"])
    return {
        "transactions": int(len(y_true)),
        "fraud": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, fraud_score)),
        "roc_auc": float(roc_auc_score(y_true, fraud_score)),
        "high": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                 "f1": 2 * precision * recall / max(1e-12, precision + recall)},
        "by_level": {lvl: {"transactions": int(levels.loc[lvl, "size"]) if lvl in levels.index else 0,
                           "fraud": int(levels.loc[lvl, "sum"]) if lvl in levels.index else 0}
                     for lvl in (HIGH, MEDIUM, LOW)},
    }
