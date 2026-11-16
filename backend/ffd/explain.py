"""Giải thích vì sao giao dịch bị cảnh báo (quyết định 10).

- Mức High: đóng góp SHAP của XGBoost (TreeSHAP chính xác qua `pred_contribs`, không cần thư viện shap).
- Mức Medium: đặc trưng lệch nhiều nhất so với phân phối train (|z| sau tiền xử lý của Isolation Forest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import EXPLAIN_TOP_K
from .features import feature_label


def fraud_contributions(model, X: pd.DataFrame) -> np.ndarray:
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def anomaly_contributions(prep, df: pd.DataFrame) -> np.ndarray:
    return np.abs(prep.transform(df))


def top_reasons(contrib_row: np.ndarray, cols: list[str], values: np.ndarray, source: str, k: int = EXPLAIN_TOP_K) -> list[dict]:
    out = []
    for j in np.argsort(-contrib_row)[:k]:
        if contrib_row[j] <= 0:
            continue
        out.append({"source": source, "feature": cols[j], "label": feature_label(cols[j]),
                    "value": float(values[j]), "contribution": float(contrib_row[j])})
    return out


def explain(df: pd.DataFrame, risk_level: np.ndarray, bundle: dict) -> list[list[dict]]:
    """Lý do cho từng dòng; dòng Low trả về danh sách rỗng."""
    reasons: list[list[dict]] = [[] for _ in range(len(df))]
    risk_level = np.asarray(risk_level)
    high_idx = np.flatnonzero(risk_level == "High")
    med_idx = np.flatnonzero(risk_level == "Medium")
    if len(high_idx):
        cols = bundle["feature_cols"]
        Xh = df.iloc[high_idx][cols]
        contrib, vals = fraud_contributions(bundle["xgb"], Xh), Xh.to_numpy(float)
        for i, row in enumerate(high_idx):
            reasons[row] = top_reasons(contrib[i], cols, vals[i], "fraud")
    if len(med_idx):
        cols = bundle["anomaly_cols"]
        dm = df.iloc[med_idx]
        contrib, vals = anomaly_contributions(bundle["anomaly_prep"], dm), dm[cols].to_numpy(float)
        for i, row in enumerate(med_idx):
            reasons[row] = top_reasons(contrib[i], cols, vals[i], "anomaly")
    return reasons


def reasons_to_text(reasons: list[dict]) -> str:
    if not reasons:
        return ""
    tag = "[Fraud]" if reasons[0]["source"] == "fraud" else "[Bất thường]"
    return tag + " " + "; ".join(f"{r['label']} = {r['value']:,.2f}" for r in reasons)
