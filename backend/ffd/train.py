"""Train lại hệ thống với cấu hình đã chốt trong notebook.

Chạy: python -m ffd.train
Đầu ra (thư mục artifacts/): model_bundle.joblib, report.json, test_scored.parquet
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import audit_quality, load_raw
from .explain import explain
from .features import build_features, feature_columns
from .models import (AnomalyPreprocessor, PercentileScaler, anomaly_columns, isolation_forest_raw_score,
                     train_isolation_forest, train_xgboost)
from .risk import assign_risk, choose_budget, choose_medium_x, evaluate, metrics_at_budget


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_by_time(full: pd.DataFrame) -> dict[str, pd.DataFrame]:
    is_test = full["_set"] == "test"
    is_val = (~is_test) & (full["trans_time"] >= pd.Timestamp(C.VAL_START))
    parts = {"train": full[(~is_test) & (~is_val)], "val": full[is_val], "test": full[is_test]}
    return {k: v.sort_values("trans_time", kind="stable").reset_index(drop=True) for k, v in parts.items()}


def main() -> None:
    paths = C.Paths()
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)

    log("Đọc dữ liệu")
    train_raw, test_raw = load_raw(paths.train_csv), load_raw(paths.test_csv)
    quality = {"train": audit_quality(train_raw), "test": audit_quality(test_raw)}

    log("Tạo đặc trưng")
    categories = sorted(train_raw["category"].astype(str).unique())
    feature_cols = feature_columns(categories)
    full = pd.concat([train_raw.assign(_set="train"), test_raw.assign(_set="test")], ignore_index=True)
    sp = split_by_time(build_features(full, categories))
    y = {k: v[C.TARGET].to_numpy() for k, v in sp.items()}
    X = {k: v[feature_cols] for k, v in sp.items()}

    log("Train XGBoost + class weights")
    xgb_model = train_xgboost(X["train"], y["train"], X["val"], y["val"])
    fraud = {k: xgb_model.predict_proba(v)[:, 1] for k, v in X.items()}

    log("Train Isolation Forest")
    a_cols = anomaly_columns(feature_cols)
    prep = AnomalyPreprocessor(a_cols).fit(sp["train"])
    Z = {k: prep.transform(v) for k, v in sp.items()}
    iforest = train_isolation_forest(Z["train"])
    raw = {k: isolation_forest_raw_score(iforest, v) for k, v in Z.items()}
    pct = PercentileScaler().fit(raw["train"])
    anomaly = {k: pct.transform(v) for k, v in raw.items()}

    log("Chọn ngân sách cảnh báo và X% trên validation")
    budget, reached = choose_budget(y["val"], fraud["val"], C.ALERT_BUDGETS, C.TARGET_RECALL)
    t_high = budget["threshold"]
    medium, medium_table = choose_medium_x(y["val"], fraud["val"], anomaly["val"], t_high,
                                           C.MEDIUM_X_OPTIONS, C.MEDIUM_MAX_RATE)
    t_anomaly = 1 - medium["x"]

    system = {"fraud_model": "XGBoost + class weights", "anomaly_model": "Isolation Forest",
              "alert_budget": budget["budget"], "t_high": t_high, "budget_reached_target_recall": reached,
              "target_recall": C.TARGET_RECALL, "medium_x": medium["x"], "t_anomaly": t_anomaly,
              "medium_max_rate": C.MEDIUM_MAX_RATE, "val_start": C.VAL_START}

    log("Đánh giá trên test (một lần)")
    risk = {k: assign_risk(fraud[k], anomaly[k], t_high, t_anomaly) for k in sp}
    alerts = int((risk["test"] != "Low").sum())
    report = {
        "system": system,
        "data_quality": quality,
        "splits": {k: {"rows": int(len(v)), "fraud": int(y[k].sum()),
                       "start": str(v["trans_time"].min()), "end": str(v["trans_time"].max())} for k, v in sp.items()},
        "validation": {"budgets": [metrics_at_budget(y["val"], fraud["val"], b) for b in C.ALERT_BUDGETS],
                       "medium_options": medium_table, **evaluate(y["val"], fraud["val"], risk["val"])},
        "test": {**evaluate(y["test"], fraud["test"], risk["test"]),
                 "budgets": [metrics_at_budget(y["test"], fraud["test"], b) for b in C.ALERT_BUDGETS],
                 "combined_alerts": alerts,
                 "combined_tp": int(y["test"][risk["test"] != "Low"].sum()),
                 "fraud_only_same_alerts_tp": metrics_at_budget(y["test"], fraud["test"], alerts / len(y["test"]))["tp"]},
        "feature_importance": dict(sorted(zip(feature_cols, map(float, xgb_model.feature_importances_)),
                                          key=lambda kv: -kv[1])[:15]),
    }

    bundle = {"xgb": xgb_model, "anomaly_prep": prep, "iforest": iforest, "anomaly_pct": pct,
              "feature_cols": feature_cols, "anomaly_cols": a_cols, "categories": categories, "system": system}
    joblib.dump(bundle, paths.artifact_dir / "model_bundle.joblib")
    with open(paths.artifact_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    log("Lưu điểm và lý do cho tập test")
    test = sp["test"]
    reasons = explain(test, risk["test"], bundle)
    pd.DataFrame({
        "trans_num": test["trans_num"].astype(str).values,
        "fraud_score": fraud["test"], "anomaly_score": anomaly["test"],
        "risk_level": risk["test"],
        "reasons": [json.dumps(r, ensure_ascii=False) if r else None for r in reasons],
    }).to_parquet(paths.artifact_dir / "test_scored.parquet", index=False)

    t = report["test"]
    log(f"Xong. Test PR-AUC {t['pr_auc']:.4f} | High: precision {t['high']['precision']:.1%}, recall {t['high']['recall']:.1%} "
        f"| ngân sách {budget['budget']:.1%}, X {medium['x']:.1%}")


if __name__ == "__main__":
    main()
