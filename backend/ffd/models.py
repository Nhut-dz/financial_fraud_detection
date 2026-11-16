"""Mô hình: XGBoost (Fraud) và Isolation Forest (Anomaly), kèm tiền xử lý và chuẩn hóa điểm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import ANOMALY_DROP_COLS, ANOMALY_LOG_COLS, CLIP_QUANTILES, IF_PARAMS, SEED, XGB_PARAMS


class AnomalyPreprocessor:
    """log1p các cột lệch phải -> clip theo phân vị train -> StandardScaler fit trên train."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.log_cols = [c for c in ANOMALY_LOG_COLS if c in cols]

    def _log(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.cols].astype("float64").copy()
        x[self.log_cols] = np.log1p(x[self.log_cols].clip(lower=0))
        return x

    def fit(self, df: pd.DataFrame) -> "AnomalyPreprocessor":
        x = self._log(df)
        self.lo_, self.hi_ = x.quantile(CLIP_QUANTILES[0]), x.quantile(CLIP_QUANTILES[1])
        self.scaler_ = StandardScaler().fit(x.clip(self.lo_, self.hi_, axis=1))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._log(df).clip(self.lo_, self.hi_, axis=1)
        return self.scaler_.transform(x).astype("float32")


class PercentileScaler:
    """Điểm thô -> tỷ lệ giao dịch train có điểm thấp hơn (0..1). Chỉ lưu 1.001 mốc phân vị cho gọn."""

    def fit(self, train_scores: np.ndarray) -> "PercentileScaler":
        self.q_ = np.linspace(0, 1, 1001)
        self.ref_ = np.quantile(np.asarray(train_scores), self.q_)
        return self

    def transform(self, scores) -> np.ndarray:
        return np.interp(np.asarray(scores, dtype="float64"), self.ref_, self.q_)


def anomaly_columns(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c not in ANOMALY_DROP_COLS]


def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    return IsolationForest(random_state=SEED, **IF_PARAMS).fit(X)


def isolation_forest_raw_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)


def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    spw = (y_train == 0).sum() / (y_train == 1).sum()   # class weights (quyết định 5)
    model = XGBClassifier(scale_pos_weight=spw, random_state=SEED, **XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
