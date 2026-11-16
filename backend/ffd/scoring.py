"""Chấm điểm một giao dịch mới: lấy lịch sử thẻ -> tính đặc trưng -> fraud_score, anomaly_score -> mức rủi ro -> lý do."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .explain import explain
from .features import build_features
from .history import TIME_FMT, HistoryStore
from .models import isolation_forest_raw_score
from .risk import assign_risk


class UnknownCardError(KeyError):
    pass


@dataclass
class TransactionInput:
    cc_num: int
    trans_time: pd.Timestamp
    amt: float
    category: str
    merch_lat: float
    merch_long: float
    merchant: str | None = None
    trans_num: str | None = None


class Scorer:
    def __init__(self, bundle_path: Path, store: HistoryStore):
        self.bundle = joblib.load(bundle_path)
        self.store = store
        self.system = self.bundle["system"]

    @property
    def categories(self) -> list[str]:
        return self.bundle["categories"]

    def build_row(self, tx: TransactionInput) -> pd.DataFrame:
        card = self.store.card(tx.cc_num)
        if card is None:
            raise UnknownCardError(tx.cc_num)
        hist = self.store.card_history(tx.cc_num, tx.trans_time, exclude_trans_num=tx.trans_num)
        new = pd.DataFrame([{"trans_num": tx.trans_num or "new", "cc_num": tx.cc_num, "trans_time": tx.trans_time,
                             "amt": tx.amt, "category": tx.category, "merch_lat": tx.merch_lat,
                             "merch_long": tx.merch_long}])
        df = pd.concat([hist, new], ignore_index=True)
        df["lat"], df["long"] = card["lat"], card["long"]
        feats = build_features(df, self.categories)
        return feats.iloc[[-1]].reset_index(drop=True)

    def score(self, tx: TransactionInput) -> dict:
        row = self.build_row(tx)
        b = self.bundle
        fraud = float(b["xgb"].predict_proba(row[b["feature_cols"]])[:, 1][0])
        raw = isolation_forest_raw_score(b["iforest"], b["anomaly_prep"].transform(row))
        anomaly = float(b["anomaly_pct"].transform(raw)[0])
        level = assign_risk(np.array([fraud]), np.array([anomaly]), self.system["t_high"], self.system["t_anomaly"])
        reasons = explain(row, level, b)[0]
        features = {c: float(row.at[0, c]) for c in b["feature_cols"] if not c.startswith("cat_")}
        return {
            "trans_num": tx.trans_num or uuid.uuid4().hex,
            "cc_num": tx.cc_num,
            "trans_time": tx.trans_time.strftime(TIME_FMT),
            "amt": tx.amt, "category": tx.category, "merchant": tx.merchant,
            "merch_lat": tx.merch_lat, "merch_long": tx.merch_long,
            "fraud_score": fraud,
            "anomaly_score": anomaly,
            "risk_level": str(level[0]),
            "alert": bool(level[0] != "Low"),
            "reasons": reasons,
            "features": features,
            "history_size": int(row.at[0, "card_tx_count"]),
        }
