"""Đọc dữ liệu Sparkov và kiểm tra chất lượng."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PII_COLS, TARGET

RAW_DTYPES = {
    "cc_num": "int64", "merchant": "string", "category": "string", "amt": "float64",
    "first": "string", "last": "string", "gender": "string", "street": "string",
    "city": "string", "state": "string", "zip": "int32", "lat": "float64", "long": "float64",
    "city_pop": "int32", "job": "string", "trans_num": "string", "unix_time": "int64",
    "merch_lat": "float64", "merch_long": "float64", "is_fraud": "int8",
}

# Các cột cần cho đặc trưng và cho lịch sử giao dịch
TRANSACTION_COLS = ["trans_num", "cc_num", "trans_time", "amt", "category", "merchant",
                    "lat", "long", "merch_lat", "merch_long", TARGET]


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Đặt fraudTrain.csv và fraudTest.csv vào thư mục data/.")
    df = pd.read_csv(path, dtype=RAW_DTYPES, index_col=0)
    df["trans_time"] = pd.to_datetime(df.pop("trans_date_trans_time"))
    df = df.drop(columns=[c for c in PII_COLS if c in df.columns])
    return df.sort_values("trans_time", kind="stable").reset_index(drop=True)


def audit_quality(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_trans_num": int(df["trans_num"].duplicated().sum()),
        "amt_le_0": int((df["amt"] <= 0).sum()),
        "fraud": int(df[TARGET].sum()),
    }
