"""Feature engineering. Dùng chung cho train (offline) và chấm điểm từng giao dịch (online).

Mọi đặc trưng theo thẻ chỉ dùng giao dịch có thời điểm TRƯỚC giao dịch hiện tại và không dùng nhãn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HOURS_SINCE_PREV_CAP, NIGHT_HOURS

NUMERIC_FEATURES = [
    "amt", "log_amt", "amt_ratio_card_mean", "amt_zscore_card",
    "hour", "is_night", "day_of_week", "n_tx_1h", "n_tx_24h", "n_tx_7d", "amt_sum_24h", "hours_since_prev",
    "dist_home_merch_km", "dist_prev_merch_km", "speed_kmh",
    "card_cat_share",
]

FEATURE_LABELS_VI = {
    "amt": "số tiền", "log_amt": "log số tiền", "amt_ratio_card_mean": "số tiền / trung bình của thẻ",
    "amt_zscore_card": "độ lệch số tiền so với thẻ (z)", "hour": "giờ giao dịch", "is_night": "giao dịch ban đêm",
    "day_of_week": "thứ trong tuần", "n_tx_1h": "số giao dịch 1h trước", "n_tx_24h": "số giao dịch 24h trước",
    "n_tx_7d": "số giao dịch 7 ngày trước", "amt_sum_24h": "tổng tiền 24h trước",
    "hours_since_prev": "số giờ từ giao dịch trước", "dist_home_merch_km": "khoảng cách nhà → merchant (km)",
    "dist_prev_merch_km": "khoảng cách so với giao dịch trước (km)", "speed_kmh": "tốc độ di chuyển (km/h)",
    "card_cat_share": "tỷ lệ thẻ dùng loại dịch vụ này",
}


def feature_label(col: str) -> str:
    return f"loại dịch vụ = {col[4:]}" if col.startswith("cat_") else FEATURE_LABELS_VI.get(col, col)


def feature_columns(categories: list[str]) -> list[str]:
    return NUMERIC_FEATURES + [f"cat_{c}" for c in categories]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amt"] = np.log1p(out["amt"])
    out["hour"] = out["trans_time"].dt.hour.astype("int8")
    out["day_of_week"] = out["trans_time"].dt.dayofweek.astype("int8")
    out["is_night"] = out["hour"].isin(NIGHT_HOURS).astype("int8")
    out["dist_home_merch_km"] = haversine_km(out["lat"].values, out["long"].values,
                                             out["merch_lat"].values, out["merch_long"].values)
    return out


def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """df phải chứa đủ lịch sử các thẻ liên quan. Trả về cùng thứ tự dòng ban đầu."""
    d = df.sort_values(["cc_num", "trans_time"], kind="stable").copy()
    g = d.groupby("cc_num", sort=False)

    n_prev = g.cumcount()
    sum_prev = g["amt"].cumsum() - d["amt"]
    sumsq_prev = (d["amt"] ** 2).groupby(d["cc_num"], sort=False).cumsum() - d["amt"] ** 2
    mean_prev = sum_prev / n_prev.replace(0, np.nan)
    var_prev = (sumsq_prev - n_prev * mean_prev ** 2) / (n_prev - 1).where(n_prev > 1)
    std_prev = np.sqrt(var_prev.clip(lower=0))

    d["card_tx_count"] = n_prev.astype("int32")
    d["amt_ratio_card_mean"] = (d["amt"] / mean_prev).fillna(1.0)
    d["amt_zscore_card"] = ((d["amt"] - mean_prev) / std_prev.replace(0, np.nan)).fillna(0.0).clip(-50, 50)

    prev_time = g["trans_time"].shift()
    prev_lat, prev_long = g["merch_lat"].shift(), g["merch_long"].shift()
    hours = (d["trans_time"] - prev_time).dt.total_seconds() / 3600
    dist = haversine_km(prev_lat.values, prev_long.values, d["merch_lat"].values, d["merch_long"].values)
    d["hours_since_prev"] = hours.fillna(HOURS_SINCE_PREV_CAP).clip(upper=HOURS_SINCE_PREV_CAP)
    d["dist_prev_merch_km"] = np.nan_to_num(dist, nan=0.0)
    d["speed_kmh"] = d["dist_prev_merch_km"] / np.maximum(d["hours_since_prev"], 1 / 60)

    # d đã sắp theo (cc_num, trans_time); groupby(sort=False) giữ thứ tự đó nên kết quả rolling khớp theo vị trí
    roll = d[["cc_num", "trans_time", "amt"]].groupby("cc_num", sort=False).rolling
    for window, name in [("1h", "n_tx_1h"), ("24h", "n_tx_24h"), ("7D", "n_tx_7d")]:
        r = roll(window, on="trans_time", closed="left")["amt"].count()
        d[name] = np.nan_to_num(r.to_numpy(), nan=0).astype("int32")
    r = roll("24h", on="trans_time", closed="left")["amt"].sum()
    d["amt_sum_24h"] = np.nan_to_num(r.to_numpy(), nan=0.0)

    same_cat_prev = d.groupby(["cc_num", "category"], sort=False).cumcount()
    d["card_cat_share"] = (same_cat_prev / n_prev.replace(0, np.nan)).fillna(0.0)
    return d.loc[df.index]


def add_category_onehot(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    out = df.copy()
    cat = out["category"].astype(str)
    for c in categories:
        out[f"cat_{c}"] = (cat == c).astype("int8")
    return out


def build_features(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Pipeline đặc trưng đầy đủ. df cần các cột: cc_num, trans_time, amt, category, lat, long, merch_lat, merch_long."""
    out = add_stateless_features(df)
    out = add_card_history_features(out)
    return add_category_onehot(out, categories)
