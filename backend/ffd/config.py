"""Cấu hình chung: đường dẫn, tham số mô hình và các quyết định nghiên cứu đã chốt."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default))


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: _path("FFD_DATA_DIR", PROJECT_ROOT / "data"))
    artifact_dir: Path = field(default_factory=lambda: _path("FFD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
    db_path: Path = field(default_factory=lambda: _path("FFD_DB_PATH", PROJECT_ROOT / "artifacts" / "ffd.sqlite"))

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "fraudTrain.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "fraudTest.csv"


TARGET = "is_fraud"
SEED = 42

# Cột định danh cá nhân: không dùng (quyết định 4)
PII_COLS = ["first", "last", "street"]

# Validation = các tháng cuối của fraudTrain (quyết định 3)
VAL_START = "2020-03-21"

# Giờ đêm xác định từ EDA trên train
NIGHT_HOURS = (22, 23, 0, 1, 2, 3)
HOURS_SINCE_PREV_CAP = 720.0

# Ngân sách cảnh báo và Recall mục tiêu (quyết định 7)
ALERT_BUDGETS = (0.005, 0.01, 0.02, 0.05)
TARGET_RECALL = 0.80

# Mức Medium: X% bất thường nhất so với train, trần số cảnh báo Medium (quyết định 8)
MEDIUM_X_OPTIONS = (0.005, 0.01, 0.02, 0.05)
MEDIUM_MAX_RATE = 0.01

XGB_PARAMS = {
    "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
    "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "aucpr",
    "early_stopping_rounds": 50, "n_jobs": -1,
}
IF_PARAMS = {"n_estimators": 300, "max_samples": 256, "n_jobs": -1}

# Tiền xử lý cho Isolation Forest
ANOMALY_LOG_COLS = ["amt_ratio_card_mean", "n_tx_1h", "n_tx_24h", "n_tx_7d", "amt_sum_24h",
                    "hours_since_prev", "dist_prev_merch_km", "speed_kmh"]
ANOMALY_DROP_COLS = ["amt"]
CLIP_QUANTILES = (0.001, 0.999)

EXPLAIN_TOP_K = 3
