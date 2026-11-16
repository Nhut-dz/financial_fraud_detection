"""Tạo cơ sở dữ liệu lịch sử SQLite từ dataset và điểm của tập test.

Chạy sau `python -m ffd.train`: python -m ffd.init_db
"""
from __future__ import annotations

import time

import pandas as pd

from . import config as C
from .data import load_raw
from .history import HistoryStore


def main() -> None:
    paths = C.Paths()
    scored_path = paths.artifact_dir / "test_scored.parquet"
    if not scored_path.exists():
        raise FileNotFoundError("Chưa có artifacts/test_scored.parquet. Chạy `python -m ffd.train` trước.")

    t0 = time.time()
    train_raw, test_raw = load_raw(paths.train_csv), load_raw(paths.test_csv)
    full = pd.concat([train_raw.assign(source="train"), test_raw.assign(source="test")], ignore_index=True)
    full["trans_num"] = full["trans_num"].astype(str)

    cards = (full.sort_values("trans_time").groupby("cc_num", as_index=False)
             .agg(lat=("lat", "first"), long=("long", "first"), city=("city", "first"), state=("state", "first")))
    scored = pd.read_parquet(scored_path)
    tx = full[["trans_num", "cc_num", "trans_time", "amt", "category", "merchant", "merch_lat", "merch_long",
               C.TARGET, "source"]].merge(scored, on="trans_num", how="left")
    for col in ("category", "merchant", "source", "risk_level", "reasons"):
        tx[col] = tx[col].astype(object).where(tx[col].notna(), None)

    HistoryStore(paths.db_path).create(tx, cards)
    print(f"Đã tạo {paths.db_path}: {len(tx):,} giao dịch, {len(cards):,} thẻ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
