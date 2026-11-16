"""Kho lịch sử giao dịch (SQLite). Nạp từ dataset; API đọc lịch sử thẻ để tính đặc trưng và ghi giao dịch mới."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

TIME_FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    cc_num INTEGER PRIMARY KEY,
    lat REAL NOT NULL,
    long REAL NOT NULL,
    city TEXT,
    state TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    trans_num TEXT PRIMARY KEY,
    cc_num INTEGER NOT NULL,
    trans_time TEXT NOT NULL,
    amt REAL NOT NULL,
    category TEXT NOT NULL,
    merchant TEXT,
    merch_lat REAL NOT NULL,
    merch_long REAL NOT NULL,
    is_fraud INTEGER,
    source TEXT NOT NULL,
    fraud_score REAL,
    anomaly_score REAL,
    risk_level TEXT,
    reasons TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_card_time ON transactions (cc_num, trans_time);
CREATE INDEX IF NOT EXISTS ix_tx_risk ON transactions (risk_level, fraud_score);
CREATE INDEX IF NOT EXISTS ix_tx_source ON transactions (source);
"""


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ khởi tạo
    def create(self, transactions: pd.DataFrame, cards: pd.DataFrame) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        with self.connect() as con:
            con.executescript(SCHEMA)
            cards.to_sql("cards", con, if_exists="append", index=False)
            tx = transactions.copy()
            tx["trans_time"] = tx["trans_time"].dt.strftime(TIME_FMT)
            tx.to_sql("transactions", con, if_exists="append", index=False, chunksize=50_000)

    # ------------------------------------------------------------------ đọc
    def card(self, cc_num: int) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM cards WHERE cc_num = ?", (cc_num,)).fetchone()
        return dict(row) if row else None

    def card_history(self, cc_num: int, before: pd.Timestamp, exclude_trans_num: str | None = None) -> pd.DataFrame:
        """Mọi giao dịch của thẻ trước thời điểm `before` (đủ để tính trung bình, độ lệch, tần suất)."""
        sql = ("SELECT trans_num, cc_num, trans_time, amt, category, merch_lat, merch_long FROM transactions "
               "WHERE cc_num = ? AND trans_time < ? AND trans_num != ? ORDER BY trans_time")
        with self.connect() as con:
            df = pd.read_sql_query(sql, con, params=(cc_num, before.strftime(TIME_FMT), exclude_trans_num or ""))
        df["trans_time"] = pd.to_datetime(df["trans_time"])
        return df

    def recent_transactions(self, cc_num: int, limit: int = 20) -> list[dict]:
        sql = ("SELECT trans_num, trans_time, amt, category, merchant, is_fraud, source, fraud_score, anomaly_score, "
               "risk_level FROM transactions WHERE cc_num = ? ORDER BY trans_time DESC LIMIT ?")
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, (cc_num, limit)).fetchall()]

    def transaction(self, trans_num: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM transactions WHERE trans_num = ?", (trans_num,)).fetchone()
        return _decode(dict(row)) if row else None

    def alerts(self, level: str | None, limit: int, offset: int) -> tuple[int, list[dict]]:
        where = "risk_level IN ('High', 'Medium')" if level is None else "risk_level = ?"
        params: tuple = () if level is None else (level,)
        with self.connect() as con:
            total = con.execute(f"SELECT COUNT(*) FROM transactions WHERE {where}", params).fetchone()[0]
            rows = con.execute(
                f"SELECT * FROM transactions WHERE {where} ORDER BY trans_time DESC LIMIT ? OFFSET ?",
                params + (limit, offset)).fetchall()
        return total, [_decode(dict(r)) for r in rows]

    def samples(self, kind: str, n: int) -> list[dict]:
        cond = {"fraud": "is_fraud = 1", "legit": "is_fraud = 0", "random": "1 = 1"}[kind]
        sql = f"SELECT * FROM transactions WHERE source = 'test' AND {cond} ORDER BY RANDOM() LIMIT ?"
        with self.connect() as con:
            return [_decode(dict(r)) for r in con.execute(sql, (n,)).fetchall()]

    def stats(self) -> dict:
        with self.connect() as con:
            by_source = {r[0]: r[1] for r in con.execute("SELECT source, COUNT(*) FROM transactions GROUP BY source")}
            by_level = {r[0]: {"transactions": r[1], "fraud": r[2]} for r in con.execute(
                "SELECT risk_level, COUNT(*), SUM(is_fraud) FROM transactions WHERE risk_level IS NOT NULL GROUP BY risk_level")}
            cards = con.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        return {"cards": cards, "by_source": by_source, "by_risk_level": by_level}

    # ------------------------------------------------------------------ ghi
    def insert_scored(self, tx: dict) -> None:
        cols = ["trans_num", "cc_num", "trans_time", "amt", "category", "merchant", "merch_lat", "merch_long",
                "is_fraud", "source", "fraud_score", "anomaly_score", "risk_level", "reasons"]
        row = {**tx, "reasons": json.dumps(tx.get("reasons") or [], ensure_ascii=False)}
        with self.connect() as con:
            con.execute(f"INSERT OR REPLACE INTO transactions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [row.get(c) for c in cols])


def _decode(row: dict) -> dict:
    if row.get("reasons"):
        row["reasons"] = json.loads(row["reasons"])
    else:
        row["reasons"] = []
    return row
