"""FastAPI cho hệ thống FFD. Chạy: uvicorn ffd.api.main:app --reload (trong thư mục backend/)."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import Paths
from ..history import HistoryStore
from ..scoring import Scorer, TransactionInput, UnknownCardError

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = Paths()
    for p, hint in [(paths.artifact_dir / "model_bundle.joblib", "python -m ffd.train"),
                    (paths.db_path, "python -m ffd.init_db")]:
        if not p.exists():
            raise RuntimeError(f"Thiếu {p}. Chạy `{hint}` trước.")
    store = HistoryStore(paths.db_path)
    STATE["store"] = store
    STATE["scorer"] = Scorer(paths.artifact_dir / "model_bundle.joblib", store)
    with open(paths.artifact_dir / "report.json", encoding="utf-8") as f:
        STATE["report"] = json.load(f)
    yield
    STATE.clear()


app = FastAPI(title="Financial Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FFD_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- schemas
class TransactionRequest(BaseModel):
    cc_num: int = Field(..., description="Số thẻ đã có trong lịch sử")
    trans_time: datetime | None = Field(None, description="Mặc định: ngay sau giao dịch gần nhất của thẻ")
    amt: float = Field(..., gt=0)
    category: str
    merch_lat: float = Field(..., ge=-90, le=90)
    merch_long: float = Field(..., ge=-180, le=180)
    merchant: str | None = None
    trans_num: str | None = Field(None, description="Truyền khi chấm lại một giao dịch đã có, để loại nó khỏi lịch sử")
    save: bool = Field(False, description="Lưu giao dịch vào lịch sử sau khi chấm")


class Reason(BaseModel):
    source: Literal["fraud", "anomaly"]
    feature: str
    label: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    trans_num: str
    cc_num: int
    trans_time: str
    amt: float
    category: str
    fraud_score: float
    anomaly_score: float
    risk_level: Literal["High", "Medium", "Low"]
    alert: bool
    reasons: list[Reason]
    features: dict[str, float]
    history_size: int
    saved: bool


# ---------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model")
def model_info():
    return STATE["report"]


@app.get("/api/categories")
def categories():
    return STATE["scorer"].categories


@app.get("/api/stats")
def stats():
    return STATE["store"].stats()


@app.post("/api/score", response_model=ScoreResponse)
def score(req: TransactionRequest):
    store: HistoryStore = STATE["store"]
    scorer: Scorer = STATE["scorer"]
    if req.category not in scorer.categories:
        raise HTTPException(422, f"category không hợp lệ. Chọn một trong: {', '.join(scorer.categories)}")
    t = pd.Timestamp(req.trans_time) if req.trans_time else None
    if t is None:
        recent = store.recent_transactions(req.cc_num, limit=1)
        t = pd.Timestamp(recent[0]["trans_time"]) + pd.Timedelta(minutes=5) if recent else pd.Timestamp.now().floor("s")
    tx = TransactionInput(cc_num=req.cc_num, trans_time=t.tz_localize(None) if t.tzinfo else t, amt=req.amt,
                          category=req.category, merch_lat=req.merch_lat, merch_long=req.merch_long,
                          merchant=req.merchant, trans_num=req.trans_num)
    try:
        result = scorer.score(tx)
    except UnknownCardError:
        raise HTTPException(404, f"Không có thẻ {req.cc_num} trong lịch sử")
    if req.save:
        store.insert_scored({**result, "is_fraud": None, "source": "api"})
    return {**result, "saved": req.save}


@app.get("/api/cards/{cc_num}")
def card(cc_num: int, limit: int = Query(20, ge=1, le=200)):
    store: HistoryStore = STATE["store"]
    info = store.card(cc_num)
    if info is None:
        raise HTTPException(404, f"Không có thẻ {cc_num}")
    return {"card": info, "transactions": store.recent_transactions(cc_num, limit)}


@app.get("/api/transactions/{trans_num}")
def transaction(trans_num: str):
    tx = STATE["store"].transaction(trans_num)
    if tx is None:
        raise HTTPException(404, "Không tìm thấy giao dịch")
    return tx


@app.get("/api/alerts")
def alerts(level: Literal["High", "Medium"] | None = None,
           limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    total, items = STATE["store"].alerts(level, limit, offset)
    return {"total": total, "items": items}


@app.get("/api/samples")
def samples(kind: Literal["fraud", "legit", "random"] = "random", n: int = Query(10, ge=1, le=50)):
    return STATE["store"].samples(kind, n)
