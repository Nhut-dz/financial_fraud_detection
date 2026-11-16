"""Kiểm thử tích hợp. Cần đã chạy `python -m ffd.train` và `python -m ffd.init_db`."""
from __future__ import annotations

import pandas as pd
import pytest

from ffd.config import Paths

PATHS = Paths()
pytestmark = pytest.mark.skipif(
    not (PATHS.artifact_dir / "model_bundle.joblib").exists() or not PATHS.db_path.exists(),
    reason="Chưa có artifacts: chạy python -m ffd.train và python -m ffd.init_db",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from ffd.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def offline_scores():
    return pd.read_parquet(PATHS.artifact_dir / "test_scored.parquet").set_index("trans_num")


def _rescore(client, tx):
    body = {k: tx[k] for k in ("cc_num", "trans_time", "amt", "category", "merch_lat", "merch_long", "merchant", "trans_num")}
    r = client.post("/api/score", json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("kind", ["fraud", "legit"])
def test_online_scoring_matches_offline(client, offline_scores, kind):
    # Chấm lại giao dịch test qua API (tính đặc trưng từ SQLite) phải ra cùng điểm với lúc train (tính hàng loạt)
    samples = client.get("/api/samples", params={"kind": kind, "n": 15}).json()
    for tx in samples:
        online = _rescore(client, tx)
        offline = offline_scores.loc[tx["trans_num"]]
        assert online["fraud_score"] == pytest.approx(offline["fraud_score"], abs=1e-5)
        assert online["anomaly_score"] == pytest.approx(offline["anomaly_score"], abs=1e-5)
        assert online["risk_level"] == offline["risk_level"]


def test_health_and_metadata(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert len(client.get("/api/categories").json()) == 14
    report = client.get("/api/model").json()
    assert report["test"]["pr_auc"] > 0.9


def test_alerts_have_reasons(client):
    data = client.get("/api/alerts", params={"level": "High", "limit": 5}).json()
    assert data["total"] > 0
    for item in data["items"]:
        assert item["risk_level"] == "High"
        assert item["reasons"] and item["reasons"][0]["source"] == "fraud"


def test_unknown_card_returns_404(client):
    r = client.post("/api/score", json={"cc_num": 1, "amt": 10, "category": "misc_net", "merch_lat": 0, "merch_long": 0})
    assert r.status_code == 404


def test_invalid_category_returns_422(client):
    tx = client.get("/api/samples", params={"n": 1}).json()[0]
    r = client.post("/api/score", json={"cc_num": tx["cc_num"], "amt": 10, "category": "abc",
                                        "merch_lat": 0, "merch_long": 0})
    assert r.status_code == 422


def test_score_and_save_new_transaction(client):
    tx = client.get("/api/samples", params={"n": 1}).json()[0]
    body = {"cc_num": tx["cc_num"], "amt": 2500, "category": "shopping_net",
            "merch_lat": tx["merch_lat"], "merch_long": tx["merch_long"], "save": True}
    res = client.post("/api/score", json=body).json()
    assert res["saved"] and 0 <= res["fraud_score"] <= 1
    stored = client.get(f"/api/transactions/{res['trans_num']}").json()
    assert stored["source"] == "api" and stored["risk_level"] == res["risk_level"]
