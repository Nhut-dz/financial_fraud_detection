# Financial Fraud Detection (FFD)

Hệ thống phát hiện giao dịch thẻ tín dụng gian lận trên dữ liệu **Sparkov** (Kaggle "Credit Card Transactions Fraud Detection Dataset").

- **Fraud Detection (chính):** XGBoost + class weights → xác suất gian lận.
- **Anomaly Detection (hỗ trợ):** Isolation Forest → mức bất thường so với hành vi train.
- **Mức rủi ro:** High / Medium / Low + lý do cho từng giao dịch bị cảnh báo.
- **Ứng dụng:** FastAPI (chấm điểm online từ lịch sử thẻ trong SQLite) + React/TypeScript UI.

## Kiến trúc

```
CSV (data/) ──► python -m ffd.train   ──► artifacts/model_bundle.joblib, report.json, test_scored.parquet
            └─► python -m ffd.init_db ──► artifacts/ffd.sqlite (thẻ + lịch sử giao dịch + điểm test)

React UI ──HTTP──► FastAPI ──► Scorer: lấy lịch sử thẻ (SQLite) + giao dịch mới
                                   → build_features (cùng hàm với lúc train)
                                   → XGBoost + Isolation Forest → mức rủi ro + lý do
```

Đặc trưng online được tính bằng **đúng hàm `build_features`** dùng khi train, nên điểm chấm qua API trùng với điểm offline (có test kiểm tra).

### Quy tắc mức rủi ro

| Mức | Điều kiện | Ý nghĩa |
|---|---|---|
| High | Điểm fraud ≥ ngưỡng của ngân sách cảnh báo 0,5% | Nghi vấn gian lận |
| Medium | Không thuộc High, mức bất thường thuộc top 1% so với train | Bất thường, cần xem xét (không đồng nghĩa gian lận) |
| Low | Còn lại | — |

Ngân sách được chọn trên validation: ngân sách nhỏ nhất đạt Recall ≥ 80%.

## Kết quả trên tập test (đánh giá một lần)

| Chỉ số | Giá trị |
|---|---|
| PR-AUC | 0,960 |
| ROC-AUC | 0,9993 |
| Recall mức High | 82,9% |
| Precision mức High | 97,0% |
| F1 mức High | 0,894 |

Kết luận trung thực: đặc trưng địa lý không có giá trị trên Sparkov (merchant nằm trong ±1° quanh nhà); kết hợp Isolation Forest không tăng số gian lận bắt được ở cùng số cảnh báo, nên Medium chỉ là lớp "cần xem xét".

## Cài đặt và chạy

Yêu cầu: Python 3.10+, Node.js 20+.

### 1. Dữ liệu

Tải `fraudTrain.csv` và `fraudTest.csv` từ Kaggle, đặt vào thư mục `data/`.

### 2. Backend

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
python -m ffd.train      # ~2 phút: huấn luyện, chọn ngưỡng, đánh giá test
python -m ffd.init_db    # ~30 giây: nạp lịch sử giao dịch vào SQLite
uvicorn ffd.api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/health` | Trạng thái |
| GET | `/api/model` | Báo cáo mô hình, ngưỡng, kết quả |
| GET | `/api/categories` | Danh sách loại dịch vụ |
| GET | `/api/stats` | Thống kê DB |
| POST | `/api/score` | Chấm điểm một giao dịch (tuỳ chọn `save`) |
| GET | `/api/cards/{cc_num}` | Thông tin + lịch sử gần nhất của thẻ |
| GET | `/api/transactions/{trans_num}` | Chi tiết một giao dịch |
| GET | `/api/alerts?level=High&limit=&offset=` | Danh sách cảnh báo |
| GET | `/api/samples?kind=fraud\|legit\|random` | Giao dịch mẫu từ tập test |

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/score -H 'Content-Type: application/json' \
  -d '{"cc_num": 60416207185, "amt": 950, "category": "shopping_net", "merch_lat": 43.2, "merch_long": -108.5}'
```

Biến môi trường (tuỳ chọn): `FFD_DATA_DIR`, `FFD_ARTIFACT_DIR`, `FFD_DB_PATH`, `FFD_CORS_ORIGINS`.

### 3. Frontend

```bash
cd frontend
cp .env.example .env     # chỉnh VITE_API_URL nếu API không ở localhost:8000
npm install
npm run dev              # http://localhost:5173
```

### 4. Test

```bash
pip install -r backend/requirements-dev.txt
cd backend && pytest
```

Test kiểm tra: điểm online khớp offline, metadata mô hình, lý do cảnh báo, lỗi 404/422, chấm và lưu giao dịch.

## Cấu trúc

```
backend/ffd/
  config.py     cấu hình, đường dẫn, tham số
  data.py       đọc dữ liệu, bỏ cột định danh/nhân khẩu học
  features.py   đặc trưng hành vi theo thẻ (cửa sổ thời gian, số tiền, địa lý, ban đêm)
  models.py     XGBoost, Isolation Forest, chuẩn hoá điểm bất thường
  risk.py       ngân sách cảnh báo, mức rủi ro, đánh giá
  explain.py    lý do: SHAP top-3 (fraud), độ lệch đặc trưng (anomaly)
  train.py      pipeline huấn luyện
  history.py    kho lịch sử SQLite
  init_db.py    nạp dữ liệu vào SQLite
  scoring.py    chấm điểm online
  api/main.py   FastAPI
frontend/src/   React + TypeScript (Chấm điểm, Cảnh báo, Mô hình & đánh giá)
```
