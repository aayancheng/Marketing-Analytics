# m02_clv — Customer Lifetime Value Prediction

Predicts 12-month forward customer lifetime value (CLV) in GBP using the UCI Online Retail II dataset. LightGBM Regressor with 22 RFM + behavioral features, SHAP explainability, and a React dashboard.

## Key Results

| Metric | LightGBM | BG/NBD Baseline | Naive Mean |
|--------|----------|-----------------|------------|
| MAE    | £1,228   | £1,426          | £2,551     |
| RMSE   | £3,063   | £3,019          | £6,774     |
| Spearman r | 0.60 | —               | —          |
| Top-decile lift | 5.33x | —          | —          |

## Quick Start

```bash
# 1. Activate venv
source /Users/aayan/MarketingAnalytics/.venv/bin/activate

# 2. Run data pipeline + feature engineering (if not already done)
cd /Users/aayan/MarketingAnalytics/m02_clv
python src/data_pipeline.py
python src/feature_engineering.py
python src/train.py

# 3. Start FastAPI (from model directory)
python -m uvicorn src.api.main:app --port 8002 --reload &

# 4. Start Express proxy
cd app/server && npm start &

# 5. Start React dev server
cd ../client && npm run dev
# Open http://localhost:5174
```

## Ports

| Service | Port |
|---------|------|
| FastAPI | 8002 |
| Express | 3002 |
| Vite    | 5174 |

## Project Structure

```
m02_clv/
├── src/
│   ├── data_pipeline.py        # Load, clean, temporal split
│   ├── feature_engineering.py  # 22 RFM + behavioral features
│   ├── train.py                # LightGBM + BG/NBD + SHAP
│   └── api/
│       ├── main.py             # FastAPI endpoints
│       └── schemas.py          # Pydantic models
├── app/
│   ├── server/                 # Express proxy
│   └── client/                 # React + TailwindCSS + Recharts
├── models/                     # Trained artifacts
├── data/processed/             # Pipeline outputs
├── notebooks/                  # EDA + training + diagnostics
└── docs/                       # Model card, documentation, deck
```

## App Views

1. **Customer Lookup** — Search by ID, view CLV prediction + SHAP explanations
2. **CLV Simulator** — What-if sliders for recency, frequency, spend, velocity
3. **Portfolio Explorer** — Scatter plot of all customers, segment filtering, top-20 table
