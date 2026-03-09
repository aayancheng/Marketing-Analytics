# Marketing Analytics

A portfolio of end-to-end marketing analytics models — each with a trained ML pipeline, FastAPI backend, and React dashboard.

## Models

| # | Model | Status | Algorithm | Key Metric |
|---|-------|--------|-----------|------------|
| m01 | [Time to Engage](m01_time_to_engage/) | ✅ Complete | LightGBM Classifier | Reference implementation — optimal email send-time prediction across 168 weekly slots |
| m02 | [Customer Lifetime Value](m02_clv/) | ✅ Complete | LightGBM Regressor + BG/NBD | MAE=£1,228, top-decile lift=5.33x |
| m03 | [Churn Prediction](m03_churn/) | ✅ Complete | LightGBM Classifier | AUC=0.788, top-20% lift=2.29x, top-20% capture=46% |
| m04 | [Marketing Mix Model](m04_mmm/) | ✅ Complete | Bayesian MMM (PyMC-Marketing) | MAPE=3.9%, R²=0.91, 5-channel decomposition |
| m05 | Next Best Offer | — | TBD | — |

## Model Details

### m01 — Time to Engage
Predicts the optimal email send-time for each customer across a 7×24 (168-slot) weekly grid. LightGBM classifier with per-slot scoring, SHAP explanations, and a heatmap dashboard.

### m02 — Customer Lifetime Value
Predicts 12-month customer value using a LightGBM Regressor with BG/NBD baseline. Features include RFM metrics, purchase velocity, and tenure. Dashboard includes portfolio scatter, what-if sliders, and segment explorer.

### m03 — Churn Prediction
Binary churn classifier using LightGBM with isotonic calibration and cost-optimal threshold optimization. Features include contract type, tenure, charges, and service usage. Dashboard includes risk lookup, what-if simulator, and segment explorer.

### m04 — Marketing Mix Model
Bayesian Media Mix Model using PyMC-Marketing with GeometricAdstock + LogisticSaturation across 5 channels (TV, OOH, Print, Facebook, Search). Serves pre-computed JSON artifacts (no live MCMC). Dashboard includes revenue decomposition, channel ROAS with HDI, response curves, budget simulator, and optimal allocation.

## Stack

Each model follows the same three-tier architecture:

- **FastAPI** — ML inference + REST API (`src/api/`)
- **Express** — proxy layer (`app/server/`)
- **React + Vite** — dashboard with TailwindCSS + Recharts (`app/client/`)

## Port Conventions

| Model | FastAPI | Express | Vite |
|-------|---------|---------|------|
| m01   | 8001    | 3001    | 5173 |
| m02   | 8002    | 3002    | 5174 |
| m03   | 8003    | 3003    | 5175 |
| m04   | 8004    | 3004    | 5176 |
| m05   | 8005    | 3005    | 5177 |

## Quick Start

```bash
# Single shared Python venv at repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run any model (replace N with model number)
cd m0N_<topic>
../.venv/bin/python -m uvicorn src.api.main:app --port 800N &
cd app/server && npm start &
cd app/client && npm run dev
# Open http://localhost:517X
```

See each model's README for model-specific instructions.

## Building New Models

Use the `/analytics-project` skill (see [skills/analytics-project.md](skills/analytics-project.md)) to scaffold a new model from scratch. It orchestrates 6 phases: research → data engineering → model building → app building → documentation → app testing.
