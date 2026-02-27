# Marketing Analytics

A portfolio of end-to-end marketing analytics models — each with a trained ML pipeline, FastAPI backend, and React dashboard.

## Models

| # | Model | Status | Algorithm | Key Metric |
|---|-------|--------|-----------|------------|
| m01 | [Time to Engage](m01_time_to_engage/) | ✅ Complete | LightGBM Classifier | Reference implementation |
| m02 | [Customer Lifetime Value](m02_clv/) | ✅ Complete | LightGBM Regressor | MAE=£1,228, top-decile lift=5.33x |
| m03 | [Churn Prediction](m03_churn/) | ✅ Complete | LightGBM Classifier | AUC=0.788, top-20% lift=2.29x |
| m04 | Marketing Mix Modeling | — | TBD | — |
| m05 | Next Best Offer | — | TBD | — |

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

## Setup

```bash
# Single shared Python venv at repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See each model's README for quick-start instructions.

## Building New Models

Use the `/analytics-project` skill (see [skills/analytics-project.md](skills/analytics-project.md)) to scaffold a new model from scratch. It orchestrates 5 phases: research → data engineering → model building → app building → documentation.
