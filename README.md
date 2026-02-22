# TimeToEngage — Email Send-Time Optimizer

Predicts the optimal day and hour to send marketing emails for maximum engagement. Uses a LightGBM classifier across a 168-slot grid (7 days x 24 hours) with SHAP explainability and isotonic probability calibration.

Built on the [UCI Online Retail II](https://doi.org/10.24432/C5CG6D) dataset with synthetic campaign events.

## Features

- **Customer Lookup** — Search any customer to see their top-3 send windows, engagement heatmap, and RFM profile
- **What-If Simulator** — Adjust customer features with sliders to explore predicted engagement in real time, with SHAP feature contribution charts
- **Segment Explorer** — Browse RFM segments with summary stats, scatter plots, and customer tables

## Architecture

```
React (Vite, port 5173)  →  Express proxy (port 3001)  →  FastAPI ML service (port 8000)
```

- **FastAPI** — Loads model artifacts at startup, scores 168 time slots per request, returns calibrated probabilities + SHAP explanations
- **Express proxy** — Forwards `/api/*` to FastAPI, serves markdown docs
- **React frontend** — Component-based with sidebar navigation, TailwindCSS (violet/pink/indigo theme), Headless UI, Recharts

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML | Python 3.11+, LightGBM, SHAP, scikit-learn, pandas |
| API | FastAPI, Pydantic, uvicorn |
| Proxy | Node.js, Express |
| Frontend | React 18, Vite 5, TailwindCSS 3, Headless UI, Recharts, lucide-react |

## Quick Start

### Prerequisites

- Python 3.11–3.13
- Node.js 22+

### Setup

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download UCI dataset
python scripts/fetch_data.py

# Node dependencies (installs both server + client workspaces)
npm install
```

### Build ML Pipeline

```bash
source .venv/bin/activate
python src/data_pipeline.py          # Clean transactions + synthesize campaign events
python src/feature_engineering.py    # Compute 20 features, RFM segments, 168-slot grids
python src/train.py                  # Train LightGBM, calibrate probabilities, save models
```

### Run the App

Start all three services (each in a separate terminal):

```bash
# Terminal 1 — FastAPI ML service
source .venv/bin/activate
uvicorn src.api.main:app --port 8000 --reload

# Terminal 2 — Express proxy
node app/server/index.js

# Terminal 3 — React frontend
cd app/client && npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

### Integration Smoke Test

Boots all 3 services, validates endpoint contracts and UI availability, then shuts everything down:

```bash
./scripts/integration_smoke.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/customer/{id}` | Customer profile + top-3 send windows + 7x24 heatmap |
| POST | `/api/predict` | What-if prediction from manual feature input |
| GET | `/api/customers?page=&per_page=&segment=` | Paginated customer list with optional segment filter |
| GET | `/api/segments` | RFM segment summary statistics |

## Project Structure

```
src/
  data_pipeline.py          # Data cleaning + synthetic event generation
  feature_engineering.py    # 20 features, RFM segments, 168-slot grids
  train.py                  # LightGBM training, calibration, SHAP
  api/
    main.py                 # FastAPI endpoints
    schemas.py              # Pydantic request/response models
app/
  server/index.js           # Express proxy
  client/src/
    App.jsx                 # Layout shell with sidebar routing
    components/             # 12 reusable UI components
    views/                  # LookupView, WhatIfView, SegmentView
    lib/                    # API client, hooks, constants
models/                     # Trained artifacts (gitignored)
data/                       # Raw + processed datasets (gitignored)
docs/                       # Model card, data dictionary, validation report
scripts/                    # Integration smoke test
notebooks/                  # EDA, training, diagnostics
```

## Data

Uses the UCI Online Retail II dataset (id=502).

> Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
