# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email send-time optimization platform. Predicts the best day/hour (168-slot grid: 7 days × 24 hours) to send marketing emails using a LightGBM binary classifier with SHAP explainability and isotonic probability calibration. Built on UCI Online Retail II data with synthetic campaign events.

## Commands

### First-Time Setup
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/fetch_data.py        # download UCI dataset to data/raw/
npm install                          # installs app/server + app/client workspaces
```

### ML Pipeline (run in order)
```bash
source .venv/bin/activate
python src/data_pipeline.py          # clean transactions + synthesize campaign events
python src/feature_engineering.py    # compute 20 features, RFM segments, 168-slot grids
python src/train.py                  # train LightGBM, calibrate probabilities, save models/
```

### Running Services (three terminals)
```bash
uvicorn src.api.main:app --port 8000 --reload   # FastAPI ML service
node app/server/index.js                          # Express proxy (port 3001)
cd app/client && npm run dev                      # Vite React frontend (port 5173)
```

### Testing
```bash
./scripts/integration_smoke.sh   # boots all 3 services, validates HTTP contracts, shuts down
```
No unit test framework is configured. The smoke test is the only automated test.

### Build Frontend
```bash
cd app/client && npm run build   # output: app/client/dist/
```

## Architecture

**Three-tier stack:**
- **FastAPI** (`src/api/main.py`, port 8000) — loads model artifacts at startup (lifespan), scores 168 time slots per customer, returns calibrated probabilities + SHAP explanations
- **Express proxy** (`app/server/index.js`, port 3001) — forwards `/api/*` to FastAPI, serves `docs/*.md` at `/api/docs/{docname}`
- **React frontend** (`app/client/src/`, port 5173) — component-based app with sidebar navigation and 3 views: Lookup, What-If, Segments. Uses TailwindCSS, Headless UI, Recharts, lucide-react, axios

**ML pipeline** (`src/`):
- `data_pipeline.py` — cleans raw transactions (UK-only filter), generates synthetic campaign events with open/click/purchase outcomes
- `feature_engineering.py` — computes 20 features (RFM + behavioral), builds per-customer 168-slot feature grids
- `train.py` — time-based train/val/test split (18m train, 6m test), LightGBM training, isotonic calibration, SHAP explainer

**Model artifacts** in `models/`: `lgbm_time_to_engage.pkl`, `shap_explainer.pkl`, `probability_calibrator.pkl`

**API endpoints** (FastAPI):
- `GET /api/customer/{id}` — profile + top-3 windows + 7×24 heatmap
- `POST /api/predict` — what-if prediction from manual feature input
- `GET /api/customers?page=&per_page=&segment=` — paginated customer list
- `GET /api/segments` — RFM segment summary

## Frontend Architecture

**Component-based structure** (`app/client/src/`):
- `App.jsx` — layout shell: sidebar + content area, view routing via `useState`, responsive mobile sidebar toggle
- `lib/api.js` — axios instance + all API call functions
- `lib/hooks.js` — custom hooks: `useCustomer`, `usePredict`, `useSegments`
- `lib/constants.js` — shared constants, heatmap color interpolation, error formatting
- `components/` — 12 reusable components: Sidebar, Card, StatCard, Heatmap, TopSlots, ShapChart, SliderControl, CustomerSelect (Headless UI Combobox), SegmentPicker (Headless UI Listbox), DataTable, ErrorBanner, LoadingSpinner
- `views/` — LookupView, WhatIfView, SegmentView

**Styling:** TailwindCSS 3 with custom theme (violet/pink/indigo palette) defined in `tailwind.config.js`. PostCSS configured in `postcss.config.js`. Custom heatmap grid styles in `index.css` `@layer components`.

**Responsive:** Sidebar collapses to hamburger toggle below `md` (768px) breakpoint with slide-in overlay and dark backdrop.

## Key Technical Details

- Python 3.11–3.13, Node.js 22+
- npm workspaces (root `package.json` manages `app/server` and `app/client`)
- Frontend: React 18, Vite 5, TailwindCSS 3, @headlessui/react 2, lucide-react, Recharts, no TypeScript
- No linting, formatting, Docker, or CI/CD configured
- Data files in `data/`, model artifacts in `models/`, build output `app/client/dist/`, and `docs/rendered/` are gitignored
- Pydantic schemas for API validation live in `src/api/schemas.py`
