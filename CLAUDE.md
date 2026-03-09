# CLAUDE.md — MarketingAnalytics Workspace

This file provides guidance to Claude Code when working anywhere in this repository.

## Workspace Structure

```
/Users/aayan/MarketingAnalytics/
├── .venv/                    ← single shared Python venv for ALL models
├── requirements.txt          ← consolidated deps for all models
├── shared/data/raw/          ← datasets shared between models
├── m01_time_to_engage/       ← reference implementation (time-slot classification)
├── m02_clv/                  ← LightGBM regression, BG/NBD baseline
├── m03_churn/                ← LightGBM classification, isotonic calibration
├── m04_mmm/                  ← Bayesian MMM (PyMC-Marketing), pre-computed JSONs
└── m05_next_best_offer/
```

**Always use the root venv:**
```bash
/Users/aayan/MarketingAnalytics/.venv/bin/python
/Users/aayan/MarketingAnalytics/.venv/bin/pip
```

## Port Conventions

| Model | FastAPI | Express | Vite |
|-------|---------|---------|------|
| m01   | 8001    | 3001    | 5173 |
| m02   | 8002    | 3002    | 5174 |
| m03   | 8003    | 3003    | 5175 |
| m04   | 8004    | 3004    | 5176 |
| m05   | 8005    | 3005    | 5177 |

## Running a Model's Services

Always run uvicorn **from the model's own directory** (not the workspace root):

```bash
cd m0N_<topic>
/Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port 800N --reload
```

```bash
cd m0N_<topic>/app/server && npm start          # Express proxy
cd m0N_<topic>/app/client && npm run dev        # Vite frontend
```

## Completed Models

| Model | Status | Notes |
|-------|--------|-------|
| m01_time_to_engage | ✅ Complete | Reference implementation |
| m02_clv | ✅ Complete | LightGBM Regressor MAE=£1,228, top-decile lift=5.33x, Playwright-tested |
| m03_churn | ✅ Complete | LightGBM AUC=0.7883, top-20% lift=2.29x, Playwright-tested |
| m04_mmm | ✅ Complete | Bayesian MMM (PyMC-Marketing), MAPE=3.9%, R²=0.91, Playwright-tested |
| m05_next_best_offer | — | Not yet built |

## Architecture Conventions (all models)

**Three-tier stack:**
- **FastAPI** (`src/api/main.py`) — loads model artifacts at startup via lifespan, scores all entities at startup and caches in `app.state`
- **Express proxy** (`app/server/index.js`) — forwards `/api/*` to FastAPI
- **React + Vite** (`app/client/src/`) — TailwindCSS, Recharts, Headless UI, axios; 3 views: Lookup, What-If, Segments

**ML pipeline** (`src/`):
- `data_pipeline.py` — load_raw(), clean(), save_processed()
- `feature_engineering.py` — computes features; exports `FEATURE_COLUMNS` constant (shared with API)
- `train.py` — stratified split, training, isotonic calibration, SHAP, threshold optimization, saves to `models/`

## Critical Code Patterns

### sys.path fix in main.py (required for module resolution)

```python
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
```

### API response field naming (schemas.py)

- `PaginatedResponse`: use `items` (list) and `pages` (int) — **not** `customers`/`total_pages`
- Entity detail endpoint (`GET /api/<entity>/{id}`): return `churn_probability`, `risk_tier`, `shap_factors` at the **top level** — **not** nested under a `"prediction"` key
- `PredictRequest`: include snake_case convenience fields (`contract_type`, `internet_service`, `payment_method`) that map to one-hot columns server-side

### api.js field mapping (frontend)

- `fetchAllCustomers`: use `first.items` and `first.pages`
- `useSegments` hook: use `resp.items`
- `fetchPrediction`: map PascalCase UI keys → snake_case API params: `Contract→contract_type`, `InternetService→internet_service`, `PaymentMethod→payment_method`
- axios `baseURL`: use absolute `"http://localhost:<express_port>"` (not relative)

### LookupView rendering condition

```jsx
const riskTier = data?.risk_tier   // read directly, NOT data?.prediction?.risk_tier
// render when: profile && riskTier  (NOT profile && prediction)
```

### CLV-specific API patterns (m02_clv)

- Predict endpoint returns `predicted_clv`, `clv_segment`, `shap_factors` (no `risk_tier` / churn fields)
- `GET /api/portfolio` — returns all customers for scatter plot (recency vs CLV)
- Cold-start fallback: customers with < 2 purchases get median CLV (£622)
- WhatIfView uses numeric sliders (recency_days, frequency, avg_order_value, days_since_first, purchase_velocity), not dropdowns
- BG/NBD lifetimes fix: `recency=tenure_days`, `T=tenure_days+recency_days`

### MMM-specific patterns (m04_mmm)

- MMM object is **not picklable** — API serves pre-computed JSON files, not a live model
- Pre-computed artifacts in `models/precomputed/`: `decomposition.json`, `roas.json`, `response_curves.json`, `adstock.json`, `simulator_params.json`, `optimal_allocation.json`
- `POST /api/simulate` computes predicted revenue from proposed channel spends using adstock/saturation/calibrated betas
- Frontend has 4 views: Decomposition, Channel Performance, Budget Simulator, Optimal Allocation (not the standard Lookup/What-If/Segments)
- `feature_engineering.py` exports `CHANNEL_COLUMNS`, `CONTROL_COLUMNS`, `TARGET_COLUMN` (not `FEATURE_COLUMNS`)
- `train.py` uses `_fallback_export()` when `export_precomputed.py` fails — generates equivalent JSONs from raw posterior samples
- PyMC-Marketing v0.18: `mmm.fit_result` is xarray Dataset (posterior), `mmm.idata` is ArviZ InferenceData

## Common Bugs Checklist

| Bug | Symptom | Fix |
|-----|---------|-----|
| Customer dropdown empty | "No customers found" after typing | `fetchAllCustomers` must use `first.items` / `first.pages` |
| Segment table empty | Segment view shows no rows | `useSegments` must use `resp.items` |
| Profile panel never appears | Selecting entity shows nothing | `LookupView` reads `data?.risk_tier` directly; condition on `riskTier` not `prediction` |
| Simulator gauge stuck | Dropdown change doesn't update score | `fetchPrediction` must map PascalCase→snake_case |
| `ModuleNotFoundError: No module named 'src'` | Startup crash | Add `sys.path.insert` at top of `main.py`; run uvicorn from model dir |
| `pyarrow` not found | Pipeline crashes on parquet save | `pip install pyarrow` explicitly even if in requirements.txt |
| `\ in f-string` SyntaxWarning | Python 3.12+ warning/error | Use intermediate variables instead of backslash inside f-strings |
| `MODULE_TYPELESS_PACKAGE_JSON` | Express ESM warning | Add `"type": "module"` to `app/server/package.json` |

## Reference Implementation

Use `m01_time_to_engage/` as the canonical pattern source for:
- Code structure, naming, API design
- React component library and TailwindCSS theme
- Notebook naming conventions (`01_eda_`, `02_model_training`, `03_model_diagnostics`)
- Document formats (`model_card.md`, `model_documentation.md`, `executive_deck.md`)
