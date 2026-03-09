# m04_mmm Session Log

## Session A — Phase 1 (Research) + Phase 2 (Data Engineering) — 2026-03-06

### Completed

**Phase 1: Research**
- Created `project_state.json` with model_category=regression, model_type=PyMC-Marketing Bayesian MMM
- Wrote `docs/research_brief.md` — MMM theory, framework comparison (PyMC-Marketing vs Robyn vs LightweightMMM vs Meridian), decision rationale

**Phase 2: Data Engineering**
- `src/data_generator.py` — synthetic data generator with known true parameters (adstock alphas, saturation lambdas, betas)
- `src/feature_engineering.py` — adds trend + Fourier seasonality; exports CHANNEL_COLUMNS, CONTROL_COLUMNS
- `scripts/fetch_data.py` — thin wrapper to generate and save data
- `docs/data_dictionary.md` — column definitions for raw, processed, and true_params
- `notebooks/01_eda_mmm.ipynb` — 8-section EDA notebook

### Data Verified
- `data/synthetic/mmm_weekly_data.csv`: 208 rows, 10 columns
- `data/synthetic/true_params.json`: all ground-truth parameters
- `data/processed/mmm_features.csv`: 208 rows, 15 columns (with trend + Fourier)
- Revenue range: 166,531 - 331,499 EUR/week (realistic)

### Next: Phase 3 (Model Building)

This is the heaviest phase — MCMC sampling with PyMC-Marketing (5-15 min).

Files to create:
- `src/train.py` — fit MMM, validate convergence, export pre-computed JSONs
- `src/export_precomputed.py` — extract decomposition, ROAS, response curves, adstock, simulator params
- `src/budget_optimizer.py` — standalone scipy optimizer (fallback)
- `notebooks/02_model_training.ipynb`
- `notebooks/03_model_diagnostics.ipynb`
- `models/precomputed/*.json` (6 files)
- `models/metadata.json`

---

## Session B — Phase 3 (Model Building) — 2026-03-07 to 2026-03-08

### Completed

**Phase 3: Model Building**

**Training pipeline (`src/train.py`):**
- PyMC-Marketing MMM with GeometricAdstock(l_max=8) + LogisticSaturation + yearly_seasonality=2
- MCMC: 4 chains x 1000 draws, pymc sampler (numpyro unavailable on macOS ARM)
- Train/test split: 156 weeks train, 52 weeks test
- Convergence: max R-hat 1.007, min ESS bulk 2032, min ESS tail 1384
- Out-of-sample: MAPE 3.9%, R2 0.91
- Parameter recovery: 9/10 within 94% HDI (only TV adstock alpha missed)

**Supporting modules:**
- `src/export_precomputed.py` — JSON export functions for 6 artifact types
- `src/budget_optimizer.py` — standalone scipy-based budget optimizer

**Pre-computed JSON artifacts (`models/precomputed/`):**
- `decomposition.json` (44KB) — 156 weekly channel contributions + totals + percentages
- `roas.json` — ROAS with 94% HDI per channel
- `response_curves.json` (22KB) — 50-point spend-vs-contribution curves
- `adstock.json` — decay vectors + alpha estimates per channel
- `simulator_params.json` — point estimates for FastAPI /api/simulate
- `optimal_allocation.json` — current vs optimal budget allocation

**Other artifacts:**
- `models/metadata.json` — convergence stats, MAPE, R2, parameter recovery
- `models/mmm_trace/trace.nc` (107MB) — full ArviZ InferenceData

**Notebooks:**
- `notebooks/02_model_training.ipynb` — 27 cells: data loading, split, MMM config, fitting, convergence, out-of-sample, parameter recovery, decomposition
- `notebooks/03_model_diagnostics.ipynb` — 17 cells: decomposition (stacked area + pie), ROAS, response curves, adstock decay, parameter recovery, budget optimisation

### Key Technical Notes
- PyMC-Marketing v0.18: `mmm.fit_result` is xarray Dataset (posterior), `mmm.idata` is ArviZ InferenceData
- MMM object is not picklable — API serves pre-computed JSONs, not live model
- `_fallback_export()` in train.py handles export_precomputed.py failures gracefully

### Next: Phase 4 (App Building)

Files to create:
- `src/api/main.py` — FastAPI with lifespan loading pre-computed JSONs
- `src/api/schemas.py` — Pydantic models
- `app/server/index.js` — Express proxy (port 3004 -> 8004)
- `app/client/` — React + Vite + TailwindCSS (port 5176)
- 4 views: Decomposition, Channel Performance, Budget Simulator, Optimal Allocation

---

## Session C — Phase 4 (App Building) — 2026-03-08

### Completed

**Phase 4: App Building**

**FastAPI Backend (`src/api/main.py`):**
- Lifespan loads all 7 pre-computed JSON files at startup
- 6 GET endpoints serving decomposition, ROAS, response curves, adstock, optimal allocation, metadata
- POST `/api/simulate` — computes predicted revenue from proposed channel spends using adstock/saturation/calibrated betas
- Saturation warnings (moderate > 0.6, high > 0.8)
- Pydantic v2 schemas in `src/api/schemas.py`

**Express Proxy (`app/server/index.js`):**
- Port 3004 → FastAPI 8004
- Forwards all 8 API routes

**React Frontend (`app/client/`):**
- 4 views: Decomposition, Channel Performance, Budget Simulator, Optimal Allocation
- Recharts visualisations: stacked AreaChart, PieChart, BarChart with error bars, LineCharts, grouped BarChart
- Budget lock toggle: adjusting one channel proportionally scales others to maintain total budget
- Saturation warning badges (amber/red)
- Channel colour palette consistent across all charts
- TailwindCSS styling matching m02 brand system

**Verification:**
- `npm run build` — passed (1.4s)
- `/health` — returns OK with channels list
- `/api/roas` — returns 5 channels with HDI intervals
- `/api/simulate` — returns predicted revenue with channel contributions
- Common Bugs Checklist — all items pass

### Next: Phase 5 (Documentation)

Files to create:
- `docs/model_card.md` — purpose, users, data, metrics, limitations
- `docs/model_documentation.md` — comprehensive technical documentation
- `docs/executive_deck.md` — Marp deck, regression category, light theme
- `README.md` — quick start with correct ports

### Resume Command
```
/analytics-project "marketing mix model" --project-dir /Users/aayan/zzLearnAndCreate/MarketingAnalytics/m04_mmm
```

---

## Session D — Phase 5 (Documentation) — 2026-03-08

### Completed

**Phase 5: Documentation**

**Technical Documentation:**
- `docs/model_card.md` — already existed from Phase 4; reviewed and confirmed complete (purpose, users, data, metrics, limitations, ethical considerations)
- `docs/model_documentation.md` — comprehensive 11-section technical report covering: executive summary, literature review (Jin et al. 2017, PyMC-Marketing, convergence diagnostics, Meta Robyn), data description, feature engineering, methodology (Bayesian MMM theory, MCMC config, frequentist comparison), model performance, channel decomposition and ROAS, response curves and saturation, API reference (8 endpoints), deployment architecture, future work

**Executive Deck:**
- `docs/executive_deck.md` — 15-slide Marp presentation with `theme: default` (regression category light styling)
- Slides: title, executive summary, business problem, solution overview, dataset, methodology, ROAS rankings, model performance, channel decomposition, ROI estimate, response curves, application views, budget recommendations, next steps, thank you
- Uses actual metrics from metadata.json and ROAS data (not invented numbers)
- Business-audience framing: media spend efficiency, channel attribution, diminishing returns

**README:**
- `README.md` — quick start guide with correct ports (8004/3004/5176), project structure, API endpoint table, retraining instructions

### Next: Phase 6 (App Testing with Playwright)

Resume: `/analytics-project "marketing mix model" --project-dir /Users/aayan/zzLearnAndCreate/MarketingAnalytics/m04_mmm`

---

## Session E — Phase 6 (App Testing with Playwright) — 2026-03-08

### Completed

**Phase 6: App Testing**

All three servers started successfully (FastAPI :8004, Express :3004, Vite :5176).

**View Test Results:**

| View | Status | Notes |
|------|--------|-------|
| Decomposition | PASS | 4 stat cards (Total Revenue, Media Contribution, Top Channel, Avg Weekly Revenue), stacked area chart with 6-source legend, pie chart |
| Channel Performance | PASS | ROAS bar chart with 94% HDI error bars, 5 response curve mini-charts, 5 adstock decay charts with alpha estimates |
| Budget Simulator | PASS | 5 channel sliders, predicted revenue updates on slider change (TV 19.7k→35k caused EUR 248,963→251,042, +0.8%), channel contributions bar chart, lock budget toggle |
| Optimal Allocation | PASS | Current vs optimal grouped bar chart, recommendations table (5 channels), revenue lift banner. Fixed inconsistent optimal_revenue placeholder |

**Bug Found & Fixed:**
- `optimal_allocation.json` had a placeholder `optimal_revenue` (1.05x current) that didn't match actual allocation data. Fixed both the precomputed JSON and `train.py` to compute lift from simulator params when allocations differ.

**Console Errors:** None (only favicon 404 on initial load, acceptable)

### All Phases Complete
