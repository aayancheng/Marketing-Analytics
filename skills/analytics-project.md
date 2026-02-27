# Analytics Project Builder

Use when invoked as `/analytics-project "<topic>" [--project-dir <path>]` to build a complete analytics project from scratch for any business domain, or to resume an in-progress project.

## Workspace Convention

All projects live inside `/Users/aayan/MarketingAnalytics/` as numbered model folders:

```
/Users/aayan/MarketingAnalytics/
├── .venv/                         ← single shared Python venv for ALL models
├── requirements.txt               ← consolidated deps for all models
├── shared/data/raw/               ← datasets shared between models
├── m01_time_to_engage/            ← reference implementation (use as pattern source)
├── m02_clv/
├── m03_churn/                     ← completed example
├── m04_mmm/
├── m05_next_best_offer/
└── model[N]_<topic>_brief.md
```

**Port convention** — assign based on model number N, no port conflicts:
| Model | FastAPI | Express | Vite |
|-------|---------|---------|------|
| m01   | 8001    | 3001    | 5173 |
| m02   | 8002    | 3002    | 5174 |
| m03   | 8003    | 3003    | 5175 |
| m04   | 8004    | 3004    | 5176 |
| m05   | 8005    | 3005    | 5177 |

**Reference implementation** — always read from `/Users/aayan/MarketingAnalytics/m01_time_to_engage/` for code patterns, component library, API design, and document formats.

## Overview

Orchestrates a 5-phase build + Playwright app test with session checkpointing:

| Phase | What Gets Built |
|-------|----------------|
| 1. Research | Domain brief, data source selection, model type decision |
| 2. Data Engineering | Data acquisition, EDA notebook, pipeline scripts |
| 3. Model Building | Training + diagnostics notebooks, trained model artifacts |
| 4. App Building | FastAPI backend + React frontend |
| 5. Documentation | Model card, technical docs, executive deck |
| 6. App Testing | Start all 3 servers, Playwright smoke test all views |

Progress is tracked in `<project_dir>/project_state.json`. Safe to stop and resume across multiple sessions.

---

## Step 1: Parse Arguments and Initialize State

Extract from the invocation:
- `topic` — the business domain string (e.g., "credit risk", "customer lifetime value")
- `project_dir` — if not provided, determine the next model number by listing `/Users/aayan/MarketingAnalytics/m*/` and default to `/Users/aayan/MarketingAnalytics/m0N_<topic-as-slug>/`

**Create project directory:**
```bash
mkdir -p <project_dir>/{data/{raw,processed,synthetic},notebooks,src/api,app/{server,client/src/{components,views,lib}},models,docs,scripts}
```

**Read or create `<project_dir>/project_state.json`:**

If the file does not exist, create it:
```json
{
  "topic": "<topic>",
  "project_dir": "<project_dir>",
  "created_at": "<today-YYYY-MM-DD>",
  "phases": {
    "research":         { "status": "pending", "completed_at": null },
    "data_engineering": { "status": "pending", "completed_at": null },
    "modeling":         { "status": "pending", "completed_at": null },
    "app_building":     { "status": "pending", "completed_at": null },
    "documentation":    { "status": "pending", "completed_at": null }
  },
  "decisions": {
    "model_type": null,
    "data_source": null,
    "feature_categories": [],
    "stack": {}
  },
  "ports": {
    "fastapi": <800N>,
    "express": <300N>,
    "vite":    <517N+2>
  }
}
```

**Announce current status** — list each phase with ✓ (completed) or → (pending/in_progress).

**Proceed to Step 2.**

---

## Step 2: Execute Next Pending Phase

Read `project_state.json`. For each phase in order (research → data_engineering → modeling → app_building → documentation), skip phases with `"status": "completed"`. Execute the first `"pending"` phase. After completion, update its status to `"completed"` with today's date. Continue to the next phase if token budget allows.

---

## Phase 1: Research

**Mark phase as `in_progress` in state file before starting.**

Dispatch 3 Task agents in a SINGLE message (parallel execution):

**Agent 1 — Domain Research** (`subagent_type: general-purpose`):
```
Research the business domain "<topic>" for a data science/ML project.

Find and summarize:
1. Domain overview: what business problem does this solve, how it creates value
2. Regulatory/compliance context (especially if financial, healthcare, or insurance)
3. Standard industry KPIs and success metrics used to evaluate models
4. Benchmark ML approaches: what model types are industry standard for this domain
5. 2-3 key papers or industry reports (title + why relevant)

Return structured markdown with clear H2 sections for each point.
```

**Agent 2 — Data Sources** (`subagent_type: general-purpose`):
```
Find available public datasets and APIs for "<topic>" analytics.

Research:
1. List 5 public datasets (UCI ML Repository, Kaggle, FRED, government open data, domain-specific)
   For each: name, URL, size estimate, key columns, license
2. Any relevant paid or freemium data APIs
3. Recommend the single best starting dataset with rationale (availability, richness, real-world relevance)

Return a markdown table of options followed by recommendation paragraph.
```

**Agent 3 — Existing Patterns** (`subagent_type: Explore`):
```
Explore /Users/aayan/MarketingAnalytics/m01_time_to_engage/ to document reusable patterns for a new analytics project.

Examine and summarize:
1. src/data_pipeline.py — data cleaning and synthesis patterns
2. src/feature_engineering.py — feature computation patterns
3. src/train.py — model training, calibration, SHAP artifact saving
4. src/api/main.py + schemas.py — FastAPI endpoint structure and response field naming
5. app/client/src/ — React component list and structure
6. notebooks/ — notebook naming convention and section structure
7. docs/model_card.md + model_documentation.md — document formats

For each file, note: purpose, key functions/components, what to adapt for a new domain.
```

**After all 3 agents complete:**

1. Synthesize results into `<project_dir>/docs/research_brief.md`:

```markdown
# Research Brief: <topic>

## Domain Overview
[from Agent 1]

## KPIs & Success Metrics
[from Agent 1]

## Data Sources
[from Agent 2 — table + recommendation]

## Model Candidates
[from Agent 1 — with recommendation based on table below]

## Recommended Stack
[derived from model type selection]

## Reference Patterns
[from Agent 3 — what to reuse from reference project]
```

2. Select model type using this table:

| Domain | Model Type | Key Libraries |
|--------|-----------|---------------|
| Credit risk / scorecard | XGBoost + WoE binning | xgboost, optbinning |
| Customer lifetime value | LightGBM Regressor (primary) + BG/NBD + Gamma-Gamma (baseline) | lightgbm, lifetimes, shap |
| Fraud detection | Isolation Forest + threshold optimization | scikit-learn, imbalanced-learn, shap |
| Churn prediction | Survival analysis or LightGBM | lifelines or lightgbm, shap |
| Marketing send-time | LightGBM 168-slot grid classifier | lightgbm, shap |
| Demand forecasting | XGBoost or Prophet | xgboost or prophet |
| Recommendation | Collaborative filtering or LightGBM ranking | implicit or lightgbm |
| Default | LightGBM + SHAP | lightgbm, shap |

3. Update `project_state.json` decisions block.

**CLV-specific note:** If model type is LightGBM Regressor (CLV), all downstream phases differ from classifiers:
- Phase 3: use `log1p()` target transform + `expm1()` inverse; metrics are MAE/RMSE/MAPE/Spearman; no threshold sweep; add BG/NBD baseline comparison
- Phase 4 FastAPI: predict endpoint returns `predicted_clv`, `clv_segment`, `shap_factors` (no `risk_tier`); add `/api/portfolio` endpoint (all customers for scatter plot)
- Phase 4 React: replace heatmap/gauge with scatter plot (recency vs CLV); WhatIfView uses numeric sliders (recency, frequency, AOV) not dropdowns

4. Create project directory skeleton (if not already created in Step 1).

**Mark phase as `completed` in state file with today's date. Proceed to Phase 2 if budget allows.**

---

## Phase 2: Data Engineering

**Mark phase as `in_progress`. Read `decisions` from `project_state.json`.**

Dispatch 2 Task agents in a SINGLE message (parallel):

**Agent 1 — Data Acquisition** (`subagent_type: Bash`):
```
Download the dataset: <decisions.data_source>

Reference pattern: /Users/aayan/MarketingAnalytics/m01_time_to_engage/scripts/fetch_data.py

Write a download script to <project_dir>/scripts/fetch_data.py
If Kaggle CLI is unavailable, generate a realistic synthetic dataset that:
- Matches the original dataset's schema, row count, and column distributions
- Preserves the key domain correlations (e.g., for churn: month-to-month contract → higher churn rate)
- Uses random_state=42 and a logistic/statistical model to generate labels realistically

Execute it to populate <project_dir>/data/raw/
Report: file name, size, row count, column list, class balance (if classification)
```

**Agent 2 — Pipeline, Features, and Notebooks** (`subagent_type: general-purpose`):
```
Write data pipeline, feature engineering scripts, EDA notebook, and data dictionary for "<topic>".

Reference implementations — read these files first:
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/src/data_pipeline.py
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/src/feature_engineering.py
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/notebooks/01_eda_and_synthesis.ipynb
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/docs/data_dictionary.md

Feature categories: <decisions.feature_categories>

Write to <project_dir>/:
1. src/data_pipeline.py — load_raw(), clean(), save_processed(), run_tests()
2. src/feature_engineering.py — compute features, FEATURE_COLUMNS constant (shared with API)
3. notebooks/01_eda_<topic_slug>.ipynb — distributions, target analysis, correlations
4. docs/data_dictionary.md — raw + engineered columns + target

Important: export FEATURE_COLUMNS as a module-level constant in feature_engineering.py.
Return complete file contents.
```

**After agents complete:**

Run the pipeline using the root venv:
```bash
/Users/aayan/MarketingAnalytics/.venv/bin/pip install pyarrow  # always install explicitly
/Users/aayan/MarketingAnalytics/.venv/bin/python <project_dir>/src/data_pipeline.py
/Users/aayan/MarketingAnalytics/.venv/bin/python <project_dir>/src/feature_engineering.py
```

Update root `/Users/aayan/MarketingAnalytics/requirements.txt` with any new model-specific deps.

**Mark phase as `completed`. Proceed to Phase 3 if budget allows.**

---

## Phase 3: Model Building

**Mark phase as `in_progress`. Read `decisions.model_type` from `project_state.json`.**

Dispatch 2 Task agents in a SINGLE message (parallel):

**Agent 1 — Training Script + Notebook** (`subagent_type: general-purpose`):
```
Write a training script and Jupyter notebook for <decisions.model_type> model on <topic>.

Reference:
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/src/train.py
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/notebooks/02_optimal_time_engagement_model.ipynb

Write to <project_dir>/:
1. src/train.py — includes: stratified split, baseline models, primary model, isotonic calibration,
   threshold optimization (cost sweep if classifier), SHAP TreeExplainer, save all artifacts to models/,
   auto-generate docs/validation_report.md
2. notebooks/02_model_training.ipynb — interactive version of train.py with inline plots

Artifacts to save to <project_dir>/models/:
- primary model pkl
- calibrator pkl
- shap_explainer pkl
- metadata.json (feature columns, metrics, optimal_threshold)

IMPORTANT: avoid backslash escapes inside f-strings (Python 3.12+ SyntaxWarning).
Use intermediate variables instead of f"\n{val}" patterns.

Return complete file contents.
```

**Agent 2 — Diagnostics Notebook** (`subagent_type: general-purpose`):
```
Write a model diagnostics Jupyter notebook for <decisions.model_type> on <topic>.

Reference: /Users/aayan/MarketingAnalytics/m01_time_to_engage/notebooks/03_client_engagement_diagnostics.ipynb

Sections for notebooks/03_model_diagnostics.ipynb:
1. Load artifacts and test set
2. Confusion matrix at default and cost-optimal thresholds
3. SHAP summary beeswarm + force plots for 3 examples (high/medium/low risk)
4. Business simulation (dollar value of top-N% targeting)
5. Failure mode analysis
6. Slice analysis by key feature segments

Return complete notebook JSON.
```

**After agents complete:**

Run training using root venv:
```bash
cd <project_dir>
/Users/aayan/MarketingAnalytics/.venv/bin/python src/train.py
```

Verify artifacts exist:
```bash
ls <project_dir>/models/
```

Record final metrics in `project_state.json` under a `"metrics"` key.

**Mark phase as `completed`. Proceed to Phase 4 if budget allows.**

---

## Phase 4: App Building

**Mark phase as `in_progress`.**

Read ports from `project_state.json` (fastapi_port, express_port, vite_port).

Dispatch 2 Task agents in a SINGLE message (parallel):

**Agent 1 — FastAPI Backend** (`subagent_type: general-purpose`):
```
Write a FastAPI backend for <topic> analytics.

Reference implementations — read these files first:
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/src/api/main.py
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/src/api/schemas.py
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/app/server/index.js

Write to <project_dir>/:
1. src/api/main.py — FastAPI with lifespan loading of all model artifacts.

   CRITICAL — add this sys.path fix at the top of main.py so it resolves src.api.schemas
   correctly when run from the model's own directory:
   ```python
   import sys, os as _os
   sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
   ```

   Endpoints:
   - GET /health → {"status":"ok","model":"<name>","<entity>s":<count>}
   - GET /api/<entity>s?page=&per_page=&segment= → PaginatedResponse
   - GET /api/<entity>/{id} → full profile + prediction + shap_factors (FLAT response, not nested)
   - POST /api/predict → what-if score from feature dict
   - POST /api/simulate → proposed_changes to existing entity → new score + delta
   - GET /api/segments → tier summary stats

   At startup, score ALL entities and cache results in app.state for fast /api/<entity>s queries.

2. src/api/schemas.py — Pydantic v2 models.

   CRITICAL naming conventions:
   - PaginatedResponse must use field names `items` (list) and `pages` (int) — NOT `customers`/`total_pages`
   - Prediction response: return churn_probability, risk_tier, shap_factors at the TOP LEVEL
     of the response (NOT nested under a "prediction" key)
   - Include snake_case convenience string fields on PredictRequest:
     `contract_type`, `internet_service`, `payment_method` (etc.) that auto-set one-hot columns server-side

3. app/server/index.js — Express proxy on port <express_port> → FastAPI <fastapi_port>
4. app/server/package.json — add "type": "module" to suppress Node ES module warning

Return complete file contents.
```

**Agent 2 — React Frontend** (`subagent_type: general-purpose`):
```
Write a React + TailwindCSS frontend for <topic> analytics.

Reference ALL files from /Users/aayan/MarketingAnalytics/m01_time_to_engage/app/client/src/ — read them before writing.

Dev server port: <vite_port>
API proxy target: http://localhost:<express_port>

CRITICAL — api.js field mapping conventions:
- axios baseURL must be absolute: "http://localhost:<express_port>" (NOT relative — it bypasses the Vite proxy and goes directly to Express, which is correct)
- fetchAllCustomers must use `first.items` (NOT `first.customers`) and `first.pages` (NOT `first.total_pages`)
- useSegments hook must use `resp.items` (NOT `resp.customers`)
- fetchPrediction must map PascalCase UI state keys to snake_case API convenience fields:
  Contract→contract_type, InternetService→internet_service, PaymentMethod→payment_method
- LookupView must read churn_probability and risk_tier DIRECTLY from data (NOT from data.prediction)
  Use: `const riskTier = data?.risk_tier` and render when `profile && riskTier` (not `profile && prediction`)

Domain-specific customizations:
- Replace Heatmap (7×24 grid) with appropriate <topic> visualization
- Add custom gauge or key chart for the primary model output
- Keep SHAP bar chart, what-if simulator, segment browser (adapt labels/fields)

Write all files to <project_dir>/app/client/:
- index.html, src/main.jsx, src/App.jsx, src/index.css
- src/components/ — all adapted components including domain-specific chart
- src/views/ — LookupView, WhatIfView, SegmentView
- src/lib/api.js, hooks.js, constants.js
- package.json (vite port: <vite_port>), vite.config.js, tailwind.config.js, postcss.config.js

Return complete file contents.
```

**After agents complete:**

1. Write all files to disk
2. Install npm dependencies:
```bash
cd <project_dir>/app/server && npm install
cd <project_dir>/app/client && npm install
```
3. Build the React client to verify no compile errors:
```bash
cd <project_dir>/app/client && npm run build
```
4. Smoke test the API (run uvicorn FROM the model directory, not the root):
```bash
cd <project_dir> && /Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port <fastapi_port> --log-level warning &
sleep 4
curl -s http://localhost:<fastapi_port>/health
curl -s "http://localhost:<fastapi_port>/api/<entity>s?page=1&per_page=3" | python3 -m json.tool
# Kill test server
pkill -f "uvicorn src.api.main"
```

**Mark phase as `completed`. Proceed to Phase 5 if budget allows.**

---

## Phase 5: Documentation

**Mark phase as `in_progress`.**

Dispatch 2 Task agents in a SINGLE message (parallel):

**Agent 1 — Technical Documentation** (`subagent_type: general-purpose`):
```
Write technical model documentation for <topic> analytics.

Reference formats (read these files):
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/docs/model_card.md
- /Users/aayan/MarketingAnalytics/m01_time_to_engage/docs/model_documentation.md

Source material to read from <project_dir>:
- docs/research_brief.md
- docs/data_dictionary.md
- docs/validation_report.md
- src/train.py

Write:
1. <project_dir>/docs/model_card.md — concise (1-2 pages): purpose, intended users,
   training data, performance metrics, limitations, ethical considerations
2. <project_dir>/docs/model_documentation.md — comprehensive (10-15 pages): data, methodology,
   training, evaluation, SHAP drivers, deployment, API reference, future work
```

**Agent 2 — Executive Deck** (`subagent_type: general-purpose`):
```
Write a Marp-format markdown executive presentation for <topic> analytics.

Reference format: read /Users/aayan/MarketingAnalytics/m01_time_to_engage/docs/ for tone and structure.

Source material: docs/research_brief.md and docs/validation_report.md from <project_dir>

15 slides covering:
1. Title + tagline
2. Business opportunity / cost of the problem
3. Solution overview (non-technical)
4. Data used
5. Methodology (high level)
6. Key model findings / top features
7. Model performance in business terms (lift, ROI)
8. The application (screenshot description)
9. Retention simulator demo
10-13. Segment insights + business recommendations
14. ROI estimate
15. Next steps

Write to <project_dir>/docs/executive_deck.md
```

**After agents complete:**

Write `<project_dir>/README.md` with quick start using correct ports.

**Mark phase as `completed`. Proceed to Phase 6.**

---

## Phase 6: App Testing with Playwright

**Start all three servers in background:**

```bash
# Kill any stale processes first
pkill -f "uvicorn src.api.main" 2>/dev/null
pkill -f "node.*<express_port>" 2>/dev/null
pkill -f "vite.*<vite_port>" 2>/dev/null
sleep 1

# FastAPI — run FROM the model directory
cd <project_dir>
/Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port <fastapi_port> --log-level warning > /tmp/<model_slug>_fastapi.log 2>&1 &

# Express proxy
cd <project_dir>/app/server && npm start > /tmp/<model_slug>_express.log 2>&1 &

# Vite dev server
cd <project_dir>/app/client && npm run dev > /tmp/<model_slug>_vite.log 2>&1 &

# Wait for all to be ready
sleep 5
curl -s http://localhost:<fastapi_port>/health
```

**Run Playwright tests using the MCP browser tools in this sequence:**

1. **Navigate and verify initial load:**
   - `browser_navigate` to `http://localhost:<vite_port>`
   - `browser_take_screenshot` — verify app renders, sidebar shows 3 nav items
   - Check `browser_console_messages` for errors (favicon 404 is acceptable; anything else must be fixed)

2. **Test Customer/Entity Lookup view:**
   - Wait for `Loading <entities>...` to disappear (customers list loaded from API)
   - If dropdown shows "No customers found" after typing: check `api.js` for field naming bugs —
     `fetchAllCustomers` must use `first.items` and `first.pages`
   - Click the combobox, type first few chars of a real entity ID (get from `curl /api/<entity>s`)
   - Select a result — wait for the profile panel to appear
   - If profile panel does NOT appear: check that `LookupView` reads `data?.risk_tier` directly
     (not `data?.prediction?.risk_tier`)
   - `browser_take_screenshot` — verify: gauge/chart, stat cards, SHAP bar chart, recommended action

3. **Test What-If / Risk Simulator view:**
   - Navigate to simulator via sidebar
   - `browser_take_screenshot` — verify gauge shows initial prediction
   - Change a high-impact dropdown (e.g., Contract Type → Two year for churn model)
   - Wait 2 seconds for debounced prediction
   - If gauge doesn't update: check `fetchPrediction` maps PascalCase keys to snake_case API params
   - `browser_take_screenshot` — verify score changed and SHAP bars updated

4. **Test Segment Explorer view:**
   - Navigate to segment explorer
   - `browser_take_screenshot` — verify KPI cards and data table load
   - Change segment filter to a different tier
   - `browser_take_screenshot` — verify table updates with new segment's data

5. **Fix any bugs found**, then re-test the affected view.

6. **Final screenshot of each passing view.** Report pass/fail for each.

---

## Step 3: Session Report

**If all phases completed:**

```
## ✅ Analytics Project Complete: <topic>

Project location: <project_dir>

Deliverables:
  ✓ Research:         docs/research_brief.md
  ✓ Data Engineering: notebooks/01, src/data_pipeline.py, src/feature_engineering.py
  ✓ Model Building:   notebooks/02-03, src/train.py, models/
  ✓ App:              src/api/, app/server/, app/client/
  ✓ Documentation:    docs/model_card.md, docs/model_documentation.md, docs/executive_deck.md
  ✓ App Testing:      All 3 views verified with Playwright

To run:
  cd <project_dir>
  /Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port <fastapi_port> &
  cd app/server && npm start &
  cd app/client && npm run dev
  # Open http://localhost:<vite_port>
```

**If session ended mid-workflow:**

```
## Analytics Project Progress: <topic>

  ✓ [Completed phases]
  ⏸ Stopped after: <last completed phase>
  → Next: <next phase>

Resume command:
  /analytics-project "<topic>" --project-dir <project_dir>
```

---

## Common Bugs Checklist (check before declaring Phase 4 complete)

These bugs appeared in the m03 build and are likely to recur:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Customer dropdown empty | "No customers found" even after loading | `fetchAllCustomers` must use `first.items` not `first.customers`, `first.pages` not `first.total_pages` |
| Segment table empty | Segment view shows no rows | `useSegments` hook must use `resp.items` not `resp.customers` |
| Profile panel never appears | Selecting a customer shows nothing | `LookupView` must use `data?.risk_tier` directly; condition on `profile && riskTier` not `profile && prediction` |
| Simulator gauge stuck | Changing dropdown doesn't update score | `fetchPrediction` must map `Contract`→`contract_type`, `InternetService`→`internet_service`, `PaymentMethod`→`payment_method` |
| ModuleNotFoundError on startup | `No module named 'src'` | Add `sys.path.insert` at top of `main.py`; run uvicorn from model dir not root |
| pyarrow not found | Pipeline crashes saving parquet | Run `pip install pyarrow` explicitly even if in requirements.txt |
| f-string SyntaxWarning | `\ in f-string` Python 3.12+ | Use intermediate variables instead of backslash inside f-strings |
| Express ESM warning | `MODULE_TYPELESS_PACKAGE_JSON` | Add `"type": "module"` to `app/server/package.json` |
