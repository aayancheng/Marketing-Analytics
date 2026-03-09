# Analytics Project Builder

Use when invoked as `/analytics-project "<topic>" [--project-dir <path>]` to build a complete analytics project from scratch for any business domain, or to resume an in-progress project.

## Session Management — MANDATORY

Building a full project exceeds a single context window. This skill uses **hard breakpoints** after each phase. At every breakpoint you MUST:

1. Update `project_state.json` with phase status, decisions, and metrics
2. Write a `<project_dir>/SESSION_LOG.md` entry summarising what was completed and what comes next
3. Print the resume command and STOP — do not continue to the next phase

The user will start a new conversation and invoke `/analytics-project` again. The skill reads `project_state.json` to resume from the correct phase.

**Exception:** Phases 1 (Research) and 2 (Data Engineering) may run in the same session if Phase 1 completes quickly. All other phases get their own session.

---

## Workspace Convention

All projects live inside `/Users/aayan/MarketingAnalytics/` as numbered model folders:

```
/Users/aayan/MarketingAnalytics/
├── .venv/                         ← single shared Python venv for ALL models
├── requirements.txt               ← consolidated deps for all models
├── shared/data/raw/               ← datasets shared between models
├── m01_time_to_engage/            ← reference (classification, email timing)
├── m02_clv/                       ← reference (regression, customer value)
├── m03_churn/                     ← reference (classification, churn risk)
├── m04_mmm/
├── m05_next_best_offer/
```

**Port convention** — assign based on model number N:
| Model | FastAPI | Express | Vite |
|-------|---------|---------|------|
| m01   | 8001    | 3001    | 5173 |
| m02   | 8002    | 3002    | 5174 |
| m03   | 8003    | 3003    | 5175 |
| m04   | 8004    | 3004    | 5176 |
| m05   | 8005    | 3005    | 5177 |

**Reference implementations:**
- **Classification** (binary outcome, probability output): read from `m03_churn/` (most recent, cleanest)
- **Regression** (continuous outcome, value output): read from `m02_clv/`
- **Component library & React patterns**: read from `m01_time_to_engage/app/client/src/`
- **Notebook structure & naming**: read from `m01_time_to_engage/notebooks/` and `m03_churn/notebooks/`

---

## Overview

Orchestrates a 6-phase build with session checkpointing:

| Phase | What Gets Built | Session |
|-------|-----------------|---------|
| 1. Research | Domain brief, data source, model type decision | Session A |
| 2. Data Engineering | Data acquisition, EDA notebook, pipeline scripts | Session A (if budget) or B |
| 3. Model Building | Training + diagnostics notebooks, trained artifacts | Own session |
| 4. App Building | FastAPI backend + React frontend | Own session |
| 5. Documentation | Model card, technical docs, executive deck | Own session |
| 6. App Testing | Start all 3 servers, Playwright smoke test all views | Own session |

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
    "documentation":    { "status": "pending", "completed_at": null },
    "app_testing":      { "status": "pending", "completed_at": null }
  },
  "decisions": {
    "model_type": null,
    "model_category": null,
    "data_source": null,
    "feature_categories": [],
    "stack": {}
  },
  "ports": {
    "fastapi": "<800N>",
    "express": "<300N>",
    "vite":    "<517N+2>"
  }
}
```

**If resuming** (file exists): read it, announce current status (✓ completed / → next / · pending for each phase), and jump to Step 2.

**Proceed to Step 2.**

---

## Step 2: Execute Next Pending Phase

Read `project_state.json`. For each phase in order, skip `"completed"` phases. Execute the first `"pending"` phase. After completion, update status, write session log, and **STOP at the breakpoint** unless this is Phase 1 continuing into Phase 2.

---

## Phase 1: Research

**Mark phase as `in_progress` in state file before starting.**

Dispatch 3 agents in a SINGLE message (parallel):

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
Explore the completed models in /Users/aayan/MarketingAnalytics/ to document reusable patterns.

Read and summarise these files from m03_churn/ (classification reference):
1. src/data_pipeline.py — data cleaning patterns
2. src/feature_engineering.py — feature computation + FEATURE_COLUMNS constant
3. src/train.py — training, calibration, SHAP, threshold optimization, metadata.json
4. src/api/main.py + schemas.py — FastAPI structure, response field naming, lifespan
5. app/client/src/ — component list and structure
6. notebooks/ — naming convention (01_eda, 02_model_training, 03_model_diagnostics)
7. docs/ — model_card.md, model_documentation.md, executive_deck.md formats

Also read m02_clv/src/api/main.py and m02_clv/src/train.py for regression-specific patterns.

For each file: purpose, key functions, what to adapt for a new domain.
```

**After all 3 agents complete:**

1. **Determine model category** — this drives all downstream decisions:

| Category | When to Use | Reference Model |
|----------|-------------|-----------------|
| `classification` | Binary outcome (churn, fraud, default, click) | m03_churn |
| `regression` | Continuous outcome (CLV, revenue, spend) | m02_clv |
| `time_slot` | Per-slot scoring grid (send-time, scheduling) | m01_time_to_engage |

2. **Select model type** using this table:

| Domain | Category | Model Type | Key Libraries |
|--------|----------|-----------|---------------|
| Credit risk / scorecard | classification | XGBoost + WoE binning | xgboost, optbinning |
| Customer lifetime value | regression | LightGBM Regressor + BG/NBD baseline | lightgbm, lifetimes, shap |
| Fraud detection | classification | Isolation Forest + threshold optimization | scikit-learn, imbalanced-learn |
| Churn prediction | classification | LightGBM Classifier | lightgbm, shap |
| Marketing send-time | time_slot | LightGBM 168-slot grid | lightgbm, shap |
| Demand forecasting | regression | XGBoost or Prophet | xgboost or prophet |
| Recommendation | classification | Collaborative filtering or LightGBM ranking | implicit or lightgbm |
| Marketing mix | regression | Bayesian MMM (PyMC-Marketing) | pymc-marketing, arviz |
| Default | classification | LightGBM + SHAP | lightgbm, shap |

3. Synthesize results into `<project_dir>/docs/research_brief.md`
4. Update `project_state.json` decisions block with `model_type`, `model_category`, `data_source`, `feature_categories`

**Mark phase as `completed` with today's date.**

**BREAKPOINT DECISION:** If research completed quickly (< 30% of context used), continue to Phase 2. Otherwise, write session log and STOP with resume command.

---

## Phase 2: Data Engineering

**Mark phase as `in_progress`. Read `decisions` from `project_state.json`.**

Dispatch 2 agents in a SINGLE message (parallel):

**Agent 1 — Data Acquisition** (`subagent_type: general-purpose`):
```
Download the dataset: <decisions.data_source>

Reference pattern: /Users/aayan/MarketingAnalytics/m03_churn/scripts/fetch_data.py

Write a download script to <project_dir>/scripts/fetch_data.py
If Kaggle CLI is unavailable, generate a realistic synthetic dataset that:
- Matches the original dataset's schema, row count, and column distributions
- Preserves key domain correlations (e.g., for churn: month-to-month contract → higher churn rate)
- Uses random_state=42 and a logistic/statistical model to generate labels realistically

Execute it to populate <project_dir>/data/raw/
Report: file name, size, row count, column list, class balance (if classification)
```

**Agent 2 — Pipeline, Features, and Notebooks** (`subagent_type: general-purpose`):
```
Write data pipeline, feature engineering scripts, EDA notebook, and data dictionary for "<topic>".

Model category: <decisions.model_category>

Reference implementations — read the CORRECT reference based on model category:
- Classification: /Users/aayan/MarketingAnalytics/m03_churn/src/data_pipeline.py and feature_engineering.py
- Regression: /Users/aayan/MarketingAnalytics/m02_clv/src/data_pipeline.py and feature_engineering.py
- Also read: /Users/aayan/MarketingAnalytics/m03_churn/notebooks/01_eda_churn.ipynb for notebook format

Feature categories: <decisions.feature_categories>

Write to <project_dir>/:
1. src/__init__.py — empty (required for module resolution)
2. src/data_pipeline.py — load_raw(), clean(), save_processed(), run_tests()
3. src/feature_engineering.py — compute features, FEATURE_COLUMNS constant (shared with API)
4. notebooks/01_eda_<topic_slug>.ipynb — distributions, target analysis, correlations
5. docs/data_dictionary.md — raw + engineered columns + target

IMPORTANT: export FEATURE_COLUMNS as a module-level constant in feature_engineering.py.
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

**Mark phase as `completed`.**

**BREAKPOINT — write session log and STOP:**
```
## Session Log — Phase 2 Complete

Completed: Research + Data Engineering
Data: <row_count> rows, <feature_count> features, target=<target_description>
Next: Phase 3 (Model Building)

Resume: /analytics-project "<topic>" --project-dir <project_dir>
```

---

## Phase 3: Model Building

**Mark phase as `in_progress`. Read `decisions.model_type` and `decisions.model_category` from `project_state.json`.**

### Category-Specific Training Approach

**Classification** (reference: m03_churn):
- Stratified train/test split (80/20) preserving class balance
- Baseline: majority-class naive + Logistic Regression
- Primary: LightGBM with `scale_pos_weight` for class imbalance
- Isotonic/Platt probability calibration
- Cost-optimal threshold sweep (FN cost >> FP cost)
- Metrics: AUC-ROC, PR-AUC, Brier Score, F1, top-20% lift, top-20% capture rate
- Confusion matrix at default (0.5) and cost-optimal thresholds

**Regression** (reference: m02_clv):
- Temporal or random train/test split (80/20)
- Baseline: naive mean + domain-specific probabilistic model (e.g., BG/NBD for CLV)
- Primary: LightGBM Regressor with `log1p()` target transform, `expm1()` inverse
- No probability calibration needed
- Metrics: MAE, RMSE, MAPE, Spearman rank correlation, top-decile lift
- Segment thresholds based on percentiles (90th, 75th, 40th, 10th)

Dispatch 2 agents in a SINGLE message (parallel):

**Agent 1 — Training Script + Notebook** (`subagent_type: general-purpose`):
```
Write a training script and Jupyter notebook for <decisions.model_type> model on <topic>.
Model category: <decisions.model_category>

Reference (read the correct one based on category):
- Classification: /Users/aayan/MarketingAnalytics/m03_churn/src/train.py
- Regression: /Users/aayan/MarketingAnalytics/m02_clv/src/train.py

Write to <project_dir>/:
1. src/train.py — includes: split, baseline models, primary model, calibration (if classification),
   threshold optimization (if classification), SHAP TreeExplainer, save all artifacts to models/,
   auto-generate docs/validation_report.md
2. notebooks/02_model_training.ipynb — interactive version of train.py with inline plots

Artifacts to save to <project_dir>/models/:
- primary model pkl (e.g., lgbm_<slug>.pkl)
- calibrator pkl (classification only)
- shap_explainer.pkl
- metadata.json — MUST contain:
  {
    "FEATURE_COLUMNS": [...],
    "train_rows": N,
    "test_rows": N,
    "metrics": { ... },  // AUC/MAE etc based on category
    "timestamp": "ISO-8601",
    // Classification only:
    "cost_optimal_threshold": 0.XX,
    "scale_pos_weight": X.XX,
    // Regression only:
    "segment_thresholds": {"Champions": 90, "High Value": 75, ...},
    "median_value": X.XX,
    "top_decile_lift": X.XX
  }

IMPORTANT: avoid backslash escapes inside f-strings (Python 3.12+ SyntaxWarning).
Use intermediate variables instead of f"\n{val}" patterns.
```

**Agent 2 — Diagnostics Notebook** (`subagent_type: general-purpose`):
```
Write a model diagnostics Jupyter notebook for <decisions.model_type> on <topic>.
Model category: <decisions.model_category>

Reference: /Users/aayan/MarketingAnalytics/m03_churn/notebooks/03_model_diagnostics.ipynb

Sections for notebooks/03_model_diagnostics.ipynb:

Classification:
1. Load artifacts and test set
2. Confusion matrix at default and cost-optimal thresholds
3. SHAP summary beeswarm + force plots for 3 examples (high/medium/low risk)
4. Lift chart — top-N% targeting vs random
5. Business simulation (dollar/pound value of top-N% targeting)
6. Failure mode analysis (worst false negatives, worst false positives)
7. Slice analysis by key feature segments

Regression:
1. Load artifacts and test set
2. Actual vs predicted scatter + residuals plot
3. SHAP summary beeswarm + force plots for 3 examples (high/mid/low value)
4. Decile lift chart — actual value by predicted decile
5. Segment distribution and per-segment MAE
6. Cold-start analysis (customers with thin history)
7. Error analysis by key feature segments
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

**Mark phase as `completed`.**

**BREAKPOINT — write session log and STOP:**
```
## Session Log — Phase 3 Complete

Completed: Model Building
Model: <model_type>
Key metrics: <top 3 metrics with values>
Artifacts: <list files in models/>
Next: Phase 4 (App Building)

Resume: /analytics-project "<topic>" --project-dir <project_dir>
```

---

## Phase 4: App Building

**Mark phase as `in_progress`.**

Read ports and model_category from `project_state.json`.

### Category-Specific API & Frontend Design

**Classification API** (reference: m03_churn/src/api/):
- Entity detail returns: `churn_probability`, `risk_tier`, `shap_factors` at TOP LEVEL (not nested)
- Risk tiers: threshold-based (e.g., High ≥ 0.50, Medium 0.16–0.49, Low < 0.16)
- PredictRequest: snake_case convenience fields (contract_type, internet_service, payment_method) that map to one-hot columns server-side
- WhatIfView: dropdowns for categoricals + sliders for numerics

**Regression API** (reference: m02_clv/src/api/):
- Entity detail returns: `predicted_clv`, `clv_segment`, `shap_factors` at TOP LEVEL
- Segments: percentile-based (Champions ≥ 90th, High Value 75–90th, Growing 40–75th, Occasional 10–40th, Dormant < 10th)
- Add `/api/portfolio` endpoint (all customers for scatter plot)
- Cold-start fallback: entities with thin history get median value
- WhatIfView: numeric sliders only (recency, frequency, avg_order_value, velocity)

Dispatch 2 agents in a SINGLE message (parallel):

**Agent 1 — FastAPI Backend** (`subagent_type: general-purpose`):
```
Write a FastAPI backend for <topic> analytics.
Model category: <decisions.model_category>

Reference — read the CORRECT reference based on category:
- Classification: /Users/aayan/MarketingAnalytics/m03_churn/src/api/main.py and schemas.py
- Regression: /Users/aayan/MarketingAnalytics/m02_clv/src/api/main.py and schemas.py
- Express proxy: /Users/aayan/MarketingAnalytics/m03_churn/app/server/index.js

Write to <project_dir>/:
1. src/api/__init__.py — empty
2. src/api/main.py — FastAPI with lifespan loading of all model artifacts.

   CRITICAL — add this sys.path fix at the top of main.py:
   ```python
   import sys, os as _os
   sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
   ```

   Endpoints:
   - GET /health → {"status":"ok","model":"<name>","<entity>s":<count>}
   - GET /api/<entity>s?page=&per_page=&segment= → PaginatedResponse
   - GET /api/<entity>/{id} → full profile + prediction + shap_factors (FLAT, not nested)
   - POST /api/predict → what-if score from feature dict
   - GET /api/segments → tier summary stats
   - (Regression only) GET /api/portfolio → all entities for scatter plot

   At startup, score ALL entities and cache results in app.state.

3. src/api/schemas.py — Pydantic v2 models.

   CRITICAL naming conventions:
   - PaginatedResponse: use `items` (list) and `pages` (int) — NOT `customers`/`total_pages`
   - Entity detail: return prediction fields at TOP LEVEL (NOT nested under "prediction" key)
   - PredictRequest: include snake_case convenience fields that map to one-hot columns server-side

4. app/server/index.js — Express proxy on port <express_port> → FastAPI <fastapi_port>
5. app/server/package.json — MUST include "type": "module"
```

**Agent 2 — React Frontend** (`subagent_type: general-purpose`):
```
Write a React + TailwindCSS frontend for <topic> analytics.
Model category: <decisions.model_category>

Reference — read ALL files from:
- Component library: /Users/aayan/MarketingAnalytics/m01_time_to_engage/app/client/src/components/
- Category-specific views:
  - Classification: /Users/aayan/MarketingAnalytics/m03_churn/app/client/src/views/
  - Regression: /Users/aayan/MarketingAnalytics/m02_clv/app/client/src/views/

Dev server port: <vite_port>
API proxy target: http://localhost:<express_port>

CRITICAL — api.js field mapping conventions:
- axios baseURL: absolute "http://localhost:<express_port>" (NOT relative)
- fetchAllCustomers: use `first.items` and `first.pages` (NOT `first.customers`/`first.total_pages`)
- useSegments hook: use `resp.items` (NOT `resp.customers`)
- fetchPrediction: map PascalCase UI keys → snake_case API params
- LookupView: read prediction fields DIRECTLY from data (NOT from data.prediction)
  Classification: `const riskTier = data?.risk_tier` — render when `profile && riskTier`
  Regression: `const clvSegment = data?.clv_segment` — render when `profile && clvSegment`

Shared components (adapt from m01):
Card, CustomerSelect, DataTable, ErrorBanner, LoadingSpinner,
SegmentPicker, ShapChart, Sidebar, SliderControl, StatCard

Domain-specific components:
- Classification: ChurnGauge or RiskGauge (probability dial), RiskFactors
- Regression: CLVBadge (segment label with color), scatter plot (recency vs value)
- Time-slot: Heatmap (7x24 grid), TopSlots

Write all files to <project_dir>/app/client/:
- index.html, src/main.jsx, src/App.jsx, src/index.css
- src/components/, src/views/, src/lib/
- package.json, vite.config.js, tailwind.config.js, postcss.config.js
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
4. Smoke test the API (run uvicorn FROM the model directory):
```bash
cd <project_dir> && /Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port <fastapi_port> --log-level warning &
sleep 4
curl -s http://localhost:<fastapi_port>/health
curl -s "http://localhost:<fastapi_port>/api/<entity>s?page=1&per_page=3" | python3 -m json.tool
pkill -f "uvicorn src.api.main"
```

**Before marking complete, check the Common Bugs Checklist at the end of this skill.**

**Mark phase as `completed`.**

**BREAKPOINT — write session log and STOP:**
```
## Session Log — Phase 4 Complete

Completed: App Building
Stack: FastAPI :<fastapi_port> + Express :<express_port> + Vite :<vite_port>
Build: npm run build passed / failed
API smoke: /health returned OK / error
Next: Phase 5 (Documentation)

Resume: /analytics-project "<topic>" --project-dir <project_dir>
```

---

## Phase 5: Documentation

**Mark phase as `in_progress`.**

Dispatch 2 agents in a SINGLE message (parallel):

**Agent 1 — Technical Documentation** (`subagent_type: general-purpose`):
```
Write technical model documentation for <topic> analytics.
Model category: <decisions.model_category>

Reference formats (read these files from the matching reference model):
- Classification: /Users/aayan/MarketingAnalytics/m03_churn/docs/model_card.md and model_documentation.md
- Regression: /Users/aayan/MarketingAnalytics/m02_clv/docs/model_card.md and model_documentation.md

Source material to read from <project_dir>:
- docs/research_brief.md, docs/data_dictionary.md, docs/validation_report.md
- src/train.py, models/metadata.json

Write:
1. <project_dir>/docs/model_card.md — concise (1-2 pages): purpose, intended users,
   training data, performance metrics, limitations, ethical considerations
2. <project_dir>/docs/model_documentation.md — comprehensive (10-15 pages): data, methodology,
   training, evaluation, SHAP drivers, deployment, API reference, future work
```

**Agent 2 — Executive Deck** (`subagent_type: general-purpose`):
```
Write a Marp-format executive presentation for <topic> analytics.
Model category: <decisions.model_category>

CRITICAL — Styling convention based on model category:
- Classification: use `theme: uncover` with dark lead slides (#0f172a background, #3b82f6 blue accents, white/gray text)
- Regression: use `theme: default` with header/footer bars and professional light styling

Reference deck (read the MATCHING reference):
- Classification: /Users/aayan/MarketingAnalytics/m03_churn/docs/executive_deck.md
- Regression: /Users/aayan/MarketingAnalytics/m02_clv/docs/executive_deck.md

Source material: docs/research_brief.md, docs/validation_report.md, models/metadata.json

Executive tone — frame for business stakeholders, not data scientists:
- Classification: frame around cost reduction and risk mitigation
  ("capture X% of churners by contacting top 20%", "reduce cost from $X to $Y")
- Regression: frame around value optimisation and portfolio allocation
  ("top 10% generates X% of revenue", "redirect spend from low-value to high-value")

Structure (15 slides):
1. Title + tagline (model, dataset, team, date)
2. Executive Summary (core insight blockquote + Without/With Model table)
3. Business problem (cost of inaction)
4. Solution overview (ML-powered, three tiers, recommended actions)
5. Dataset overview (source, volume, split)
6. Methodology (feature pipeline diagram, feature groups)
7. Key findings — SHAP drivers (top 5 with business interpretation)
8. Model performance (hero metric in <div class="metric">, full metrics table)
9. Business impact (lift metric, policy comparison table)
10. ROI estimate (conservative assumptions table + revenue calculation)
11. Risk tiers / segments with operational playbook
12. The application (three-panel description + daily workflow)
13. Diagnostics / sensitivity analysis
14. Next steps (numbered, actionable, with cross-references to other models)
15. Thank You + Key Takeaway summary

Write to <project_dir>/docs/executive_deck.md
```

**After agents complete:**

Write `<project_dir>/README.md` with quick start using correct ports.

**Mark phase as `completed`.**

**BREAKPOINT — write session log and STOP:**
```
## Session Log — Phase 5 Complete

Completed: Documentation
Docs: model_card.md, model_documentation.md, executive_deck.md, README.md
Next: Phase 6 (App Testing with Playwright)

Resume: /analytics-project "<topic>" --project-dir <project_dir>
```

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
   - Check `browser_console_messages` for errors (favicon 404 is acceptable)

2. **Test Customer/Entity Lookup view:**
   - Wait for loading to complete (customer list loaded from API)
   - If dropdown shows "No customers found": fix `api.js` — `fetchAllCustomers` must use `first.items` and `first.pages`
   - Click the combobox, type first few chars of a real entity ID (get from `curl /api/<entity>s`)
   - Select a result — wait for profile panel
   - If profile panel does NOT appear: fix `LookupView` — must read prediction fields directly from data (not nested)
   - `browser_take_screenshot` — verify: gauge/badge, stat cards, SHAP bar chart

3. **Test What-If / Simulator view:**
   - Navigate via sidebar
   - `browser_take_screenshot` — verify gauge/value shows initial prediction
   - Change a high-impact input
   - Wait 2 seconds for debounced prediction
   - If gauge doesn't update: fix `fetchPrediction` field mapping
   - `browser_take_screenshot` — verify score changed

4. **Test Segment Explorer view:**
   - Navigate via sidebar
   - `browser_take_screenshot` — verify KPI cards and data table load
   - Change segment filter
   - `browser_take_screenshot` — verify table updates

5. **Fix any bugs found**, then re-test the affected view.

6. **Final screenshot of each passing view.** Report pass/fail for each.

7. **Kill servers and clean up:**
```bash
pkill -f "uvicorn src.api.main" 2>/dev/null
pkill -f "node.*<express_port>" 2>/dev/null
pkill -f "vite.*<vite_port>" 2>/dev/null
```

**Mark phase as `completed`.**

---

## Step 3: Session Report

**If all phases completed:**

Update `project_state.json` with all phases completed. Then print:

```
## Analytics Project Complete: <topic>

Project location: <project_dir>

Deliverables:
  ✓ Research:         docs/research_brief.md
  ✓ Data Engineering: notebooks/01_eda, src/data_pipeline.py, src/feature_engineering.py
  ✓ Model Building:   notebooks/02-03, src/train.py, models/
  ✓ App:              src/api/, app/server/, app/client/
  ✓ Documentation:    docs/model_card.md, model_documentation.md, executive_deck.md
  ✓ App Testing:      All 3 views verified with Playwright

To run:
  cd <project_dir>
  /Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port <fastapi_port> &
  cd app/server && npm start &
  cd app/client && npm run dev
  # Open http://localhost:<vite_port>
```

Also update the root CLAUDE.md completed models table.

---

## Common Bugs Checklist (check before declaring Phase 4 complete)

These bugs recurred across m01, m02, and m03 builds:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Customer dropdown empty | "No customers found" even after loading | `fetchAllCustomers` must use `first.items` not `first.customers`, `first.pages` not `first.total_pages` |
| Segment table empty | Segment view shows no rows | `useSegments` hook must use `resp.items` not `resp.customers` |
| Profile panel never appears | Selecting a customer shows nothing | `LookupView` must read prediction fields directly from data — condition on `profile && riskTier` (classification) or `profile && clvSegment` (regression), NOT `profile && prediction` |
| Simulator gauge stuck | Changing input doesn't update score | `fetchPrediction` must map PascalCase UI keys → snake_case API params |
| ModuleNotFoundError on startup | `No module named 'src'` | Add `sys.path.insert` at top of `main.py`; run uvicorn from model dir |
| pyarrow not found | Pipeline crashes saving parquet | Run `pip install pyarrow` explicitly |
| f-string SyntaxWarning | `\ in f-string` Python 3.12+ | Use intermediate variables instead of backslash inside f-strings |
| Express ESM warning | `MODULE_TYPELESS_PACKAGE_JSON` | Add `"type": "module"` to `app/server/package.json` |
| Missing __init__.py | Import errors in src/ | Create empty `src/__init__.py` and `src/api/__init__.py` |
| Nested prediction response | Frontend can't read fields | API must return prediction fields at TOP LEVEL of entity response, never nested under `"prediction"` |

---

## Model Artifact Checklist (verify before declaring Phase 3 complete)

Every model must produce these files in `<project_dir>/models/`:

| File | Required | Contents |
|------|----------|----------|
| `lgbm_<slug>.pkl` | Always | Trained LightGBM model |
| `shap_explainer.pkl` | Always | SHAP TreeExplainer fitted to training data |
| `metadata.json` | Always | FEATURE_COLUMNS, train/test rows, metrics, timestamp |
| `probability_calibrator.pkl` | Classification only | Isotonic or Platt calibrator |
| `diagnostics_summary.json` | Optional | Summary stats from notebook 03 |
