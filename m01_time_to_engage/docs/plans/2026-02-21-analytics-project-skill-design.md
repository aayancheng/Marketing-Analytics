# Analytics Project Skill — Design Document

**Date:** 2026-02-21
**Status:** Approved

---

## Context

The current MarketingAnalytics project demonstrates a complete analytics workflow: data acquisition, feature engineering, LightGBM model training, FastAPI backend, React frontend, and comprehensive documentation. The goal is to package this workflow as a reusable Claude Code skill so that any new business topic (credit risk, CLV, fraud detection, etc.) can be stood up end-to-end with consistent structure, adaptive model choices, and multi-day resumability.

---

## Skill Overview

**File:** `~/.claude/skills/analytics-project.md`
**Invocation:** `/analytics-project "<topic>" --project-dir <path>`
**Example:** `/analytics-project "credit risk" --project-dir /Users/aayan/CreditRisk`

---

## Architecture

### State File: `project_state.json`

Created on first run at `<project-dir>/project_state.json`. Read on every subsequent invocation to determine which phases are complete and where to resume.

```json
{
  "topic": "credit risk",
  "project_dir": "/Users/aayan/CreditRisk",
  "created_at": "2026-02-21",
  "phases": {
    "research":          { "status": "pending|in_progress|completed", "completed_at": null },
    "data_engineering":  { "status": "pending", "completed_at": null },
    "modeling":          { "status": "pending", "completed_at": null },
    "app_building":      { "status": "pending", "completed_at": null },
    "documentation":     { "status": "pending", "completed_at": null }
  },
  "decisions": {
    "model_type": null,
    "data_source": null,
    "feature_categories": [],
    "stack": {}
  }
}
```

Resume logic: On invocation, read state file → skip `completed` phases → continue from first `pending` or `in_progress` phase.

---

## The 5 Phases

Phases run sequentially. Within each phase, independent subtasks dispatch parallel Task agents.

### Phase 1 — Research

**Goal:** Understand the domain, select the tech stack, identify data sources.

Parallel agents:
- **Agent 1** (WebSearch): Domain overview, regulatory context, industry KPIs, benchmark models, published papers
- **Agent 2** (WebSearch): Available public datasets (UCI, Kaggle, FRED, etc.) and APIs
- **Agent 3** (Explore): Existing project at `/Users/aayan/MarketingAnalytics/project/` — identify reusable patterns, components, scripts

**Outputs:**
- `docs/research_brief.md` — domain summary, KPIs, data sources table, model candidates with rationale
- `project_state.json` `decisions` populated: `model_type`, `data_source`, `feature_categories`, `stack`

**Adaptive stack examples:**
| Topic | Model Type | Key Libraries |
|-------|-----------|---------------|
| Credit risk | XGBoost + scorecard/WoE | xgboost, optbinning, scikit-learn |
| CLV | BG/NBD + GG spend model | lifetimes, pymc |
| Fraud detection | Isolation Forest + SHAP | scikit-learn, imbalanced-learn, shap |
| Marketing send-time | LightGBM 168-slot classifier | lightgbm, shap |
| Churn | Survival analysis or gradient boosting | lifelines or lightgbm |

All topics share: FastAPI backend, React + TailwindCSS frontend, Jupyter notebooks, pandas/numpy.

---

### Phase 2 — Data Engineering

**Goal:** Acquire data, clean it, engineer features, produce EDA notebook.

Parallel agents (first wave):
- **Agent 1** (Bash + general-purpose): Download/fetch dataset based on `decisions.data_source`
- **Agent 2** (general-purpose): Design data schema and feature specification document

Sequential (second wave, after data exists):
- Build `notebooks/01_eda_and_synthesis.ipynb` (EDA + synthetic augmentation if needed)
- Write `src/data_pipeline.py` (cleaning, transforms)
- Write `src/feature_engineering.py` (domain-specific features)
- Write `docs/data_dictionary.md`

**Outputs:**
- `data/raw/` populated
- `data/processed/` (cleaned features)
- `notebooks/01_eda_and_synthesis.ipynb`
- `src/data_pipeline.py`, `src/feature_engineering.py`
- `docs/data_dictionary.md`

---

### Phase 3 — Model Building

**Goal:** Train model, calibrate, compute explainability artifacts.

Parallel agents:
- **Agent 1** (general-purpose): Build `notebooks/02_model_training.ipynb` + `src/train.py` using `decisions.model_type`
- **Agent 2** (general-purpose): Build `notebooks/03_model_diagnostics.ipynb` + `docs/validation_report.md`

Sequential (after notebooks written):
- Execute notebooks via `jupyter nbconvert --execute --to notebook`
- Save model artifacts to `models/`

**Outputs:**
- `notebooks/02_model_training.ipynb` (executed)
- `notebooks/03_model_diagnostics.ipynb` (executed)
- `src/train.py`
- `models/` (trained model + calibrator + explainer pickles)
- `docs/validation_report.md`

---

### Phase 4 — App Building

**Goal:** Stand up FastAPI backend and React frontend served via Express proxy.

Parallel agents:
- **Agent 1** (general-purpose): FastAPI backend (`src/api/main.py`, `src/api/schemas.py`) — endpoints adapted to model's input/output interface
- **Agent 2** (general-purpose): React frontend (`app/client/src/`) — components adapted from current project (Heatmap → topic-appropriate visualization, SHAP chart, lookup/what-if/segment views)

Reference components from `/Users/aayan/MarketingAnalytics/project/app/` for structure and patterns.

**Outputs:**
- `app/server/index.js` (Express proxy)
- `app/client/src/` (React components, views, hooks, API client)
- `package.json` workspace config
- Working end-to-end app (FastAPI on :8000, Express on :3001, React on :5173)

---

### Phase 5 — Documentation

**Goal:** Produce model card, full model documentation, and executive deck.

Parallel agents:
- **Agent 1** (general-purpose): `docs/model_card.md` + `docs/model_documentation.md` — reference `/Users/aayan/MarketingAnalytics/project/docs/model_card.md` and `model_documentation.md` for format
- **Agent 2** (general-purpose): `docs/executive_deck.md` — reference `/Users/aayan/MarketingAnalytics/project/docs/marketing_analytics_deck.md` for format and tone

**Outputs:**
- `docs/model_card.md`
- `docs/model_documentation.md`
- `docs/executive_deck.md`

---

## Generated Project Structure

```
<project-dir>/
├── project_state.json
├── .gitignore
├── requirements.txt
├── package.json
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_eda_and_synthesis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_diagnostics.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── api/
│       ├── main.py
│       └── schemas.py
├── app/
│   ├── server/
│   └── client/
├── models/
└── docs/
    ├── research_brief.md
    ├── data_dictionary.md
    ├── synthesis_methodology.md
    ├── validation_report.md
    ├── model_card.md
    ├── model_documentation.md
    └── executive_deck.md
```

---

## Multi-Day Token Budget

Each session, the skill:
1. Reads `project_state.json`
2. Announces which phases are complete and which is next
3. Executes one or two phases within the session's token budget
4. Writes `completed` status + timestamp to state file on phase completion
5. Ends session with a summary of what's done and what remains

Phases are roughly sized by token cost:
- Research: ~1 session
- Data Engineering: ~1 session
- Model Building: ~1-2 sessions (notebook execution is token-light but compute-heavy)
- App Building: ~1-2 sessions
- Documentation: ~1 session

---

## Skill File Structure

The `.md` skill file is organized as:

```
# Analytics Project Builder

## Overview + invocation syntax

## Step 1: Read or initialize state file
## Step 2: Announce current status
## Step 3: Execute next pending phase
  ### Phase 1 — Research
  ### Phase 2 — Data Engineering
  ### Phase 3 — Model Building
  ### Phase 4 — App Building
  ### Phase 5 — Documentation
## Step 4: Update state file
## Step 5: Report completion + next steps
```

---

## Reference Files (reuse from current project)

| What | Path |
|------|------|
| Notebook structure | `/Users/aayan/MarketingAnalytics/project/notebooks/` |
| Data pipeline pattern | `/Users/aayan/MarketingAnalytics/project/src/data_pipeline.py` |
| Feature engineering pattern | `/Users/aayan/MarketingAnalytics/project/src/feature_engineering.py` |
| Train script pattern | `/Users/aayan/MarketingAnalytics/project/src/train.py` |
| FastAPI pattern | `/Users/aayan/MarketingAnalytics/project/src/api/main.py` |
| React components | `/Users/aayan/MarketingAnalytics/project/app/client/src/` |
| Model card format | `/Users/aayan/MarketingAnalytics/project/docs/model_card.md` |
| Model doc format | `/Users/aayan/MarketingAnalytics/project/docs/model_documentation.md` |
| Executive deck format | `/Users/aayan/MarketingAnalytics/project/docs/marketing_analytics_deck.md` |
| Express proxy | `/Users/aayan/MarketingAnalytics/project/app/server/index.js` |

---

## Verification

After the skill is created:
1. Create a test project dir: `mkdir -p /tmp/test-analytics`
2. Invoke: `/analytics-project "test topic" --project-dir /tmp/test-analytics`
3. Verify `project_state.json` is created with correct structure
4. Verify Phase 1 runs research agents and produces `docs/research_brief.md`
5. Kill session, re-invoke, verify Phase 1 is skipped and Phase 2 begins
6. Run full workflow for a real topic (e.g., "credit risk") and verify all 5 deliverables are produced
