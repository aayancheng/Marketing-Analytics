# How to Use the `analytics-project` Skill

**For:** Data scientists and analysts starting a new analytics project
**Skill file:** `~/.claude/skills/analytics-project.md`

---

## Quick Start

Open Claude Code in any directory and run:

```
/analytics-project "credit risk" --project-dir ~/CreditRisk
```

That's it. Claude will build a complete, production-ready analytics project — research through documentation — across one or more sessions.

---

## What It Does

The skill orchestrates **5 sequential phases**, each one feeding the next:

| Phase | What Claude Builds | Agent Teams |
|-------|-------------------|-------------|
| **1. Research** | Domain brief, data source shortlist, model type selection | 3 parallel agents |
| **2. Data Engineering** | Data download, EDA notebook, pipeline scripts, data dictionary | 2 parallel agents |
| **3. Model Building** | Training + diagnostics notebooks, trained model artifacts | 2 parallel agents |
| **4. App Building** | FastAPI backend + React frontend served via Express | 2 parallel agents |
| **5. Documentation** | Model card, technical docs, executive deck, README | 2 parallel agents |

### Example Invocations

```bash
# Credit risk scorecard
/analytics-project "credit risk" --project-dir ~/CreditRisk

# Customer lifetime value
/analytics-project "customer lifetime value" --project-dir ~/CLV

# Fraud detection
/analytics-project "fraud detection" --project-dir ~/FraudModel

# Demand forecasting
/analytics-project "demand forecasting" --project-dir ~/DemandForecast
```

---

## Multi-Day Workflow

Each session, Claude reads `project_state.json` to know where it left off:

```json
{
  "topic": "credit risk",
  "phases": {
    "research":         { "status": "completed", "completed_at": "2026-02-21" },
    "data_engineering": { "status": "completed", "completed_at": "2026-02-22" },
    "modeling":         { "status": "in_progress", "completed_at": null },
    "app_building":     { "status": "pending",   "completed_at": null },
    "documentation":    { "status": "pending",   "completed_at": null }
  }
}
```

**Day 1** — Run the skill, complete Research and Data Engineering
**Day 2** — Resume: `decisions` from Phase 1 automatically feed Phase 3's model choice
**Day 3** — Resume: App Building uses the model's input/output interface
**Day 4** — Resume: Documentation reads all prior artifacts

To resume, use the exact same command:
```
/analytics-project "credit risk" --project-dir ~/CreditRisk
```
Claude skips completed phases and picks up from `modeling`.

---

## How the Skill Makes Claude Better

Without a skill, asking Claude to "build a credit risk model" produces generic code that ignores your existing project structure, makes arbitrary tech stack choices, and can't span multiple sessions.

The skill solves this in four ways:

### 1. Research Before Building

Phase 1 always runs three agents in parallel before any code is written:
- One researches domain context, regulations, and benchmark approaches
- One finds the best available public datasets with licenses and column lists
- One reads **this project's** actual code to extract reusable patterns

This means Claude picks XGBoost + WoE binning for credit risk (industry standard) rather than defaulting to a generic classifier — and it knows exactly which functions from `src/data_pipeline.py` to adapt rather than starting from scratch.

### 2. Adaptive Stack Selection

The skill contains a domain-to-model mapping table:

| Domain | Model Type |
|--------|-----------|
| Credit risk | XGBoost + WoE binning |
| CLV | BG/NBD + Gamma-Gamma spend model |
| Fraud | Isolation Forest + threshold tuning |
| Churn | Survival analysis or LightGBM |
| Send-time | LightGBM 168-slot grid classifier |

Claude reads this table during Research and populates `decisions.model_type` in the state file. Every subsequent phase reads that decision — the training notebook, the FastAPI schemas, the React frontend — all adapt to the chosen model type automatically.

### 3. Parallel Agent Teams

Within each phase, the skill dispatches multiple Task agents simultaneously in a single message. For example, Phase 4 (App Building) runs:

- **Agent 1** writing FastAPI endpoints adapted to the model's input/output interface
- **Agent 2** reading and adapting the React component library

Both agents work at the same time. This cuts wall-clock time roughly in half for phases with independent subtasks.

### 4. Anchored to a Reference Implementation

Every agent prompt in the skill references `/Users/aayan/MarketingAnalytics/project/` as the canonical source of patterns. This means:

- New projects get the same notebook structure (01_eda → 02_training → 03_diagnostics)
- The FastAPI endpoint design matches the existing API patterns
- React components adapt from the actual component library rather than being invented from scratch
- Model cards and executive decks match the existing document formats

The new project inherits quality from the reference project rather than starting cold.

---

## What Gets Built

After all 5 phases, `~/CreditRisk/` (or your chosen directory) contains:

```
~/CreditRisk/
├── project_state.json          ← resume checkpoint
├── requirements.txt
├── package.json
├── README.md
├── data/
│   ├── raw/                    ← downloaded dataset
│   ├── processed/              ← cleaned features
│   └── synthetic/              ← augmented data (if needed)
├── notebooks/
│   ├── 01_eda_and_synthesis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_diagnostics.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── api/
│       ├── main.py             ← FastAPI (port 8000)
│       └── schemas.py
├── app/
│   ├── server/                 ← Express proxy (port 3001)
│   └── client/                 ← React + TailwindCSS (port 5173)
├── models/
│   ├── model.pkl
│   ├── calibrator.pkl
│   └── shap_explainer.pkl
└── docs/
    ├── research_brief.md
    ├── data_dictionary.md
    ├── validation_report.md
    ├── model_card.md
    ├── model_documentation.md
    └── executive_deck.md
```

To run the finished app:
```bash
cd ~/CreditRisk
python -m uvicorn src.api.main:app --port 8000 &
cd app/server && npm start &
cd app/client && npm run dev
# Open http://localhost:5173
```

---

## Tips

**Token budget:** Each phase uses roughly one session's budget. Research and Documentation are lighter; Model Building (which executes notebooks) and App Building (many files) are heavier. If a session ends mid-phase, simply re-run the same command — the skill restarts the interrupted phase from the beginning.

**Changing direction:** Edit `project_state.json` directly to re-run a completed phase. Set its `"status"` back to `"pending"` and invoke the skill again.

**Custom project directory:** Always specify `--project-dir` to keep topics isolated. Without it, the skill defaults to `~/Analytics/<topic-slug>/`.

**Reference project:** The skill is anchored to `/Users/aayan/MarketingAnalytics/project/`. If you move that project, update the paths in `~/.claude/skills/analytics-project.md`.
