# Churn Propensity Analytics — Model 3

Predicts the probability that a telecom subscriber will churn (cancel service), using a LightGBM binary classifier trained on the IBM Telco Customer Churn dataset. Delivers a full-stack web app with customer risk lookup, what-if simulation, and segment explorer.

## Quick Start

```bash
# From /Users/aayan/MarketingAnalytics/

# 1. Activate the shared virtual environment
source .venv/bin/activate

# 2. Install Python dependencies (if not already installed)
pip install -r requirements.txt

# 3. Run the data pipeline + feature engineering
python m03_churn/src/data_pipeline.py
python m03_churn/src/feature_engineering.py

# 4. Train the model
python m03_churn/src/train.py
# → Saves models/ artifacts, prints AUC-ROC

# 5. Start the FastAPI prediction service (port 8003)
cd m03_churn && python -m uvicorn src.api.main:app --port 8003

# 6. Start the Express proxy (port 3003) — new terminal
cd m03_churn/app/server && npm start

# 7. Start the React UI (port 5175) — new terminal
cd m03_churn/app/client && npm run dev

# Open http://localhost:5175
```

## Architecture

Python data pipeline (`src/data_pipeline.py` → `src/feature_engineering.py` → `src/train.py`) produces trained LightGBM + SHAP artifacts served by a FastAPI microservice (port 8003). An Express proxy (port 3003) fronts the Python API, and a React + TailwindCSS SPA (port 5175) provides the UI: customer churn risk gauge, top SHAP risk drivers, what-if retention simulator, and segment explorer.

## Model Performance

| Model | AUC-ROC | Notes |
|---|---|---|
| Naive baseline | 0.50 | Majority-class constant |
| Logistic Regression | 0.8162 | All 32 features |
| **LightGBM (calibrated)** | **0.7883** | Primary model |

- **Top-20% lift**: 2.29× — contact top-20% scored subscribers to capture 45.7% of actual churners
- **Cost-optimal threshold**: 0.16 (FN=$200, FP=$20 cost model)
- **Brier score**: 0.155 (calibrated probabilities)

## Documentation

- [Research Brief](docs/research_brief.md) — domain overview, dataset selection
- [Data Dictionary](docs/data_dictionary.md) — all 32 features with formulas
- [Model Card](docs/model_card.md) — concise model summary, limitations, fairness
- [Model Documentation](docs/model_documentation.md) — full technical deep-dive
- [Validation Report](docs/validation_report.md) — auto-generated training metrics
- [Executive Deck](docs/executive_deck.md) — stakeholder presentation (Marp format)

## Project Structure

```
m03_churn/
├── data/
│   ├── raw/          WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/    customers_clean.parquet, customer_features.parquet
├── notebooks/
│   ├── 01_eda_churn.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_diagnostics.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── api/main.py, schemas.py
├── app/
│   ├── client/       React + Vite + TailwindCSS (port 5175)
│   └── server/       Express proxy (port 3003)
├── models/           lgbm_churn.pkl, calibrator.pkl, shap_explainer.pkl
├── docs/             model_card.md, model_documentation.md, etc.
├── scripts/          fetch_data.py
└── project_state.json
```
