# Project Brief: Model 3 — Churn Propensity Model
**Version:** 1.0 | **Stack:** Python (ML) + React/Node.js (App) | **License:** Apache 2.0 (IBM dataset)

---

## 1. Project Overview

Build a binary classifier that predicts the probability that a customer will churn (leave/cancel) within the next billing cycle. The model uses the IBM Telco Customer Churn dataset — the most widely benchmarked public churn dataset in existence, with well-established community benchmarks for validation.

The project delivers:
1. A trained LightGBM classifier with SHAP interpretability
2. A React/Node.js SPA with a churn risk scorecard, customer lookup, what-if sliders, and a retention simulation panel
3. Full model card and documentation

**Business framing:** A California-based telecom provider serving 7,043 customers wants to identify which subscribers are at risk of cancelling their service so the retention team can intervene with targeted offers before the churn event occurs. The cost of intervention is far lower than the cost of acquisition.

---

## 2. Dataset

### 2a. Dataset — IBM Telco Customer Churn
| Property | Value |
|---|---|
| **Source** | IBM Sample Data / Kaggle |
| **Kaggle URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 (20 features + 1 target) |
| **Churn rate** | 26.5% positive (churned), 73.5% negative (retained) — moderate class imbalance |
| **License** | IBM Sample Data (free to use for education and analysis) |
| **Geography** | California, USA |

### 2b. Full Column Schema
| Column | Type | Description | Notes |
|---|---|---|---|
| `customerID` | string | Unique customer identifier | Drop before modeling |
| `gender` | binary string | Male / Female | |
| `SeniorCitizen` | integer (0/1) | Whether customer is a senior | Already binary |
| `Partner` | binary string | Has a partner (Yes/No) | |
| `Dependents` | binary string | Has dependents (Yes/No) | |
| `tenure` | integer | Months with the company | **Top predictor** — range: 0–72 |
| `PhoneService` | binary string | Has phone service (Yes/No) | |
| `MultipleLines` | string | Multiple phone lines | Yes / No / No phone service |
| `InternetService` | string | Internet type | DSL / Fiber optic / No |
| `OnlineSecurity` | string | Online security add-on | Yes / No / No internet service |
| `OnlineBackup` | string | Online backup add-on | Yes / No / No internet service |
| `DeviceProtection` | string | Device protection add-on | Yes / No / No internet service |
| `TechSupport` | string | Tech support add-on | Yes / No / No internet service |
| `StreamingTV` | string | Streaming TV add-on | Yes / No / No internet service |
| `StreamingMovies` | string | Streaming movies add-on | Yes / No / No internet service |
| `Contract` | string | Contract type | **Top predictor** — Month-to-month / One year / Two year |
| `PaperlessBilling` | binary string | Paperless billing (Yes/No) | |
| `PaymentMethod` | string | Payment method | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | float | Monthly charge (USD) | Range: $18.25–$118.75; mean ~$64.76 |
| `TotalCharges` | string → float | Total charges to date | **Contains 11 missing values** — coerce to numeric, impute |
| `Churn` | binary string | **TARGET** — Yes/No | Yes = churned; encode as 1/0 |

### 2c. Known Data Quality Issues
- `TotalCharges` is stored as string type in the raw CSV; convert with `pd.to_numeric(errors='coerce')` — this reveals 11 null values for customers with `tenure == 0` → impute with 0 or drop
- `customerID` column must be dropped before modeling
- All binary Yes/No columns need label encoding
- `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` have a third category: "No phone service" / "No internet service" — encode as 0 (same as "No") or keep as separate dummy
- No other nulls in the dataset

---

## 3. Modeling Target

| Property | Value |
|---|---|
| **Prediction target** | `Churn` — binary: 1 if churned, 0 if retained |
| **Definition of churn** | Customer terminated service (indicated in dataset label) |
| **Scope** | Single-period prediction: predict current churn status (no forward-looking window needed — labels are pre-assigned) |
| **Output** | Per-customer churn probability (0–1) + binary prediction at threshold |
| **Threshold** | Optimize threshold for business objective — see Section 5c |

---

## 4. Feature Engineering

### 4a. Derived Features (beyond raw columns)
| Feature | Description | Formula |
|---|---|---|
| `has_family` | Combined partner/dependents signal | `1 if Partner==1 OR Dependents==1` |
| `num_services` | Count of active add-on services | Sum of PhoneService + MultipleLines + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies (encoded as 0/1) |
| `monthly_per_tenure` | Average monthly spend rate | `MonthlyCharges / (tenure + 1)` |
| `total_charges_gap` | Difference between expected and actual total | `MonthlyCharges * tenure - TotalCharges` — flags billing anomalies |
| `is_month_to_month` | Binary: highest-risk contract type | `1 if Contract == 'Month-to-month'` |
| `is_fiber_optic` | Binary: fiber optic users churn more | `1 if InternetService == 'Fiber optic'` |
| `is_electronic_check` | Binary: highest churn payment method | `1 if PaymentMethod == 'Electronic check'` |
| `tenure_bucket` | Categorical: tenure stage | `0–12 months=New`, `13–36=Developing`, `37–72=Loyal` |

### 4b. Encoding Strategy
- Binary Yes/No columns → LabelEncoder (0/1)
- `InternetService`, `Contract`, `PaymentMethod` → pd.get_dummies (one-hot)
- `MultipleLines`, `OnlineSecurity`, etc. (3-category) → map {"No internet service": 0, "No": 0, "Yes": 1}
- `SeniorCitizen` → already 0/1, no change

---

## 5. Model Approach

### 5a. Primary Model — LightGBM Classifier
| Property | Value |
|---|---|
| **Algorithm** | `lightgbm.LGBMClassifier` |
| **Validation** | Stratified 5-fold cross-validation + held-out 20% test set (stratified by Churn) |
| **Class imbalance** | `scale_pos_weight = 73.5/26.5 ≈ 2.77`; also test SMOTE as alternative |
| **Primary metric** | AUC-ROC |
| **Secondary metrics** | F1 score (at optimized threshold), Precision, Recall, PR-AUC |
| **Interpretability** | SHAP TreeExplainer — global feature importance + per-customer waterfall |

**LightGBM Hyperparameter Starting Point:**
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 2.77,
    'verbose': -1,
    'random_state': 42
}
```

### 5b. Baseline Models
| Model | Expected AUC | Purpose |
|---|---|---|
| Logistic Regression | ~0.85 | Interpretable baseline |
| Decision Tree (max_depth=3) | ~0.78 | Transparent rule extraction |
| Always-predict-majority | 0.50 | Floor |

### 5c. Threshold Optimization for Business Use
The default 0.5 threshold is rarely optimal for churn. Use cost-sensitive threshold tuning:

```
Cost matrix (illustrative):
  - False Negative (miss a churner): Cost = $200 (lost customer lifetime value)
  - False Positive (intervene on non-churner): Cost = $20 (discount/offer cost)
  - Optimal threshold: minimize (FN * 200 + FP * 20) across thresholds
```
Output both the AUC-optimized threshold (0.5) and the business cost-optimized threshold.

### 5d. Published Benchmarks on This Exact Dataset
Based on multiple peer-reviewed papers and community analyses (2024–2026):
| Model | AUC-ROC | F1 Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.852 | ~0.62 | ~0.81 |
| Random Forest | 0.887 | ~0.65 | ~0.82 |
| XGBoost | 0.932 | 0.84 | 0.84 |
| **LightGBM (target)** | **0.930** | **0.84** | **0.84** |
| Stacking Ensemble | 0.975 | 0.82 | 0.95 |

**Target to beat:** AUC-ROC > 0.90, F1 > 0.80 on test set.

### 5e. Key SHAP Drivers (from literature)
Top predictors consistently identified across studies:
1. `Contract` (month-to-month = highest risk)
2. `tenure` (low tenure = high risk)
3. `MonthlyCharges` (higher = more risk)
4. `TotalCharges` (proxy for tenure × spend)
5. `InternetService` (Fiber optic users churn more)
6. `PaymentMethod` (electronic check users churn more)
7. `TechSupport` (lack of tech support increases risk)

---

## 6. App Requirements — React/Node.js SPA

### 6a. Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS |
| Backend API | Node.js + Express |
| Charts | Recharts (bar chart, gauge, scatter) |
| SHAP display | Horizontal waterfall bar chart |
| ML serving | Python FastAPI microservice |
| Model artifact | `lgbm_churn.pkl` via joblib |

### 6b. App Layout — Three Panels

**Panel 1 — Customer Churn Risk Lookup**
- Input: CustomerID search with autocomplete
- Output: Churn risk gauge (0–100%) with color zones: Green < 30%, Amber 30–60%, Red > 60%
- Risk tier badge: Low / Medium / High / Critical
- Key risk drivers: top 3 SHAP factors displayed as plain-English bullets ("High monthly charges are increasing churn risk by 18%")
- Recommended action tag: Watch / Soft outreach / Priority retention / Urgent intervention

**Panel 2 — What-If Retention Simulator**
Allow the retention team to simulate the impact of a retention offer on a specific customer:

| Slider / Input | What It Simulates |
|---|---|
| Contract upgrade (Month-to-month → 1yr → 2yr) | Offer contract lock-in incentive |
| Monthly charge reduction (−$0 to −$30) | Offer a discount |
| Add TechSupport (Yes/No toggle) | Bundle tech support into plan |
| Add OnlineSecurity (Yes/No toggle) | Bundle security into plan |
| Tenure increment (+3 / +6 / +12 months simulated) | Time-based retention trajectory |

- Real-time API update on each change → new churn probability displayed
- Side-by-side: "Current Risk: 78%" → "After Offer: 34%"
- Display estimated revenue saved: `churn_risk_reduction * avg_customer_LTV`

**Panel 3 — Portfolio Risk Dashboard**
- Histogram: distribution of churn probabilities across all 7,043 customers
- Scatter: tenure (x) vs. monthly charges (y), colored by churn probability decile
- KPI cards: # High-Risk Customers | Total MRR at Risk | Avg Churn Probability
- Segment table: churn rate by Contract type × Internet Service (3×3 heatmap grid)
- Filter: by contract type, internet service, senior citizen, tenure bucket

### 6c. API Endpoints
```
GET  /api/customer/:id          → churn probability, SHAP values, risk tier, recommended action
POST /api/predict               → body: customer feature dict → probability + SHAP + tier
POST /api/simulate              → body: customer features + proposed changes → new probability + delta
GET  /api/portfolio             → all customers with predicted churn probabilities
GET  /api/portfolio/segments    → churn rate by contract × internet service grid
GET  /api/customers/search?q=   → autocomplete
```

---

## 7. Model Documentation Requirements

| Document | Content |
|---|---|
| **Model Card** | Purpose, intended users (retention ops team), training data (IBM Telco, 7,043 customers, California), performance vs. baselines, threshold choice and business rationale, known limitations (telecom-specific — may not generalize to other industries), fairness note (gender should not be used as a decision feature in production) |
| **Data Dictionary** | All 21 raw columns + 8 engineered features with type, values, encoding applied |
| **Threshold Optimization Report** | Cost matrix assumptions, threshold sweep plot, business case for chosen threshold |
| **SHAP Analysis** | Global beeswarm plot, top 10 feature importances, 3 anonymized customer case studies (low/medium/high risk) |
| **Format** | Markdown in `/docs` + rendered in app About tab |

---

## 8. Constraints and Assumptions

- Dataset represents a single snapshot in time (not longitudinal) — model predicts current-state churn risk, not future churn window
- `gender` is included in the dataset but should be flagged in docs as a protected attribute and excluded from the retention action scoring (include in model as control but don't use in offer targeting)
- `SeniorCitizen` is a sensitive attribute — document usage carefully
- 11 missing `TotalCharges` values → impute with 0 (all have `tenure == 0`, no charges accrued yet)
- No true time-based validation possible with this static dataset; use stratified k-fold as proxy
- Random seed: 42 for all stochastic operations

---

## 9. Deliverables

```
project/
├── data/
│   └── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda_churn.ipynb
├── src/
│   ├── data_pipeline.py          # Cleaning + encoding
│   ├── feature_engineering.py    # Derived features
│   ├── train.py                  # LightGBM + SHAP + threshold optimization
│   └── api/main.py               # FastAPI service
├── app/
│   ├── server/                   # Node.js/Express
│   └── client/                   # React + Tailwind
├── models/lgbm_churn.pkl
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   ├── threshold_report.md
│   └── shap_analysis.md
├── requirements.txt
├── package.json
└── README.md
```

---

## 10. Getting Started Commands for Claude Code

```bash
# Download dataset (requires Kaggle CLI)
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip

# Install Python deps
pip install lightgbm shap pandas numpy scikit-learn imbalanced-learn fastapi uvicorn joblib

# Install Node + React deps
npm install express cors axios
npx create-react-app app/client
cd app/client && npm install recharts tailwindcss lucide-react
```

---

*Dataset citation: IBM Watson Analytics Sample Data. Telco Customer Churn. Available at: https://www.kaggle.com/datasets/blastchar/telco-customer-churn*
