# Project Brief: Model 2 — Customer Lifetime Value (CLV) Prediction
**Version:** 1.0 | **Stack:** Python (ML) + React/Node.js (App) | **License:** CC BY 4.0 (dataset)

---

## 1. Project Overview

Build a machine learning model that predicts the **expected total purchase value (GBP) per customer over the next 12 months** using the UCI Online Retail II dataset. The model leverages RFM (Recency, Frequency, Monetary) features plus behavioral and temporal signals derived from 2 years of transactional history.

The project delivers:
1. A data preprocessing and feature engineering pipeline
2. A trained LightGBM regressor (primary) with BG/NBD as interpretable baseline
3. A React/Node.js single-page app with customer lookup, what-if sliders, and portfolio explorer
4. A model card and full documentation

**Business framing:** The retailer wants to identify high-CLV customers for premium retention treatment, and flag at-risk high-value customers before they lapse — enabling smarter budget allocation across the customer base.

---

## 2. Dataset

### 2a. Dataset — UCI Online Retail II
| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **UCI ID** | 502 |
| **Direct download** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Kaggle mirror** | https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci |
| **Python fetch** | `from ucimlrepo import fetch_ucirepo; online_retail_ii = fetch_ucirepo(id=502)` |
| **Format** | Excel (.xlsx), two sheets — Year 2009-2010, Year 2010-2011 |
| **Raw rows** | ~1,067,371 transaction line items across both sheets |
| **Date range** | 01 Dec 2009 – 09 Dec 2011 (2 full years) |
| **Unique customers** | ~5,878 after removing null CustomerID rows |
| **Countries** | 38 countries; ~90% UK-based |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |

### 2b. Raw Column Schema
| Column | Type | Description | Notes |
|---|---|---|---|
| `Invoice` | string | 6-digit transaction ID | Prefix 'C' = cancellation; exclude for CLV |
| `StockCode` | string | 5-digit product code | Some non-product codes: POST, D, M, BANK CHARGES → filter |
| `Description` | string | Product name | Some nulls; not used in modeling |
| `Quantity` | integer | Units per line item | Negative = returns → exclude |
| `InvoiceDate` | datetime | Transaction timestamp | Hour + minute available |
| `Price` | float | Unit price in GBP (£) | May be named `UnitPrice` in v1 — handle both |
| `Customer ID` | float | 5-digit customer ID | ~25% null → drop |
| `Country` | string | Customer country | |

### 2c. Known Data Quality Issues (handle in preprocessing)
- `Customer ID` null for ~25% of rows → drop all null-CustomerID rows
- `Quantity` < 0 and/or `Invoice` starting with 'C' → cancellations/returns → exclude from revenue calculation (but cancellation rate can be a feature)
- Duplicate rows (~0.97%) → deduplicate on (Invoice, StockCode, CustomerID)
- Non-product StockCodes (POST, D, M, DOT, BANK CHARGES, AMAZONFEE) → filter out
- Some `Price` values are 0 or negative → treat as adjustments, exclude from revenue
- `InvoiceDate` outlier: transactions at exactly 00:00:00 may be bulk system entries
- Inconsistent `Price` for same `StockCode` across transactions → expected; use as-is

### 2d. Train/Test Split Strategy (Temporal, Out-of-Time)
This is critical for a CLV model — do NOT use random split.

```
Observation Window:  01 Dec 2009 → 30 Nov 2010  (12 months) → features derived here
Prediction Window:   01 Dec 2010 → 09 Dec 2011  (12 months) → ground truth CLV computed here
Reference Date:      30 Nov 2010 (used for recency calculation)
```

**Eligible customers:** Those with at least **2 purchases** in the observation window. Single-purchase customers are scored separately with a cold-start fallback (median CLV prediction).

**Expected training set size:** ~3,500–4,000 customers with sufficient history.

---

## 3. Modeling Target

| Property | Value |
|---|---|
| **Prediction target** | `clv_12m` — total GBP revenue per customer in prediction window |
| **Definition** | `sum(Quantity * Price)` for all non-cancelled invoices in Dec 2010 – Dec 2011 |
| **Scope** | Gross revenue in GBP; not margin-adjusted; returns excluded |
| **Distribution** | Highly right-skewed (expect median ~£200, mean ~£800, max > £50,000 due to wholesalers) |
| **Transformation** | Apply `log1p(clv_12m)` for training; inverse-transform predictions for display |
| **Zero-CLV customers** | Customers who did not purchase in prediction window → CLV = 0 (churned); include in training set |

---

## 4. Feature Engineering

All features are derived from the **observation window only** (Dec 2009 – Nov 2010).

### 4a. Core RFM Features
| Feature | Description | Formula |
|---|---|---|
| `recency_days` | Days since last purchase (from reference date) | `(reference_date - last_invoice_date).days` |
| `frequency` | Number of distinct invoices | `nunique(Invoice)` |
| `monetary_total` | Total spend in observation window (GBP) | `sum(Quantity * Price)` |
| `monetary_avg` | Average order value | `monetary_total / frequency` |
| `monetary_max` | Highest single order value | `max(invoice_total)` per customer |

### 4b. Behavioral Features
| Feature | Description | Formula |
|---|---|---|
| `tenure_days` | Days from first to last purchase | `(last_date - first_date).days` |
| `purchase_velocity` | Orders per month of active tenure | `frequency / (tenure_days / 30.44)` |
| `inter_purchase_days_avg` | Average days between consecutive orders | Mean of sorted invoice date deltas |
| `inter_purchase_days_std` | Consistency of purchase cadence | Std dev of date deltas |
| `unique_products` | Number of distinct StockCodes bought | `nunique(StockCode)` |
| `unique_product_categories` | Breadth of category engagement | Derived from StockCode prefix patterns |
| `cancellation_rate` | Proportion of invoices that are cancellations | `cancelled_invoices / total_invoices` |
| `avg_quantity_per_item` | Bulk vs. retail buyer signal | `mean(Quantity)` per line item |
| `uk_customer` | Binary: UK customer | 1 if Country == 'United Kingdom' |

### 4c. Temporal Features
| Feature | Description |
|---|---|
| `acquisition_month` | Month of first purchase (1–12) — seasonality of cohort |
| `acquisition_quarter` | Quarter of first purchase (1–4) |
| `purchased_in_q4` | Binary: has any purchase in Oct–Dec (gifting season signal) |
| `weekend_purchase_ratio` | Proportion of purchases on weekends |
| `evening_purchase_ratio` | Proportion of purchases after 17:00 |

### 4d. Derived RFM Score Features
| Feature | Description |
|---|---|
| `rfm_recency_score` | Quintile rank of recency (5=most recent) |
| `rfm_frequency_score` | Quintile rank of frequency (5=most frequent) |
| `rfm_monetary_score` | Quintile rank of monetary (5=highest spend) |
| `rfm_combined_score` | `rfm_recency_score + rfm_frequency_score + rfm_monetary_score` |

### 4e. Known Benchmark Features (from published research on this dataset)
Based on the ResearchGate paper (2025) using UCI Online Retail: recency, engagement rate, return rate, and discount dependency are identified as top predictors. The features above cover all of these.

---

## 5. Model Approach

### 5a. Primary Model — LightGBM Regressor
| Property | Value |
|---|---|
| **Algorithm** | LightGBM Regressor (`lgb.LGBMRegressor`) |
| **Target transformation** | `log1p(clv_12m)` during training; `expm1()` for predictions |
| **Validation** | Out-of-time: train on observation window features, evaluate on prediction window actuals |
| **Primary metric** | MAE (Mean Absolute Error) in GBP |
| **Secondary metrics** | RMSE, MAPE, Spearman correlation, decile lift (top decile capture rate) |
| **Interpretability** | SHAP TreeExplainer — waterfall chart per customer |

**LightGBM Hyperparameter Starting Point:**
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1
}
```

### 5b. Baseline Model — BG/NBD + Gamma-Gamma (Probabilistic)
| Property | Value |
|---|---|
| **Package** | `lifetimes` Python library |
| **Purpose** | Interpretable benchmark; also useful for documentation narrative |
| **Inputs** | recency, frequency, T (customer age), monetary |
| **Output** | Expected transactions (BG/NBD) × expected order value (Gamma-Gamma) = CLV estimate |
| **Limitation** | Assumes stationary purchase rate; weaker for highly seasonal data |

### 5c. Naive Baseline
Predict every customer's CLV = average CLV of training set. This is the floor to beat.

### 5d. Benchmark Context (from published research on UCI Online Retail)
Based on multiple GitHub and academic sources using this exact dataset:
- BG/NBD baseline MAE: ~£125–£155 per customer
- LightGBM with RFM features: RMSE ~£108, MAPE ~8.7% (best published result)
- Random Forest: RMSE ~£129.7, MAPE ~11.2%
- Linear regression: RMSE ~£154.5, MAPE ~14.9%
- **Target to beat:** RMSE < £130, MAPE < 12%

### 5e. Customer Segmentation (for app display)
After scoring, assign each customer a CLV tier:
| Segment | CLV Range | Label |
|---|---|---|
| Champions | Top 10% CLV | 🏆 Champions |
| High Value | 75th–90th percentile | ⭐ High Value |
| Mid Tier | 40th–75th percentile | 📈 Growing |
| Low Value | 10th–40th percentile | 💤 Occasional |
| Churned/Dormant | CLV = 0 in prediction window | ⚠️ Dormant |

---

## 6. App Requirements — React/Node.js SPA

### 6a. Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS |
| Backend API | Node.js + Express |
| Charts | Recharts (scatter plot, bar chart, gauge) |
| SHAP display | Horizontal bar chart rendered from SHAP values |
| ML serving | Python FastAPI microservice (called by Node backend) |
| Model artifact | `lgbm_clv.pkl` + `bgn bd_model.pkl` via `joblib` |

### 6b. App Layout — Single Page, Three Sections

**Section 1 — Customer CLV Lookup**
- Input: CustomerID search field with autocomplete
- Output: CLV prediction card showing:
  - Predicted 12-month CLV (£ value with confidence range ± 1 std)
  - CLV segment badge (Champions / High Value / Growing / Occasional / Dormant)
  - Percentile rank: "This customer is in the top X% by predicted CLV"
  - RFM summary bar: recency / frequency / monetary scores visualized as mini gauges
- SHAP waterfall chart: "Why this CLV?" — top 5 features pushing prediction up/down from base value

**Section 2 — What-If Sliders**
Allows marketing team to explore hypotheticals: "What if we re-engaged this customer and they bought again?"

| Slider | Range | Default |
|---|---|---|
| Recency (days since last purchase) | 1 – 365 | Customer's actual value |
| Frequency (number of orders) | 1 – 50 | Customer's actual value |
| Average Order Value (£) | £10 – £2,000 | Customer's actual AOV |
| Purchase Velocity (orders/month) | 0.1 – 5.0 | Customer's actual velocity |
| Cancellation Rate | 0% – 50% | Customer's actual rate |

- On slider change → real-time API call → predicted CLV updates with animation
- Side-by-side before/after CLV bar chart
- Label: "Revenue uplift potential: +£X if recency improved to Y days"

**Section 3 — Portfolio Explorer**
- Scatter plot: all customers plotted by `recency_days` (x) vs. `predicted_clv_12m` (y), colored by CLV segment
- Interactive: hover shows customer ID, CLV, segment, top SHAP driver
- Filter bar: filter by segment, country (UK / non-UK), CLV tier
- Summary metrics row: Total predicted portfolio CLV | Top 10% share of CLV | # Dormant high-value customers
- Table: top 20 customers by predicted CLV with segment, recency, and recommended action tag

### 6c. API Endpoints (Node.js → Python FastAPI)
```
GET  /api/customer/:id            → CLV prediction, SHAP values, RFM scores, segment
POST /api/predict                 → body: {recency, frequency, aov, velocity, cancel_rate} → CLV + SHAP
GET  /api/portfolio               → all customer predictions for scatter plot
GET  /api/segments/summary        → aggregate stats per segment
GET  /api/customers/search?q=     → autocomplete for CustomerID lookup
```

---

## 7. Model Documentation Requirements

| Document | Content |
|---|---|
| **Model Card** | Purpose, business use case (budget allocation, retention targeting), training data summary, feature list, performance table vs. baselines, limitations (UK B2B wholesaler skew, no demographic data, synthetic channel data), fairness note |
| **Data Dictionary** | All raw + engineered features with type, range, derivation formula, and business meaning |
| **Methodology** | Observation/prediction window design rationale, log transformation justification, BG/NBD vs. LightGBM trade-off discussion |
| **Segment Definition Table** | How predicted CLV maps to tiers, percentile cutoffs |
| **Validation Report** | MAE/RMSE/MAPE table vs. baselines, decile lift chart, calibration by-decile plot (predicted vs. actual CLV), residuals by customer segment |
| **Format** | Markdown files in `/docs` folder + rendered in app's About/Methodology tab |

---

## 8. Constraints and Assumptions

- CLV is **gross revenue in GBP**, not margin-adjusted (no margin data available)
- Returns/cancellations are excluded from revenue computation; included as a behavioral feature (`cancellation_rate`)
- Customers with fewer than 2 purchases in observation window are excluded from training; at inference they receive a cold-start response: "Insufficient purchase history — predicted CLV: £[median]"
- No channel or product-level breakdown in v1
- `Price` column is assumed to be unit price in GBP — no currency conversion applied
- The dataset is biased toward UK B2B wholesale customers (large bulk orders); this inflates the high end of CLV distribution and should be called out in the model card
- Random seed: 42 for all stochastic operations

---

## 9. Deliverables

```
project/
├── data/
│   ├── raw/                        # UCI Online Retail II xlsx files (both sheets)
│   ├── processed/
│   │   ├── transactions_clean.csv  # Cleaned, filtered transaction table
│   │   ├── customer_features.csv   # One row per customer, all features
│   │   └── clv_labels.csv          # Ground truth CLV per customer (prediction window)
├── notebooks/
│   └── 01_eda_and_clv_analysis.ipynb
├── src/
│   ├── data_pipeline.py            # Cleaning + window splitting
│   ├── feature_engineering.py      # RFM + behavioral features
│   ├── train_lgbm.py               # LightGBM training + SHAP
│   ├── train_bgnbd.py              # BG/NBD + Gamma-Gamma baseline
│   └── api/
│       └── main.py                 # FastAPI prediction + SHAP service
├── app/
│   ├── server/                     # Node.js/Express backend
│   └── client/                     # React frontend (Tailwind, Recharts)
├── models/
│   ├── lgbm_clv.pkl
│   └── bgnbd_model.pkl
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── segment_definitions.md
│   └── validation_report.md
├── requirements.txt
├── package.json
└── README.md
```

---

## 10. Getting Started Commands for Claude Code

```bash
# 1. Fetch dataset
pip install ucimlrepo lifetimes lightgbm shap pandas numpy scikit-learn fastapi uvicorn joblib openpyxl

# 2. Download UCI Online Retail II
python -c "
from ucimlrepo import fetch_ucirepo
import pandas as pd
d = fetch_ucirepo(id=502)
df = d.data.features
df.to_csv('data/raw/online_retail_ii.csv', index=False)
print(f'Downloaded {len(df)} rows')
"

# 3. Install Node + React deps
npm install express cors axios
npx create-react-app app/client
cd app/client && npm install recharts tailwindcss @headlessui/react lucide-react
```

---

## 11. Key References

- Dataset: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- BG/NBD Model: Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. *Marketing Science*, 24(2), 275–284.
- Community benchmark: ResearchGate (2025) — LightGBM achieves RMSE ~£108, MAPE ~8.7% on UCI Online Retail with RFM + behavioral features and OOT validation.
- `lifetimes` Python library: https://lifetimes.readthedocs.io
