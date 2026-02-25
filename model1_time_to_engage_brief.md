# Project Brief: Model 1 — Optimal Time-to-Engage
**Version:** 1.0 | **Stack:** Python (ML) + React/Node.js (App) | **License:** CC BY 4.0 (base dataset)

---

## 1. Project Overview

Build a machine learning model that predicts the **optimal hour and day of week to contact each customer** to maximize purchase conversion probability. The model uses the UCI Online Retail II transactional dataset as its behavioral foundation, with a **synthetically generated email campaign layer** added on top to simulate realistic outreach timing data.

The project delivers:
1. A data synthesis pipeline (UCI transactions → simulated email send/open/convert events)
2. A trained LightGBM classifier
3. A React/Node.js single-page app with what-if sliders and customer lookup
4. A model card and full documentation

**Business framing:** A UK-based online gift retailer wants to reduce email blast waste by personalizing send times per customer based on their historical purchase behavior patterns.

---

## 2. Dataset

### 2a. Base Dataset — UCI Online Retail II
| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **UCI ID** | 502 |
| **Direct download** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Kaggle mirror** | https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci |
| **Python fetch** | `from ucimlrepo import fetch_ucirepo; online_retail_ii = fetch_ucirepo(id=502)` |
| **Format** | Excel (.xlsx), two sheets (Year 2009-2010, Year 2010-2011) |
| **Raw rows** | ~1,067,371 transaction line items |
| **Date range** | 01 Dec 2009 – 09 Dec 2011 (2 full years) |
| **Unique customers** | ~5,900 (after removing rows with null CustomerID) |
| **Countries** | 38 countries; ~90% UK-based |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |

### 2b. Raw Column Schema (UCI Online Retail II)
| Column | Type | Description | Notes |
|---|---|---|---|
| `Invoice` | string (nominal) | 6-digit transaction ID | Prefix 'C' = cancellation |
| `StockCode` | string (nominal) | 5-digit product code | |
| `Description` | string (nominal) | Product name | 1,454 nulls in v1; more in v2 |
| `Quantity` | integer | Units per line item | Negative = returns/cancellations |
| `InvoiceDate` | datetime | Transaction timestamp | **Key field** — has hour + minute |
| `Price` | float | Unit price in GBP (£) | Some negative (adjustments) |
| `Customer ID` | float (nominal) | 5-digit customer ID | ~25% null — drop these rows |
| `Country` | string (nominal) | Customer country | |

### 2c. Known Data Quality Issues (handle in preprocessing)
- `Customer ID` is null for ~25% of rows → drop; non-nullable for modeling
- `Quantity` < 0 and `Price` < 0 → cancellations/returns → filter out for engagement modeling
- `Invoice` starting with 'C' → cancellation → exclude
- Duplicate rows exist (~0.97% per community analysis) → deduplicate
- Some `StockCode` values are non-product codes ('POST', 'D', 'M', 'BANK CHARGES') → filter
- `InvoiceDate` hours range from ~06:00 to ~20:00 (business hours, B2B wholesalers dominant)

### 2d. Synthetic Email Campaign Layer (to be generated)
Since UCI Online Retail II has no email send data, Claude Code must **synthesize a campaign events table** on top of the transaction data using the following logic:

**Synthesis rules:**
```
For each customer with >= 5 transactions:
  - Generate N email send events (N ~ Poisson(lambda=2) per month of activity)
  - Send time = sampled from truncated normal centered on customer's modal purchase hour ± 3h
  - Day of week = weighted by customer's historical purchase day distribution
  - open_event = Bernoulli(p) where p = base_open_rate * time_alignment_score
    base_open_rate = 0.25 (industry benchmark for retail email)
    time_alignment_score = 1.0 if sent within ±2h of modal hour, else decays to 0.4
  - click_event = Bernoulli(p=0.35) given open
  - purchase_event = Bernoulli(p=0.12) given click
  - Add Gaussian noise to prevent perfect signal
```

**Resulting synthetic campaign table schema:**
| Column | Type | Description |
|---|---|---|
| `customer_id` | string | From UCI CustomerID |
| `campaign_id` | string | Synthetic campaign identifier |
| `send_datetime` | datetime | Simulated send timestamp |
| `send_hour` | integer | 0–23 |
| `send_dow` | integer | 0=Monday … 6=Sunday |
| `opened` | binary | 1 if email opened |
| `clicked` | binary | 1 if link clicked (given open) |
| `purchased` | binary | 1 if purchase within 48h (given click) |
| `channel` | string | 'email' (v1 scope) |

**Total synthetic rows expected:** ~80,000–120,000 campaign send events across ~4,500 eligible customers.

---

## 3. Modeling Target

| Property | Value |
|---|---|
| **Prediction target** | Probability of email open (binary: `opened = 1`) |
| **Positive outcome** | Email opened within 24 hours of send |
| **Secondary target** | Probability of purchase conversion (for SHAP narrative) |
| **Granularity** | Per customer × per candidate send slot |
| **Output** | Top 3 recommended send windows (hour + day combinations) per customer |
| **Framing** | Binary classifier trained per (customer_features, send_hour, send_dow) pair |

---

## 4. Feature Engineering

### 4a. Customer Behavioral Features (derived from UCI transactions)
| Feature | Description | Derivation |
|---|---|---|
| `modal_purchase_hour` | Most frequent purchase hour | `mode(InvoiceDate.hour)` per customer |
| `modal_purchase_dow` | Most frequent purchase day of week | `mode(InvoiceDate.dayofweek)` |
| `purchase_hour_entropy` | Spread of purchase hours (low = habitual) | Shannon entropy of hour distribution |
| `avg_daily_txn_count` | Average transactions per active day | txn_count / active_days |
| `recency_days` | Days since last transaction | `(reference_date - last_invoice_date).days` |
| `frequency` | Number of distinct invoices | `nunique(Invoice)` |
| `monetary_total` | Total spend in GBP | `sum(Quantity * Price)` |
| `tenure_days` | Days between first and last purchase | `(last_date - first_date).days` |
| `country_uk` | Binary: customer in UK | 1 if Country == 'United Kingdom' |
| `unique_products` | Number of distinct StockCodes purchased | `nunique(StockCode)` |
| `cancellation_rate` | Proportion of cancelled invoices | cancelled_invoices / total_invoices |

### 4b. Send Slot Features (per candidate time window)
| Feature | Description |
|---|---|
| `send_hour` | Hour of day (0–23) |
| `send_dow` | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | Binary: send_dow >= 5 |
| `is_business_hours` | Binary: 9 <= send_hour <= 17 |
| `hour_delta_from_modal` | Absolute difference: `abs(send_hour - modal_purchase_hour)` |
| `dow_match` | Binary: send_dow == modal_purchase_dow |
| `industry_open_rate_by_hour` | External prior: average retail email open rate by hour (from Brevo/Mailchimp benchmarks) |

### 4c. Interaction Features
| Feature | Description |
|---|---|
| `hour_x_entropy` | `send_hour * purchase_hour_entropy` — habitual customers penalized less for off-peak |
| `recency_x_frequency` | `recency_days * frequency` — RFM interaction |

---

## 5. Model Approach

| Property | Value |
|---|---|
| **Primary algorithm** | LightGBM BinaryClassifier (via `lightgbm` Python package) |
| **Baseline** | Logistic Regression on send_hour + send_dow only |
| **Naive baseline** | Uniform random send slot (AUC target to beat: 0.50) |
| **Validation strategy** | Time-based split: train on campaign events from first 18 months, test on final 6 months |
| **Primary metric** | AUC-ROC (target: > 0.70 on synthetic test set) |
| **Secondary metric** | Top-3 hit rate: % of customers where optimal actual slot falls in model's top-3 recommendations |
| **Class imbalance** | Expected open rate ~25%; handle with `scale_pos_weight` in LightGBM |
| **Interpretability** | SHAP values (TreeExplainer) for each prediction |

### 5a. LightGBM Hyperparameter Starting Point
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
    'scale_pos_weight': 3.0,  # ~75% negative, 25% positive
    'verbose': -1
}
```

### 5b. Benchmark Context
Based on community research using similar UCI transaction data + simulated engagement:
- Logistic regression baseline: AUC ~0.62–0.65
- LightGBM with RFM + time features: AUC ~0.72–0.78 expected
- Key predictive features expected: `hour_delta_from_modal`, `modal_purchase_hour`, `recency_days`, `purchase_hour_entropy`

---

## 6. App Requirements — React/Node.js SPA

### 6a. Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS |
| Backend API | Node.js + Express |
| Charts | Recharts or Chart.js |
| SHAP display | Horizontal bar chart from SHAP values served via API |
| ML serving | Python FastAPI microservice (called by Node backend) OR pre-computed lookup table |
| Model artifact | `.pkl` via `joblib`; loaded in Python service |

### 6b. App Layout — Single Page, Tabbed Sections

**Section 1 — Customer Lookup**
- Input: CustomerID text field (validated against known IDs)
- Output: Customer profile card showing recency, frequency, total spend, modal hour, modal day
- Output: Top 3 recommended send windows (e.g., "Tuesday 10:00–11:00 | Confidence: 82%")
- Visual: 7×24 heatmap of predicted open probability by day × hour (color-coded, interactive hover)

**Section 2 — What-If Sliders**
- Sliders for: Recency (days since last purchase), Frequency (number of orders), Modal Purchase Hour (0–23), Purchase Hour Entropy (0=habitual, 1=random)
- On slider change: real-time API call updates the heatmap and top-3 recommendations
- Show SHAP waterfall chart explaining why the recommended slot scored highest

**Section 3 — Segment Explorer**
- Dropdown: filter customers by RFM segment (Champions, Loyal, At Risk, Hibernating)
- Scatter plot: all customers in segment plotted by modal_hour vs. open_rate, colored by recency bucket
- Table: top 10 customers in segment with their recommended send slots

### 6c. API Endpoints (Node.js → Python service)
```
GET  /api/customer/:id          → customer profile + top-3 slots + heatmap data
POST /api/predict               → body: {recency, frequency, modal_hour, entropy} → top-3 + SHAP
GET  /api/customers             → paginated list for segment explorer
GET  /api/segments              → RFM segment summary stats
```

---

## 7. Model Documentation Requirements

| Document | Content |
|---|---|
| **Model Card** | Purpose, intended users (marketing ops team), training data summary, performance metrics, known limitations, fairness note (UK-centric dataset) |
| **Data Dictionary** | All raw + engineered features with type, range, derivation formula |
| **Synthesis Methodology** | Full explanation of how email campaign events were generated, assumptions made, and why the synthetic signal is a valid proxy |
| **Validation Report** | AUC-ROC curve, confusion matrix at 0.5 threshold, top-3 hit rate by customer segment |
| **Format** | Markdown files in `/docs` folder + rendered in app's "About" section |

---

## 8. Constraints and Assumptions

- Only email channel in scope for v1
- UK customers only for core model training (filter `Country == 'United Kingdom'`); other countries can be scored but flagged
- Minimum 5 historical transactions required per customer to generate a prediction; fewer → "insufficient data" response
- CustomerID is anonymized; no PII in dataset
- Synthetic campaign data is seeded with a fixed random seed (42) for reproducibility
- Negative quantities (returns) and cancelled invoices (Invoice starting with 'C') are excluded from behavioral feature computation
- `Price` column may be named `UnitPrice` in v1 of the dataset — handle both column names in preprocessing

---

## 9. Deliverables

```
project/
├── data/
│   ├── raw/                        # UCI Online Retail II xlsx files
│   ├── processed/                  # Cleaned transaction table
│   └── synthetic/                  # Generated campaign events table
├── notebooks/
│   └── 01_eda_and_synthesis.ipynb  # EDA + synthesis pipeline exploration
├── src/
│   ├── data_pipeline.py            # Cleaning + synthesis script
│   ├── feature_engineering.py      # Feature derivation
│   ├── train.py                    # LightGBM training + SHAP computation
│   └── api/
│       └── main.py                 # FastAPI prediction service
├── app/
│   ├── server/                     # Node.js/Express backend
│   └── client/                     # React frontend
├── models/
│   └── lgbm_time_to_engage.pkl     # Trained model artifact
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   ├── synthesis_methodology.md
│   └── validation_report.md
├── requirements.txt
├── package.json
└── README.md
```

---

## 10. Getting Started Commands for Claude Code

```bash
# 1. Fetch dataset
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=502); d.data.features.to_csv('data/raw/online_retail_ii.csv')"

# 2. Install Python deps
pip install lightgbm shap pandas numpy scikit-learn fastapi uvicorn joblib openpyxl

# 3. Install Node deps
npm install express cors axios

# 4. Install React deps
npx create-react-app app/client
cd app/client && npm install recharts tailwindcss @headlessui/react
```

---

*Dataset citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D*
