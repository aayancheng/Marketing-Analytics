# Research Brief: Customer Lifetime Value (CLV)

## Domain Overview

Customer Lifetime Value (CLV) is the estimated net revenue a business will earn from a customer over the entire duration of the relationship. CLV prediction transforms this from a historical average into a forward-looking, customer-level score.

**Core business problems solved:**
- **Marketing budget allocation**: Direct spend toward high-CLV customer channels (target CLV:CAC ratio ≥ 3:1)
- **Retention prioritization**: Identify which customers are worth re-engagement (5–7x cheaper to retain vs acquire; 5% retention improvement → up to 95% profit lift)
- **Personalization**: Assign promotion budgets proportional to predicted value
- **Portfolio management**: Tiered account management for wholesale vs. individual buyers

**UCI Online Retail II context**: UK wholesale/gifting retailer with predominantly B2B customers. CLV prediction distinguishes high-frequency wholesale accounts from occasional individual buyers.

## KPIs & Success Metrics

### Regression Metrics
| Metric | Target | Why |
|--------|--------|-----|
| MAE | < £130 | Interpretable in £; robust to outliers |
| RMSE | < £130 | Penalizes whale customer under-prediction |
| MAPE | < 12% | Scale-independent across segments |
| Spearman r | > 0.5 | Correct customer ordering matters more than absolute accuracy |

### Business Metrics
| Metric | Target |
|--------|--------|
| Top-decile lift | 3–5x (top 10% predicted should hold 30–50% of actual CLV) |
| Decile calibration | Predicted vs actual CLV aligned per decile |

## Data Sources

| Dataset | Size | Key Columns | License | CLV Suitability |
|---------|------|-------------|---------|-----------------|
| **UCI Online Retail II** (selected) | 90 MB, ~1M rows | Invoice, StockCode, Quantity, Price, CustomerID, Country | CC BY 4.0 | Excellent — 2-year window enables 12m observation → 12m prediction split |
| CDNOW Transactions | 2 MB, 70K rows | CustomerID, Date, NumCDs, DollarValue | Public domain | Gold-standard BG/NBD benchmark; too small for ML |
| Ta-Feng Grocery | 20 MB, 818K rows | Date, CustomerID, ProductCategory, Quantity, Price | Public | 4-month window too short for 12m CLV |
| Olist Brazilian E-Commerce | 43 MB, 100K orders | OrderID, CustomerID, PaymentValue, ReviewScore | CC BY-NC-SA 4.0 | Median customer has 1 order only |
| REES46 Multi-Category | 4.3 GB, 285M events | EventTime, EventType, ProductID, Price, UserID | CC BY-SA 4.0 | High potential but requires heavy preprocessing |

**Recommendation**: UCI Online Retail II — already available at `shared/data/raw/online_retail_ii.csv`. Two-year span enables proper temporal split, ~4,300 repeat UK buyers, CC BY 4.0 license.

## Model Candidates

| Approach | Type | Strengths | Limitations |
|----------|------|-----------|-------------|
| **LightGBM Regressor** (primary) | ML | Fast, handles skewed targets with log1p, SHAP explainability | Requires feature engineering |
| **BG/NBD + Gamma-Gamma** (baseline) | Probabilistic | Interpretable, no training data beyond RFM, well-validated | Assumes stationary purchase rates |
| Naive mean baseline | Statistical | Floor to beat | No customer differentiation |
| XGBoost Regressor | ML | Widely used in Kaggle CLV | Slower than LightGBM, similar performance |
| LSTM / Seq2Seq | Deep Learning | Captures temporal patterns | Needs 1000s of transactions per customer |

**Selected**: LightGBM Regressor (primary) + BG/NBD + Gamma-Gamma (interpretable baseline)

## Recommended Stack

| Component | Choice |
|-----------|--------|
| Primary model | LightGBM Regressor (`log1p` target) |
| Baseline | BG/NBD + Gamma-Gamma via `lifetimes` |
| Target | `clv_12m` = sum(Quantity × Price) in prediction window |
| Features | RFM + behavioral + temporal (~20 columns) |
| Interpretability | SHAP TreeExplainer |
| Serving | FastAPI (8002) → Express proxy (3002) → React/Vite (5174) |

## Reference Patterns (from m01_time_to_engage)

**Reuse directly:**
- FastAPI lifespan pattern (load artifacts, score all entities at startup, cache in app.state)
- PaginatedResponse schema (items/pages convention)
- Express proxy structure (port swap only)
- React component library (Sidebar, Card, StatCard, CustomerSelect, SliderControl, ShapChart, DataTable, SegmentPicker)
- lib/api.js fetch pattern (fetchAllCustomers uses items/pages)
- lib/hooks.js (useCustomer, usePredict, useSegments — same logic)
- sys.path.insert fix in main.py

**Adapt for CLV:**
- Replace heatmap (7×24 grid) with scatter plot (recency vs CLV)
- Replace TopSlots with CLV prediction card (single £ value)
- WhatIfView: numeric sliders (recency, frequency, AOV, velocity, cancel_rate) instead of dropdowns
- Metrics: MAE/RMSE/MAPE/Spearman instead of AUC/Brier
- No isotonic calibration (regression, not classification)
- Cold-start: < 2 purchases → return median CLV

## Key References

1. **Fader, Hardie & Lee (2005)** — "Counting Your Customers" the Easy Way. Marketing Science 24(2). Foundational BG/NBD paper.
2. **Fader & Hardie (2013)** — The Gamma-Gamma Model of Monetary Value. Companion monetary model for BG/NBD.
3. **Bauer & Jannach (2021)** — Improved CLV Prediction with Seq2Seq Learning. ACM TKDD. Best recent comparison of probabilistic, tree-based, and deep learning CLV approaches.
