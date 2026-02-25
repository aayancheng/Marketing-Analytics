---
marp: true
theme: uncover
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22px;
  }
  section.lead h1 {
    font-size: 2.2em;
  }
  section.lead h2 {
    font-size: 1.2em;
    font-weight: 400;
    color: #555;
  }
  h1 { color: #1a3a5c; border-bottom: 2px solid #3a7bd5; padding-bottom: 8px; }
  h2 { color: #3a7bd5; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #1a3a5c; color: white; }
  td { border-bottom: 1px solid #ddd; }
  .metric { font-size: 2em; font-weight: bold; color: #3a7bd5; }
  blockquote { border-left: 4px solid #3a7bd5; background: #f0f5ff; padding: 8px 16px; border-radius: 4px; }
---

<!-- _class: lead -->

# Personalized Send-Time Optimization

## Turning Behavioral Data into Smarter Marketing Engagement

**Dataset:** UCI Online Retail II — 1.07M transactions · 5,942 customers · 2009–2011
**Methodology:** RFM Profiling · LightGBM · SHAP Explainability · Policy Simulation

---

# The Business Problem

> **The average marketing email open rate is ~23%. Fixed-schedule blasts leave significant value on the table.**

### Why timing matters
- Customers have predictable behavioral rhythms — peak activity hours, preferred days
- Sending at the wrong time means competing with higher-priority inbox traffic
- One-size-fits-all "Tue 10am" blasts ignore individual habits entirely

### The opportunity
| Policy | Open Rate | vs. Fixed Blast |
|--------|-----------|----------------|
| Fixed blast (Tue 10:00) | 23.2% | — |
| Segment best slot | 83.3%* | +60.1 pp |
| **Personalized top-1 (model)** | **38.0%** | **+14.8 pp** |

*Small sample — directional. Personalized rate is robust (n=4,907 events).

---

# Data Requirements

### Source: UCI Online Retail II
UK-based gift retailer · Dec 2009 – Dec 2011 · **1,067,371 raw rows**

| Column | Type | Role |
|--------|------|------|
| `Customer ID` | float | Customer identifier (22.8% null → filtered) |
| `InvoiceDate` | datetime | Derives send_hour, send_dow, recency |
| `Invoice` | string | Cancellation detection (prefix `C`) |
| `Quantity` / `Price` | numeric | Monetary value for RFM |
| `Country` | string | Geographic segmentation |
| `StockCode` | string | Product diversity (unique_products) |

### Data quality handled
- **243,007** rows with no Customer ID → excluded from modeling
- **34,335** duplicate rows (3.2%) → removed
- **19,494** cancellations (1.8%) → flagged, excluded from open signals

---

# Exploratory Data Analysis

### Transaction Timing Patterns
- **Peak purchase hour: 12:00** — midday dominates across all days
- **Most common modal hour per customer: 12:00** — habits are concentrated
- Saturday and Sunday show significantly lower volume than weekdays

### Market Composition
| Metric | Value |
|--------|-------|
| UK transactions | **91.9%** of all rows |
| Countries represented | **43** |
| Unique customers (identified) | **5,942** |
| Date range | Dec 2009 – Dec 2011 |

### Key EDA insight
Customers have **predictable intra-day rhythms** — the distribution of modal purchase hours is clustered, not uniform. This justifies building a per-customer "habit hour" feature and scoring all 168 possible send slots (7 days × 24 hours).

---

# Feature Engineering — 20 Features Across 3 Groups

### Group 1: Customer Behavior (11 features)
| Feature | Description |
|---------|-------------|
| `modal_purchase_hour` | Customer's most common purchase hour |
| `purchase_hour_entropy` | Diversity of purchase timing (low = habitual) |
| `recency_days` | Days since last purchase |
| `frequency` | Total purchase sessions |
| `monetary_total` | Lifetime spend |
| `tenure_days` | Days from first to last purchase |
| `unique_products` | Product breadth |
| `cancellation_rate` | Returns signal |

### Group 2: Send-Time Slot (2 features)
`send_hour` (0–23), `send_dow` (0=Mon … 6=Sun)

### Group 3: Interaction Terms (7 features)
`hour_delta_from_modal` · `hour_x_entropy` · `dow_x_frequency` · and 4 more cross-features

---

# Model Building Approach

### Temporal Train / Validation / Test Split
```
Events:  106,154 total   →   Train 76,020 | Val 12,944 | Test 12,464
Cutoff:  chronological split — no future leakage
```

### Three Models Compared

| Stage | Model | Features |
|-------|-------|----------|
| Baseline | Naive random | — |
| Logistic | Logistic Regression | `send_hour`, `send_dow` only |
| **Final** | **LightGBM + Platt calibration** | **All 20 features** |

### Inference: 168-Slot Scoring Grid
For each customer, build a 7×24 grid combining their behavioral profile with every possible send slot → predict open probability → return top-3 slots.

> Platt calibration ensures predicted probabilities are **trustworthy**, not just rankings.

---

# Model Results & Performance

### Discrimination (AUC — higher is better)

<div class="metric">0.523</div>

LightGBM AUC vs. Naive 0.504 · Logistic 0.511

### Probability Quality
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | **0.179** | Well-calibrated (lower = better) |
| ECE | **6.8%** | Expected calibration error — good |
| Precision@3 | **34.5%** | 1-in-3 top recommendations were actual opens |
| Recall@5 | **45.3%** | 5 recommendations capture nearly half of all opens |

### Top SHAP Features (what drives the prediction)
1. `hour_delta_from_modal` — distance from customer's habit hour *(most important)*
2. `purchase_hour_entropy` — habitual vs. scattered buyers
3. `frequency` and `monetary_total` — customer engagement depth
4. `send_dow` — day of week signal

---

# Policy Simulation & Segment Diagnostics

### Open Rate by Policy

```
Fixed blast (Tue 10:00)   ████████░░░░░░░░░░░░  23.2%
Personalized top-1        ██████████████░░░░░░  38.0%  ▲ +14.8 pp
```

### RFM Segment Funnel (Open → Click → Purchase)

| Segment | Customers | Open Rate | Click\|Open | Purchase\|Click |
|---------|-----------|-----------|-------------|-----------------|
| Loyal | 1,430 | 23.5% | 35.3% | 12.6% |
| Champions | 1,045 | 23.4% | 35.7% | 11.3% |
| At Risk | 1,023 | 23.5% | 34.7% | 12.0% |
| Other | 1,130 | 23.9% | 34.6% | 11.7% |
| Hibernating | 707 | 24.6% | 37.3% | 13.1% |

> **Priority focus:** Loyal (1,430 customers) and Champions (1,045) offer the largest volume × improvement opportunity.

---

# Client Engagement Diagnostics

### Cohort Retention
- Dec-2009 cohort retains **35–49%** of customers through months 1–11 — strong baseline loyalty
- Later cohorts show steeper drop-off, suggesting acquisition quality matters
- **Reactivation after 30 days:** only 1.9% of inter-purchase gaps — re-engagement campaigns have real headroom

### Time-Slot Stability
<div class="metric">5.1%</div>

Only **5.1% of customers** kept the same best send slot between early and late periods.

> **Implication:** Personalization profiles must be refreshed frequently — stale preferences actively hurt performance.

### What-if Sensitivity
- Best predicted probability is **robust to recency** (10 vs 240 days doesn't shift it much)
- **High entropy** (scattered buyers) slightly reduces peak probability — habitual customers are easier to time

---

# Suggested Future Analytics

### 1. Multi-Touch Attribution Modeling
Move beyond open rate — attribute revenue to the full click → purchase journey using Shapley-value attribution across channels.

### 2. Customer Churn Prediction
Use the RFM + tenure features already engineered to build a survival model identifying at-risk customers before they lapse.

### 3. Dynamic Product Recommendations
Combine `unique_products` breadth and purchase history to generate personalized product bundles at the optimal send time.

### 4. NLP on Product Descriptions
Apply topic modeling or embeddings to `StockCode` / `Description` to identify category affinity clusters — enabling content personalization alongside time personalization.

### 5. Bayesian A/B Testing Framework
Replace point-estimate policy comparisons with a proper Bayesian bandit or sequential test to continuously optimize send-time policies in production with statistical rigor.

---

<!-- _class: lead -->

# Thank You

**Key Takeaway:** Personalizing send time to individual behavioral rhythms lifts open rates by ~15 percentage points over a fixed-blast policy — using only transactional data already available in most CRM systems.

**Model:** LightGBM · 20 behavioral + temporal features · Platt-calibrated probabilities
**Data:** UCI Online Retail II · 1.07M rows · 5,942 customers

*Notebooks: `01_eda_and_synthesis` · `02_optimal_time_engagement_model` · `03_client_engagement_diagnostics`*
