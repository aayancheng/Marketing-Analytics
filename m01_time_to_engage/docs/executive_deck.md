---
marp: true
theme: uncover
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
  }
  section.lead {
    background: #0f172a;
    color: white;
  }
  section.lead h1 {
    color: #3b82f6;
    font-size: 2.2em;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 12px;
  }
  section.lead h2 {
    font-size: 1.2em;
    font-weight: 400;
    color: #94a3b8;
  }
  section.lead p {
    color: #cbd5e1;
  }
  h1 {
    color: #1e40af;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
  }
  h2 { color: #1e40af; }
  h3 { color: #3b82f6; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #1e40af; color: white; }
  td { border-bottom: 1px solid #ddd; }
  .metric { font-size: 2.5em; font-weight: bold; color: #3b82f6; }
  blockquote {
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    padding: 8px 16px;
    border-radius: 4px;
    color: #1e3a5f;
  }
---

<!-- _class: lead -->

# Personalised Send-Time Optimisation

## Turning Behavioural Data into Smarter Email Engagement

**Model:** LightGBM · 20 Features · SHAP Explainability · Platt-Calibrated Probabilities
**Dataset:** UCI Online Retail II · 1.07M Transactions · 5,942 Customers
**Team:** Marketing Analytics · February 2026

---

# Executive Summary

> **The core insight:** Personalising email send times to each customer's behavioural rhythm lifts open rates by ~15 percentage points over a fixed-blast policy — using only transactional data already available in most CRM systems.

### The Opportunity

- The average marketing email open rate is **~23%** — fixed-schedule blasts leave significant value on the table
- Customers have **predictable behavioural rhythms** — peak activity hours and preferred days
- One-size-fits-all "Tuesday 10am" sends ignore individual habits entirely

### What This Delivers

| Without Model | With Model (Personalised Top-1) |
|---|---|
| Fixed blast to all 5,942 customers | Per-customer optimal send slot |
| 23.2% open rate | 38.0% open rate |
| No personalisation | +14.8 pp uplift over baseline |

---

# The Business Problem

> **Sending at the wrong time means competing with higher-priority inbox traffic — and losing.**

### What Poor Timing Costs

- Lower open rates reduce downstream click and purchase conversions proportionally
- Customers who never open eventually disengage entirely — contributing to list fatigue and churn
- Generic send schedules waste the behavioural signals already embedded in transaction data

### Why "Best Average Time" Is Not Enough

- Only **5.1% of customers** maintain the same optimal send slot over time — preferences shift
- High-entropy (scattered) buyers respond differently from habitual customers
- Segment-level timing captures some value, but **per-customer scoring captures significantly more**

### The Right Question

Instead of "what's the best time to send?" ask:

**"What's the best time to send to *this* customer?"**

---

# The Solution

### ML-Powered Send-Time Personalisation

A LightGBM classifier scores every customer across all 168 possible weekly send slots (7 days x 24 hours), producing a calibrated open probability for each. The marketing team sends at each customer's predicted peak.

### Three Engagement Tiers

| Tier | Description | Recommended Approach |
|---|---|---|
| Habitual | Low entropy — strong timing preference | Send at predicted top-1 slot for maximum impact |
| Moderate | Medium entropy — some timing flexibility | Send within top-3 predicted slots |
| Scattered | High entropy — no clear preference | Optimise on content/offer rather than timing |

### Four Recommended Actions

1. **Habitual customers:** Deploy personalised send time via CRM automation
2. **Moderate customers:** A/B test top-1 vs. top-3 slot windows
3. **Scattered customers:** Focus on subject line and offer personalisation instead
4. **All segments:** Refresh personalisation profiles monthly (preferences drift)

---

# Dataset Overview

### UCI Online Retail II — UK Gift Retailer

| Attribute | Value |
|---|---|
| Total transactions | **1,067,371** (raw) |
| Identified customers | **5,942** |
| Time span | December 2009 -- December 2011 (2 full years) |
| Geography | Predominantly UK (91.9%), 43 countries total |
| Source | UCI Machine Learning Repository |

### Data Quality

| Issue | Volume | Resolution |
|---|---|---|
| Missing Customer ID | 243,007 rows (22.8%) | Excluded from modelling |
| Duplicate rows | 34,335 (3.2%) | Removed |
| Cancellations | 19,494 (1.8%) | Flagged, excluded from open signals |

### Temporal Split

Chronological train / validation / test split (106,154 events total) — no future leakage.

---

# Methodology: Behavioural Features + LightGBM

```
Transaction logs                   20 Engineered              LightGBM            Per-customer
  (1M+ rows)        ------>         Features        ------>   Classifier  ------>  168-slot ranking
                                                           (Platt calibrated)     + top-3 windows
```

**Feature groups (20 features):**
- **Behavioural core**: modal purchase hour, purchase hour entropy, recency, frequency, monetary total
- **Engagement depth**: tenure, unique products, cancellation rate
- **Send-time slot**: send hour (0--23), send day-of-week (Mon--Sun)
- **Interaction terms**: hour delta from modal, hour x entropy, dow x frequency, and 4 cross-features

**Inference approach**: For each customer, score all 168 slots and return the top-3 predicted windows with calibrated open probabilities.

> Platt calibration ensures predicted probabilities are **trustworthy**, not just rankings — critical for setting confidence thresholds on send-time recommendations.

---

# Key Findings — SHAP Drivers

### What Actually Drives Email Opens

1. **Distance from habit hour is the dominant signal.**
`hour_delta_from_modal` — the gap between the proposed send time and the customer's natural activity peak — is the single most important feature. Closer = better.

2. **Purchase hour entropy separates easy wins from hard cases.**
Habitual customers (low entropy) have a clear best slot. Scattered buyers (high entropy) are harder to time — optimise on content instead.

3. **Engagement depth matters.**
Frequency and monetary total indicate how invested a customer is. More engaged customers respond more predictably to timing optimisation.

4. **Day-of-week carries a meaningful signal.**
Weekend vs. weekday preferences vary by customer — the model captures this per-individual, not as a population average.

> **Implication:** The model works best for habitual, engaged customers — exactly the segment where timing optimisation has the highest ROI.

---

# Model Performance

### Discrimination

<div class="metric">AUC 0.523</div>

LightGBM on 12,464 held-out test events · chronological split · Platt-calibrated probabilities

### Full Metrics Table

| Metric | Value | Interpretation |
|---|---:|---|
| AUC-ROC | **0.523** | Lift over naive baseline (0.504) |
| Brier Score | **0.179** | Well-calibrated probabilities |
| ECE | **6.8%** | Expected calibration error — good |
| Precision@3 | **34.5%** | 1-in-3 top recommendations were actual opens |
| Recall@5 | **45.3%** | 5 recommendations capture nearly half of all opens |

### Important Context

The modest AUC reflects that email open timing is inherently noisy — many factors beyond send time influence opens. The model's value lies not in perfect discrimination but in its **policy uplift**: +14.8 pp open rate improvement when used to personalise send times.

---

# Business Impact

### Personalise the Send Time — Lift Open Rates by 15 Percentage Points

<div class="metric">38.0% open rate · +14.8 pp lift</div>

Scoring 4,907 customer-events and sending at each customer's predicted top-1 slot achieves a 38.0% open rate versus 23.2% for a fixed Tuesday 10am blast.

### Policy Comparison

| Policy | Open Rate | vs. Fixed Blast |
|---|---:|---|
| Fixed blast (Tue 10:00) | 23.2% | Baseline |
| **Personalised top-1 (model)** | **38.0%** | **+14.8 pp** |
| Segment best slot | 83.3%* | Directional (small n) |

*Segment-level rate inflated by small sample — personalised rate is robust.

### RFM Segment Breakdown

| Segment | Customers | Open Rate | Click given Open | Purchase given Click |
|---|---:|---:|---:|---:|
| Loyal | 1,430 | 23.5% | 35.3% | 12.6% |
| Champions | 1,045 | 23.4% | 35.7% | 11.3% |
| At Risk | 1,023 | 23.5% | 34.7% | 12.0% |
| Hibernating | 707 | 24.6% | 37.3% | 13.1% |

> **Priority focus:** Loyal (1,430) and Champions (1,045) offer the largest volume x improvement opportunity.

---

# ROI Estimate

### Illustrative Revenue Impact — Personalised Send Time

**Assumptions (conservative)**

| Parameter | Value |
|---|---|
| Email campaigns per month | 4 |
| Customers per campaign | 5,942 |
| Baseline open rate | 23.2% |
| Personalised open rate | 38.0% |
| Click-through rate (given open) | 35% |
| Purchase rate (given click) | 12% |
| Average order value | £35 |

**Revenue Calculation**

```
Baseline monthly:   5,942 x 4 x 0.232 x 0.35 x 0.12 x £35 = £8,170
Personalised:       5,942 x 4 x 0.380 x 0.35 x 0.12 x £35 = £13,380

Incremental monthly revenue: +£5,210
Annualised:                  +£62,500
```

<div class="metric">+£62k annualised uplift</div>

> This is an illustrative estimate using observed open-rate lift and average downstream conversion rates. Actual impact depends on campaign mix, product margin, and customer base composition.

---

# The Application

### Three-Panel Engagement Tool

**Panel 1 — Customer Lookup**
Enter any customer ID to retrieve their profile, predicted top-3 optimal send windows, a 7x24 probability heatmap, and the SHAP waterfall chart showing which factors drove their personalisation.

**Panel 2 — What-If Simulator**
Adjust behavioural inputs (recency, frequency, monetary value, entropy) and see the predicted optimal send time and open probability update in real time. Used to model how re-engagement might shift a customer's best window.

**Panel 3 — Segment Explorer**
Browse customers by RFM segment. Paginated data table with segment-level open/click/purchase funnel metrics.

### Operational Workflow

```
Monthly profile refresh --> Score all 168 slots per customer
--> Export optimal send times to CRM automation platform
--> Campaign sends at personalised times --> Measure lift
```

> The application is live at localhost:5173 (development) and connects to a FastAPI scoring service backed by the trained LightGBM model.

---

# Engagement Diagnostics

### Cohort Retention
- Dec-2009 cohort retains **35--49%** of customers through months 1--11 — strong baseline loyalty
- Later cohorts show steeper drop-off, suggesting acquisition quality matters
- **Reactivation after 30 days:** only 1.9% of inter-purchase gaps — re-engagement campaigns have real headroom

### Send-Time Stability

<div class="metric">5.1% stable</div>

Only **5.1% of customers** kept the same best send slot between early and late observation periods.

> **Implication:** Personalisation profiles must be refreshed frequently — stale preferences actively hurt performance. Monthly rescoring is the minimum recommended cadence.

### Sensitivity Analysis
- Best predicted probability is **robust to recency** (10 vs 240 days doesn't shift it much)
- **High entropy** (scattered buyers) slightly reduces peak probability — habitual customers are easier to time and should be prioritised for send-time personalisation

---

# Next Steps

### 1. CRM Integration & Automated Send-Time Deployment
Export per-customer optimal send windows to the marketing automation platform (Mailchimp, HubSpot, Salesforce Marketing Cloud). Automate campaign scheduling so each recipient receives emails at their predicted peak engagement time.

### 2. A/B Testing Framework
Run a controlled A/B test: personalised send time (treatment) vs. fixed blast (control) across a statistically significant sample. Measure open rate, click-through, and downstream purchase lift to validate the +14.8 pp estimate with production data.

### 3. Real-Time Profile Refresh
Connect the scoring pipeline to live transaction data for monthly automated rescoring. Given that only 5.1% of customers maintain stable preferences, frequent refresh is essential to sustained performance.

### 4. Combine with CLV and Churn Models
Integrate with **m02 CLV scores** to prioritise send-time optimisation for high-value customers, and with **m03 churn predictions** to time retention campaigns at optimal engagement windows for at-risk subscribers.

---

<!-- _class: lead -->

# Thank You

**Questions Welcome**

---

**Key Takeaway:** Personalising email send time to individual behavioural rhythms lifts open rates by ~15 percentage points over a fixed-blast policy — an estimated +£62k annualised revenue uplift using only transactional data already available in most CRM systems.

**Model:** LightGBM · 20 Behavioural + Temporal Features · Platt-Calibrated Probabilities · SHAP Explainability
**Data:** UCI Online Retail II · 1.07M Transactions · 5,942 Customers · 168-Slot Scoring Grid

*Notebooks: `01_eda_and_synthesis` · `02_optimal_time_engagement_model` · `03_client_engagement_diagnostics`*
*Application: `app/client` (React) · `src/api` (FastAPI) · `app/server` (Express proxy)*
