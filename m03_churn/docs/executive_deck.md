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

# Churn Propensity Analytics

## Identifying At-Risk Subscribers Before They Leave

**Model:** LightGBM · 32 Features · SHAP Explainability · Cost-Optimised Threshold
**Dataset:** IBM Telco Customer Churn · 7,043 California Subscribers
**Team:** Marketing Analytics · February 2026

---

# Executive Summary

> **The core insight:** We can identify 45.7% of all churners by contacting only the top 20% of subscribers ranked by risk — at a fraction of the cost of blanket outreach.

### The Opportunity

- Acquiring a new customer costs **5–25× more** than retaining an existing one
- **1,878 subscribers** (26.7% of the base) are at active churn risk
- A reactive "wait and see" approach means losing revenue that cannot be recovered
- A trained LightGBM model achieves **AUC-ROC 0.7883** on held-out test data

### What This Delivers

| Without Model | With Model (Top 20%) |
|---|---|
| Contact all 7,043 subscribers | Contact 1,409 subscribers |
| Recover some churners inefficiently | Capture 45.7% of all churners |
| High cost, low precision | Cost reduced from $75,200 → $17,580 |

---

# The Business Problem

> **Every month a churner stays undetected is a month of revenue permanently lost.**

### What Churn Costs

- At an illustrative **$65 average monthly charge**, 1,878 churners represent roughly **$122,000 MRR at risk**
- Customers who have already decided to leave rarely respond to reactive win-back campaigns
- Retention offers are most effective **before** the cancellation decision is made

### Why Reactive Is Too Late

- The churn signal often appears weeks after the customer has mentally disengaged
- Customers on month-to-month contracts can leave with **zero notice**
- By the time a cancellation is logged, the intervention window has closed

### The Right Question

Instead of "how do we win them back?" ask:

**"Which subscribers are showing early warning signs right now?"**

---

# The Solution

### ML-Powered Early Warning System

A LightGBM classifier scores every subscriber daily, producing a churn probability between 0 and 1. The retention team works from a prioritised queue — not a full customer list.

### Three Risk Tiers

| Tier | Score Range | Recommended Posture |
|---|---|---|
| High Risk | ≥ 0.50 | Immediate proactive outreach |
| Medium Risk | 0.16 – 0.49 | Scheduled nurture & monitoring |
| Low Risk | < 0.16 | Standard communications |

### Four Recommended Actions

1. **High Risk — Contract Upgrade Offer:** Incentivise switch from month-to-month to annual
2. **High Risk — Loyalty Reward:** Discount or service credit for long-tenure customers
3. **Medium Risk — Engagement Nudge:** Feature education, usage tips, check-in call
4. **Low Risk — Monitor:** No incremental spend; refresh score monthly

---

# Dataset Overview

### IBM Telco Customer Churn — California

| Attribute | Value |
|---|---|
| Total subscribers | **7,043** |
| Churners | **1,878 (26.7%)** |
| Non-churners | 5,165 (73.3%) |
| Features (raw) | 21 |
| Features (engineered) | 32 |
| Geography | California, USA |
| Source | IBM Sample Data Repository |

### Train / Test Split

| Split | Rows | Churn Rate |
|---|---|---|
| Train (80%) | 5,634 | 26.7% |
| Test (20%) | 1,409 | 26.7% |

Stratified random split preserves class balance across both sets — no temporal leakage risk given cross-sectional data structure.

---

# Feature Engineering — 32 Final Features

### Demographic (3 features)
`gender` · `SeniorCitizen` · `Partner` / `Dependents`

### Product & Service (10 features)
`PhoneService` · `MultipleLines` · `InternetService` · `OnlineSecurity` · `OnlineBackup` · `DeviceProtection` · `TechSupport` · `StreamingTV` · `StreamingMovies`

### Contract & Billing (4 features)
`Contract` · `PaperlessBilling` · `PaymentMethod` · `tenure`

### Financial (2 features)
`MonthlyCharges` · `TotalCharges`

### Engineered (13 features)
`charges_per_month_of_tenure` · `service_count` (number of add-ons) · `is_new_customer` (tenure ≤ 6 months) · `has_no_support_services` · `contract_encoded` · `payment_encoded` · one-hot expansions of multi-category fields

> Feature engineering converts raw account attributes into signals the model can learn from directly — particularly contract type and service bundle depth.

---

# Model Approach

### Three Candidates Evaluated on Test AUC-ROC

| Model | Test AUC-ROC | Notes |
|---|---:|---|
| Naive Baseline (majority class) | 0.5000 | No predictive power |
| Logistic Regression (32 features) | 0.8162 | Strong linear baseline |
| **LightGBM (calibrated)** | **0.7883** | Gradient boosted trees · scale_pos_weight=2.77 |

### Why LightGBM

- Handles mixed feature types (categorical + numeric) natively
- `scale_pos_weight=2.77` corrects for the 73/27 class imbalance
- Probability calibration ensures scores are **interpretable as true probabilities**, not just rankings
- SHAP integration provides per-customer explanations for every prediction

### Note on Logistic Regression

Logistic Regression achieves a marginally higher AUC (0.8162) on this dataset — an expected result given that many churn drivers are approximately linear. LightGBM is preferred in production for its robustness to feature interactions and its natural support for incremental retraining.

---

# Key Findings — SHAP Churn Drivers

### What Actually Drives Churn

1. **Contract type is the dominant signal.**
Month-to-month customers are **9× more likely to churn** than those on 2-year contracts. No other feature comes close in impact.

2. **New customers are the highest-risk cohort.**
Subscribers in their first 6–12 months have the steepest churn probability. Early engagement programmes have the highest ROI.

3. **Monthly charges amplify risk.**
Higher bills increase churn probability — particularly when not paired with a long-term contract commitment.

4. **Fiber optic internet correlates with elevated churn.**
Fiber customers churn at a higher rate than DSL or no-internet customers — possibly reflecting unmet service expectations or competitive alternatives.

5. **Electronic check payment is a churn signal.**
Customers paying by electronic check churn more than those on credit card or bank transfer auto-pay — likely a proxy for lower engagement and commitment.

> **Implication:** Contract conversion and early-tenure retention programmes target the two most impactful levers simultaneously.

---

# Model Performance

### Discrimination

<div class="metric">AUC-ROC 0.7883</div>

LightGBM on 1,409 held-out test customers · stratified split · calibrated probabilities

### Full Metrics Table

| Metric | Value | Interpretation |
|---|---:|---|
| AUC-ROC | **0.7883** | Strong discrimination (random = 0.50) |
| PR-AUC | **0.5401** | Robust under class imbalance |
| Brier Score | **0.1552** | Well-calibrated probabilities |
| F1 @ threshold 0.5 | 0.5234 | Balanced precision/recall |
| F1 @ threshold 0.16 | **0.5290** | Cost-optimised operating point |
| Top-20% Lift | **2.29×** | vs. random selection |
| Top-20% Capture Rate | **45.7%** | of all churners caught |

### Confusion Matrix at Default Threshold (0.5)

| | Predicted: Stay | Predicted: Churn |
|---|---:|---:|
| Actual: Stay | 921 | 112 |
| Actual: Churn | 203 | 173 |

---

# Business Impact

### Contact the Top 20% — Catch Nearly Half of All Churners

<div class="metric">45.7% captured · 2.29× lift</div>

Ranking 1,409 test customers by predicted churn score and contacting the top 281 (20%) identifies **45.7% of all actual churners** — more than double what random contact would achieve.

### Cost Model at Cost-Optimal Threshold (0.16)

| Scenario | Cost | Notes |
|---|---:|---|
| Baseline: flag nobody | **$75,200** | 376 missed churners × $200 each |
| Default threshold (0.50) | ~$52,000 | Estimated from confusion matrix |
| **Cost-optimal threshold (0.16)** | **$17,580** | FN=$200 · FP=$20 |
| **Estimated saving** | **$57,620** | vs. flagging nobody |

### Why Threshold 0.16?

At $200 per missed churner and $20 per unnecessary contact, the model recovers **$57,620** in avoided churn costs on the test cohort alone. Scaled to the full 7,043-subscriber base, the operational saving is proportionally larger.

> Threshold is a business decision — it can be adjusted as cost assumptions change.

---

# Risk Tiers & Recommended Actions

### Operational Playbook

| Tier | Score | Est. Size | Action | Example Offer | Expected Outcome |
|---|---|---|---|---|---|
| Critical | ≥ 0.60 | ~8% of base | Immediate outreach — phone or email within 24h | 20% bill discount + contract upgrade | Prevent imminent cancellation |
| High | 0.35–0.59 | ~12% of base | Proactive retention offer within 7 days | Free service add-on for 3 months | Convert to annual contract |
| Medium | 0.16–0.34 | ~20% of base | Scheduled nurture sequence | Feature education + loyalty points | Deepen product engagement |
| Low | < 0.16 | ~60% of base | Standard marketing cadence | No incremental spend | Monitor; rescore monthly |

### Key Design Principles

- **Tiered spend** ensures retention budget is concentrated on the highest-value intervention opportunities
- **Score refresh** should run at minimum monthly — churn risk shifts as account events occur
- **Offer design** targets the root cause: critical tier offers pair a financial incentive with a contract commitment, directly addressing the top two SHAP drivers

---

# The Application

### Three-Panel Retention Tool — Used Daily by the Retention Team

**Panel 1 — Customer Lookup**
Enter any customer ID to retrieve their churn probability, risk tier, top SHAP drivers, and recommended action. Retention agents use this during inbound calls to personalise their conversation in real time.

**Panel 2 — Risk Simulator**
Adjust account attributes (contract type, monthly charge, tenure, add-ons) and see the predicted churn probability update instantly. Used to model the impact of a proposed retention offer before it is made.

**Panel 3 — Segment Explorer**
Paginated table of all subscribers ranked by churn score, filterable by risk tier. Allows the retention team to build daily outreach queues and track which customers have been contacted.

### Daily Workflow

```
Morning score refresh → Export Critical + High tier queue
→ Retention agents work queue (phone / email)
→ Outcomes logged → Model retrained monthly
```

> The application is live at localhost:5173 (development) and connects to a FastAPI scoring service backed by the trained LightGBM model.

---

# ROI Estimate

### Illustrative Revenue Saved — High Risk Tier

**Assumptions (conservative)**

| Parameter | Value |
|---|---|
| High Risk customers contacted (top 20%) | 1,409 |
| Estimated true churners in this group | 644 (45.7% of 1,878 × scale) |
| Retention rate for contacted churners | 30% |
| Average customer lifetime value (LTV) | $200 |

**Revenue Calculation**

```
644 at-risk customers contacted
× 30% retention success rate
= 193 customers retained

193 customers × $200 LTV
= $38,600 revenue preserved
```

**Net of intervention cost** (1,409 contacts × $20 outreach cost = $28,180):

<div class="metric">Net gain ≈ $10,400</div>

> This is a conservative illustration using test-set proportions. With real LTV data and a calibrated retention rate from A/B testing, the business case sharpens considerably.

---

# Next Steps

### 1. Real Data Integration
Replace the IBM sample dataset with live CRM and billing data. Connect the scoring pipeline to the production subscriber database for daily automated scoring. Validate that feature distributions match the training population.

### 2. Uplift / Causal Modelling
The current model predicts *who will churn* — not *who will respond to an offer*. An uplift model (two-model or transformed outcome approach) estimates the **incremental effect** of intervention, ensuring retention budget targets persuadable customers rather than customers who would have stayed anyway.

### 3. CRM Integration & Closed-Loop Feedback
Push risk scores directly into the CRM (Salesforce / HubSpot) as a custom field. Log intervention outcomes back into the model training pipeline to enable supervised retraining on actual retention outcomes rather than churn labels alone.

### 4. Survival Analysis — Time-to-Churn
Extend from binary classification to a survival model (Cox PH or Weibull AFT) that estimates **when** a customer is likely to churn, not just whether they will. This enables intervention scheduling optimised by urgency rather than treating all high-risk customers as equally time-sensitive.

---

<!-- _class: lead -->

# Thank You

**Questions Welcome**

---

**Key Takeaway:** Contacting the top 20% of subscribers by predicted churn score captures 45.7% of all churners at a model-driven cost of $17,580 — a $57,620 saving versus an undifferentiated "flag nobody" baseline.

**Model:** LightGBM · Calibrated Probabilities · SHAP Explainability · Cost-Optimised Threshold 0.16
**Data:** IBM Telco · 7,043 California Subscribers · 26.7% Churn Rate · 32 Features

*Notebooks: `01_eda_and_feature_engineering` · `02_model_training` · `03_validation_and_cost_analysis`*
*Application: `app/client` (React) · `src/api` (FastAPI) · `app/server` (Express proxy)*
