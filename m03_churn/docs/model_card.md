# Model Card — Model 3: Churn Propensity

**Project:** Marketing Analytics — Churn Propensity Scoring
**Model:** LightGBM Binary Classifier (calibrated)
**Version:** 1.0 | **Date:** 2026-02-25
**Authors:** Marketing Analytics Team

---

## Purpose

Predict the probability that an individual telecom subscriber will churn (terminate service), enabling proactive retention intervention before the customer disengages.

---

## Intended Users

- **Retention operations team** — use risk scores to prioritize daily outreach queues and allocate intervention budget
- **Customer success managers** — use per-customer SHAP explanations to tailor retention conversations to the specific reasons a customer is flagged as high risk

---

## Business Context

A California telecom provider serving 7,043 subscribers. At the observed 26.7% churn rate, the business loses approximately one in four customers per period. Each missed churner costs an estimated $200 in lost lifetime value; each unnecessary outreach costs $20. At the cost-optimal threshold of 0.16, the model reduces total intervention cost from $75,200 (flag nobody) to $17,580 — an estimated **$57,620 saving per scoring cycle**.

---

## Training Data

| Property | Value |
|---|---|
| Dataset | IBM Telco Customer Churn (Kaggle, CC0 / IBM Sample) |
| Source file | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers (single snapshot) |
| Churn rate | 26.7% positive (1,869 churners / 5,174 retained) |
| Geography | California telecom provider |
| License | CC0 public domain / IBM sample dataset |

---

## Features

32 features total: 25 encoded from 21 raw columns + 7 engineered features.

**Raw encoded (21 columns → 25 after one-hot encoding):**
Demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account tenure (`tenure`), service subscriptions (PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies), internet service type (one-hot: DSL / Fiber optic / No), contract type (one-hot: Month-to-month / One year / Two year), payment method (one-hot: 4 options), billing fields (`MonthlyCharges`, `TotalCharges`, `PaperlessBilling`).

**Engineered (7 features):**
`has_family`, `num_services`, `monthly_per_tenure`, `total_charges_gap`, `is_month_to_month`, `is_fiber_optic`, `is_electronic_check`.

See `docs/data_dictionary.md` for full definitions and encoding rules.

---

## Output

Each scored customer receives three values:

| Output | Description |
|---|---|
| `churn_probability` | Calibrated probability of churn in range [0, 1] |
| `risk_tier` | Categorical label (see table below) |
| `recommended_action` | Operational instruction for the retention team |

**Risk tiers:**

| Tier | Score Range | Recommended Action |
|---|---|---|
| High Risk | > 0.60 | Urgent intervention |
| Medium-High Risk | 0.40 – 0.60 | Priority retention offer |
| Medium-Low Risk | 0.20 – 0.40 | Soft outreach |
| Low Risk | < 0.20 | Monitor |

---

## Performance

| Model | AUC-ROC | PR-AUC | Brier Score | F1 @ 0.5 |
|---|---:|---:|---:|---:|
| Naive baseline | 0.5000 | — | — | — |
| Logistic Regression (32 features) | 0.8162 | — | — | — |
| **LightGBM (calibrated)** | **0.7883** | **0.5401** | **0.1552** | **0.5234** |

**LightGBM @ threshold 0.5:** Precision = 0.6070, Recall = 0.4601

**Decision threshold:** The default threshold of 0.50 optimizes F1. The **cost-optimal threshold of 0.16** (based on $200 FN / $20 FP cost ratio) is recommended for operational use — it raises recall to 0.923, capturing 92% of churners at the expense of more false positives.

**Top-20% decile lift:** 2.29x — customers in the top-scored quintile churn at 61.3%, versus 26.7% overall. Targeting only this quintile captures **45.7% of all churners**.

---

## Top Predictive Drivers (SHAP)

1. **Contract type** — Month-to-month customers churn at ~3x the rate of two-year contract customers
2. **Tenure** — Shorter tenure strongly predicts churn; long-tenured customers are anchored
3. **MonthlyCharges** — Higher monthly spend correlates with elevated churn risk
4. **InternetService: Fiber optic** — Fiber customers show disproportionately high churn despite premium pricing
5. **PaymentMethod: Electronic check** — Electronic check payers have the highest observed churn rate of any payment method

---

## Known Limitations

- **Static snapshot:** The dataset is a single point-in-time observation with no longitudinal time series and no forward-looking survival window. The model predicts current-state churn, not time-to-churn.
- **Telecom-specific:** Features and learned patterns are specific to a California telecom provider. The model should not be assumed to generalize to SaaS, retail, or other verticals without revalidation.
- **Synthetic / sample data:** The Kaggle IBM dataset is a curated sample, not live production data. Real production telemetry would improve AUC-ROC and reduce distributional shift risk at deployment.
- **Class imbalance:** At 26.7% churn, the minority class is meaningful but not severe. Imbalance is addressed via `scale_pos_weight = 2.77` in LightGBM; SMOTE is available as an alternative.
- **LightGBM underperforms logistic regression on AUC-ROC** (0.7883 vs. 0.8162) on this dataset. This is expected for low-row-count, well-structured tabular data. LightGBM is retained as the production model for its superior calibration (Brier Score 0.155), native SHAP support, and better threshold-level business metrics.

---

## Fairness & Risk

| Attribute | Status | Constraint |
|---|---|---|
| `gender` | Included as a model feature | **MUST NOT** be used to offer differential retention incentives. ECOA and applicable state law prohibit conditioning financial product offers on gender. The feature may inform aggregate analysis but must not drive individual offer decisions. |
| `SeniorCitizen` | Included as a model feature; sensitive attribute | Document all retention interventions disaggregated by senior/non-senior status. Review for disparate impact before production deployment. |
| Other demographics | `Partner`, `Dependents` included | Low regulatory sensitivity; monitor for proxy discrimination. |

**Disparate impact analysis is recommended before production deployment.** Compare predicted churn rates and actual intervention rates across gender and SeniorCitizen groups to verify that retention resources are not disproportionately withheld from any protected class.

---

## Maintenance

| Trigger | Action |
|---|---|
| PSI > 0.2 on any key feature (`tenure`, `MonthlyCharges`, `Contract_Month-to-month`) | Retrain model on refreshed data |
| Actual churn rate deviates > 5 pp from training base rate (26.7%) | Recalibrate probability calibrator; consider full retrain |
| New contract types, payment methods, or service bundles added | Re-engineer features; retrain |
| Quarterly (minimum) | Refresh customer feature table; re-score all customers |

---

*For full technical documentation see [`model_documentation.md`](model_documentation.md). For feature definitions see [`data_dictionary.md`](data_dictionary.md). For raw metrics see [`validation_report.md`](validation_report.md).*
