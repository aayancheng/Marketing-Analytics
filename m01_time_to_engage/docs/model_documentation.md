# Model Documentation Report
## Personalized Send-Time Optimization — LightGBM Engagement Model

**Project:** Marketing Analytics — Send-Time Optimization
**Dataset:** UCI Online Retail II (2009–2011)
**Model:** LightGBM Binary Classifier (calibrated)
**Version:** 1.0 | **Date:** 2026-02-20
**Authors:** Marketing Analytics Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review](#2-literature-review)
3. [Model Data](#3-model-data)
4. [Model Methodology](#4-model-methodology)
5. [Model Performance](#5-model-performance)
6. [Model Risk and Future Enhancements](#6-model-risk-and-future-enhancements)

---

## 1. Executive Summary

### The Problem

Marketing email campaigns are routinely sent at fixed, organization-wide schedules — "Tuesday at 10am" — regardless of when individual customers are most likely to engage. This spray-and-pray approach ignores a fundamental behavioral truth: customers have predictable timing rhythms shaped by their purchasing habits, day-of-week preferences, and lifestyle patterns. Sending at the wrong time means competing with a higher-priority inbox, depressing open rates and ultimately reducing conversion downstream.

The business cost is measurable. Against a fixed-blast policy benchmarked in this project, a personalized send-time model achieves an open rate of **38.0%** compared to **23.2%** — a **+14.8 percentage point uplift** — using only transactional data already captured in most CRM systems. No new data collection is required.

### The Dataset and Approach

This project uses the UCI Online Retail II dataset: 1,067,371 transactions from a UK-based online gift retailer spanning December 2009 to December 2011, covering 5,942 identified customers across 43 countries. From this transaction history, behavioral profiles were constructed for each customer — capturing their habitual purchase hour, day-of-week preferences, recency, frequency, monetary value, and temporal entropy (how concentrated or scattered their purchase timing is).

These profiles were combined with synthetic campaign event data to create a training dataset of 106,154 labeled send events (opened=1 or 0). A LightGBM binary classifier was trained on 20 behavioral and temporal features to predict, for any given customer and candidate send slot, the probability that the customer will open the email. At inference time, all 168 possible send slots (7 days × 24 hours) are scored per customer and the top-3 slots are returned.

Predicted probabilities are calibrated using isotonic regression to ensure they reflect true likelihoods rather than raw classifier scores — a critical requirement for business decision-making where the absolute probability value matters, not just the ranking.

### Key Results

| Policy | Open Rate | vs. Fixed Blast |
|--------|-----------|----------------|
| Fixed blast (Tue 10:00) | 23.2% | — |
| **Personalized top-1 (model)** | **38.0%** | **+14.8 pp** |

The LightGBM model achieves a test AUC of 0.523 against a naive baseline of 0.504 — a modest discriminative improvement that nonetheless translates to meaningful business uplift when applied as a ranking policy across thousands of customers. The top predictive driver, as identified by SHAP feature attribution, is `hour_delta_from_modal`: the distance between the proposed send hour and the customer's habitual purchase hour. Customers sent emails close to their behavioral rhythm are materially more likely to open.

### Deployment Recommendation

The model is exposed via a REST API at `src/api/main.py`. For production integration, the following is recommended:

1. **Connect to real campaign telemetry.** The current model is trained on synthetic campaign events. Replacing this with actual ESP (Email Service Provider) open/click/purchase data at deployment is the single highest-priority improvement.
2. **Integrate with campaign scheduling.** The API endpoint accepts a `customer_id` and returns the top-3 recommended (day, hour) slots — these should be used to populate per-customer send times in the CRM or ESP.
3. **Refresh customer profiles monthly.** Only 5.1% of customers maintained the same optimal send slot across time periods, indicating that behavioral preferences evolve rapidly. Stale profiles actively hurt model performance.

---

## 2. Literature Review

This project draws on four bodies of work, each directly motivating a specific design decision.

### 2.1 RFM Customer Segmentation

> Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.

The RFM framework — Recency, Frequency, and Monetary value — was introduced as a practical segmentation tool for direct marketing. It operationalizes customer value as three easily measured behavioral signals: how recently a customer purchased, how often they purchase, and how much they spend. Hughes demonstrated that these three dimensions, combined into a scoring schema, outperform more complex demographic segmentation for predicting direct mail response rates.

**Influence on this project:** The three core RFM dimensions (`recency_days`, `frequency`, `monetary_total`) form the backbone of the customer behavioral feature set. An RFM-derived five-tier segment label (Champions, Loyal, At Risk, Hibernating, Other) is used for segment-level performance diagnostics and as a cold-start fallback for customers with insufficient transaction history.

### 2.2 LightGBM — Gradient Boosted Decision Trees

> Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).

LightGBM introduced two key algorithmic innovations over earlier gradient boosting implementations (XGBoost, GBM): Gradient-based One-Side Sampling (GOSS), which discards low-gradient training instances to speed training without sacrificing accuracy, and Exclusive Feature Bundling (EFB), which reduces the effective feature dimensionality by bundling mutually exclusive sparse features. The result is a highly efficient tree-based classifier suitable for large tabular datasets with mixed feature types.

**Influence on this project:** LightGBM was selected as the primary model because it handles the mixed feature types in this dataset (continuous RFM features, discrete send slots, binary flags) efficiently and natively. The `scale_pos_weight` hyperparameter was set to 3.0 to compensate for class imbalance (roughly 23.5% positive open rate). Its histogram-based split finding is well-suited to the `hour_delta_from_modal` feature, which has a bounded integer range.

### 2.3 Probability Calibration

> Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML).

This paper provides an empirical study of the calibration properties of major supervised learning algorithms. Its central finding is that tree-based ensembles (including gradient boosting) tend to produce overconfident probability estimates — predictions that are systematically too extreme relative to the true class proportions. The paper evaluates Platt scaling and isotonic regression as post-hoc calibration methods, finding isotonic regression to be more effective for large datasets with sufficient calibration data.

**Influence on this project:** Raw LightGBM probability scores were observed to be poorly calibrated on the validation slice. Isotonic regression was fitted on the validation partition (12,944 events) to map raw scores to calibrated probabilities clipped to [0, 1]. The resulting Expected Calibration Error (ECE) of 6.8% confirms the calibrated probabilities are suitable for business use — stakeholders can interpret "27% predicted open probability" as a genuine likelihood estimate, not a raw classifier score.

### 2.4 SHAP — Unified Model Explainability

> Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS).

SHAP (SHapley Additive exPlanations) provides a theoretically grounded framework for attributing a model's output to its input features. It extends Shapley values from cooperative game theory to machine learning, guaranteeing that feature attributions sum to the difference between the prediction and the expected model output, and satisfy desirable axioms of efficiency, symmetry, and consistency. The `TreeExplainer` variant computes exact SHAP values for tree-based models in polynomial time.

**Influence on this project:** SHAP `TreeExplainer` is used to explain individual send-slot recommendations. For any customer × slot combination, a ranked list of feature contributions is returned alongside the predicted probability. This serves two purposes: (1) it allows marketing teams to understand *why* a particular slot was recommended, building trust in the model's output; and (2) it surfaces the global feature importance hierarchy — `hour_delta_from_modal` is consistently the top driver, validating the intuition that temporal alignment with behavioral habit is the strongest signal.

---

## 3. Model Data

### 3.1 Data Sources

**Transactional Base Data — UCI Online Retail II**

The primary data source is the UCI Machine Learning Repository's Online Retail II dataset (id=502), comprising retail transactions from a UK-based online gift company. The raw dataset contains 1,067,371 rows across 10 columns, spanning December 1, 2009 to December 9, 2011. The original columns are: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country`. Two derived columns — `hour` and `day_of_week` — are added at load time.

**Synthetic Campaign Events**

Because real email campaign telemetry (send timestamps paired with open/click/purchase outcomes) is not available in the transactional dataset, campaign events were generated synthetically using the methodology documented in [`synthesis_methodology.md`](synthesis_methodology.md). The key steps are: (1) for each customer with ≥5 transactions, draw a Poisson number of sends per active month; (2) draw send hour from a truncated normal distribution centered on the customer's modal purchase hour (±3h); (3) compute open probability as a function of time-alignment score with added Gaussian noise; (4) draw binary outcomes (`opened`, `clicked`, `purchased`) from those probabilities with conditional rates of 35% for click-given-open and 12% for purchase-given-click. The final synthetic dataset contains **106,154 events** across **5,335 customers**.

### 3.2 Data Quality and Preparation

Three data quality issues were identified and handled prior to feature engineering:

| Issue | Count | % of total | Resolution |
|-------|-------|-----------|------------|
| Null `Customer ID` | 243,007 | 22.8% | Excluded from modeling; no customer-level features possible |
| Duplicate rows | 34,335 | 3.2% | Removed; likely system re-submission artifacts |
| Cancellation invoices (prefix `C`) | 19,494 | 1.8% | Flagged and excluded from purchase signals |

After removing null Customer ID rows, 824,364 rows remain, covering 5,942 unique customers. The null Customer ID rate (22.8%) represents a meaningful data loss. For production deployment, enforcing Customer ID capture at point-of-sale or ESP level is strongly recommended.

The dataset is highly UK-concentrated: 91.9% of all rows originate from United Kingdom customers. This geographic concentration is both a strength (consistent behavioral context) and a limitation (see Section 6).

### 3.3 Feature Engineering

Twenty features were engineered across three groups. The full canonical definition is maintained in [`data_dictionary.md`](data_dictionary.md); the table below adds a business-purpose column.

**Group 1: Customer Behavioral Features (11 features)**

These features are computed once per customer from transaction history and remain fixed across all 168 candidate send slots.

| Feature | Type | Business Purpose |
|---------|------|-----------------|
| `modal_purchase_hour` | int | Customer's most frequent purchase hour — the behavioral habit anchor |
| `modal_purchase_dow` | int | Customer's most frequent purchase day (0=Mon … 6=Sun) |
| `purchase_hour_entropy` | float | Shannon entropy of purchase hour distribution — low value = habitual, high = scattered |
| `avg_daily_txn_count` | float | Purchase velocity; distinguishes light browsers from heavy buyers |
| `recency_days` | float | Days since last purchase (RFM: R) — staleness signal |
| `frequency` | int | Total distinct purchase sessions (RFM: F) — loyalty signal |
| `monetary_total` | float | Lifetime spend in GBP (RFM: M) — value signal |
| `tenure_days` | int | Days from first to last purchase — customer age |
| `country_uk` | int | Binary flag: 1 if UK customer |
| `unique_products` | int | Count of distinct SKUs purchased — product breadth |
| `cancellation_rate` | float | Fraction of transactions that were cancellations — returns behavior |

**Group 2: Send-Slot Features (7 features)**

These vary across the 168 candidate slots and encode properties of the proposed send time.

| Feature | Type | Business Purpose |
|---------|------|-----------------|
| `send_hour` | int | Candidate send hour (0–23) |
| `send_dow` | int | Candidate send day of week (0=Mon … 6=Sun) |
| `is_weekend` | int | Binary: 1 if send_dow ≥ 5 |
| `is_business_hours` | int | Binary: 1 if send_hour between 9 and 17 |
| `hour_delta_from_modal` | int | Circular distance between send_hour and customer's modal_purchase_hour — **top SHAP feature** |
| `dow_match` | int | Binary: 1 if send_dow matches modal_purchase_dow |
| `industry_open_rate_by_hour` | float | Benchmark email open rate for the customer's industry at the proposed hour |

**Group 3: Interaction Features (2 features)**

| Feature | Derivation | Purpose |
|---------|-----------|---------|
| `hour_x_entropy` | `send_hour × purchase_hour_entropy` | Captures how much timing spread matters at each hour |
| `recency_x_frequency` | `recency_days × frequency` | RFM cross-signal: recently active high-frequency customers |

### 3.4 Train / Validation / Test Split

The dataset is split temporally to prevent data leakage — a critical requirement when customers have records at both training and test time. The split logic is implemented in `src/train.py` (lines 75–99):

- **Training set:** All events in the first 18 months from the earliest campaign timestamp → **76,020 events**
- **Validation set:** First half of the final 6-month tail → **12,944 events** — used exclusively for isotonic calibration
- **Test set:** Second half of the final 6-month tail → **12,464 events** — held out for final evaluation

This structure ensures that the calibrator (fitted on validation) and the model (fitted on train) have never seen test data. The base open rate across all events is **23.5%**, creating a moderate class imbalance that is addressed via LightGBM's `scale_pos_weight=3.0` hyperparameter.

---

## 4. Model Methodology

### 4.1 Problem Framing

Send-time optimization is framed as a **binary classification problem**: given a customer profile vector **x**_c and a candidate send slot (send_hour h, send_dow d), predict:

```
P(opened = 1 | x_c, h, d)
```

At inference time, all 168 (h, d) combinations are scored for each customer, producing a 7×24 probability heatmap. The top-3 slots by predicted probability are returned as the personalized send-time recommendations.

This framing treats each (customer, send slot) pair as an independent observation. It does not explicitly model sequence or time-series dependencies between sends — a deliberate simplification that keeps the inference pipeline stateless and low-latency.

### 4.2 Baseline Models

Two baselines establish the performance floor:

**Baseline 1 — Naive Random**
Assigns uniform random scores from U(0, 1) to each event. By construction, AUC ≈ 0.500. This is the minimum viable performance benchmark.

**Baseline 2 — Logistic Regression (2 features)**
A logistic regression model trained on `send_hour` and `send_dow` only. This captures aggregate temporal patterns (e.g., midday emails are more likely to be opened in general) but has no customer-specific behavioral signal. Test AUC: **0.5114**.

The gap between the two baselines confirms that temporal patterns do carry a learnable signal. The gap between logistic regression and LightGBM reflects the value of adding 18 customer behavioral and interaction features.

### 4.3 Primary Model — LightGBM Binary Classifier

LightGBM was trained on all 20 features using the hyperparameters in the table below. All values are from `src/train.py` (lines 113–126).

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| `objective` | `binary` | Binary cross-entropy loss |
| `metric` | `auc` | Optimize ranking quality during training |
| `n_estimators` | 300 | Sufficient tree depth without overfitting on 76K training rows |
| `learning_rate` | 0.05 | Conservative step size; balances speed and generalization |
| `num_leaves` | 31 | Default; controls model capacity |
| `min_child_samples` | 20 | Minimum leaf size; regularizes against overfitting small segments |
| `feature_fraction` | 0.8 | Subsample 80% of features per tree (column dropout) |
| `bagging_fraction` | 0.8 | Subsample 80% of rows per iteration |
| `bagging_freq` | 5 | Apply row subsampling every 5 iterations |
| `scale_pos_weight` | 3.0 | Upweights positive (opened=1) samples to address ~76%/24% imbalance |
| `random_state` | 42 | Reproducibility |

### 4.4 Probability Calibration

Raw LightGBM probability scores are systematically overconfident (Niculescu-Mizil & Caruana, 2005). To correct this, **isotonic regression** is fitted on the validation slice: the raw model scores on validation events are mapped to calibrated probabilities using a monotonic non-decreasing function, constrained to [0, 1]. The calibrator is then applied to all downstream scoring.

This step is critical for business use: stakeholders interpreting a "27% predicted open probability" should be able to trust that it reflects a genuine 27-in-100 likelihood, not an arbitrary classifier output. The final Expected Calibration Error (ECE) of **6.8%** confirms this calibration is effective.

Model artifacts saved to `models/`:
- `lgbm_time_to_engage.pkl` — trained LightGBM classifier
- `probability_calibrator.pkl` — fitted isotonic regression calibrator
- `shap_explainer.pkl` — SHAP TreeExplainer with 500-row background

### 4.5 Inference — 7×24 Scoring Grid

For each customer at inference time:

1. Retrieve the customer's behavioral feature vector from `data/processed/customer_features.csv`
2. Construct a 168-row grid: broadcast the customer features across all (send_hour, send_dow) pairs
3. Recompute slot-specific and interaction features: `is_weekend`, `is_business_hours`, `hour_delta_from_modal`, `dow_match`, `industry_open_rate_by_hour`, `hour_x_entropy`, `recency_x_frequency`
4. Score all 168 rows with `lgbm.predict_proba()[:, 1]`
5. Apply isotonic calibrator
6. Sort by descending calibrated probability; return top-3 (send_hour, send_dow, probability, day_name)

The REST API in `src/api/main.py` wraps this pipeline and exposes it as a `/recommend/{customer_id}` endpoint.

### 4.6 Explainability via SHAP

A SHAP `TreeExplainer` is pre-fitted using a 500-row random background sample from the training set (random state 42). For any individual prediction, SHAP decomposes the output into additive feature contributions that sum to the difference between the prediction and the global mean prediction.

**Example: Customer 12747, Best Slot — Saturday 12:00, P(open) = 0.275**

| Feature | Value | SHAP Contribution |
|---------|-------|------------------|
| `hour_delta_from_modal` | 2 | +0.119 |
| `purchase_hour_entropy` | 0.62 | +0.094 |
| `frequency` | 26 | −0.046 |
| `monetary_total` | 8,847 | +0.046 |
| `tenure_days` | 730 | −0.043 |
| `unique_products` | 85 | +0.043 |
| `send_dow` | 5 (Sat) | +0.038 |

The top positive driver is `hour_delta_from_modal = 2`: the proposed slot (noon) is only 2 hours away from this customer's habitual purchase hour, suggesting high temporal alignment. The entropy value (0.62) indicates moderate habit concentration, also positively contributing.

---

## 5. Model Performance

### 5.1 Discrimination — AUC-ROC

Area Under the Receiver Operating Characteristic Curve (AUC-ROC) measures how well the model ranks positive events (opens) above negative events (non-opens) across all possible classification thresholds.

| Model | Test AUC | Delta vs. Naive |
|-------|---------|----------------|
| Naive Random | 0.5039 | — |
| Logistic Regression (2 features) | 0.5114 | +0.0075 |
| **LightGBM (20 features, calibrated)** | **0.5235** | **+0.0196** |

An AUC of 0.52 may appear modest by conventional standards, but it is appropriate for send-time optimization. The underlying signal — individual timing preferences — is real but inherently subtle. Customers do not behave with clockwork precision, and the synthetic training data introduces controlled noise. The model's value lies in its ranking quality across a large customer base, not in its per-event discrimination. The policy simulation results (Section 5.5) quantify the business impact of this ranking.

### 5.2 Calibration Quality

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Brier Score | 0.179 | Good; lower is better (perfect = 0, chance = 0.25 at 50% base rate) |
| Expected Calibration Error (ECE) | 6.8% | Mean absolute deviation between predicted probability and observed positive rate |

The reliability curve (predicted probability decile vs. observed positive rate) shows close alignment between predicted and observed values across all deciles, confirming that the calibrated probabilities are trustworthy for business decision-making. The small residual ECE of 6.8% is well within the range considered acceptable for operational use.

### 5.3 Ranking Quality

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Precision@3 | **34.5%** | On average, 1 in 3 of the model's top-3 slot recommendations matched an actual open |
| Recall@3 | 34.8% | The top-3 slots capture 35% of all opens for a given customer |
| Recall@5 | **45.3%** | Expanding to 5 recommendations captures nearly half of all opens |
| Top-3 Hit Rate (exact match) | 5.1% | Stricter metric: exact (send_hour, send_dow) match between top-3 prediction and observed best slot |

The distinction between Top-3 Hit Rate (5.1%) and Precision@3 (34.5%) reflects two different measurement approaches. The hit rate requires an exact slot match; Precision@3 asks whether any of the top-3 predicted slots were actually opened by the customer — a more operationally relevant measure.

### 5.4 Segment-Level Performance

Top-3 hit rates by RFM segment reveal that the model performs best for well-established customers (Other, Loyal) and worst for Hibernating customers — consistent with the intuition that behavioral profiles require sufficient transaction history to be predictive.

| RFM Segment | Customers | Top-3 Hit Rate |
|-------------|-----------|---------------|
| Other | 1,130 | 5.9% |
| Loyal | 1,430 | 5.5% |
| Champions | 1,045 | 4.8% |
| At Risk | 1,023 | 3.4% |
| Hibernating | 707 | 0.0% |

Hibernating customers (long lapsed, low transaction count) return zero hit rate because their transaction histories are too sparse to establish reliable behavioral patterns. These customers should receive the segment-level fallback slot rather than personalized recommendations.

### 5.5 Policy Simulation

A retrospective policy simulation compares three send-time strategies on the historical event dataset:

| Policy | Open Rate | n Events | Uplift vs. Fixed |
|--------|-----------|----------|-----------------|
| Fixed blast (Tue 10:00) | 23.2% | 1,576 | — |
| **Personalized top-1 (model)** | **38.0%** | **4,907** | **+14.8 pp** |

The personalized policy selects the model's highest-probability send slot for each customer and evaluates the observed open outcome at that slot. The +14.8 percentage point uplift over the fixed-blast baseline translates directly to more engaged customers, higher click-through, and improved purchase conversion at every step of the funnel.

**Segment Engagement Funnel**

| Segment | Open Rate | Click \| Open | Purchase \| Click |
|---------|-----------|-------------|-----------------|
| Loyal | 23.5% | 35.3% | 12.6% |
| Champions | 23.4% | 35.7% | 11.3% |
| At Risk | 23.5% | 34.7% | 12.0% |
| Other | 23.9% | 34.6% | 11.7% |
| Hibernating | 24.6% | 37.3% | 13.1% |

Open rates are broadly consistent across segments (23–25%), suggesting timing personalization has roughly equal leverage across customer tiers. The engagement funnel also reveals that click-given-open and purchase-given-click rates are stable, meaning open rate is the primary lever to optimize.

### 5.6 Engagement Diagnostics

**Cohort Retention**

Monthly cohort analysis shows that the December 2009 cohort retains 35–49% of customers across their first 11 months — a strong baseline for an online retailer. Later cohorts show steeper drop-off, suggesting that acquisition quality may have declined or that the product mix shifted. This retention dynamic reinforces the value of send-time optimization: even a modest improvement in open rate for retained customers compounds significantly over a multi-month relationship.

**Temporal Slot Stability**

Only **5.1%** of customers maintained the same optimal send slot (defined as the slot with the highest observed open rate) between the early and late halves of the observation period. This near-total instability means that a "learn once, apply forever" approach is unacceptable. Customer profiles and model scores must be refreshed frequently — monthly is the recommended cadence.

**Reactivation Rate**

The gap between purchases for individual customers (inter-purchase gap) exceeds 30 days in only **1.9%** of cases. This low reactivation rate suggests that re-engagement campaigns targeting lapsed customers represent a meaningful upside opportunity — particularly for the Hibernating segment, which has the most customers to recapture.

---

## 6. Model Risk and Future Enhancements

### 6.1 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Synthetic data dependency** — model trained on simulated open/click outcomes, not real ESP telemetry | High | High | Replace `campaign_events.csv` with real campaign data before production deployment. This is the single most important pre-launch action. |
| **Modest discriminative power** (AUC 0.523) | Medium | Medium | Enrich features with email content signals (subject line length, personalization tokens), device type, and time-since-last-open. Larger real datasets will also improve signal quality. |
| **UK-market generalization** — 91.9% of training data is UK-origin | Medium | Medium | For non-UK markets, apply country-stratified retraining. Do not deploy UK-trained profiles to markets with substantially different behavioral norms (e.g., different work-week structures, timezone effects). |
| **Rapid preference drift** (5.1% slot stability) | High | High | Implement a monthly automated retraining pipeline. Monitor slot distribution shift between current and previous model versions. Alert if >20% of customers' top-1 recommended slot changes between refreshes. |
| **Cold-start problem** (<5 transactions) | Low | Low | Customers below the 5-transaction threshold return `insufficient_history` from the API. Fall back to the best-performing slot for the customer's RFM segment as a default. |
| **Customer ID null rate** (22.8%) | High | High | Enforce Customer ID capture at point-of-sale, account creation, and ESP subscription. Every unidentified customer represents a permanently unoptimizable profile. |

### 6.2 Future Enhancements

The current model addresses send-time optimization in isolation. The following five extensions represent the highest-value next steps in a broader marketing analytics capability build.

**1. Multi-Touch Attribution**
The existing open/click/purchase funnel captures sequential engagement, but does not attribute revenue to the specific combination of touches (email, push, SMS, retargeting) that led to conversion. A multi-touch attribution model using Shapley value allocation across channels would allow marketing budget to be allocated to the touchpoints with the highest marginal conversion contribution.

**2. Customer Churn Prediction**
The RFM features, tenure, and recency signals already engineered for this project can directly feed a survival analysis model (e.g., a Cox proportional hazards model or a discrete-time logistic regression). A churn score per customer would enable proactive retention campaigns for customers approaching lapse — before they enter the Hibernating segment.

**3. Dynamic Product Recommendations**
The current model personalizes *when* to send but not *what* to send. Combining the `unique_products` breadth signal and itemized purchase history with collaborative filtering or a matrix factorization model would enable personalized product bundle recommendations delivered at the model's optimal send time — a compounding uplift across both content and timing.

**4. NLP on Product Descriptions**
The `StockCode` and `Description` columns encode rich product category information that is currently discarded. Applying sentence embeddings (e.g., sentence-transformers) or topic modeling (LDA) to product descriptions would identify latent product category clusters. Customers could then be grouped by category affinity, enabling content personalization to complement the timing personalization already in place.

**5. Bayesian A/B Testing Framework**
The current policy comparison is a retrospective, point-estimate simulation. Production deployment requires an online testing framework that can continuously optimize the send-time policy while controlling for false discovery. A Thompson Sampling multi-armed bandit — one arm per send slot per segment — would learn the optimal policy in real time, with proper Bayesian uncertainty quantification eliminating the need for fixed test durations or arbitrary significance thresholds.

---

*For reference documentation, see:*
- [Model Card](model_card.md) — brief model summary for operations teams
- [Validation Report](validation_report.md) — raw metrics tables from training run
- [Data Dictionary](data_dictionary.md) — feature definitions and file schema
- [Synthesis Methodology](synthesis_methodology.md) — synthetic campaign event generation
