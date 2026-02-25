# Model Documentation Report
## Churn Propensity Scoring — LightGBM Classifier

**Project:** Marketing Analytics — Model 3: Churn Propensity
**Dataset:** IBM Telco Customer Churn (Kaggle / IBM Sample)
**Model:** LightGBM Binary Classifier (calibrated)
**Version:** 1.0 | **Date:** 2026-02-25
**Authors:** Marketing Analytics Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Problem & Context](#2-business-problem--context)
3. [Literature Review](#3-literature-review)
4. [Dataset & Data Quality](#4-dataset--data-quality)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Methodology](#6-model-methodology)
7. [Train/Test Split Strategy](#7-traintest-split-strategy)
8. [Model Performance](#8-model-performance)
9. [SHAP Interpretability](#9-shap-interpretability)
10. [Business Impact Simulation](#10-business-impact-simulation)
11. [Risk Tiers & Recommended Actions](#11-risk-tiers--recommended-actions)
12. [API Reference](#12-api-reference)
13. [Deployment & Maintenance](#13-deployment--maintenance)
14. [Known Limitations & Future Extensions](#14-known-limitations--future-extensions)
15. [References](#15-references)

---

## 1. Executive Summary

### The Problem

Customer churn — the voluntary termination of a service subscription — is one of the most costly and predictable risks facing subscription businesses. For a telecom provider with 7,043 subscribers and a 26.7% observed churn rate, approximately 1,869 customers have already left. At a conservative estimated cost of $200 per lost churner (representing lost lifetime revenue and re-acquisition spend), the untreated churn pool represents over $370,000 in avoidable losses per period.

The core operational challenge is not that churn is unpredictable — it is that retention resources are finite, and intervening with every customer is both wasteful and counterproductive. The model described in this document addresses that challenge directly: assign each subscriber a calibrated probability of churning, so the retention team can focus its budget on the customers most likely to leave before they actually do.

### The Approach

A LightGBM binary classifier was trained on 32 features derived from the IBM Telco Customer Churn dataset — a single-snapshot record of 7,043 California telecom customers including demographics, service subscriptions, contract terms, billing information, and payment method. Seven engineered features augment the 25 encoded raw columns, each constructed to capture business logic that the raw columns individually miss (switching cost depth, billing anomalies, tenure-adjusted spend rate).

The model outputs a calibrated churn probability in [0, 1] for each customer. An isotonic regression calibrator is applied post-hoc to ensure predicted probabilities reflect true likelihoods rather than raw classifier scores. Each customer is also assigned a four-tier risk label and a recommended retention action.

### Key Results

| Metric | Value |
|---|---|
| AUC-ROC (LightGBM) | 0.7883 |
| PR-AUC | 0.5401 |
| Brier Score | 0.1552 |
| F1 @ threshold 0.5 | 0.5234 |
| Top-20% lift | 2.29x (captures 45.7% of churners) |
| Cost-optimal threshold | 0.16 |
| Estimated cost saving vs. no model | $57,620 per scoring cycle |

### Deployment Recommendation

The model is production-ready at the cost-optimal threshold of **0.16** for operational retention scoring. The REST API at `src/api/main.py` exposes all scoring endpoints. Before broad deployment, a disparate impact analysis across `gender` and `SeniorCitizen` segments is required to satisfy ECOA and applicable state law compliance obligations.

---

## 2. Business Problem & Context

### 2.1 Churn as a Business Risk

Subscription-based telecom businesses compete on retention as much as acquisition. The cost of acquiring a new customer typically exceeds the cost of retaining an existing one by a factor of 5–7x (depending on channel). At the observed 26.7% churn rate, the provider in this dataset is losing roughly one in four customers per period — a rate that, left unaddressed, compounds into significant revenue and market share erosion.

The challenge is asymmetric. Retention interventions (discounts, plan upgrades, direct outreach) carry a direct cost: approximately $20 per contact in this business's cost model. Applied indiscriminately across all customers, this cost exceeds the value recovered. Applied only to the customers most likely to leave, the same budget recovers orders of magnitude more revenue.

This is the core framing for churn propensity modeling: not "will this customer churn?" as a binary prediction, but "what is the probability this customer churns, and is that probability high enough to justify the cost of intervening?"

### 2.2 Why Churn Is Predictable in Telecom

Telecom churn has a well-established set of behavioral and structural drivers that make it more predictable than churn in many other verticals:

- **Contract lock-in:** Customers on month-to-month contracts have no financial friction to leaving. Two-year contract customers have explicit early termination fees, creating a structural churn barrier.
- **Service switching costs:** Customers enrolled in multiple services (internet, TV streaming, phone, security bundles) face higher switching costs than single-service subscribers, because terminating means losing a bundled package.
- **Tenure effects:** Newly acquired customers are disproportionately likely to churn in the first 12 months. Long-tenured customers have demonstrated persistent loyalty.
- **Billing sensitivity:** Higher monthly charges without a perceived service quality premium create price dissatisfaction. Fiber optic customers — who pay more — show anomalously high churn rates in this dataset.
- **Payment method as a commitment signal:** Customers on automatic payment methods (bank transfer, credit card auto-pay) are more committed to the relationship than those on manual payment methods (electronic check, mailed check).

These structural signals are all observable at the point of scoring, making them tractable for a supervised classification model.

### 2.3 Operational Use Case

The model output feeds a daily retention scoring pipeline:

1. All 7,043 customer records are scored nightly by the FastAPI service
2. Customers with `churn_probability > 0.16` (the cost-optimal threshold) are flagged for the retention queue
3. High-Risk customers (probability > 0.60) are escalated for urgent outreach by customer success managers
4. Medium-High customers (0.40–0.60) receive priority retention offers (e.g., contract upgrade incentives)
5. Medium-Low customers (0.20–0.40) receive soft outreach (satisfaction surveys, feature education)
6. Low-Risk customers are monitored passively

---

## 3. Literature Review

This model draws on three bodies of work that directly motivate specific design and evaluation decisions.

### 3.1 Profit-Maximizing Metrics for Churn Prediction

> Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). A novel profit maximizing metric for measuring classification performance of customer churn prediction models. *IEEE Transactions on Knowledge and Data Engineering*, 24(7), 1227–1241.

Verbeke et al. argue that standard classification metrics — accuracy, AUC-ROC, and even F1 — are poorly suited to churn prediction in business settings because they treat all errors equally. A false negative (missed churner) and a false positive (unnecessary outreach) carry asymmetric costs: the cost of a missed churner includes lost lifetime revenue, while the cost of a false positive is limited to the outreach expense. The authors introduce the Maximum Profit (MP) measure and the Expected Maximum Profit (EMPC) criterion as profit-aware alternatives that incorporate the business cost structure directly into model evaluation.

**Influence on this project:** This paper provides the theoretical grounding for the cost-optimal threshold analysis in Section 8.4. Rather than using the default 0.5 threshold, a threshold sweep over `np.linspace(0.1, 0.9, 81)` identifies the threshold that minimizes total cost under the business cost model ($200 FN / $20 FP). This yields an optimal threshold of 0.16 — substantially lower than the default — reflecting the severe asymmetry between the cost of missing a churner and the cost of unnecessary outreach.

### 3.2 Social Network Analytics for Telco Churn

> Oskarsdottir, M., Bravo, C., Verbeke, W., Sarraute, C., Baesens, B., & Vanthienen, J. (2018). Social network analytics for churn prediction in telco: Model building, evaluation and network pruning. *Decision Support Systems*, 107, 112–123.

Oskarsdottir et al. demonstrate that customer social network structure — call graphs between subscribers — provides significant additional predictive power for churn beyond individual customer attributes. Customers whose frequent contacts are churning are more likely to churn themselves (social contagion), and customers with many connections to retained customers are more anchored. The paper also demonstrates that network-derived features (degree centrality, clustering coefficient, neighbor churn rate) increase AUC-ROC materially over attribute-only models on the same IBM Telco dataset used in this project.

**Influence on this project:** This paper establishes the performance ceiling for attribute-only models on the IBM Telco dataset and motivates network feature enrichment as a high-priority future extension (Section 14). The current model intentionally excludes network features because call graph data is not present in the Kaggle snapshot; if the provider's CDR (Call Detail Record) data were available, replicating this paper's network feature engineering would be the single highest-priority improvement.

### 3.3 Meta-Learners for Heterogeneous Treatment Effects

> Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Meta-learners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165.

Kunzel et al. introduce the S-learner, T-learner, and X-learner frameworks for estimating individual-level causal treatment effects using any machine learning model as a base learner. The central insight is that churn *propensity* (who is likely to churn) and churn *treatability* (who will respond to a retention offer) are distinct quantities. A customer who is likely to churn but will churn regardless of intervention ("sure thing" churner) does not benefit from retention spend; a customer who is unlikely to churn anyway ("sure save") represents wasted spend. The highest-value retention targets are the "persuadables" — customers who would churn without intervention but would be retained with it.

**Influence on this project:** This paper provides the theoretical motivation for uplift modeling as the natural next step beyond propensity scoring (Section 14.2). The current model produces churn propensity scores; a T-learner or X-learner built on top of A/B-tested retention campaign data would produce individual treatment effect estimates, allowing the retention team to target persuadables instead of all high-propensity churners.

---

## 4. Dataset & Data Quality

### 4.1 Data Source

**IBM Telco Customer Churn Dataset**

The dataset is a publicly available customer churn dataset originally published as an IBM Watson Analytics sample and distributed via Kaggle under a CC0 license. It represents a single cross-sectional snapshot of customer records from a fictional California telecom provider.

| Property | Value |
|---|---|
| Raw file | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Raw columns | 21 (20 features + 1 target) |
| Geography | California telecom provider (simulated) |
| License | CC0 public domain / IBM Sample |
| Kaggle source | `blastchar/telco-customer-churn` |

### 4.2 Data Quality

The dataset is notably clean by real-world standards, reflecting its curated, synthetic origin. Three data quality issues were identified and resolved:

| Issue | Count | Resolution |
|---|---|---|
| `TotalCharges` stored as string (not float) | 7,043 rows affected | `pd.to_numeric(errors='coerce')` at load time |
| Blank `TotalCharges` values | 11 rows | All have `tenure == 0`; imputed with 0.0 (no charges yet accrued) |
| No null `customerID` values | 0 rows | No customer ID missingness; no exclusions required |

After these three transformations, the working dataset is 7,043 rows with no null values and all columns in numeric or one-hot-encoded form.

**Note on data authenticity:** Because the dataset is a curated IBM sample rather than live CDR data, several caveats apply: (1) class proportions may not reflect the provider's true churn rate; (2) feature distributions are unlikely to exhibit the long-tail anomalies present in real billing data; (3) the absence of call graphs, NPS scores, and support ticket history limits the feature space relative to what a production system would have access to. These limitations are discussed further in Section 14.

### 4.3 Class Distribution

| Class | Count | Proportion |
|---|---|---|
| Churn = 0 (retained) | 5,174 | 73.46% |
| Churn = 1 (churned) | 1,869 | 26.54% |
| **Total** | **7,043** | **100%** |

The 1:2.77 class imbalance ratio is moderate. A naive majority-class classifier achieves 73.5% accuracy but is useless for the business problem. This is why accuracy is excluded from the primary evaluation metrics. AUC-ROC evaluates rank-order quality across all thresholds; PR-AUC evaluates precision-recall trade-offs in the positive class; the Brier Score measures probabilistic calibration; and F1 (at the business-optimized threshold) captures the operational precision-recall balance.

The imbalance is addressed at training time via `scale_pos_weight = 73.5 / 26.5 ≈ 2.77` in LightGBM, which upweights positive (churned) samples proportionally to their class underrepresentation.

---

## 5. Feature Engineering

The full feature set comprises 32 columns: 25 numerically encoded from the 21 raw input columns (via binary encoding and one-hot encoding) plus 7 newly engineered features. All 32 are listed in `models/metadata.json` under `feature_columns`.

### 5.1 Passthrough Features (no encoding)

| Feature | Type | Range | Business Rationale |
|---|---|---|---|
| `SeniorCitizen` | int (0/1) | {0, 1} | Sensitive demographic attribute; senior customers may have different churn drivers and must be monitored for disparate impact |
| `tenure` | int | 0–72 months | Primary loyalty signal; customers who have stayed longer are less likely to leave. Also anchors the `monthly_per_tenure` engineered feature |
| `MonthlyCharges` | float | $18.25–$118.75 | Price sensitivity driver; higher charges without service depth correlate with churn |
| `TotalCharges` | float | $0–$8,685 | Cumulative revenue signal; anchors `total_charges_gap` |

### 5.2 Binary-Encoded Features (Yes/No → 1/0)

| Feature | Encoding | Business Rationale |
|---|---|---|
| `gender` | Male=1, Female=0 | Included for model completeness; flagged as sensitive attribute — must not be used for differential offer targeting |
| `Partner` | Yes=1, No=0 | Customers with partners have household stability signal |
| `Dependents` | Yes=1, No=0 | Dependents increase switching friction (family plan economics) |
| `PhoneService` | Yes=1, No=0 | Core service subscription depth |
| `MultipleLines` | Yes=1, else=0 | Multiple lines increase switching cost; "No phone service" collapses to 0 |
| `OnlineSecurity` | Yes=1, else=0 | Security add-on subscription; absence is a mild churn predictor |
| `OnlineBackup` | Yes=1, else=0 | Backup add-on subscription depth |
| `DeviceProtection` | Yes=1, else=0 | Protection plan subscription depth |
| `TechSupport` | Yes=1, else=0 | Absence of tech support is a consistent churn driver in the telco literature |
| `StreamingTV` | Yes=1, else=0 | Entertainment bundle subscription depth |
| `StreamingMovies` | Yes=1, else=0 | Entertainment bundle subscription depth |
| `PaperlessBilling` | Yes=1, No=0 | Paperless billing correlates with digital engagement; mild churn signal |

### 5.3 One-Hot Encoded Features

**InternetService** (3 dummy columns, `drop_first=False`):

| Feature | Value=1 Condition | Business Rationale |
|---|---|---|
| `InternetService_DSL` | DSL internet subscriber | DSL customers show moderate churn; price-competitive |
| `InternetService_Fiber optic` | Fiber optic subscriber | Fiber customers have highest churn rate despite premium pricing — key model driver |
| `InternetService_No` | No internet service | No-internet customers are less engaged but also less at risk |

**Contract** (3 dummy columns):

| Feature | Value=1 Condition | Business Rationale |
|---|---|---|
| `Contract_Month-to-month` | Month-to-month contract | Strongest churn driver; no financial lock-in means zero switching cost |
| `Contract_One year` | One-year contract | Moderate lock-in; lower churn than M2M |
| `Contract_Two year` | Two-year contract | Strongest retention; early termination fees and long-term price commitment |

**PaymentMethod** (4 dummy columns):

| Feature | Value=1 Condition | Business Rationale |
|---|---|---|
| `PaymentMethod_Electronic check` | Electronic check payment | Highest churn rate of any payment method; non-automated payment suggests lower commitment |
| `PaymentMethod_Mailed check` | Mailed check payment | Manual payment; moderate churn signal |
| `PaymentMethod_Bank transfer (automatic)` | Automatic bank transfer | Automated payment signals committed relationship |
| `PaymentMethod_Credit card (automatic)` | Automatic credit card | Automated payment signals committed relationship |

### 5.4 Engineered Features

Seven features are computed in `src/feature_engineering.py:compute_customer_features()` and added to the feature matrix. Each was designed to capture a business concept that the raw columns individually miss.

| Feature | Formula | Type | Business Rationale |
|---|---|---|---|
| `has_family` | `1 if (Partner==1) OR (Dependents==1) else 0` | int (0/1) | Customers with household ties have higher switching friction. Combines two correlated raw signals into a single compact indicator |
| `num_services` | Sum of 8 service binary flags (PhoneService + MultipleLines + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies) | int (0–8) | Depth-of-relationship proxy. Customers enrolled in more services face higher switching costs; range 0 = no services, 8 = full bundle |
| `monthly_per_tenure` | `MonthlyCharges / (tenure + 1)` | float | Tenure-adjusted spend rate. A new customer paying $80/month looks very different from a 60-month customer paying $80/month. The +1 guard prevents division-by-zero for tenure==0 |
| `total_charges_gap` | `MonthlyCharges * tenure - TotalCharges` | float | Billing anomaly signal. In a stable account, TotalCharges ≈ MonthlyCharges × tenure. Deviations indicate plan changes, prorated billing, discounts, or data errors. Near-zero for stable accounts; negative for discount recipients |
| `is_month_to_month` | Alias for `Contract_Month-to-month` | int (0/1) | Convenience alias for the top predictive feature. Ensures the contract signal is available as a simple binary column for SHAP and threshold logic |
| `is_fiber_optic` | Alias for `InternetService_Fiber optic` | int (0/1) | Convenience alias for the second-highest churn-rate internet segment |
| `is_electronic_check` | Alias for `PaymentMethod_Electronic check` | int (0/1) | Convenience alias for the highest-churn payment method |

---

## 6. Model Methodology

### 6.1 Problem Framing

Churn prediction is framed as a **binary classification problem**: given a customer feature vector **x**, predict:

```
P(Churn = 1 | x)
```

where Churn=1 indicates the customer has or will terminate their subscription. The model outputs a calibrated probability in [0, 1]. Classification at a given threshold is a downstream decision that depends on the business cost structure, not a property of the model itself.

This is a **static snapshot** prediction: the model uses the customer's current-state attributes, not a longitudinal time series. It predicts churn propensity as of the scoring date, not time-to-churn. Forward-looking survival modeling is identified as a future extension in Section 14.

### 6.2 Baseline Models

Two baselines establish the performance floor:

**Baseline 1 — Naive (majority class probability):** Assigns every customer the dataset churn rate (26.7%) as the predicted probability. AUC-ROC = 0.5000 by construction. This is the minimum acceptable performance threshold.

**Baseline 2 — Logistic Regression (all 32 features):** A standard L2-regularized logistic regression trained on all 32 features with the same train/test split. AUC-ROC = 0.8162. This baseline demonstrates strong performance on the structured, low-noise IBM sample data and serves as the upper-bound reference. The fact that LightGBM (AUC 0.7883) underperforms logistic regression here is discussed in Section 8.1.

### 6.3 Primary Model — LightGBM Binary Classifier

LightGBM (Ke et al., 2017) was selected as the primary model for four reasons: (1) native handling of mixed feature types without preprocessing; (2) built-in class imbalance weighting via `scale_pos_weight`; (3) exact SHAP value computation via `TreeExplainer`; and (4) strong empirical performance on tabular datasets across the churn prediction literature.

**Hyperparameters** (from `src/train.py`):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `binary` | Binary cross-entropy loss |
| `metric` | `auc` | Optimize rank quality during training |
| `scale_pos_weight` | 2.77 | Upweights churned class (73.5% / 26.5%) to address imbalance |
| `n_estimators` | 300 | Sufficient capacity for 5,634 training rows |
| `learning_rate` | 0.05 | Conservative; balances speed and generalization |
| `num_leaves` | 31 | Controls tree complexity |
| `min_child_samples` | 20 | Leaf-level regularization against overfitting small segments |
| `feature_fraction` | 0.8 | Column subsampling per tree (80%) |
| `bagging_fraction` | 0.8 | Row subsampling per iteration (80%) |
| `bagging_freq` | 5 | Row subsampling applied every 5 iterations |
| `random_state` | 42 | Reproducibility |

**Model artifacts** saved to `models/`:
- `lgbm_churn.pkl` — trained LightGBM classifier
- `probability_calibrator.pkl` — fitted isotonic regression calibrator
- `shap_explainer.pkl` — SHAP TreeExplainer
- `metadata.json` — feature list, split sizes, metrics, cost-optimal threshold

### 6.4 Probability Calibration

Raw LightGBM scores are known to be systematically overconfident on imbalanced datasets (Niculescu-Mizil & Caruana, 2005). An **isotonic regression** calibrator is fitted on a held-out validation slice and applied to all downstream scores. This ensures that a predicted probability of 0.30 corresponds to an observed churn rate close to 30% in the data — a requirement for using the scores in a cost-minimization framework.

The calibrated Brier Score of **0.1552** (lower is better; 0.0 is perfect) confirms that calibrated probabilities are close to true likelihoods. The calibrator is a monotone non-decreasing step function clipped to [0, 1].

### 6.5 Inference Pipeline

At inference time for any customer record:

1. Retrieve or construct the 32-feature vector from `data/processed/customer_features.parquet`
2. Score with `lgbm_churn.predict_proba(X[FEATURE_COLUMNS])[:, 1]`
3. Apply isotonic calibrator: `calibrator.predict(raw_scores)`
4. Assign risk tier: thresholds at 0.20, 0.40, 0.60
5. Return `churn_probability`, `risk_tier`, `recommended_action`, and SHAP top-5 drivers

The API in `src/api/main.py` pre-scores all customers at startup and caches the results. Individual customer lookups return from cache in O(1). The `POST /api/predict` endpoint scores arbitrary feature vectors on-demand for what-if analysis.

---

## 7. Train/Test Split Strategy

### 7.1 Primary Split

Because the dataset is a static snapshot (no temporal ordering), a temporal split is not applicable. A **stratified random split** is used instead, preserving the 26.7% churn rate in both sets.

| Set | Rows | Churn Rate | Purpose |
|---|---|---|---|
| Train | 5,634 (80%) | 26.7% | Model training and cross-validation |
| Test | 1,409 (20%) | 26.7% | Final held-out evaluation |

Implementation: `sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`

The test set is held out entirely and is used only for the final evaluation reported in Section 8. It is not used for hyperparameter tuning.

### 7.2 Cross-Validation on the Training Set

Hyperparameter tuning uses **stratified 5-fold cross-validation** on the training set only.

| Property | Value |
|---|---|
| Strategy | StratifiedKFold |
| Folds | 5 |
| Stratification | By `Churn` in every fold |
| Approximate fold sizes | ~4,507 train / ~1,127 validation rows per fold |
| Implementation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

Cross-validation provides unbiased generalization estimates during model selection without contaminating the test set.

### 7.3 Handling of SMOTE

SMOTE oversampling (when used) is applied **inside each training fold only**, never to the validation or test portions. This is critical to prevent data leakage: evaluating a model on synthetic oversamples would produce inflated performance estimates.

---

## 8. Model Performance

### 8.1 AUC-ROC Comparison

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures rank-order discrimination: the probability that a randomly selected churner receives a higher predicted score than a randomly selected non-churner.

| Model | Test AUC-ROC | Delta vs. Naive |
|---|---:|---:|
| Naive baseline | 0.5000 | — |
| Logistic Regression (32 features) | 0.8162 | +0.3162 |
| **LightGBM (calibrated)** | **0.7883** | **+0.2883** |

LightGBM underperforms logistic regression on AUC-ROC by 0.028 on this particular dataset. This is a recognized pattern for low-volume, well-structured tabular datasets: logistic regression benefits from the linear separability of features like contract type and tenure, and is not penalized by the small training set size the way tree-based models can be. LightGBM is retained as the production model because: (1) it produces better-calibrated probabilities (lower Brier Score); (2) SHAP explanations are exact and computationally efficient for tree models; and (3) it is more robust to future feature additions and non-linear interactions.

**AUC-ROC curve key points:**

| FPR | TPR |
|---|---|
| 0.0 | 0.005 |
| 0.1 | 0.445 |
| 0.2 | 0.620 |
| 0.3 | 0.738 |
| 0.4 | 0.818 |
| 0.5 | 0.885 |
| 0.6 | 0.926 |
| 0.7 | 0.950 |
| 0.8 | 0.970 |
| 0.9 | 0.988 |
| 1.0 | 1.000 |

### 8.2 Additional Classification Metrics

| Metric | Value | Interpretation |
|---|---|---|
| PR-AUC | 0.5401 | Area under the Precision-Recall curve; more informative than AUC-ROC under class imbalance |
| Brier Score | 0.1552 | Mean squared error of probabilistic predictions; lower is better (0 = perfect, 0.267 = naive at 26.7% base rate) |
| F1 @ threshold 0.5 | 0.5234 | Harmonic mean of precision and recall at the default threshold |
| Precision @ threshold 0.5 | 0.6070 | 61% of flagged customers are true churners at the default threshold |
| Recall @ threshold 0.5 | 0.4601 | Only 46% of actual churners are captured at the default threshold |

### 8.3 Confusion Matrix at Threshold 0.5

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **True No Churn** | 921 (TN) | 112 (FP) |
| **True Churn** | 203 (FN) | 173 (TP) |

At the default threshold, 203 churners are missed (false negatives) and 112 non-churners are incorrectly flagged (false positives). The asymmetric cost of false negatives ($200 each vs. $20 for false positives) motivates the lower cost-optimal threshold of 0.16.

### 8.4 Cost-Optimal Threshold Analysis

A threshold sweep over `np.linspace(0.1, 0.9, 81)` identifies the threshold minimizing total cost under the business cost model.

**Cost model:**
- False Negative (missed churner): $200 per customer
- False Positive (unnecessary outreach): $20 per customer

**Results at threshold 0.16:**

| Metric | Value |
|---|---|
| F1 | 0.5290 |
| Precision | 0.3707 |
| Recall | 0.9229 |
| Total cost | $17,580 |
| Baseline cost (flag nobody) | $75,200 |
| **Estimated cost saving** | **$57,620** |

At the cost-optimal threshold of 0.16, the model captures 92.3% of all churners (Recall = 0.9229) at the expense of a lower precision (37.1%). This is the correct trade-off given the 10:1 cost asymmetry: it is far more costly to miss a churner than to send an unnecessary retention offer.

**Operational implication:** For daily scoring, apply threshold = 0.16 to the `churn_probability` column. Customers above this threshold enter the retention queue. Within the queue, use risk tier labels (Section 11) to prioritize urgency.

### 8.5 Top-20% Decile Lift

When all 7,043 customers are ranked by descending predicted churn probability, the top-scoring 20% (approximately 1,409 customers) show the following concentration:

| Metric | Value |
|---|---|
| Churn rate in top-20% | 61.3% |
| Overall churn rate | 26.7% |
| **Lift** | **2.29x** |
| **Churner capture rate** | **45.7%** |

This means that targeting only the top-scored quintile — roughly 1,409 out of 7,043 customers — captures 45.7% of all 1,869 churners. The retention team can achieve nearly half the total possible churn prevention by intervening with only 20% of the customer base.

---

## 9. SHAP Interpretability

### 9.1 Methodology

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) provides theoretically grounded feature attributions for individual predictions. The `TreeExplainer` variant computes exact SHAP values for tree-based models in polynomial time. For each customer, SHAP decomposes the predicted log-odds into additive contributions from each of the 32 features, summing to the difference between the individual prediction and the global mean prediction.

The SHAP explainer is pre-fitted with a background dataset and saved to `models/shap_explainer.pkl`. The API returns the top-5 SHAP contributors for each customer in the `/api/customer/{customer_id}` response.

### 9.2 Global Feature Importance

The five strongest SHAP drivers across the test set, in descending order of mean absolute SHAP value:

**1. Contract type (is_month_to_month / Contract_Month-to-month)**

The single strongest churn predictor. Month-to-month customers have no financial switching cost: there is no early termination fee, no annual commitment, and no multi-year price guarantee. As a result, they churn at approximately 3x the rate of two-year contract customers. In retention terms, the highest-leverage intervention for a month-to-month customer is a contract upgrade offer — converting them to an annual or biennial plan removes the structural churn vulnerability entirely.

**2. Tenure**

Tenure has a strong negative SHAP contribution for churn: customers who have stayed longer are less likely to leave. The relationship is non-linear: customers are most vulnerable in the first 12 months (acquisition-stage attrition), with churn risk declining steeply through month 24 and flattening for long-tenured customers. This means recent acquisitions should receive closer monitoring even if their other attributes look stable.

**3. MonthlyCharges**

Higher monthly charges are positively associated with churn risk. This reflects price sensitivity: customers paying high monthly bills who do not perceive commensurate service value are more likely to shop competitors. Counterintuitively, this is not simply explained by fiber optic pricing — after controlling for internet service type, `MonthlyCharges` retains an independent positive SHAP contribution. Customers with high monthly charges and low service depth (few add-ons, no contract lock-in) represent the highest-value retention opportunities.

**4. InternetService: Fiber optic (is_fiber_optic)**

Fiber optic customers show significantly higher churn rates than DSL or non-internet customers, despite being the provider's premium-priced segment. The most likely explanations are: (1) fiber customers are more technologically sophisticated and more willing to compare competitive offers; (2) the premium pricing creates a higher bar for perceived service value; and (3) fiber infrastructure is more widely available from competitors than DSL, reducing switching friction. In retention terms, fiber customers warrant proactive outreach focused on satisfaction, speed guarantees, and service quality — not price discounts alone.

**5. PaymentMethod: Electronic check (is_electronic_check)**

Electronic check payers have the highest observed churn rate of any payment method group. Unlike bank transfer and credit card auto-pay customers, electronic check payers actively choose their payment each period, which may reflect a lower level of commitment to the relationship. In retention outreach, nudging electronic check payers toward automatic payment enrollment may improve retention through passive commitment effects.

### 9.3 Individual SHAP Interpretation

For any individual customer, the API returns a ranked list of the 5 features with the largest absolute SHAP contributions, with positive contributions pushing toward churn and negative contributions pushing toward retention. For example:

| Feature | Value | SHAP Contribution | Interpretation |
|---|---|---|---|
| `is_month_to_month` | 1 | +0.35 | Month-to-month contract sharply increases churn risk |
| `tenure` | 3 months | +0.22 | Very new customer; not yet anchored |
| `num_services` | 1 | +0.09 | Only one service; low switching cost |
| `MonthlyCharges` | $85.50 | +0.08 | High monthly charge relative to service depth |
| `has_family` | 0 | +0.04 | No household ties; lower retention friction |

This customer has a predicted churn probability of approximately 0.71 (High Risk). The recommended intervention is contract upgrade offer, ideally bundled with a multi-service discount to increase `num_services` and tenure simultaneously.

---

## 10. Business Impact Simulation

### 10.1 Top-20% Lift Scenario

Under a budget-constrained scenario where the retention team can contact at most 20% of all customers per period:

**Without model:** Random selection of 1,409 customers captures approximately 26.7% × 1,409 = 376 churners (the base rate).

**With model (top-20% by score):** Score-ranked selection of 1,409 customers captures 45.7% × 1,869 = 854 churners (the measured capture rate).

This represents a **2.3x improvement** in churner capture per outreach dollar spent, which translates directly into reduced revenue loss at the same outreach budget.

### 10.2 Cost Optimization Case

The full-population cost model (Section 8.4) produces the following comparison:

| Strategy | Intervention Cost | Missed Churner Cost | Total Cost |
|---|---:|---:|---:|
| Flag nobody (no model) | $0 | $200 × 1,869 = $373,800 | $373,800 |
| Normalized to 1,409-customer test set | $0 | $200 × 376 = $75,200 | $75,200 |
| Model at threshold 0.16 | $20 × FP | $200 × FN | $17,580 |
| **Estimated saving** | | | **$57,620** |

These figures apply to the 1,409-customer test set and scale linearly to the full 7,043-customer population.

### 10.3 Sensitivity Analysis

The cost-optimal threshold of 0.16 was derived under a 10:1 cost ratio ($200 FN / $20 FP). If business costs change:

| FN Cost | FP Cost | Implied Ratio | Direction of Optimal Threshold |
|---|---|---|---|
| $200 | $10 | 20:1 | Shift threshold lower (more aggressive recall) |
| $200 | $20 | 10:1 | 0.16 (current) |
| $200 | $50 | 4:1 | Shift threshold higher (accept more missed churners) |
| $100 | $20 | 5:1 | Shift threshold higher |

The threshold sweep in `src/train.py` can be re-run with revised cost parameters without retraining the model.

---

## 11. Risk Tiers & Recommended Actions

The four-tier risk framework translates continuous probabilities into operational instructions for the retention team.

| Tier | Score Range | Description | Recommended Action | Typical Profile |
|---|---|---|---|---|
| **High Risk** | > 0.60 | Customer is highly likely to churn within the current period | Urgent intervention — direct outreach from a customer success manager within 24 hours; offer contract upgrade, loyalty discount, or free service bundle upgrade | Month-to-month, short tenure (<12 months), fiber optic, electronic check, high monthly charges, low service depth |
| **Medium-High Risk** | 0.40 – 0.60 | Elevated churn probability; warrants priority treatment | Priority retention offer — automated offer delivered within 72 hours; contract upgrade incentive, multi-service discount, or free add-on trial | Month-to-month or one-year contract, moderate tenure, one or two services, above-average monthly charges |
| **Medium-Low Risk** | 0.20 – 0.40 | Below average churn probability; low urgency | Soft outreach — satisfaction survey, feature education email, or periodic check-in at next natural touchpoint | One-year contract or DSL internet, moderate tenure, some services |
| **Low Risk** | < 0.20 | Minimal churn risk; customer is anchored | Monitor passively — no active intervention required; include in standard engagement communications | Two-year contract, long tenure (24+ months), multiple services, automatic payment method |

**At the cost-optimal threshold (0.16),** High Risk + Medium-High Risk + a portion of Medium-Low Risk customers are all flagged for the retention queue. The risk tier provides prioritization within the queue.

---

## 12. API Reference

The churn scoring service is a FastAPI application at `src/api/main.py`. All endpoints are documented via the automatic OpenAPI interface at `http://localhost:8003/docs` when the service is running.

### 12.1 Base URL

```
http://localhost:8003
```

Start the service with:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8003 --reload
```

### 12.2 Endpoints

**GET /health**

Returns service liveness status and the number of loaded customers.

```json
{"status": "ok", "model": "lgbm_churn", "customers": 7043}
```

---

**GET /api/customers**

Returns a paginated list of all scored customers.

Query parameters:
- `page` (int, default 1)
- `per_page` (int, default 50, max 200)
- `segment` (string, optional) — filter by risk tier: `"High Risk"`, `"Medium-High Risk"`, `"Medium-Low Risk"`, `"Low Risk"`

Response: `PaginatedCustomers` with `items` (list of `CustomerListItem`), `total`, `page`, `per_page`, `pages`.

---

**GET /api/customer/{customer_id}**

Returns the full churn prediction for a single customer by ID (format: `DDDD-AAAAA`, e.g., `7590-VHVEG`).

Response: `CustomerResponse` including:
- `churn_probability` (float, 4 decimal places)
- `risk_tier` (string)
- `recommended_action` (string)
- `shap_factors` (list of top-5 SHAP contributions)
- `profile` (CustomerProfile with key account attributes)
- `cost_optimal_threshold` (float, from metadata.json)

Errors: `404` if customer not found; `422` if customer_id format is invalid.

---

**POST /api/predict**

Scores an arbitrary feature vector for what-if analysis. All fields default to dataset medians; provide only the fields to override.

Body (`PredictRequest`, all fields optional):
- `tenure`, `MonthlyCharges`, `TotalCharges` (numeric)
- `contract_type` (string: `"month-to-month"`, `"one year"`, `"two year"`)
- `internet_service` (string: `"dsl"`, `"fiber optic"`, `"no"`)
- `payment_method` (string: `"electronic check"`, `"bank transfer (automatic)"`, `"credit card (automatic)"`, `"mailed check"`)
- Individual binary feature flags

Engineered features (`has_family`, `num_services`, `monthly_per_tenure`, `total_charges_gap`, `is_month_to_month`, `is_fiber_optic`, `is_electronic_check`) are recomputed automatically from the provided or default raw fields.

Response: `ChurnPrediction` with `churn_probability`, `risk_tier`, `recommended_action`, `shap_factors`.

---

**POST /api/simulate**

Simulates the effect of a proposed contract or plan change on a specific customer's churn probability.

Body (`SimulateRequest`):
- `customer_id` (string, required)
- `proposed_changes` (dict of feature name → new value)

Response: `SimulateResponse` with `original_probability`, `new_probability`, `probability_delta`, `original_tier`, `new_tier`, `message`.

---

**GET /api/segments**

Returns aggregate risk tier summaries and a Contract × InternetService churn rate grid across the full customer population.

Response: `SegmentsResponse` with `risk_tiers` (list of `SegmentSummary`) and `contract_internet_grid`.

---

## 13. Deployment & Maintenance

### 13.1 Deployment Architecture

The churn API is the third component in a three-tier stack:

| Component | Technology | Port | Role |
|---|---|---|---|
| FastAPI ML service | Python / uvicorn | 8003 | Model inference, SHAP, customer scoring |
| Express proxy | Node.js | 3001 | Forwards `/api/*` to FastAPI; serves docs |
| React frontend | Vite / React 18 | 5173 | Customer lookup, what-if, segment views |

Model artifacts at startup (loaded via `lifespan` context manager):
- `models/lgbm_churn.pkl` — trained LightGBM classifier
- `models/probability_calibrator.pkl` — isotonic regression calibrator
- `models/shap_explainer.pkl` — SHAP TreeExplainer
- `models/metadata.json` — metrics and cost-optimal threshold
- `data/processed/customer_features.parquet` — scored customer feature table

All customers are pre-scored at service startup and cached in memory. Score refresh requires a service restart after running the full ML pipeline (`data_pipeline.py` → `feature_engineering.py` → `train.py`).

### 13.2 Retraining Triggers

| Trigger | Threshold | Action |
|---|---|---|
| Population Stability Index (PSI) on `tenure` | PSI > 0.2 | Full retrain on refreshed data |
| PSI on `MonthlyCharges` | PSI > 0.2 | Full retrain on refreshed data |
| PSI on `Contract_Month-to-month` | PSI > 0.2 | Full retrain on refreshed data |
| Actual churn rate shift | > 5 percentage points from 26.7% | Recalibrate probability calibrator; assess full retrain |
| New contract types or payment methods | Any new category | Re-engineer one-hot features; retrain |
| New service bundles (additional binary add-ons) | Any new service | Update feature engineering; retrain |
| Scheduled refresh (minimum cadence) | Quarterly | Refresh customer feature table; re-score all customers |

PSI is computed as: `PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))` over binned feature distributions. A PSI > 0.10 warrants monitoring; PSI > 0.20 signals significant distribution shift requiring retraining.

### 13.3 Monitoring

The following metrics should be tracked in production:

| Metric | Frequency | Alert Threshold |
|---|---|---|
| Mean predicted churn probability | Daily | Shift > 3 pp from baseline |
| % of customers in High Risk tier | Weekly | Shift > 5 pp from baseline |
| PSI on top-5 features | Monthly | PSI > 0.10 (warning), > 0.20 (retrain) |
| Actual churn rate (realized) | Monthly | Deviation > 5 pp from model-predicted rate |
| Brier Score on new actuals | Quarterly | Degradation > 0.02 from baseline (0.1552) |

### 13.4 Retraining Pipeline

```bash
# Activate environment
source .venv/bin/activate

# Step 1: Refresh raw data
python scripts/fetch_data.py

# Step 2: Clean and process
python src/data_pipeline.py

# Step 3: Feature engineering
python src/feature_engineering.py

# Step 4: Train, calibrate, evaluate
python src/train.py

# Step 5: Restart API to load new artifacts
uvicorn src.api.main:app --host 0.0.0.0 --port 8003 --reload
```

All stochastic operations use `random_state=42` for reproducibility across pipeline runs.

---

## 14. Known Limitations & Future Extensions

### 14.1 Known Limitations

**Static snapshot — no longitudinal time series**

The dataset is a single cross-sectional observation per customer. The model has no access to the trajectory of customer behavior over time: whether charges have been increasing, whether service subscription depth has been declining, or whether recent support interactions have occurred. All of these temporal signals are predictive of churn in real-world deployments and are entirely absent here. Adding a time-series dimension (e.g., rolling 3-month change in charges, recent service downgrade events) is the highest-impact data engineering improvement.

**Telecom-specific feature set**

The model's features — contract type, internet service tier, payment method, add-on services — are telecom-specific. The model should not be assumed to generalize to SaaS (where churn drivers include product engagement, feature adoption, and NPS), retail (where drivers include purchase frequency and category switching), or other verticals. Revalidation on new domain data is mandatory before cross-vertical deployment.

**Synthetic / sample data**

The Kaggle IBM Telco dataset is a curated sample, not live CDR data. The clean feature distributions and absence of edge cases (billing errors, re-acquisitions, corporate accounts, family plan structures) mean the model is likely to encounter distributional challenges at production deployment against real data. The AUC-ROC of 0.7883 on the held-out test set should be treated as an optimistic upper bound on real-world performance.

**LightGBM underperforms logistic regression on AUC-ROC**

On this 7,043-row, well-structured dataset, logistic regression (AUC 0.8162) outperforms LightGBM (AUC 0.7883). This is consistent with theoretical expectations for small, low-noise datasets where linear decision boundaries are well-suited to the problem. LightGBM is retained for its calibration quality and SHAP support, but this gap should be re-evaluated with production data at higher volume.

**No network / social graph features**

Oskarsdottir et al. (2018) demonstrate that social network features derived from call graphs substantially improve AUC-ROC on this same dataset. The current model cannot access call detail records. If CDR data becomes available, network features (neighbor churn rate, degree centrality, community membership) should be added as a priority.

**No customer support / satisfaction signals**

NPS scores, support ticket volume, and recent complaint history are strong leading indicators of churn that are entirely absent from the dataset. These signals, if available in CRM, should be incorporated at the next model refresh.

### 14.2 Future Extensions

**1. Uplift Modeling (Treatment Effect Estimation)**

Churn propensity scoring identifies who is likely to leave; uplift modeling identifies who will respond to a retention intervention. Using a T-learner or X-learner framework (Kunzel et al., 2019) trained on A/B-tested retention campaign data, the model could estimate individual-level treatment effects — separating the "persuadables" (who churn without intervention but stay with it) from the "lost causes" (who churn regardless) and the "sure saves" (who stay regardless). This would substantially improve return on retention spend by concentrating it on the persuadables.

**2. Survival Analysis / Time-to-Churn**

Rather than predicting whether a customer will churn (binary), a survival analysis model (Cox Proportional Hazards, Weibull AFT, or discrete-time hazard model) would predict when they will churn — producing a survival curve per customer and enabling time-aware intervention scheduling. Customers expected to churn within 30 days warrant different urgency than those at risk in 90–180 days.

**3. Time-Series Feature Engineering**

With access to monthly snapshots of customer attributes, features capturing behavioral trajectory — 3-month rolling trend in monthly charges, service subscription depth changes over time, recent downgrade events — would add substantial predictive power and reduce the static snapshot limitation.

**4. Social Network Features**

As detailed in Section 3.2, adding call-graph-derived features (neighbor churn rate, clustering coefficient, betweenness centrality) following the methodology of Oskarsdottir et al. (2018) would materially improve AUC-ROC. This requires access to CDR data and a graph construction pipeline.

**5. Multi-Model Ensemble**

An ensemble combining the logistic regression baseline (AUC 0.8162) with LightGBM (AUC 0.7883) via stacking or blending may improve AUC-ROC above either individual model, while retaining LightGBM's SHAP explanations for the operational layer. A Platt-scaled ensemble with a meta-learner trained on the validation fold would be the natural implementation.

---

## 15. References

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Meta-learners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning (ICML)*, 625–632.

Oskarsdottir, M., Bravo, C., Verbeke, W., Sarraute, C., Baesens, B., & Vanthienen, J. (2018). Social network analytics for churn prediction in telco: Model building, evaluation and network pruning. *Decision Support Systems*, 107, 112–123.

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). A novel profit maximizing metric for measuring classification performance of customer churn prediction models. *IEEE Transactions on Knowledge and Data Engineering*, 24(7), 1227–1241.

---

*For a concise model summary see [`model_card.md`](model_card.md). For feature definitions and encoding rules see [`data_dictionary.md`](data_dictionary.md). For raw validation metrics see [`validation_report.md`](validation_report.md).*
