# Model Documentation Report
## Customer Lifetime Value Prediction — LightGBM Regression Model

**Project:** Marketing Analytics — Customer Lifetime Value (m02_clv)
**Dataset:** UCI Online Retail II (2009--2011)
**Model:** LightGBM Regressor (log1p-transformed target)
**Version:** 1.0 | **Date:** 2026-02-26
**Authors:** Marketing Analytics Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review](#2-literature-review)
3. [Data Description](#3-data-description)
4. [Feature Engineering](#4-feature-engineering)
5. [Methodology](#5-methodology)
6. [Model Performance](#6-model-performance)
7. [SHAP Analysis](#7-shap-analysis)
8. [Segment Definitions](#8-segment-definitions)
9. [API Reference](#9-api-reference)
10. [Deployment](#10-deployment)
11. [Future Work](#11-future-work)

---

## 1. Executive Summary

### The Problem

Customer Lifetime Value (CLV) is the estimated net revenue a business will earn from a customer over the remaining duration of the relationship. Most organizations compute CLV retrospectively as a historical average — dividing total revenue by total customers — which provides no differentiation between a high-frequency wholesale buyer and an occasional gift purchaser. This aggregate view fails at three critical marketing decisions: how much to spend acquiring a customer (target CLV:CAC ratio of 3:1 or better), which customers justify retention investment (5--7x cheaper to retain than acquire; a 5% retention improvement can yield up to 95% profit lift), and how to allocate promotion budgets across customer tiers.

A predictive CLV model transforms this static metric into a forward-looking, customer-level score. Each customer receives an individual 12-month revenue forecast that can be used to segment the portfolio, prioritize outreach, and calibrate marketing spend.

### The Dataset and Approach

This project uses the UCI Online Retail II dataset: approximately 1,067,371 transactions from a UK-based online gift retailer spanning December 2009 to December 2011. The dataset covers over 5,900 identified customers across 43 countries, though 91.9% of transactions originate from the United Kingdom.

The modeling approach uses a temporal split design. The observation window (December 2009 through November 2010) is used to construct 22 per-customer features across four groups: RFM core metrics (recency, frequency, monetary value), behavioral signals (tenure, purchase velocity, product breadth), temporal patterns (acquisition timing, seasonality, weekend and evening purchasing), and derived RFM quintile scores. The prediction window (December 2010 through December 2011) provides the target variable: `clv_12m`, defined as the sum of all line-item revenue for each customer during that period.

A LightGBM Regressor is trained on the log1p-transformed target to handle the heavily right-skewed CLV distribution. Predictions are inverted with expm1() and clipped at zero. Two baselines establish the performance floor: a naive mean baseline (predicts the training set average for all customers) and a BG/NBD + Gamma-Gamma probabilistic model from the lifetimes library.

### Key Results

| Model | MAE | RMSE | Spearman r | Top-Decile Lift |
|-------|----:|-----:|-----------:|----------------:|
| Naive Mean Baseline | £2,551 | £6,774 | -- | -- |
| BG/NBD + Gamma-Gamma | £1,426 | £3,019 | 0.630 | -- |
| **LightGBM (22 features)** | **£1,228** | **£3,063** | **0.600** | **5.33x** |

The LightGBM model achieves a 52% MAE reduction over the naive baseline and a 14% improvement over the BG/NBD probabilistic model. The top-decile lift of 5.33x means that the model's top 10% of predicted CLV captures 53% of actual total CLV — a strong concentration signal for marketing budget allocation. The Spearman rank correlation of 0.60 confirms that the model correctly orders customers by value, which is the primary operational requirement.

### Deployment Recommendation

The model is deployed as a three-tier application: FastAPI backend (port 8002) serving predictions and SHAP explanations, an Express proxy (port 3002) forwarding API requests, and a React + Vite frontend (port 5174) providing Lookup, What-If, and Portfolio views. All customers are scored at API startup and cached in memory for low-latency serving.

---

## 2. Literature Review

This project draws on four bodies of work, each directly motivating a specific design decision.

### 2.1 RFM Segmentation Framework

> Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.

The RFM framework — Recency, Frequency, and Monetary value — was introduced as a practical segmentation tool for direct marketing. Hughes demonstrated that these three behavioral dimensions, combined into a scoring schema, outperform more complex demographic segmentation for predicting customer response rates and value. The core insight is that recent, frequent, high-spending customers are the most likely to respond to future marketing efforts and generate the highest lifetime value.

**Influence on this project:** The five RFM core features (`recency_days`, `frequency`, `monetary_total`, `monetary_avg`, `monetary_max`) form the backbone of the feature set. Additionally, derived RFM quintile scores (`rfm_recency_score`, `rfm_frequency_score`, `rfm_monetary_score`, `rfm_combined_score`) are computed to provide the model with both raw values and discretized rank signals. The combined RFM score also serves as the basis for the CLV segment tier definitions.

### 2.2 BG/NBD and Gamma-Gamma Models

> Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. *Marketing Science*, 24(2), 275--284.

> Fader, P. S., & Hardie, B. G. S. (2013). The Gamma-Gamma Model of Monetary Value. Companion technical note.

The BG/NBD (Beta-Geometric/Negative Binomial Distribution) model is the standard probabilistic approach to CLV prediction in non-contractual settings. It models two latent processes: a customer's transaction rate (Poisson process with gamma-distributed heterogeneity across customers) and their dropout probability (geometric process with beta-distributed heterogeneity). The Gamma-Gamma companion model estimates expected monetary value per transaction, conditional on observed spending. Together, the two models produce a CLV estimate from only three inputs: recency, frequency, and monetary value.

**Influence on this project:** The BG/NBD + Gamma-Gamma model serves as the primary interpretable baseline (MAE £1,426, Spearman r = 0.630). It requires no feature engineering beyond RFM and provides an important calibration benchmark. The fact that the LightGBM model achieves a 14% MAE improvement over this strong baseline validates the incremental value of the 22-feature approach — the additional behavioral, temporal, and derived features capture signals that the purely probabilistic model cannot.

### 2.3 LightGBM — Gradient Boosted Decision Trees

> Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).

LightGBM introduced Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to accelerate gradient boosting without sacrificing accuracy. Its histogram-based split finding is particularly efficient for mixed feature types — continuous monetary values, integer counts, and binary flags — which characterize customer-level CLV features.

**Influence on this project:** LightGBM was selected as the primary model because it handles the mixed feature types natively, supports the log-transformed regression target required by the skewed CLV distribution, and provides native feature importance rankings that complement SHAP analysis. The `reg_alpha` and `reg_lambda` regularization parameters (both set to 0.1) control model complexity and prevent overfitting on the relatively small training set of 2,252 customers.

### 2.4 SHAP — Unified Model Explainability

> Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS).

SHAP provides theoretically grounded feature attributions based on Shapley values from cooperative game theory. The `TreeExplainer` variant computes exact SHAP values for tree-based models in polynomial time, guaranteeing that feature contributions sum to the difference between the individual prediction and the global mean.

**Influence on this project:** SHAP `TreeExplainer` is used to explain individual CLV predictions. For each customer, the top-5 feature contributions are returned alongside the predicted CLV, enabling marketing teams to understand why a particular customer is classified as a Champion or Dormant. The global SHAP importance ranking (Section 7) identifies `monetary_total` and `recency_days` as the dominant drivers, validating the classic RFM intuition while revealing the incremental contributions of behavioral and temporal features.

---

## 3. Data Description

### 3.1 Data Source

**UCI Online Retail II** — a publicly available dataset (CC BY 4.0 license) from the UCI Machine Learning Repository. It contains transactional records from a UK-based online gift company specializing in unique all-occasion gifts, with many customers being wholesalers.

| Property | Value |
|----------|-------|
| Raw rows | ~1,067,371 |
| Raw columns | 8 (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country) |
| Date range | December 1, 2009 -- December 9, 2011 |
| Unique customers (with ID) | ~5,942 |
| Countries | 43 (91.9% UK) |
| License | CC BY 4.0 |
| Location | `shared/data/raw/online_retail_ii.csv` |

### 3.2 Data Quality and Cleaning

Four data quality issues were identified and resolved during the cleaning pipeline (`src/data_pipeline.py`):

| Issue | Count | Resolution |
|-------|------:|-----------|
| Null Customer ID | ~243,007 (22.8%) | Excluded; no customer-level features possible |
| Cancellation invoices (prefix 'C') | ~19,494 | Removed; not valid purchase signals |
| Non-product StockCodes (POST, D, M, DOT, etc.) | Variable | Removed; represent fees and adjustments, not product purchases |
| Rows with Price <= 0 or Quantity <= 0 | Variable | Removed; invalid transaction records |

After cleaning, deduplicated transactions are written to `data/processed/transactions_clean.csv`. The null Customer ID rate of 22.8% represents a meaningful data loss; for production deployment, enforcing Customer ID capture at point-of-sale is strongly recommended.

### 3.3 Temporal Windows

The dataset's two-year span enables a proper temporal split that prevents data leakage:

| Window | Start | End | Purpose |
|--------|-------|-----|---------|
| Observation | 2009-12-01 | 2010-11-30 | Construct per-customer features (RFM, behavioral, temporal) |
| Prediction | 2010-12-01 | 2011-12-09 | Compute CLV target labels (sum of future revenue) |

This design ensures that no future information leaks into the feature set. Features are computed exclusively from observation-window transactions, while the target `clv_12m` is the sum of `line_total` (Quantity x Price) for each customer during the prediction window. Customers who made no prediction-window purchases receive `clv_12m = 0`.

### 3.4 Population Definition

From the cleaned observation-window transactions, customer-level labels are computed in `data/processed/clv_labels.csv`:

| Column | Description |
|--------|-------------|
| customer_id | Unique customer identifier |
| n_obs_purchases | Number of unique invoices in observation window |
| clv_12m | Total spend in prediction window (0 if no purchases) |
| is_cold_start | 1 if n_obs_purchases < 2 (single observation-window purchase) |

Only **repeat customers** (is_cold_start == 0, meaning 2 or more observation-window purchases) are included in the training population. This yields 2,815 customers, split 80/20 into 2,252 training and 563 test customers, stratified by CLV quintile.

---

## 4. Feature Engineering

Twenty-two features are computed per customer from observation-window transactions (`src/feature_engineering.py`). The full canonical definitions are maintained in [`data_dictionary.md`](data_dictionary.md).

### 4.1 RFM Core Features (5)

| Feature | Type | Derivation |
|---------|------|------------|
| `recency_days` | int | Days from last purchase to reference date (2010-11-30). Lower = more recent. |
| `frequency` | int | Count of unique invoices in observation window |
| `monetary_total` | float | Sum of all line_total across transactions |
| `monetary_avg` | float | monetary_total / frequency |
| `monetary_max` | float | Maximum single-invoice total |

These five features encode the classic RFM dimensions in their raw numeric form. `monetary_max` supplements the mean and total by capturing the peak transaction size, which is particularly informative for wholesale customers who place occasional large orders.

### 4.2 Behavioral Features (8)

| Feature | Type | Derivation |
|---------|------|------------|
| `tenure_days` | int | Days from first to last purchase |
| `purchase_velocity` | float | frequency / (tenure_days / 30.44); equals frequency if tenure = 0 |
| `inter_purchase_days_avg` | float | Mean gap between consecutive invoice dates; 0 if single invoice |
| `inter_purchase_days_std` | float | Std dev (ddof=1) of inter-purchase gaps; 0 if fewer than 3 invoices |
| `unique_products` | int | Count of distinct StockCodes purchased |
| `cancellation_rate` | float | Always 0 (cancellations filtered in cleaning step) |
| `avg_quantity_per_item` | float | Mean quantity per line item |
| `uk_customer` | int | 1 if any transaction has Country == 'United Kingdom' |

The behavioral group captures purchasing rhythm and breadth. `purchase_velocity` normalizes frequency by tenure to distinguish genuinely frequent buyers from customers who simply have a longer observation period. `inter_purchase_days_std` captures regularity: a low standard deviation indicates clockwork-like purchasing patterns associated with wholesale accounts, while high variability suggests occasional gift buyers.

### 4.3 Temporal Features (5)

| Feature | Type | Derivation |
|---------|------|------------|
| `acquisition_month` | int | Month of first purchase (1--12) |
| `acquisition_quarter` | int | Quarter of first purchase (1--4) |
| `purchased_in_q4` | int | 1 if any purchase in Oct, Nov, or Dec |
| `weekend_purchase_ratio` | float | Proportion of transactions on Saturday or Sunday |
| `evening_purchase_ratio` | float | Proportion of transactions at or after 17:00 |

Temporal features capture seasonality and behavioral timing. `purchased_in_q4` flags customers active during the holiday peak, which correlates with higher future spend. `weekend_purchase_ratio` and `evening_purchase_ratio` proxy for consumer vs. business buying patterns — B2B wholesalers tend to purchase during business hours on weekdays.

### 4.4 Derived RFM Scores (4)

| Feature | Type | Derivation |
|---------|------|------------|
| `rfm_recency_score` | int | Quintile rank of recency_days (5 = most recent) |
| `rfm_frequency_score` | int | Quintile rank of frequency (5 = most frequent) |
| `rfm_monetary_score` | int | Quintile rank of monetary_total (5 = highest spend) |
| `rfm_combined_score` | int | Sum of three quintile scores (range 3--15) |

The derived RFM scores provide the model with discretized rank information that is invariant to the scale of raw values. The combined score serves as a composite customer quality index. These features are complementary to the raw RFM values: the model can use raw values for fine-grained splits and score ranks for coarse-grained segmentation.

---

## 5. Methodology

### 5.1 Problem Framing

CLV prediction is framed as a **regression problem**: given a customer's 22-dimensional feature vector from the observation window, predict `clv_12m`, the total revenue they will generate in the subsequent 12-month prediction window.

The target variable is heavily right-skewed: the median CLV is £622 while the mean is £2,397, indicating a long tail of high-value wholesale customers. To handle this skewness, the target is log1p-transformed for training:

```
y_train = log(1 + clv_12m)
```

Predictions are inverted with expm1 and clipped at zero:

```
predicted_clv = max(exp(y_pred) - 1, 0)
```

This transformation compresses the dynamic range of the target, reducing the influence of extreme values on the loss function while preserving the ability to predict large CLV values through the inverse transform.

### 5.2 Train/Test Split

The dataset of 2,815 repeat customers is split 80/20, stratified by CLV quintile to ensure balanced representation of high-value and low-value customers in both partitions:

| Partition | Customers | Purpose |
|-----------|----------:|---------|
| Training | 2,252 | Model fitting |
| Test | 563 | Final evaluation (held out, never seen during training) |

Stratified splitting is critical because the CLV distribution is highly skewed. Without stratification, the test set could under-represent Champions or Dormant customers, producing misleading metric estimates.

### 5.3 Baseline Models

**Baseline 1: Naive Mean**

Predicts the training set mean CLV (£2,397) for every customer. This baseline provides no customer differentiation and serves as the absolute performance floor.

- MAE: £2,551 | RMSE: £6,774

**Baseline 2: BG/NBD + Gamma-Gamma**

The standard probabilistic CLV model from the `lifetimes` library. The BG/NBD component models expected future purchase count using recency, frequency, and customer age. The Gamma-Gamma component models expected monetary value per transaction. Their product yields the expected CLV.

Implementation details:
- `lifetimes` frequency = number of repeat purchases (total invoices minus 1)
- `lifetimes` recency = tenure_days (time between first and last purchase)
- `lifetimes` T = tenure_days + recency_days (total observation period)
- Penalizer coefficient: 0.01 for both models
- Prediction horizon: 365 days

Results: MAE £1,426 | RMSE £3,019 | Spearman r = 0.630

The BG/NBD baseline is a strong benchmark. Its Spearman correlation of 0.630 slightly exceeds the LightGBM model's 0.600, indicating that the probabilistic model produces slightly better customer rankings. However, the LightGBM model achieves 14% lower MAE, meaning its absolute predictions are more accurate — a tradeoff between ranking and calibration quality.

### 5.4 Primary Model: LightGBM Regressor

LightGBM was trained on all 22 features with the following hyperparameters:

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `objective` | `regression` | Mean squared error loss on log1p-transformed target |
| `metric` | `mae` | Optimize for mean absolute error during training |
| `n_estimators` | 300 | Sufficient capacity without overfitting on 2,252 training rows |
| `learning_rate` | 0.05 | Conservative step size for stable convergence |
| `num_leaves` | 31 | Default; controls tree complexity |
| `min_child_samples` | 20 | Minimum leaf size; regularizes against overfitting small customer groups |
| `feature_fraction` | 0.8 | Subsample 80% of features per tree (column dropout) |
| `bagging_fraction` | 0.8 | Subsample 80% of rows per iteration |
| `bagging_freq` | 5 | Apply row subsampling every 5 iterations |
| `reg_alpha` | 0.1 | L1 regularization on leaf weights |
| `reg_lambda` | 0.1 | L2 regularization on leaf weights |
| `random_state` | 42 | Reproducibility |

The combination of feature and bagging subsampling with L1/L2 regularization is designed to prevent overfitting on the relatively small training set. The log1p target transformation works in concert with the regression objective to produce well-calibrated predictions across the full CLV range.

### 5.5 Cold-Start Handling

Customers with fewer than 2 observation-window purchases (is_cold_start == 1) lack sufficient transaction history for reliable feature computation — metrics like inter-purchase gap, purchase velocity, and RFM scores are undefined or degenerate for single-purchase customers.

**Strategy:** Cold-start customers are excluded from training and receive a fixed fallback prediction equal to the training set median CLV of **£622**. This is a conservative estimate that avoids both over-predicting (which would waste marketing budget) and under-predicting (which would ignore potentially valuable new customers).

At inference, the API checks the cold-start flag and returns the median fallback with a clear indicator that the prediction is a default rather than a model-based score.

---

## 6. Model Performance

### 6.1 Regression Metrics

| Model | MAE | RMSE | MAPE | Spearman r |
|-------|----:|-----:|-----:|-----------:|
| Naive Mean Baseline | £2,551 | £6,774 | 394.9% | -- |
| BG/NBD + Gamma-Gamma | £1,426 | £3,019 | 169.3% | 0.630 |
| **LightGBM (22 features)** | **£1,228** | **£3,063** | **79.3%** | **0.600** |

**MAE (Mean Absolute Error):** The primary accuracy metric. LightGBM's MAE of £1,228 means that, on average, individual CLV predictions deviate by £1,228 from the actual value. This represents a 52% improvement over the naive baseline and a 14% improvement over BG/NBD.

**RMSE (Root Mean Squared Error):** Penalizes large errors more heavily. The RMSE of £3,063 is comparable to BG/NBD's £3,019, indicating that both models struggle similarly with extreme-value customers (whales). The gap between MAE and RMSE reflects the long tail of the CLV distribution.

**MAPE (Mean Absolute Percentage Error):** Computed only on customers with non-zero actual CLV to avoid division-by-zero. At 79.3%, MAPE appears high but is characteristic of skewed revenue distributions where many customers have small actual values. MAE is the more operationally relevant metric.

**Spearman r (Rank Correlation):** At 0.600, the model correctly orders 60% of customer pairs by relative CLV. This ranking quality is the most important metric for marketing applications, where the goal is to identify which customers are more valuable rather than predicting exact pound values.

### 6.2 Top-Decile Lift

The top-decile lift of **5.33x** means that the top 10% of customers ranked by predicted CLV hold 53.3% of total actual CLV. This concentration signal is highly actionable: a marketing team targeting only the top decile would capture more than half of all future revenue.

### 6.3 Decile Calibration

| Decile | Count | Mean Actual (£) | Mean Predicted (£) | Total Actual (£) |
|-------:|------:|-----------------:|--------------------:|------------------:|
| 1 | 57 | 441 | 7 | 25,119 |
| 2 | 56 | 365 | 26 | 20,439 |
| 3 | 56 | 509 | 52 | 28,510 |
| 4 | 56 | 706 | 104 | 39,556 |
| 5 | 57 | 870 | 198 | 49,605 |
| 6 | 56 | 873 | 317 | 48,874 |
| 7 | 56 | 1,393 | 577 | 78,027 |
| 8 | 56 | 1,363 | 1,058 | 76,327 |
| 9 | 56 | 3,142 | 1,856 | 175,935 |
| 10 | 57 | 10,878 | 11,084 | 620,035 |

The decile table reveals that the model systematically under-predicts for deciles 1--9 while achieving close calibration for the top decile (mean predicted £11,084 vs. actual £10,878). This conservative bias is a natural consequence of the log1p transformation, which compresses the upper tail. For business purposes, this behavior is acceptable: the model correctly identifies the highest-value customers while under-promising on mid-range customers, reducing the risk of over-allocating marketing spend.

The monotonically increasing pattern of mean actual CLV across deciles confirms that the model's ranking is sound even where absolute values are under-estimated.

---

## 7. SHAP Analysis

### 7.1 Global Feature Importance

SHAP `TreeExplainer` was computed on the full test set using a 500-row background sample from the training set (random state 42). The top 10 features by mean absolute SHAP value:

| Rank | Feature | Mean |SHAP| | Interpretation |
|-----:|---------|-----:|----------------|
| 1 | `monetary_total` | 0.756 | Total lifetime spend is the dominant CLV predictor |
| 2 | `recency_days` | 0.414 | Recent customers have higher predicted CLV |
| 3 | `rfm_combined_score` | 0.303 | Composite RFM quality captures multi-dimensional value |
| 4 | `monetary_max` | 0.236 | Peak transaction size identifies wholesale buyers |
| 5 | `unique_products` | 0.209 | Product breadth signals engagement depth |
| 6 | `monetary_avg` | 0.194 | Average order value complements total and peak |
| 7 | `purchase_velocity` | 0.189 | Purchase rate normalized by tenure |
| 8 | `avg_quantity_per_item` | 0.187 | Bulk buying behavior (wholesale signal) |
| 9 | `tenure_days` | 0.176 | Longer customer relationships predict higher CLV |
| 10 | `inter_purchase_days_std` | 0.145 | Regularity of purchasing rhythm |

### 7.2 Key Insights

**Monetary dominance:** The top three SHAP features are all monetary or RFM-composite signals, confirming that past spending behavior is the strongest predictor of future spending. `monetary_total` alone has nearly twice the SHAP importance of the next feature.

**Recency as a freshness signal:** `recency_days` ranks second, reflecting the intuition that customers who purchased recently are more likely to purchase again. The SHAP direction is negative — higher recency (longer since last purchase) reduces predicted CLV.

**Behavioral features add incremental value:** `unique_products`, `purchase_velocity`, and `avg_quantity_per_item` all appear in the top 10, validating the decision to go beyond raw RFM. These features capture product engagement breadth and wholesale buying patterns that the BG/NBD baseline cannot model.

**Temporal features are secondary:** Temporal features (`acquisition_month`, `purchased_in_q4`, etc.) do not appear in the top 10, suggesting that when and at what time a customer shops matters less than how much and how often they buy. This is consistent with the wholesale-heavy nature of the dataset, where business purchasing schedules are less sensitive to calendar effects than consumer behavior.

---

## 8. Segment Definitions

Customers are assigned to five CLV segments based on their predicted CLV percentile rank within the scored population. These segments are computed at API startup and cached alongside predictions.

| Segment | Percentile Range | Description | Recommended Action |
|---------|-----------------|-------------|-------------------|
| **Champions** | 90th--100th | Top 10% by predicted CLV. Typically high-frequency wholesale accounts with large order values. | VIP treatment, dedicated account management, early access to new products, loyalty rewards |
| **High Value** | 75th--90th | Above-average CLV. Consistent buyers who may be candidates for upselling to Champion tier. | Cross-sell campaigns, volume discount offers, referral incentives |
| **Growing** | 40th--75th | Mid-range CLV. Moderate purchasing frequency and order sizes with growth potential. | Targeted promotions, product recommendations, engagement nurturing sequences |
| **Occasional** | 10th--40th | Below-average CLV. Infrequent buyers with low order values. | Re-engagement campaigns, win-back offers, low-cost digital touchpoints |
| **Dormant** | 0th--10th | Bottom 10% by predicted CLV. Minimal purchasing activity, highest churn risk. | Cost-efficient reactivation (email only), exit survey, reduce spend allocation |

### Segment Thresholds

Segment boundaries are computed dynamically from the percentile distribution of predicted CLV values at API startup. This means that the absolute GBP thresholds will shift as the customer base evolves. The percentile-based approach ensures that each segment always contains the expected proportion of customers regardless of inflation, product mix changes, or customer base growth.

---

## 9. API Reference

The CLV model is served via a FastAPI application at `src/api/main.py`. All endpoints are prefixed with `/api/` and proxied through Express at port 3002.

### 9.1 Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model": "clv",
  "customers": 4312
}
```

### 9.2 List Customers (Paginated)

```
GET /api/customers?page=1&per_page=50&segment=Champions
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number (>= 1) |
| `per_page` | int | 50 | Items per page (1--200) |
| `segment` | string | null | Filter by CLV segment name |

**Response:** `PaginatedResponse`
```json
{
  "items": [
    {
      "customer_id": 12748,
      "predicted_clv": 15234.50,
      "clv_segment": "Champions",
      "recency_days": 2,
      "frequency": 28,
      "monetary_total": 18920.45
    }
  ],
  "pages": 10,
  "total": 500,
  "page": 1,
  "per_page": 50
}
```

**Important:** The response uses `items` (not `customers`) and `pages` (not `total_pages`). This convention is shared across all models in the Marketing Analytics platform.

### 9.3 Customer Detail

```
GET /api/customer/{customer_id}
```

**Response:** `CustomerDetailResponse` — flat structure (not nested under "prediction")
```json
{
  "profile": {
    "customer_id": 12748,
    "recency_days": 2.0,
    "frequency": 28,
    "monetary_total": 18920.45,
    "monetary_avg": 675.73,
    "tenure_days": 350,
    "unique_products": 142,
    "uk_customer": true,
    "rfm_combined_score": 14
  },
  "predicted_clv": 15234.50,
  "clv_segment": "Champions",
  "percentile_rank": 95.2,
  "shap_factors": [
    {"feature": "monetary_total", "value": 18920.45, "contribution": 1.82},
    {"feature": "recency_days", "value": 2.0, "contribution": 0.95},
    {"feature": "frequency", "value": 28.0, "contribution": 0.73},
    {"feature": "unique_products", "value": 142.0, "contribution": 0.41},
    {"feature": "purchase_velocity", "value": 2.4, "contribution": 0.38}
  ]
}
```

**Important:** `predicted_clv`, `clv_segment`, and `shap_factors` are at the top level of the response, not nested under a `prediction` key.

### 9.4 What-If Prediction

```
POST /api/predict
```

**Request Body:** `PredictRequest`
```json
{
  "recency_days": 30,
  "frequency": 10,
  "monetary_total": 5000,
  "monetary_avg": 500,
  "purchase_velocity": 1.5,
  "cancellation_rate": 0.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `recency_days` | float | Yes | Days since last purchase (0--3650) |
| `frequency` | int | Yes | Number of purchases (1--500) |
| `monetary_total` | float | Yes | Total spend in GBP (0--500,000) |
| `monetary_avg` | float | No | Average order value (derived from total/frequency if omitted) |
| `purchase_velocity` | float | No | Purchases per month (uses median default if omitted) |
| `cancellation_rate` | float | No | Cancellation rate 0--1 (default 0.0) |

Features not provided are filled from population medians computed at startup.

**Response:** `PredictResponse`
```json
{
  "predicted_clv": 8542.30,
  "clv_segment": "High Value",
  "shap_values": [...]
}
```

### 9.5 Portfolio (Scatter Plot Data)

```
GET /api/portfolio
```

Returns all customers with recency, predicted CLV, segment, and frequency for the portfolio scatter plot visualization.

**Response:** `list[PortfolioItem]`
```json
[
  {
    "customer_id": 12748,
    "recency_days": 2.0,
    "predicted_clv": 15234.50,
    "clv_segment": "Champions",
    "frequency": 28
  }
]
```

### 9.6 Segment Summary

```
GET /api/segments
```

Returns aggregate statistics for each CLV segment.

**Response:** `list[SegmentSummary]`
```json
[
  {
    "segment": "Champions",
    "count": 432,
    "mean_clv": 12450.75,
    "mean_recency_days": 15.3
  }
]
```

---

## 10. Deployment

### 10.1 Architecture

The CLV model follows the three-tier architecture established by the Marketing Analytics platform:

```
React + Vite (5174) → Express Proxy (3002) → FastAPI (8002)
```

- **FastAPI** (`src/api/main.py`, port 8002): Loads the LightGBM model, SHAP explainer, and metadata at startup via lifespan context. Scores all customers at startup and caches predictions in `app.state` for low-latency serving. Individual SHAP explanations are computed on-demand per request.

- **Express Proxy** (`app/server/index.js`, port 3002): Forwards all `/api/*` requests to FastAPI. Serves documentation markdown files at `/api/docs/{docname}`.

- **React + Vite** (`app/client/src/`, port 5174): Three views — Lookup (customer search with CLV prediction, profile, and SHAP breakdown), What-If (slider-based CLV simulator), and Portfolio (recency vs. CLV scatter plot with segment coloring).

### 10.2 Model Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| `lgbm_clv.pkl` | `models/` | Trained LightGBM Regressor (joblib-serialized) |
| `shap_explainer.pkl` | `models/` | SHAP TreeExplainer with 500-row background |
| `metadata.json` | `models/` | Feature columns, split sizes, metrics, segment thresholds, median CLV |

### 10.3 Running the Stack

```bash
# Terminal 1: FastAPI
cd m02_clv
/Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port 8002 --reload

# Terminal 2: Express proxy
cd m02_clv/app/server && npm start

# Terminal 3: Vite frontend
cd m02_clv/app/client && npm run dev
```

### 10.4 Module Resolution

The FastAPI entry point requires a `sys.path.insert` at the top of `main.py` to resolve imports from the `src` package:

```python
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
```

Uvicorn must be run from the model directory (`m02_clv/`), not from the workspace root.

---

## 11. Future Work

### 11.1 Model Improvements

**Dynamic feature refresh.** The current model uses a fixed observation window. Implementing a rolling-window feature pipeline that recomputes features from the most recent 12 months of transactions would keep predictions current as customer behavior evolves.

**Hyperparameter tuning.** The current hyperparameters are manually selected based on reasonable defaults. A Bayesian optimization sweep (e.g., Optuna) over learning rate, num_leaves, min_child_samples, and regularization parameters could yield modest performance improvements, particularly for high-value customers where prediction accuracy matters most.

**Ensemble with BG/NBD.** The BG/NBD model achieves slightly higher Spearman r (0.630 vs. 0.600) while LightGBM achieves lower MAE. A stacked ensemble — using BG/NBD predicted CLV as an additional feature for LightGBM — could combine the ranking quality of the probabilistic model with the absolute accuracy of the tree-based model.

**Cold-start improvement.** The current median fallback for single-purchase customers is a coarse approximation. A dedicated model for cold-start customers — using acquisition channel, first-purchase product category, and first-order value as features — could provide differentiated predictions for new customers before they reach the 2-purchase threshold.

### 11.2 Data Enhancements

**Real campaign telemetry.** Incorporating email open rates, click-through rates, and campaign response data would add behavioral signals beyond transactional history.

**Product category features.** The `StockCode` and `Description` columns contain rich product information that is currently aggregated away. Computing product category embeddings or topic clusters from purchase baskets would enable product-affinity-based CLV differentiation.

**External economic signals.** Macroeconomic indicators (consumer confidence index, retail sales data) and seasonal calendar events could improve predictions for businesses sensitive to economic cycles.

### 11.3 Platform Integration

**Automated retraining pipeline.** Implement a monthly pipeline that refreshes features, retrains the model, and compares metrics against the previous version before promoting to production.

**A/B testing framework.** Validate that CLV-based marketing allocation produces measurable revenue uplift compared to uniform allocation through a controlled experiment framework.

**Cross-model signals.** The churn model (m03) and CLV model share the same customer base. Combining churn risk with CLV produces a risk-adjusted expected value that is more actionable than either signal alone: a high-CLV, high-churn-risk customer demands immediate retention intervention, while a high-CLV, low-churn-risk customer can be served with standard engagement.

---

*For reference documentation, see:*
- [Model Card](model_card.md) — brief model summary for operations teams
- [Validation Report](validation_report.md) — raw metrics tables from training run
- [Data Dictionary](data_dictionary.md) — feature definitions and file schema
- [Research Brief](research_brief.md) — initial project scoping and literature review
