# Model Card — Customer Lifetime Value (CLV)

**Project:** Marketing Analytics — m02_clv
**Version:** 1.0 | **Date:** 2026-02-26
**Model:** LightGBM Regressor (log1p-transformed target)

---

## Purpose

Predict the 12-month forward-looking Customer Lifetime Value (CLV) in GBP for each customer of a UK-based online gift retailer. The model transforms CLV from a backward-looking aggregate into a per-customer predictive score, enabling marketing budget allocation, retention prioritization, and tiered account management.

## Intended Users

- **Marketing operations** — allocate campaign spend proportional to predicted customer value
- **CRM and lifecycle teams** — identify Champions for loyalty programs and Dormant customers for win-back campaigns
- **Finance and planning** — forecast portfolio-level revenue from the customer base
- **Account management** — prioritize high-value wholesale accounts for personalized outreach

## Training Data

- **Source:** UCI Online Retail II (CC BY 4.0) — 1,067,371 transactions from a UK-based online gift company, December 2009 to December 2011
- **Temporal design:** Observation window (Dec 2009 -- Nov 2010) for feature construction; Prediction window (Dec 2010 -- Dec 2011) for CLV labels
- **Training population:** 2,815 customers with 2+ purchases in the observation window (repeat customers only)
- **Train/test split:** 80/20 stratified by CLV quintile (2,252 train / 563 test)
- **Cold-start exclusion:** Customers with fewer than 2 observation-window purchases are excluded from training and receive the median CLV fallback at inference

## Feature Summary

22 features across four groups:

| Group | Count | Examples |
|-------|------:|---------|
| RFM Core | 5 | recency_days, frequency, monetary_total, monetary_avg, monetary_max |
| Behavioral | 8 | tenure_days, purchase_velocity, inter_purchase_days_avg, unique_products, uk_customer |
| Temporal | 5 | acquisition_month, acquisition_quarter, purchased_in_q4, weekend_purchase_ratio, evening_purchase_ratio |
| Derived RFM Scores | 4 | rfm_recency_score, rfm_frequency_score, rfm_monetary_score, rfm_combined_score |

## Performance Metrics

| Model | MAE | RMSE | MAPE | Spearman r |
|-------|----:|-----:|-----:|-----------:|
| Naive Mean Baseline | 2,551 | 6,774 | 394.9% | -- |
| BG/NBD + Gamma-Gamma | 1,426 | 3,019 | 169.3% | 0.630 |
| **LightGBM (22 features)** | **1,228** | **3,063** | **79.3%** | **0.600** |

- **Top-decile lift:** 5.33x -- the top 10% of predicted CLV captures 53% of actual total CLV
- **LightGBM vs. Naive:** 52% MAE reduction
- **LightGBM vs. BG/NBD:** 14% MAE improvement; comparable ranking quality

## Outputs

- **predicted_clv** (float, GBP) — 12-month forward CLV estimate
- **clv_segment** (string) — one of: Champions (top 10%), High Value (75--90th pctl), Growing (40--75th), Occasional (10--40th), Dormant (bottom 10%)
- **percentile_rank** (float, 0--100) — customer's position in the CLV distribution
- **shap_factors** (list) — top-5 SHAP feature contributions explaining the prediction

## Known Limitations

- **UK-concentrated data:** 91.9% of transactions originate from United Kingdom customers. The model has not been validated for non-UK markets with different purchasing patterns or currency environments.
- **Wholesale bias:** The dataset represents a wholesale/gifting retailer; CLV distributions may not transfer to pure B2C or subscription-based businesses.
- **Cold-start gap:** Customers with a single observed purchase receive a fixed median CLV fallback (£622), which does not differentiate within this group.
- **Static features:** Features are computed from a fixed observation window. The model does not incorporate real-time behavioral updates or streaming transaction data.
- **High MAPE on low-CLV customers:** MAPE of 79.3% is driven by customers with small actual CLV values where even modest absolute errors produce large percentage errors. MAE (£1,228) is the more reliable accuracy indicator for business use.
- **No exogenous signals:** The model uses only transactional data. It does not incorporate marketing spend, campaign exposure, macroeconomic indicators, or product catalog changes.

## Ethical Considerations

- **No PII used:** Customer IDs are anonymized integer identifiers. No names, emails, or demographic attributes are used as features.
- **No protected attributes:** The model does not use age, gender, ethnicity, or income as inputs. The only geographic signal is a binary UK/non-UK flag derived from shipping country.
- **Value-based segmentation risk:** Labeling customers as "Dormant" may lead to reduced service levels for low-spend customers. Organizations should ensure that segment-based treatment policies do not create discriminatory outcomes.
- **Prediction uncertainty:** Individual CLV predictions carry substantial uncertainty (RMSE £3,063). Predictions should inform portfolio-level strategy rather than high-stakes individual decisions without human review.
