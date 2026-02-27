# Validation Report — m02_clv (Customer Lifetime Value)

## Split Definition
- Training set: 2,252 customers (80% stratified by CLV quintile)
- Test set: 563 customers (20% holdout)
- Target: clv_12m (log1p-transformed for training, expm1 for evaluation)
- Temporal design: features from Dec 2009-Nov 2010, target from Dec 2010-Dec 2011

## Model Performance Comparison

| Model | MAE | RMSE | MAPE | Spearman r |
|-------|----:|-----:|-----:|-----------:|
| Naive Mean Baseline | 2,551 | 6,774 | 394.9% | nan |
| BG/NBD + Gamma-Gamma | 1,426 | 3,019 | 169.3% | 0.6297 |
| **LightGBM (22 features)** | **1,228** | **3,063** | **79.3%** | **0.5995** |

## Top-Decile Lift
- Top decile captures 5.3x its proportional share of total CLV

## Decile Analysis

| Decile | Count | Mean Actual | Mean Predicted | Total Actual |
|-------:|------:|------------:|---------------:|-------------:|
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

## Top 10 Feature Importance (SHAP)

| Feature | Mean |SHAP| |
|---------|-----:|
| monetary_total | 0.7561 |
| recency_days | 0.4144 |
| rfm_combined_score | 0.3029 |
| monetary_max | 0.2355 |
| unique_products | 0.2089 |
| monetary_avg | 0.1939 |
| purchase_velocity | 0.1889 |
| avg_quantity_per_item | 0.1872 |
| tenure_days | 0.1759 |
| inter_purchase_days_std | 0.1447 |

## Cold-Start Strategy
- Customers with < 2 purchases in observation window: use median CLV from training set
- Median CLV (training): 622
