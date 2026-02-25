# Validation Report — Model 3: Churn Propensity

Generated: 2026-02-25 03:14 UTC

## Split Definition
- Method: Stratified random split (80% train / 20% test)
- Stratified on: `Churn` (preserves class balance across both sets)
- Train rows: 5,634
- Test rows:  1,409
- Overall churn rate: 26.7% (train) / 26.7% (test)
- Dataset: `data/processed/customer_features.parquet`

## AUC-ROC Comparison
| Model | Test AUC-ROC |
|---|---:|
| Naive Baseline (constant = majority class prob) | 0.5000 |
| Logistic Regression (all 32 features) | 0.8162 |
| LightGBM (calibrated, `scale_pos_weight=2.77`) | 0.7883 |

## LightGBM Additional Metrics
| Metric | Value |
|---|---:|
| PR-AUC | 0.5401 |
| Brier Score (calibrated) | 0.155158 |
| F1 @ threshold 0.5 | 0.5234 |
| Precision @ threshold 0.5 | 0.6070 |
| Recall @ threshold 0.5 | 0.4601 |

## Confusion Matrix @ Threshold 0.5
| | Pred No Churn | Pred Churn |
|---|---:|---:|
| True No Churn | 921 | 112 |
| True Churn    | 203 | 173 |

## Top-20% Decile Lift
- Customers ranked by predicted churn score, top 20% selected
- Churn rate in top-20%: 0.6132672089110381
- **Lift**: 2.29x versus overall churn rate
- **Capture rate**: 45.7% of all churners fall in top-20% scored customers

## Cost-Optimised Decision Threshold
### Business Cost Model
- False Negative (missed churner): \$200 per customer
- False Positive (unnecessary outreach): \$20 per customer
- Threshold sweep: `np.linspace(0.1, 0.9, 81)`

### Result
| Item | Value |
|---|---:|
| Optimal threshold | 0.16 |
| F1 at optimal threshold | 0.5290 |
| Precision at optimal threshold | 0.3707 |
| Recall at optimal threshold | 0.9229 |
| Total cost at optimal threshold | \$17,580 |
| Baseline cost (flag nobody) | \$75,200 |
| Estimated cost saving | \$57,620 |

**Business interpretation**: At the cost-optimal threshold of **0.16**, the model
targets customers most likely to churn while balancing the cost of missed churners (\$200/each)
against the cost of false alarms (\$20/each). This threshold should be used for operational
scoring rather than the default 0.5.

## AUC-ROC Curve (ASCII sample)
AUC: 0.7883

```text
  FPR 0.0 -> TPR 0.005
  FPR 0.1 -> TPR 0.445
  FPR 0.2 -> TPR 0.620
  FPR 0.3 -> TPR 0.738
  FPR 0.4 -> TPR 0.818
  FPR 0.5 -> TPR 0.885
  FPR 0.6 -> TPR 0.926
  FPR 0.7 -> TPR 0.950
  FPR 0.8 -> TPR 0.970
  FPR 0.9 -> TPR 0.988
  FPR 1.0 -> TPR 1.000
```
