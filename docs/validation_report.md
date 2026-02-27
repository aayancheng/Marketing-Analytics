# Validation Report (Temporal, Leakage-Safe)

## Split Definition
- Training window: first 18 months from first campaign timestamp
- Validation window: first half of final 6 months (for calibration)
- Test window: second half of final 6 months
- Dataset: `data/processed/event_features.csv`

## AUC Comparison
| Model | Test AUC |
|---|---:|
| Naive Random | 0.5039 |
| Logistic Regression (`send_hour`,`send_dow`) | 0.5114 |
| LightGBM (20 features, calibrated) | 0.5235 |

## Calibration
- Brier score (calibrated LightGBM): 0.178859

## Confusion Matrix @ 0.5
| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 9555 | 0 |
| True 1 | 2909 | 0 |

## Top-3 Hit Rate
- Overall: 0.0507

| RFM Segment | Top-3 Hit Rate |
|---|---:|
| Other | 0.0592 |
| Loyal | 0.0552 |
| Champions | 0.0479 |
| At Risk | 0.0337 |
| Hibernating | 0.0000 |

## AUC-ROC (ASCII sample points)
- AUC: 0.5235

```text
FPR 0.0 -> TPR 0.000
FPR 0.1 -> TPR 0.106
FPR 0.2 -> TPR 0.214
FPR 0.3 -> TPR 0.323
FPR 0.4 -> TPR 0.433
FPR 0.5 -> TPR 0.542
FPR 0.6 -> TPR 0.636
FPR 0.7 -> TPR 0.733
FPR 0.8 -> TPR 0.831
FPR 0.9 -> TPR 0.916
FPR 1.0 -> TPR 1.000
```
