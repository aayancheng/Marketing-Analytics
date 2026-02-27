# Model Card

## Purpose
Predict the best email send hour/day for each customer to maximize open probability.

## Intended Users
Marketing operations and lifecycle campaign managers.

## Data
- Base: UCI Online Retail II (UK-filtered)
- Synthetic: campaign send/open/click/purchase events
- Training table: `data/processed/event_features.csv` (event-level)

## Model
- Primary: LightGBM binary classifier
- Baselines: random, logistic regression (`send_hour`, `send_dow`)
- Probability calibration: isotonic regression

## Outputs
- Top-3 recommended send windows
- 7x24 probability heatmap
- SHAP contribution list

## Known Limitations
- UK-only behavior in v2
- Synthetic campaign outcomes may not match all real-world campaign dynamics
- Customers with <5 transactions return `insufficient_history`

## Fairness & Risk
No direct PII used; customer IDs are anonymized. Geographic generalization is intentionally limited in v2.
