# Data Dictionary

## Core Files
- `data/processed/transactions_clean.csv`: cleaned UK transactions
- `data/synthetic/campaign_events.csv`: synthetic campaign events
- `data/processed/customer_features.csv`: one row per customer + `rfm_segment`
- `data/processed/model_features.csv`: customer x 168 inference grid
- `data/processed/event_features.csv`: event-level training rows with `send_datetime`

## Feature Columns (20)
- Customer behavioral (11):
  `modal_purchase_hour`, `modal_purchase_dow`, `purchase_hour_entropy`, `avg_daily_txn_count`, `recency_days`, `frequency`, `monetary_total`, `tenure_days`, `country_uk`, `unique_products`, `cancellation_rate`
- Slot features (7):
  `send_hour`, `send_dow`, `is_weekend`, `is_business_hours`, `hour_delta_from_modal`, `dow_match`, `industry_open_rate_by_hour`
- Interaction (2):
  `hour_x_entropy`, `recency_x_frequency`

## Targets and Metadata
- Training target: `opened` (binary)
- Temporal key for split: `send_datetime`
- Customer key type: `int64`
