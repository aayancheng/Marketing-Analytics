# Data Dictionary — m02_clv (Customer Lifetime Value)

## 1. Raw Data

**Source:** UCI Online Retail II (`shared/data/raw/online_retail_ii.csv`)

| Column | Type | Description |
|--------|------|-------------|
| Invoice | str | 6-digit invoice number; prefix 'C' indicates cancellation |
| StockCode | str | 5-digit product code |
| Description | str | Product name |
| Quantity | int | Quantity purchased per line item |
| InvoiceDate | datetime | Invoice timestamp (YYYY-MM-DD HH:MM:SS) |
| Price | float | Unit price in GBP |
| Customer ID | float | 5-digit customer identifier (NaN for guest purchases) |
| Country | str | Country of customer residence |

## 2. Cleaned Transactions (`data/processed/transactions_clean.csv`)

| Column | Type | Description |
|--------|------|-------------|
| Invoice | str | Invoice number (cancellations removed) |
| StockCode | str | Product code (non-product codes removed) |
| Description | str | Product name |
| Quantity | int | Quantity (> 0 only) |
| InvoiceDate | datetime | Invoice timestamp |
| Price | float | Unit price (> 0 only) |
| customer_id | int64 | Customer identifier (no nulls) |
| Country | str | Country of residence |
| line_total | float | Quantity * Price |

**Cleaning rules applied:**
- Dropped rows with null customer_id
- Removed cancellations (Invoice starting with 'C')
- Removed non-product StockCodes: POST, D, M, DOT, BANK CHARGES, AMAZONFEE, S, PADS, C2
- Removed rows with Price <= 0 or Quantity <= 0
- Deduplicated on (Invoice, StockCode, customer_id)

## 3. Observation Transactions (`data/processed/observation_transactions.csv`)

Same schema as cleaned transactions, filtered to the observation window: **Dec 1, 2009 -- Nov 30, 2010**.

Used for computing per-customer features.

## 4. CLV Labels (`data/processed/clv_labels.csv`)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| customer_id | int64 | — | Unique customer identifier |
| n_obs_purchases | int | >= 1 | Number of unique invoices in observation window |
| clv_12m | float | >= 0 | Total spend (sum of line_total) in prediction window (Dec 1, 2010 -- Dec 9, 2011). 0 if no purchases. |
| is_cold_start | int | {0, 1} | 1 if n_obs_purchases < 2 (single purchase in observation window) |

## 5. Engineered Features (`data/processed/customer_features.csv`)

### 5.1 RFM Core Features

| Feature | Type | Range | Derivation |
|---------|------|-------|------------|
| recency_days | int | >= 0 | (reference_date - last_invoice_date).days. Reference date = 2010-11-30. Lower = more recent. |
| frequency | int | >= 1 | Count of unique invoices in observation window |
| monetary_total | float | > 0 | Sum of line_total across all transactions |
| monetary_avg | float | > 0 | monetary_total / frequency |
| monetary_max | float | > 0 | Maximum single-invoice total |

### 5.2 Behavioral Features

| Feature | Type | Range | Derivation |
|---------|------|-------|------------|
| tenure_days | int | >= 0 | (last_purchase_date - first_purchase_date).days |
| purchase_velocity | float | > 0 | frequency / (tenure_days / 30.44). If tenure_days = 0, equals frequency. |
| inter_purchase_days_avg | float | >= 0 | Mean of gaps (in days) between consecutive unique invoice dates. 0 if only one invoice. |
| inter_purchase_days_std | float | >= 0 | Standard deviation (ddof=1) of inter-purchase gaps. 0 if fewer than 3 invoices. |
| unique_products | int | >= 1 | Count of distinct StockCodes purchased |
| cancellation_rate | float | 0.0 | Always 0 since cancellations are filtered in cleaning step |
| avg_quantity_per_item | float | > 0 | Mean quantity per line item |
| uk_customer | int | {0, 1} | 1 if any transaction has Country == 'United Kingdom' |

### 5.3 Temporal Features

| Feature | Type | Range | Derivation |
|---------|------|-------|------------|
| acquisition_month | int | 1--12 | Month of first purchase |
| acquisition_quarter | int | 1--4 | Quarter of first purchase |
| purchased_in_q4 | int | {0, 1} | 1 if any purchase in Oct, Nov, or Dec |
| weekend_purchase_ratio | float | 0--1 | Proportion of transactions on Saturday or Sunday |
| evening_purchase_ratio | float | 0--1 | Proportion of transactions at or after 17:00 |

### 5.4 Derived RFM Scores

| Feature | Type | Range | Derivation |
|---------|------|-------|------------|
| rfm_recency_score | int | 1--5 | Quintile rank of recency_days (5 = most recent, 1 = least recent) |
| rfm_frequency_score | int | 1--5 | Quintile rank of frequency (5 = most frequent, 1 = least frequent) |
| rfm_monetary_score | int | 1--5 | Quintile rank of monetary_total (5 = highest spend, 1 = lowest spend) |
| rfm_combined_score | int | 3--15 | rfm_recency_score + rfm_frequency_score + rfm_monetary_score |

## 6. Training Features (`data/processed/training_features.csv`)

Contains all columns from `customer_features.csv` plus the target variable `clv_12m`. Only includes repeat customers (is_cold_start == 0, i.e., n_obs_purchases >= 2).

| Column | Type | Description |
|--------|------|-------------|
| customer_id | int64 | Customer identifier |
| (all 22 features) | various | See sections 5.1--5.4 above |
| clv_12m | float | Target: total spend in prediction window |

## 7. Target Variable

| Name | Type | Description |
|------|------|-------------|
| clv_12m | float (>= 0) | Customer Lifetime Value over the ~12-month prediction window (Dec 2010 -- Dec 2011). Computed as sum of line_total for the customer in the prediction window. Zero for customers who made no purchases in the prediction period. |

## 8. Temporal Windows

| Window | Start | End | Purpose |
|--------|-------|-----|---------|
| Observation | 2009-12-01 | 2010-11-30 | Build customer features (RFM, behavioral, temporal) |
| Prediction | 2010-12-01 | 2011-12-09 | Compute CLV labels (future spend) |

## 9. Feature Count Summary

| Category | Count |
|----------|-------|
| RFM Core | 5 |
| Behavioral | 8 |
| Temporal | 5 |
| RFM Scores | 4 |
| **Total** | **22** |
