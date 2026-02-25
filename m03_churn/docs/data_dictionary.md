# Data Dictionary — Model 3: Churn Propensity

**Dataset:** IBM Telco Customer Churn
**Source:** IBM Sample Data / Kaggle ([blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
**Raw file:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
**Processed file:** `data/processed/customers_clean.parquet`
**Feature file:** `data/processed/customer_features.parquet`
**Rows:** 7,043 customers (single snapshot, California telecom provider)
**Columns (raw):** 21 (20 features + 1 target)

---

## Section 1: Raw Columns (21 original columns)

| Column | Raw Type | Raw Values | Notes |
|---|---|---|---|
| `customerID` | string | e.g. `7590-VHVEG` | Unique customer identifier. Dropped from model feature matrix; retained for lookup joins. |
| `gender` | string | `Male`, `Female` | Binary demographic. Encoded as Male=1, Female=0. Flag as sensitive attribute — do not use as a direct targeting criterion in production interventions. |
| `SeniorCitizen` | integer | `0`, `1` | Already binary; 1 = senior citizen. Sensitive attribute — document usage. No encoding needed. |
| `Partner` | string | `Yes`, `No` | Whether the customer has a partner. Encoded Yes=1, No=0. |
| `Dependents` | string | `Yes`, `No` | Whether the customer has dependents. Encoded Yes=1, No=0. |
| `tenure` | integer | 0–72 | Months the customer has been with the company. Top predictor of churn. 0 = new customer in first billing month. |
| `PhoneService` | string | `Yes`, `No` | Has phone service. Encoded Yes=1, No=0. |
| `MultipleLines` | string | `Yes`, `No`, `No phone service` | Multiple phone lines. Third category ("No phone service") collapses to 0 (same as "No"). Encoded Yes=1, otherwise=0. |
| `InternetService` | string | `DSL`, `Fiber optic`, `No` | Internet service type. One-hot encoded into 3 dummy columns. Fiber optic correlates strongly with higher churn. |
| `OnlineSecurity` | string | `Yes`, `No`, `No internet service` | Online security add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. |
| `OnlineBackup` | string | `Yes`, `No`, `No internet service` | Online backup add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. |
| `DeviceProtection` | string | `Yes`, `No`, `No internet service` | Device protection add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. |
| `TechSupport` | string | `Yes`, `No`, `No internet service` | Tech support add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. Absence of tech support is a consistent churn driver in the literature. |
| `StreamingTV` | string | `Yes`, `No`, `No internet service` | Streaming TV add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. |
| `StreamingMovies` | string | `Yes`, `No`, `No internet service` | Streaming movies add-on. "No internet service" collapses to 0. Encoded Yes=1, otherwise=0. |
| `Contract` | string | `Month-to-month`, `One year`, `Two year` | Contract type. Top predictor of churn. Month-to-month customers churn at a significantly higher rate. One-hot encoded into 3 dummy columns. |
| `PaperlessBilling` | string | `Yes`, `No` | Paperless billing opted in. Encoded Yes=1, No=0. |
| `PaymentMethod` | string | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Payment method. Electronic check customers have the highest churn rate. One-hot encoded into 4 dummy columns. |
| `MonthlyCharges` | float | 18.25–118.75 | Monthly bill amount in USD. Mean ≈ $64.76. Higher charges correlate with higher churn. |
| `TotalCharges` | string → float | 11 blanks; otherwise numeric | Total amount charged to date. Stored as string in the raw CSV — must be cast with `pd.to_numeric(errors='coerce')`. 11 blank values exist for customers with tenure==0 (no charges have yet accrued). |
| `Churn` | string | `Yes`, `No` | **TARGET VARIABLE.** Whether the customer has churned. Encoded Yes=1, No=0. |

---

## Section 2: Cleaned and Encoded Columns

All columns below are present in `customers_clean.parquet`. The DataFrame contains 7,043 rows and all values are numeric (no nulls).

### 2a. Passthrough (no encoding change)

| Column | Type After Cleaning | Notes |
|---|---|---|
| `customerID` | string | Retained as first column for lookup; excluded from model feature matrix `X`. |
| `SeniorCitizen` | int (0/1) | Already binary in raw file; no transformation applied. |
| `tenure` | int (0–72) | Kept as-is. |
| `MonthlyCharges` | float | Kept as-is. |
| `TotalCharges` | float | Cast from string; 11 blanks imputed with 0.0 (all had tenure==0). |

### 2b. Binary Encodings (Yes/No → 1/0)

| Column | Encoding Rule | Values |
|---|---|---|
| `gender` | Male=1, Female=0 | {0, 1} |
| `Partner` | Yes=1, No=0 | {0, 1} |
| `Dependents` | Yes=1, No=0 | {0, 1} |
| `PhoneService` | Yes=1, No=0 | {0, 1} |
| `PaperlessBilling` | Yes=1, No=0 | {0, 1} |
| `Churn` | Yes=1, No=0 | {0, 1} — **Target column** |

### 2c. 3-Category Service Encodings (collapsed to binary)

All six service columns and MultipleLines use the same rule: any "No" variant (including "No phone service" and "No internet service") maps to 0; "Yes" maps to 1. This preserves interpretability: a value of 1 means the service is actively subscribed.

| Column | Encoding Rule | Values |
|---|---|---|
| `MultipleLines` | Yes=1, No=0, No phone service=0 | {0, 1} |
| `OnlineSecurity` | Yes=1, No=0, No internet service=0 | {0, 1} |
| `OnlineBackup` | Yes=1, No=0, No internet service=0 | {0, 1} |
| `DeviceProtection` | Yes=1, No=0, No internet service=0 | {0, 1} |
| `TechSupport` | Yes=1, No=0, No internet service=0 | {0, 1} |
| `StreamingTV` | Yes=1, No=0, No internet service=0 | {0, 1} |
| `StreamingMovies` | Yes=1, No=0, No internet service=0 | {0, 1} |

### 2d. One-Hot Encoded Columns

`pd.get_dummies` is applied with `drop_first=False` — all categories are retained as separate dummy columns. Dummy columns are cast to int after encoding.

**InternetService** (3 dummies):

| Column | Value=1 Condition |
|---|---|
| `InternetService_DSL` | Customer has DSL internet |
| `InternetService_Fiber optic` | Customer has Fiber optic internet |
| `InternetService_No` | Customer has no internet service |

**Contract** (3 dummies):

| Column | Value=1 Condition |
|---|---|
| `Contract_Month-to-month` | Month-to-month contract |
| `Contract_One year` | One-year contract |
| `Contract_Two year` | Two-year contract |

**PaymentMethod** (4 dummies):

| Column | Value=1 Condition |
|---|---|
| `PaymentMethod_Bank transfer (automatic)` | Automatic bank transfer payment |
| `PaymentMethod_Credit card (automatic)` | Automatic credit card payment |
| `PaymentMethod_Electronic check` | Electronic check payment |
| `PaymentMethod_Mailed check` | Mailed check payment |

---

## Section 3: Engineered Features (7 new features)

All engineered features are computed in `feature_engineering.py:compute_customer_features()` and are added to the DataFrame alongside the cleaned columns. They are included in `FEATURE_COLUMNS` and saved to `customer_features.parquet`.

| Feature | Type | Formula | Rationale |
|---|---|---|---|
| `has_family` | int (0/1) | `1 if (Partner==1) OR (Dependents==1) else 0` | Customers with family ties tend to be more stable and less likely to churn. Combines two correlated signals into one compact indicator. |
| `num_services` | int (0–8) | `PhoneService + MultipleLines + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies` | Depth-of-relationship proxy. Customers subscribed to more services have higher switching costs and lower churn propensity. Range: 0 (no services) to 8 (all services active). |
| `monthly_per_tenure` | float | `MonthlyCharges / (tenure + 1)` | Average spend rate normalised for tenure. A customer with high monthly charges but very short tenure (e.g., first billing month) looks different from a long-tenure customer paying the same amount. The +1 avoids division-by-zero for tenure==0 customers. |
| `total_charges_gap` | float | `MonthlyCharges * tenure - TotalCharges` | Billing anomaly signal. In a frictionless account, TotalCharges should approximately equal MonthlyCharges × tenure. A non-zero gap indicates plan changes, discounts, partial months, or billing irregularities. Expected to be near 0 for stable accounts; negative for customers who received discounts. |
| `is_month_to_month` | int (0/1) | `Contract_Month-to-month` (alias) | Convenience alias for the most predictive contract dummy. Month-to-month customers have no lock-in and churn at roughly 3× the rate of two-year contract customers. |
| `is_fiber_optic` | int (0/1) | `InternetService_Fiber optic` (alias) | Convenience alias. Fiber optic customers have significantly higher churn rates despite — or possibly because of — paying premium prices, suggesting price sensitivity or service quality dissatisfaction. |
| `is_electronic_check` | int (0/1) | `PaymentMethod_Electronic check` (alias) | Convenience alias. Electronic check payers are the payment-method group with the highest observed churn rate in this dataset, possibly reflecting less automated commitment than bank transfer or credit card auto-pay. |

---

## Section 4: Target Variable

| Property | Value |
|---|---|
| **Column name** | `Churn` |
| **Type** | int (0/1) |
| **Encoding** | Yes (churned) = 1; No (retained) = 0 |
| **Class distribution** | Positive (Churn=1): 1,869 customers — **26.54%** |
| | Negative (Churn=0): 5,174 customers — **73.46%** |
| **Class imbalance ratio** | Approximately 1:2.77 (positive:negative) |
| **Imbalance handling** | `scale_pos_weight = 73.5 / 26.5 ≈ 2.77` in LightGBM; alternatively SMOTE on the training fold only |
| **Definition of churn** | Customer has terminated service as recorded in the dataset snapshot |
| **Prediction scope** | Single-period current-state prediction (not a forward-looking survival window) |

**Why 26.5% matters:**
The dataset has moderate class imbalance. A naive majority-class classifier achieves 73.5% accuracy but has zero utility. Accuracy is therefore not a suitable primary metric. AUC-ROC is used as the primary metric because it evaluates rank-order calibration across all thresholds; F1 (at the business-optimised threshold) is the secondary metric.

---

## Section 5: Train/Test Split Strategy

### 5a. Primary Split

| Property | Value |
|---|---|
| **Split ratio** | 80% train / 20% test |
| **Stratification** | Stratified by `Churn` to preserve the 26.5%/73.5% class ratio in both sets |
| **Random seed** | `42` for all stochastic operations |
| **Approximate sizes** | Train: ~5,634 rows; Test: ~1,409 rows |
| **Implementation** | `sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` |

The test set is held out entirely and is used only for final evaluation. It is not used for hyperparameter tuning.

### 5b. Cross-Validation on the Training Set

| Property | Value |
|---|---|
| **Strategy** | Stratified 5-fold cross-validation |
| **Applied to** | Training set (~5,634 rows) only |
| **Stratification** | By `Churn` in every fold |
| **Purpose** | Hyperparameter tuning and model selection without contaminating the test set |
| **Implementation** | `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| **Approximate fold sizes** | Each fold: ~4,507 train / ~1,127 validation rows |

### 5c. Rationale

Because this is a static snapshot dataset (not longitudinal time-series data), time-based splitting is not applicable. Stratified k-fold provides unbiased estimates of generalisation performance while maintaining the natural class ratio in every fold, which is critical for threshold calibration and AUC computation under class imbalance.

SMOTE oversampling (if used) is applied **inside each fold** on the fold's training portion only — never to the validation or test data — to prevent data leakage.

### 5d. Performance Benchmarks (from literature, same dataset)

| Model | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| Always-predict-majority (floor) | 0.50 | — | 73.5% |
| Logistic Regression | ~0.852 | ~0.62 | ~0.81 |
| Decision Tree (max_depth=3) | ~0.780 | ~0.57 | ~0.79 |
| Random Forest | ~0.887 | ~0.65 | ~0.82 |
| XGBoost | ~0.932 | ~0.84 | ~0.84 |
| **LightGBM (target)** | **> 0.90** | **> 0.80** | **> 0.82** |

**Target:** AUC-ROC > 0.90 and F1 > 0.80 on the held-out 20% test set.
