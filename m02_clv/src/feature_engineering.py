"""
Feature Engineering: m02_clv
- Computes per-customer RFM, behavioral, temporal, and derived features
- Merges with CLV labels to produce training dataset
- Exports FEATURE_COLUMNS constant for use by train.py and API

Usage:
    cd m02_clv
    /Users/aayan/MarketingAnalytics/.venv/bin/python src/feature_engineering.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

OBSERVATION_CSV = os.path.join(PROCESSED_DIR, "observation_transactions.csv")
CLV_LABELS_CSV = os.path.join(PROCESSED_DIR, "clv_labels.csv")
CUSTOMER_FEATURES_CSV = os.path.join(PROCESSED_DIR, "customer_features.csv")
TRAINING_FEATURES_CSV = os.path.join(PROCESSED_DIR, "training_features.csv")

# Reference date: end of observation window
REFERENCE_DATE = pd.Timestamp("2010-11-30")

# ---------------------------------------------------------------------------
# FEATURE_COLUMNS — shared with API and train.py
# ---------------------------------------------------------------------------

RFM_CORE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "monetary_max",
]

BEHAVIORAL_COLUMNS = [
    "tenure_days",
    "purchase_velocity",
    "inter_purchase_days_avg",
    "inter_purchase_days_std",
    "unique_products",
    "cancellation_rate",
    "avg_quantity_per_item",
    "uk_customer",
]

TEMPORAL_COLUMNS = [
    "acquisition_month",
    "acquisition_quarter",
    "purchased_in_q4",
    "weekend_purchase_ratio",
    "evening_purchase_ratio",
]

RFM_SCORE_COLUMNS = [
    "rfm_recency_score",
    "rfm_frequency_score",
    "rfm_monetary_score",
    "rfm_combined_score",
]

FEATURE_COLUMNS: list[str] = (
    RFM_CORE_COLUMNS
    + BEHAVIORAL_COLUMNS
    + TEMPORAL_COLUMNS
    + RFM_SCORE_COLUMNS
)


# ---------------------------------------------------------------------------
# Step 1: Compute per-customer features
# ---------------------------------------------------------------------------

def compute_customer_features(observation_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-customer features from observation-window transactions."""
    print("[Step 1] Computing per-customer features...")

    df = observation_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # --- Invoice-level aggregation (for monetary_max) ---
    invoice_totals = (
        df.groupby(["customer_id", "Invoice"])["line_total"]
        .sum()
        .reset_index()
        .rename(columns={"line_total": "invoice_total"})
    )

    # --- Per-customer base aggregation ---
    cust_agg = df.groupby("customer_id").agg(
        last_date=("InvoiceDate", "max"),
        first_date=("InvoiceDate", "min"),
        frequency=("Invoice", "nunique"),
        monetary_total=("line_total", "sum"),
        unique_products=("StockCode", "nunique"),
        avg_quantity_per_item=("Quantity", "mean"),
    )

    # --- RFM Core ---
    # Normalize dates to midnight to avoid sub-day offsets producing negative recency
    cust_agg["last_date"] = cust_agg["last_date"].dt.normalize()
    cust_agg["first_date"] = cust_agg["first_date"].dt.normalize()
    cust_agg["recency_days"] = (REFERENCE_DATE - cust_agg["last_date"]).dt.days
    cust_agg["monetary_avg"] = cust_agg["monetary_total"] / cust_agg["frequency"]

    # monetary_max: max single-invoice total
    monetary_max = (
        invoice_totals.groupby("customer_id")["invoice_total"]
        .max()
        .rename("monetary_max")
    )
    cust_agg = cust_agg.join(monetary_max)

    # --- Behavioral ---
    cust_agg["tenure_days"] = (cust_agg["last_date"] - cust_agg["first_date"]).dt.days

    # purchase_velocity: frequency per 30.44-day period (handle tenure=0)
    tenure_months = cust_agg["tenure_days"] / 30.44
    cust_agg["purchase_velocity"] = np.where(
        cust_agg["tenure_days"] == 0,
        cust_agg["frequency"].astype(float),
        cust_agg["frequency"] / tenure_months,
    )

    # inter_purchase_days: avg and std of gaps between unique invoice dates
    inter_purchase_stats = _compute_inter_purchase_stats(df)
    cust_agg = cust_agg.join(inter_purchase_stats)

    # cancellation_rate: 0 since cancellations are already filtered out
    cust_agg["cancellation_rate"] = 0.0

    # uk_customer: 1 if Country == 'United Kingdom'
    uk_flag = (
        df.groupby("customer_id")["Country"]
        .agg(lambda s: 1 if (s == "United Kingdom").any() else 0)
        .rename("uk_customer")
    )
    cust_agg = cust_agg.join(uk_flag)

    # --- Temporal ---
    cust_agg["acquisition_month"] = cust_agg["first_date"].dt.month
    cust_agg["acquisition_quarter"] = cust_agg["first_date"].dt.quarter

    # purchased_in_q4: any purchase in Oct-Dec
    q4_flag = (
        df[df["InvoiceDate"].dt.month.isin([10, 11, 12])]
        .groupby("customer_id")
        .size()
        .rename("purchased_in_q4")
        .clip(upper=1)
    )
    cust_agg = cust_agg.join(q4_flag)
    cust_agg["purchased_in_q4"] = cust_agg["purchased_in_q4"].fillna(0).astype(int)

    # weekend_purchase_ratio
    df["_is_weekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_ratio = (
        df.groupby("customer_id")["_is_weekend"]
        .mean()
        .rename("weekend_purchase_ratio")
    )
    cust_agg = cust_agg.join(weekend_ratio)

    # evening_purchase_ratio (after 17:00)
    df["_is_evening"] = (df["InvoiceDate"].dt.hour >= 17).astype(int)
    evening_ratio = (
        df.groupby("customer_id")["_is_evening"]
        .mean()
        .rename("evening_purchase_ratio")
    )
    cust_agg = cust_agg.join(evening_ratio)

    # --- RFM Quintile Scores ---
    cust_agg["rfm_recency_score"] = pd.qcut(
        cust_agg["recency_days"].rank(method="first", ascending=True),
        q=5, labels=[5, 4, 3, 2, 1], duplicates="drop",
    ).astype(int)
    cust_agg["rfm_frequency_score"] = pd.qcut(
        cust_agg["frequency"].rank(method="first", ascending=True),
        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop",
    ).astype(int)
    cust_agg["rfm_monetary_score"] = pd.qcut(
        cust_agg["monetary_total"].rank(method="first", ascending=True),
        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop",
    ).astype(int)
    cust_agg["rfm_combined_score"] = (
        cust_agg["rfm_recency_score"]
        + cust_agg["rfm_frequency_score"]
        + cust_agg["rfm_monetary_score"]
    )

    # --- Finalize ---
    cust_agg = cust_agg.drop(columns=["last_date", "first_date"])
    cust_agg = cust_agg.reset_index().rename(columns={"index": "customer_id"})

    # Ensure column order: customer_id + FEATURE_COLUMNS
    cust_agg = cust_agg[["customer_id"] + FEATURE_COLUMNS]

    cust_agg["customer_id"] = cust_agg["customer_id"].astype(np.int64)

    n_customers = len(cust_agg)
    n_features = len(FEATURE_COLUMNS)
    print(f"  Computed {n_features} features for {n_customers:,} customers.")
    return cust_agg


def _compute_inter_purchase_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std of inter-purchase day gaps per customer."""
    invoice_dates = (
        df.groupby(["customer_id", "Invoice"])["InvoiceDate"]
        .min()
        .reset_index()
        .sort_values(["customer_id", "InvoiceDate"])
    )

    results = []
    for cid, group in invoice_dates.groupby("customer_id"):
        dates = group["InvoiceDate"].sort_values().values
        if len(dates) < 2:
            results.append({
                "customer_id": cid,
                "inter_purchase_days_avg": 0.0,
                "inter_purchase_days_std": 0.0,
            })
        else:
            deltas = np.diff(dates).astype("timedelta64[D]").astype(float)
            results.append({
                "customer_id": cid,
                "inter_purchase_days_avg": float(np.mean(deltas)),
                "inter_purchase_days_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            })

    stats_df = pd.DataFrame(results).set_index("customer_id")
    return stats_df


# ---------------------------------------------------------------------------
# Step 2: Build training set (merge features + CLV labels)
# ---------------------------------------------------------------------------

def build_training_set(
    customer_features: pd.DataFrame,
    clv_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge customer features with CLV labels. Only repeat customers (>= 2 purchases)."""
    print("\n[Step 2] Building training set...")

    # Filter to repeat customers only
    repeat_labels = clv_labels[clv_labels["is_cold_start"] == 0].copy()

    training = customer_features.merge(
        repeat_labels[["customer_id", "clv_12m"]],
        on="customer_id",
        how="inner",
    )

    training = training.sort_values("customer_id").reset_index(drop=True)

    n_rows = len(training)
    n_features = len(FEATURE_COLUMNS)
    target_median = training["clv_12m"].median()
    target_mean = training["clv_12m"].mean()
    target_zero = (training["clv_12m"] == 0).sum()
    zero_pct = target_zero / n_rows * 100

    print(f"  Training set: {n_rows:,} customers x {n_features} features + target")
    print(f"  Target (clv_12m) median: {target_median:,.2f}")
    print(f"  Target (clv_12m) mean:   {target_mean:,.2f}")
    print(f"  Zero-CLV customers:      {target_zero:,} ({zero_pct:.1f}%)")

    return training


# ---------------------------------------------------------------------------
# Step 3: Run validation tests
# ---------------------------------------------------------------------------

def run_tests(
    customer_features: pd.DataFrame,
    training_features: pd.DataFrame,
    clv_labels: pd.DataFrame,
) -> None:
    """Run assertions on feature engineering outputs."""
    print("\n[Tests] Running validation assertions...")

    # --- customer_features ---
    assert customer_features["customer_id"].dtype == np.int64, "customer_id must be int64"
    assert customer_features["customer_id"].is_unique, "Duplicate customer_ids"
    expected_cols = ["customer_id"] + FEATURE_COLUMNS
    assert list(customer_features.columns) == expected_cols, (
        f"Column mismatch: {list(customer_features.columns)}"
    )

    # No NaNs in features
    nan_counts = customer_features[FEATURE_COLUMNS].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    assert len(nan_cols) == 0, f"NaN features found: {nan_cols.to_dict()}"
    print("  [PASS] customer_features: no NaNs, correct schema")

    # Value range checks
    assert (customer_features["recency_days"] >= 0).all(), "Negative recency_days"
    assert (customer_features["frequency"] >= 1).all(), "frequency < 1"
    assert (customer_features["monetary_total"] > 0).all(), "monetary_total <= 0"
    assert (customer_features["tenure_days"] >= 0).all(), "Negative tenure_days"
    assert (customer_features["purchase_velocity"] > 0).all(), "purchase_velocity <= 0"
    assert (customer_features["unique_products"] >= 1).all(), "unique_products < 1"
    assert customer_features["uk_customer"].isin([0, 1]).all(), "uk_customer not binary"
    assert customer_features["purchased_in_q4"].isin([0, 1]).all(), "purchased_in_q4 not binary"
    assert customer_features["weekend_purchase_ratio"].between(0, 1).all(), "weekend_purchase_ratio outside [0,1]"
    assert customer_features["evening_purchase_ratio"].between(0, 1).all(), "evening_purchase_ratio outside [0,1]"
    assert customer_features["acquisition_month"].between(1, 12).all(), "acquisition_month outside [1,12]"
    assert customer_features["acquisition_quarter"].between(1, 4).all(), "acquisition_quarter outside [1,4]"
    assert customer_features["rfm_recency_score"].between(1, 5).all(), "rfm_recency_score outside [1,5]"
    assert customer_features["rfm_frequency_score"].between(1, 5).all(), "rfm_frequency_score outside [1,5]"
    assert customer_features["rfm_monetary_score"].between(1, 5).all(), "rfm_monetary_score outside [1,5]"
    assert customer_features["rfm_combined_score"].between(3, 15).all(), "rfm_combined_score outside [3,15]"
    print("  [PASS] Feature value ranges valid")

    # --- training_features ---
    assert "clv_12m" in training_features.columns, "clv_12m missing from training set"
    assert (training_features["clv_12m"] >= 0).all(), "Negative CLV in training set"
    assert training_features["clv_12m"].isna().sum() == 0, "NaN CLV in training set"
    n_train = len(training_features)
    assert n_train > 1_000, f"Training set too small: {n_train}"
    print(f"  [PASS] training_features: {n_train:,} rows, target >= 0, no NaNs")

    # --- No cold-start in training ---
    repeat_ids = set(clv_labels.loc[clv_labels["is_cold_start"] == 0, "customer_id"])
    train_ids = set(training_features["customer_id"])
    assert train_ids.issubset(repeat_ids), "Cold-start customers leaked into training set"
    print("  [PASS] No cold-start customers in training set")

    n_features = len(FEATURE_COLUMNS)
    print(f"\n  All assertions passed. ({n_features} features)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Feature Engineering — m02_clv (Customer Lifetime Value)")
    print("=" * 60)

    # Load inputs
    print("\n[Load] Reading observation_transactions.csv...")
    obs_df = pd.read_csv(OBSERVATION_CSV)
    obs_df["InvoiceDate"] = pd.to_datetime(obs_df["InvoiceDate"])
    obs_rows = len(obs_df)
    obs_custs = obs_df["customer_id"].nunique()
    print(f"  Loaded {obs_rows:,} rows, {obs_custs:,} unique customers.")

    print("[Load] Reading clv_labels.csv...")
    clv_labels = pd.read_csv(CLV_LABELS_CSV)
    label_rows = len(clv_labels)
    print(f"  Loaded {label_rows:,} customer labels.")

    ref_str = REFERENCE_DATE.strftime("%Y-%m-%d")
    print(f"  Reference date: {ref_str}")

    # Compute features
    customer_features = compute_customer_features(obs_df)

    # Save customer_features
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    customer_features.to_csv(CUSTOMER_FEATURES_CSV, index=False)
    feat_shape = customer_features.shape
    print(f"\n[Save] customer_features.csv -> {CUSTOMER_FEATURES_CSV}")
    print(f"  Shape: {feat_shape}")

    # Build and save training set
    training_features = build_training_set(customer_features, clv_labels)
    training_features.to_csv(TRAINING_FEATURES_CSV, index=False)
    train_shape = training_features.shape
    print(f"\n[Save] training_features.csv -> {TRAINING_FEATURES_CSV}")
    print(f"  Shape: {train_shape}")

    # Run tests
    run_tests(customer_features, training_features, clv_labels)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_feats = len(FEATURE_COLUMNS)
    n_cust = len(customer_features)
    n_train = len(training_features)
    median_clv = training_features["clv_12m"].median()
    mean_clv = training_features["clv_12m"].mean()
    print(f"  Feature columns ({n_feats}): {FEATURE_COLUMNS}")
    print(f"  Customer features: {n_cust:,} customers")
    print(f"  Training features: {n_train:,} customers")
    print(f"  Target median CLV: {median_clv:,.2f}")
    print(f"  Target mean CLV:   {mean_clv:,.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
