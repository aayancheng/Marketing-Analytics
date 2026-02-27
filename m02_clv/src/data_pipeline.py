"""
Data Pipeline: m02_clv
- Step 1: Load raw UCI Online Retail II transactions
- Step 2: Clean transactions (nulls, cancellations, non-products, dedup)
- Step 3: Temporal split into observation and prediction windows
- Step 4: Compute CLV labels per customer
- Step 5: Save processed outputs
- Step 6: Run validation assertions

Usage:
    cd m02_clv
    /Users/aayan/MarketingAnalytics/.venv/bin/python src/data_pipeline.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "..", "shared", "data", "raw", "online_retail_ii.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TRANSACTIONS_CLEAN_CSV = os.path.join(PROCESSED_DIR, "transactions_clean.csv")
OBSERVATION_CSV = os.path.join(PROCESSED_DIR, "observation_transactions.csv")
CLV_LABELS_CSV = os.path.join(PROCESSED_DIR, "clv_labels.csv")

# Non-product StockCodes to exclude
NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "AMAZONFEE", "S", "PADS", "C2",
}

# Temporal boundaries
OBSERVATION_START = pd.Timestamp("2009-12-01")
OBSERVATION_END = pd.Timestamp("2010-11-30")
PREDICTION_START = pd.Timestamp("2010-12-01")
PREDICTION_END = pd.Timestamp("2011-12-09")


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------

def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """Load the raw UCI Online Retail II CSV."""
    print("[Step 1] Loading raw transactions...")
    df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str})
    row_count = len(df)
    col_count = df.shape[1]
    print(f"  Loaded {row_count:,} rows, {col_count} columns.")
    return df


# ---------------------------------------------------------------------------
# Step 2: Clean transactions
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transactions: nulls, cancellations, non-products, dedup."""
    print("\n[Step 2] Cleaning transactions...")

    # Handle column name variants
    if "UnitPrice" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"UnitPrice": "Price"})
        print("  Renamed 'UnitPrice' to 'Price'.")

    # Standardize Customer ID column name
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "customer_id"})
    elif "CustomerID" in df.columns:
        df = df.rename(columns={"CustomerID": "customer_id"})

    # Drop null customer_id
    before = len(df)
    df = df.dropna(subset=["customer_id"])
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} rows with null customer_id. Remaining: {remaining:,}")

    # Filter out cancellations (Invoice starts with 'C')
    before = len(df)
    cancel_mask = df["Invoice"].astype(str).str.startswith("C")
    df = df[~cancel_mask].copy()
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} cancellation rows. Remaining: {remaining:,}")

    # Remove non-product StockCodes
    before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)].copy()
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} non-product StockCode rows. Remaining: {remaining:,}")

    # Drop rows with Price <= 0
    before = len(df)
    df = df[df["Price"] > 0].copy()
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} rows with Price <= 0. Remaining: {remaining:,}")

    # Drop rows with Quantity <= 0
    before = len(df)
    df = df[df["Quantity"] > 0].copy()
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} rows with Quantity <= 0. Remaining: {remaining:,}")

    # Deduplicate on (Invoice, StockCode, customer_id)
    before = len(df)
    df = df.drop_duplicates(subset=["Invoice", "StockCode", "customer_id"])
    dropped = before - len(df)
    remaining = len(df)
    print(f"  Dropped {dropped:,} duplicate rows. Remaining: {remaining:,}")

    # Add line_total
    df["line_total"] = df["Quantity"] * df["Price"]

    # Convert customer_id to int
    float_ids = df["customer_id"].astype(float)
    if (float_ids % 1 != 0).any():
        raise ValueError("Non-integer customer_id values detected after nulls dropped.")
    df["customer_id"] = float_ids.astype(np.int64)

    # Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Deterministic sort
    df = df.sort_values(
        ["customer_id", "InvoiceDate", "Invoice", "StockCode"]
    ).reset_index(drop=True)

    final_shape = df.shape
    print(f"  Final cleaned shape: {final_shape}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned transactions into observation and prediction windows.

    Observation: Dec 1, 2009 - Nov 30, 2010 (for building features)
    Prediction:  Dec 1, 2010 - Dec 9, 2011  (for computing CLV labels)
    """
    print("\n[Step 3] Temporal split...")

    obs_mask = (df["InvoiceDate"] >= OBSERVATION_START) & (
        df["InvoiceDate"] <= OBSERVATION_END + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    pred_mask = (df["InvoiceDate"] >= PREDICTION_START) & (
        df["InvoiceDate"] <= PREDICTION_END + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )

    observation_df = df[obs_mask].copy().reset_index(drop=True)
    prediction_df = df[pred_mask].copy().reset_index(drop=True)

    obs_rows = len(observation_df)
    obs_custs = observation_df["customer_id"].nunique()
    pred_rows = len(prediction_df)
    pred_custs = prediction_df["customer_id"].nunique()

    obs_min = observation_df["InvoiceDate"].min()
    obs_max = observation_df["InvoiceDate"].max()
    pred_min = prediction_df["InvoiceDate"].min()
    pred_max = prediction_df["InvoiceDate"].max()

    print(f"  Observation window: {OBSERVATION_START.date()} to {OBSERVATION_END.date()}")
    print(f"    Rows: {obs_rows:,}, Customers: {obs_custs:,}")
    print(f"    Actual date range: {obs_min} to {obs_max}")
    print(f"  Prediction window:  {PREDICTION_START.date()} to {PREDICTION_END.date()}")
    print(f"    Rows: {pred_rows:,}, Customers: {pred_custs:,}")
    print(f"    Actual date range: {pred_min} to {pred_max}")

    return observation_df, prediction_df


# ---------------------------------------------------------------------------
# Step 4: Compute CLV labels
# ---------------------------------------------------------------------------

def compute_clv_labels(
    observation_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute CLV labels for each customer in the observation window.

    - clv_12m: total spend in the prediction window (0 if no purchases)
    - Customers with >= 2 observation purchases are the primary training set
    - Cold-start customers (< 2 purchases) are flagged separately
    """
    print("\n[Step 4] Computing CLV labels...")

    # Count observation purchases per customer
    obs_purchase_counts = (
        observation_df.groupby("customer_id")["Invoice"]
        .nunique()
        .rename("n_obs_purchases")
    )

    # All customers in observation window
    all_obs_customers = obs_purchase_counts.index

    # Prediction-window revenue per customer
    pred_revenue = (
        prediction_df.groupby("customer_id")["line_total"]
        .sum()
        .rename("clv_12m")
    )

    # Build labels dataframe
    labels = pd.DataFrame({"customer_id": all_obs_customers})
    labels = labels.merge(
        obs_purchase_counts.reset_index(), on="customer_id", how="left"
    )
    labels = labels.merge(
        pred_revenue.reset_index(), on="customer_id", how="left"
    )

    # Fill NaN clv_12m with 0 (customers who didn't purchase in prediction window)
    labels["clv_12m"] = labels["clv_12m"].fillna(0.0)

    # Flag cold-start customers
    labels["is_cold_start"] = (labels["n_obs_purchases"] < 2).astype(int)

    # Sort deterministically
    labels = labels.sort_values("customer_id").reset_index(drop=True)

    total_custs = len(labels)
    repeat_custs = (labels["is_cold_start"] == 0).sum()
    cold_custs = (labels["is_cold_start"] == 1).sum()
    median_clv = labels.loc[labels["is_cold_start"] == 0, "clv_12m"].median()
    mean_clv = labels.loc[labels["is_cold_start"] == 0, "clv_12m"].mean()
    zero_clv = (labels["clv_12m"] == 0).sum()

    print(f"  Total customers in observation window: {total_custs:,}")
    print(f"  Repeat customers (>= 2 purchases):     {repeat_custs:,}")
    print(f"  Cold-start customers (< 2 purchases):  {cold_custs:,}")
    print(f"  Median CLV (repeat only):              {median_clv:,.2f}")
    print(f"  Mean CLV (repeat only):                {mean_clv:,.2f}")
    print(f"  Customers with zero prediction spend:  {zero_clv:,}")

    return labels


# ---------------------------------------------------------------------------
# Step 5: Save processed outputs
# ---------------------------------------------------------------------------

def save_processed(
    clean_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    clv_labels: pd.DataFrame,
) -> None:
    """Save all processed dataframes to data/processed/."""
    print("\n[Step 5] Saving processed outputs...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    clean_df.to_csv(TRANSACTIONS_CLEAN_CSV, index=False)
    print(f"  Saved: {TRANSACTIONS_CLEAN_CSV}")
    clean_shape = clean_df.shape
    print(f"    Shape: {clean_shape}")

    observation_df.to_csv(OBSERVATION_CSV, index=False)
    print(f"  Saved: {OBSERVATION_CSV}")
    obs_shape = observation_df.shape
    print(f"    Shape: {obs_shape}")

    clv_labels.to_csv(CLV_LABELS_CSV, index=False)
    print(f"  Saved: {CLV_LABELS_CSV}")
    labels_shape = clv_labels.shape
    print(f"    Shape: {labels_shape}")


# ---------------------------------------------------------------------------
# Step 6: Validation assertions
# ---------------------------------------------------------------------------

def run_tests(
    clean_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    clv_labels: pd.DataFrame,
) -> None:
    """Run assertions on pipeline outputs."""
    print("\n[Tests] Running validation assertions...")

    # --- transactions_clean ---
    assert clean_df["customer_id"].dtype == np.int64, "customer_id must be int64"
    assert clean_df["customer_id"].isna().sum() == 0, "customer_id has nulls"
    assert (~clean_df["StockCode"].isin(NON_PRODUCT_CODES)).all(), "Non-product StockCodes present"
    assert (~clean_df["Invoice"].astype(str).str.startswith("C")).all(), "Cancellations present"
    assert (clean_df["Price"] > 0).all(), "Rows with Price <= 0 present"
    assert (clean_df["Quantity"] > 0).all(), "Rows with Quantity <= 0 present"
    assert (clean_df["line_total"] > 0).all(), "Rows with line_total <= 0 present"
    clean_rows = len(clean_df)
    print(f"  [PASS] transactions_clean: {clean_rows:,} rows, no nulls, no cancellations")

    # --- temporal boundaries ---
    assert observation_df["InvoiceDate"].min() >= OBSERVATION_START, "Observation starts too early"
    obs_end_limit = OBSERVATION_END + pd.Timedelta(days=1)
    assert observation_df["InvoiceDate"].max() < obs_end_limit, "Observation extends past boundary"
    assert prediction_df["InvoiceDate"].min() >= PREDICTION_START, "Prediction starts too early"
    pred_end_limit = PREDICTION_END + pd.Timedelta(days=1)
    assert prediction_df["InvoiceDate"].max() < pred_end_limit, "Prediction extends past boundary"
    print("  [PASS] Temporal boundaries correct")

    # --- no overlap ---
    obs_dates = set(observation_df["InvoiceDate"].dt.date.unique())
    pred_dates = set(prediction_df["InvoiceDate"].dt.date.unique())
    overlap = obs_dates & pred_dates
    assert len(overlap) == 0, f"Temporal overlap found on dates: {overlap}"
    print("  [PASS] No temporal overlap between observation and prediction")

    # --- CLV labels ---
    assert clv_labels["customer_id"].is_unique, "Duplicate customer_ids in labels"
    assert (clv_labels["clv_12m"] >= 0).all(), "Negative CLV values found"
    assert clv_labels["clv_12m"].isna().sum() == 0, "clv_12m has nulls"
    assert clv_labels["n_obs_purchases"].isna().sum() == 0, "n_obs_purchases has nulls"
    assert set(clv_labels["is_cold_start"].unique()).issubset({0, 1}), "is_cold_start not binary"
    labels_rows = len(clv_labels)
    repeat_n = (clv_labels["is_cold_start"] == 0).sum()
    cold_n = (clv_labels["is_cold_start"] == 1).sum()
    print(f"  [PASS] CLV labels: {labels_rows:,} customers ({repeat_n:,} repeat, {cold_n:,} cold-start)")

    # --- row count sanity ---
    assert len(observation_df) > 100_000, "Observation window too small"
    assert len(prediction_df) > 100_000, "Prediction window too small"
    assert len(clv_labels) > 3_000, "Too few customers in labels"
    print("  [PASS] Row count sanity checks passed")

    # --- cold-start flag consistency ---
    cold_mask = clv_labels["is_cold_start"] == 1
    assert (clv_labels.loc[cold_mask, "n_obs_purchases"] < 2).all(), "Cold-start flag inconsistent"
    repeat_mask = clv_labels["is_cold_start"] == 0
    assert (clv_labels.loc[repeat_mask, "n_obs_purchases"] >= 2).all(), "Repeat flag inconsistent"
    print("  [PASS] Cold-start flags consistent")

    print("\n  All assertions passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Data Pipeline — m02_clv (Customer Lifetime Value)")
    print("=" * 60)

    raw_df = load_raw()
    clean_df = clean(raw_df)
    observation_df, prediction_df = temporal_split(clean_df)
    clv_labels = compute_clv_labels(observation_df, prediction_df)
    save_processed(clean_df, observation_df, clv_labels)
    run_tests(clean_df, observation_df, prediction_df, clv_labels)

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    clean_rows = len(clean_df)
    clean_custs = clean_df["customer_id"].nunique()
    obs_rows = len(observation_df)
    obs_custs = observation_df["customer_id"].nunique()
    pred_rows = len(prediction_df)
    pred_custs = prediction_df["customer_id"].nunique()
    label_rows = len(clv_labels)
    repeat_count = (clv_labels["is_cold_start"] == 0).sum()
    cold_count = (clv_labels["is_cold_start"] == 1).sum()

    print(f"  Transactions (clean):   {clean_rows:,} rows, {clean_custs:,} customers")
    print(f"  Observation window:     {obs_rows:,} rows, {obs_custs:,} customers")
    print(f"  Prediction window:      {pred_rows:,} rows, {pred_custs:,} customers")
    print(f"  CLV labels:             {label_rows:,} customers")
    print(f"    Repeat (>= 2 purch):  {repeat_count:,}")
    print(f"    Cold-start (< 2):     {cold_count:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
