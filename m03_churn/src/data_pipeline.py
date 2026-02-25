"""
Data Pipeline: Model 3 — Churn Propensity
- Step 1: Load raw IBM Telco Customer Churn CSV
- Step 2: Clean, impute, and encode all columns
- Step 3: Save processed parquet for downstream feature engineering

Usage:
    python src/data_pipeline.py
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

RAW_CSV_DEFAULT = os.path.join(RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PARQUET_DEFAULT = os.path.join(PROCESSED_DIR, "customers_clean.parquet")

# ---------------------------------------------------------------------------
# Column groups — used for encoding and validation
# ---------------------------------------------------------------------------

# Binary Yes/No columns (encode Yes=1, No=0; gender Male=1, Female=0)
_BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# 3-category service columns: "No phone service" or "No internet service" → 0, "No" → 0, "Yes" → 1
_SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Multi-category columns for one-hot encoding (all dummies retained)
_ONEHOT_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------

def load_raw(raw_path: str | None = None) -> pd.DataFrame:
    """Load the raw IBM Telco Customer Churn CSV with schema intact.

    Parameters
    ----------
    raw_path:
        Absolute path to the raw CSV. Defaults to DATA_DIR/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with TotalCharges cast to float (nulls preserved via coerce).
    """
    path = raw_path if raw_path is not None else RAW_CSV_DEFAULT
    print(f"[Step 1] Loading raw data from: {path}")

    df = pd.read_csv(path, dtype={"customerID": str})
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # TotalCharges is stored as a string in the raw file; coerce to float, exposing 11 blanks as NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_null_tc = df["TotalCharges"].isna().sum()
    print(f"  TotalCharges nulls after coerce: {n_null_tc} (expected 11)")

    return df


# ---------------------------------------------------------------------------
# Step 2: Clean and encode
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and encoding steps to produce a model-ready DataFrame.

    Operations (in order):
    1. Impute the 11 null TotalCharges (all have tenure==0) with 0.0.
    2. Encode gender: Male=1, Female=0.
    3. Encode binary Yes/No columns → 1/0.
    4. Encode 3-category service columns: any 'No *' category → 0, 'Yes' → 1.
    5. One-hot encode InternetService, Contract, PaymentMethod (all dummies kept).
    6. Drop customerID from the feature matrix (preserved as the first column for lookup).

    Parameters
    ----------
    df:
        Raw DataFrame from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned and fully numeric DataFrame.  customerID is retained as the
        first column to allow downstream joins but must be excluded from X
        at training time.
    """
    print("[Step 2] Cleaning and encoding...")
    out = df.copy()

    # --- 2a. Impute TotalCharges nulls with 0 ---
    null_mask = out["TotalCharges"].isna()
    assert (out.loc[null_mask, "tenure"] == 0).all(), (
        "Non-zero tenure customers have null TotalCharges — check raw data."
    )
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    print(f"  Imputed {null_mask.sum()} null TotalCharges → 0.0 (all had tenure==0).")

    # --- 2b. Encode gender (Male=1, Female=0) ---
    out["gender"] = (out["gender"] == "Male").astype(int)

    # --- 2c. Encode binary Yes/No columns ---
    for col in _BINARY_COLUMNS:
        out[col] = (out[col] == "Yes").astype(int)
    print(f"  Encoded binary columns: gender + {_BINARY_COLUMNS}")

    # --- 2d. Encode 3-category service columns ---
    # "No phone service", "No internet service", and "No" all map to 0; "Yes" maps to 1
    for col in _SERVICE_COLUMNS:
        out[col] = (out[col] == "Yes").astype(int)
    print(f"  Encoded 3-category service columns (No/No *service→0, Yes→1): {_SERVICE_COLUMNS}")

    # --- 2e. One-hot encode multi-category columns ---
    out = pd.get_dummies(out, columns=_ONEHOT_COLUMNS, drop_first=False)
    # Ensure dummy columns are int (get_dummies returns bool in newer pandas)
    new_dummy_cols = [c for c in out.columns if any(c.startswith(f"{base}_") for base in _ONEHOT_COLUMNS)]
    out[new_dummy_cols] = out[new_dummy_cols].astype(int)
    print(f"  One-hot encoded: {_ONEHOT_COLUMNS} → {len(new_dummy_cols)} dummy columns.")

    # --- 2f. Retain customerID as first column, remove nothing else ---
    # customerID must be excluded from the feature matrix at training time,
    # but is retained here so downstream scripts can build lookup tables.
    cols = out.columns.tolist()
    # Move customerID to front for clarity
    cols = ["customerID"] + [c for c in cols if c != "customerID"]
    out = out[cols]

    print(f"  Clean shape: {out.shape}")
    return out


# ---------------------------------------------------------------------------
# Step 3: Save processed parquet
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Persist the cleaned DataFrame to parquet with deterministic ordering.

    Parameters
    ----------
    df:
        Cleaned DataFrame from clean().
    output_path:
        Destination path. Defaults to PROCESSED_DIR/customers_clean.parquet.
    """
    path = output_path if output_path is not None else PROCESSED_PARQUET_DEFAULT
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Deterministic sort so parquet is reproducible across runs
    out = df.sort_values("customerID").reset_index(drop=True)

    out.to_parquet(path, index=False, engine="pyarrow")
    print(f"[Step 3] Saved processed data to: {path}  ({len(out):,} rows)")


# ---------------------------------------------------------------------------
# Tests / Assertions
# ---------------------------------------------------------------------------

def run_tests(df: pd.DataFrame) -> None:
    """Assert schema contracts on the cleaned DataFrame.

    Parameters
    ----------
    df:
        The cleaned DataFrame returned by clean().

    Raises
    ------
    AssertionError
        On any contract violation.
    """
    print("[Tests] Running assertions on cleaned data...")

    # Row count: 7,043 rows in raw; all 11 nulls are imputed (not dropped), so output is 7,043
    assert len(df) == 7_043, f"Expected 7,043 rows, got {len(df):,}"

    # Churn is binary integer
    assert pd.api.types.is_integer_dtype(df["Churn"]), "Churn column must be integer dtype"
    assert set(df["Churn"].unique()).issubset({0, 1}), "Churn must contain only 0 and 1"

    # No nulls anywhere
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    assert len(columns_with_nulls) == 0, (
        f"Null values found in columns: {columns_with_nulls.to_dict()}"
    )

    # tenure range 0–72
    assert df["tenure"].min() >= 0, f"tenure minimum {df['tenure'].min()} is below 0"
    assert df["tenure"].max() <= 72, f"tenure maximum {df['tenure'].max()} exceeds 72"

    # MonthlyCharges must be strictly positive
    assert (df["MonthlyCharges"] > 0).all(), "MonthlyCharges contains non-positive values"

    # gender, binary columns, and service columns must be in {0, 1}
    binary_check_cols = ["gender"] + _BINARY_COLUMNS + _SERVICE_COLUMNS
    for col in binary_check_cols:
        if col in df.columns:
            uniq = set(df[col].unique())
            assert uniq.issubset({0, 1}), f"Column '{col}' has values outside {{0, 1}}: {uniq}"

    # One-hot columns must exist
    expected_dummies = [
        "InternetService_DSL",
        "InternetService_Fiber optic",
        "InternetService_No",
        "Contract_Month-to-month",
        "Contract_One year",
        "Contract_Two year",
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ]
    for col in expected_dummies:
        assert col in df.columns, f"Expected one-hot column missing: '{col}'"

    # customerID is present for lookup
    assert "customerID" in df.columns, "customerID column is missing from cleaned output"

    print("All tests passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Data Pipeline — Model 3: Churn Propensity")
    print("=" * 60)

    raw_df = load_raw()

    clean_df = clean(raw_df)

    save_processed(clean_df)

    print()
    run_tests(clean_df)

    churn_rate = clean_df["Churn"].mean()
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Clean rows:          {len(clean_df):,}")
    print(f"  Clean columns:       {clean_df.shape[1]}")
    print(f"  Churn rate:          {churn_rate:.1%}  ({clean_df['Churn'].sum():,} churners)")
    print(f"  Retained rate:       {1 - churn_rate:.1%}  ({(clean_df['Churn'] == 0).sum():,} retained)")
    print(f"  Output:              {PROCESSED_PARQUET_DEFAULT}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
