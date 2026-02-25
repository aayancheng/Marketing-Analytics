"""
Feature Engineering: Model 3 — Churn Propensity
- Loads cleaned customer data (customers_clean.parquet)
- Adds 7 derived features to the cleaned dataset
- Computes risk segments from churn probabilities
- Supports what-if single-row scoring for the retention simulator
- Saves customer_features.parquet for model training

Usage:
    python src/feature_engineering.py
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
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "customers_clean.parquet")
FEATURES_PARQUET = os.path.join(PROCESSED_DIR, "customer_features.parquet")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Raw encoded columns (after cleaning)
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    # One-hot encoded (InternetService, Contract, PaymentMethod)
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
    # Service columns
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    # Engineered features
    "has_family",
    "num_services",
    "monthly_per_tenure",
    "total_charges_gap",
    "is_month_to_month",
    "is_fiber_optic",
    "is_electronic_check",
]

RISK_SEGMENTS = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]

# Sensible defaults for what-if scoring: median/mode values from the training population
# (representative of a typical mid-tenure, DSL customer on month-to-month with moderate charges)
_WHATIF_DEFAULTS: dict[str, float] = {
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 29,           # approx median tenure (~29 months)
    "PhoneService": 1,
    "MultipleLines": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 64.76,  # dataset mean
    "TotalCharges": 1397.0,   # approx median
    "InternetService_DSL": 0,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 1,
    "Contract_Month-to-month": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaymentMethod_Bank transfer (automatic)": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    # Engineered features — will be recomputed from raw columns if not supplied
    "has_family": 0,
    "num_services": 1,
    "monthly_per_tenure": 64.76 / 30,  # MonthlyCharges / (tenure+1)
    "total_charges_gap": 0.0,
    "is_month_to_month": 1,
    "is_fiber_optic": 0,
    "is_electronic_check": 1,
}


# ---------------------------------------------------------------------------
# Step 1: Compute derived customer features
# ---------------------------------------------------------------------------

def compute_customer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Add 7 engineered features to the cleaned customer DataFrame.

    Engineered features:
    - has_family          : 1 if Partner==1 OR Dependents==1, else 0
    - num_services        : sum of 8 binary service flags (0–8)
    - monthly_per_tenure  : MonthlyCharges / (tenure + 1)   [avoids divide-by-zero]
    - total_charges_gap   : MonthlyCharges * tenure - TotalCharges
    - is_month_to_month   : alias for Contract_Month-to-month dummy
    - is_fiber_optic      : alias for InternetService_Fiber optic dummy
    - is_electronic_check : alias for PaymentMethod_Electronic check dummy

    Parameters
    ----------
    df_clean:
        Cleaned DataFrame from data_pipeline.clean() or loaded from
        customers_clean.parquet.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all FEATURE_COLUMNS present.  customerID and
        Churn (target) columns are preserved but not included in FEATURE_COLUMNS.
    """
    print("[Step 1] Computing engineered features...")
    out = df_clean.copy()

    # has_family: partner or dependents present
    out["has_family"] = ((out["Partner"] == 1) | (out["Dependents"] == 1)).astype(int)

    # num_services: count of active service flags (already 0/1 after cleaning)
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["num_services"] = out[service_cols].sum(axis=1).astype(int)

    # monthly_per_tenure: normalise monthly charge by tenure length (+1 avoids div-by-zero)
    out["monthly_per_tenure"] = out["MonthlyCharges"] / (out["tenure"] + 1)

    # total_charges_gap: expected cumulative charges minus actual — flags anomalies / discounts
    out["total_charges_gap"] = out["MonthlyCharges"] * out["tenure"] - out["TotalCharges"]

    # Convenience aliases for the three highest-churn-risk categorical indicators
    out["is_month_to_month"] = out["Contract_Month-to-month"].astype(int)
    out["is_fiber_optic"] = out["InternetService_Fiber optic"].astype(int)
    out["is_electronic_check"] = out["PaymentMethod_Electronic check"].astype(int)

    # Verify all FEATURE_COLUMNS are now present
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing FEATURE_COLUMNS after engineering: {missing}")

    print(f"  Feature-engineered shape: {out.shape}")
    print(f"  has_family positive rate: {out['has_family'].mean():.1%}")
    print(f"  num_services mean:        {out['num_services'].mean():.2f}")
    print(f"  monthly_per_tenure mean:  {out['monthly_per_tenure'].mean():.2f}")

    return out


# ---------------------------------------------------------------------------
# Step 2: Compute risk segments from model probabilities
# ---------------------------------------------------------------------------

def compute_risk_segments(df: pd.DataFrame, churn_proba_col: str = "churn_probability") -> pd.DataFrame:
    """Assign a four-tier risk segment label based on churn probability.

    Segment thresholds:
    - > 0.60  → "High Risk"
    - 0.40–0.60 → "Medium-High Risk"
    - 0.20–0.40 → "Medium-Low Risk"
    - < 0.20  → "Low Risk"

    Parameters
    ----------
    df:
        DataFrame that already contains a churn probability column.
    churn_proba_col:
        Name of the column holding predicted churn probability (float, 0–1).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional 'risk_segment' column (string).
    """
    if churn_proba_col not in df.columns:
        raise KeyError(
            f"Column '{churn_proba_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    proba = df[churn_proba_col]

    conditions = [
        proba > 0.60,
        (proba > 0.40) & (proba <= 0.60),
        (proba > 0.20) & (proba <= 0.40),
        proba <= 0.20,
    ]
    choices = RISK_SEGMENTS  # ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]

    out = df.copy()
    out["risk_segment"] = np.select(conditions, choices, default="Low Risk")

    seg_counts = out["risk_segment"].value_counts()
    print("[Step 2] Risk segment distribution:")
    for seg in RISK_SEGMENTS:
        count = seg_counts.get(seg, 0)
        pct = count / len(out) if len(out) > 0 else 0.0
        print(f"  {seg:<22}: {count:>5,}  ({pct:.1%})")

    return out


# ---------------------------------------------------------------------------
# Step 3: Build a single-row what-if feature vector
# ---------------------------------------------------------------------------

def build_whatif_features(params: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with all FEATURE_COLUMNS for ad-hoc scoring.

    The caller supplies a partial dictionary of feature overrides; any missing
    features are filled from _WHATIF_DEFAULTS (population median/mode values).
    Engineered features (has_family, num_services, monthly_per_tenure,
    total_charges_gap, is_month_to_month, is_fiber_optic, is_electronic_check)
    are always recomputed from the raw/encoded columns so they remain consistent
    with any overrides the caller provides.

    Parameters
    ----------
    params:
        Dict mapping feature names to scalar values. Only fields in
        FEATURE_COLUMNS are used; extra keys are silently ignored.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns == FEATURE_COLUMNS, ready for model.predict().
    """
    # Start from defaults and apply caller overrides
    merged = {**_WHATIF_DEFAULTS, **{k: v for k, v in params.items() if k in _WHATIF_DEFAULTS}}

    # Recompute engineered features from raw inputs to maintain internal consistency
    merged["has_family"] = int((merged["Partner"] == 1) or (merged["Dependents"] == 1))

    service_keys = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    merged["num_services"] = int(sum(merged.get(k, 0) for k in service_keys))

    merged["monthly_per_tenure"] = merged["MonthlyCharges"] / (merged["tenure"] + 1)
    merged["total_charges_gap"] = (
        merged["MonthlyCharges"] * merged["tenure"] - merged["TotalCharges"]
    )
    merged["is_month_to_month"] = int(merged["Contract_Month-to-month"])
    merged["is_fiber_optic"] = int(merged["InternetService_Fiber optic"])
    merged["is_electronic_check"] = int(merged["PaymentMethod_Electronic check"])

    row = pd.DataFrame([{col: merged[col] for col in FEATURE_COLUMNS}])
    return row


# ---------------------------------------------------------------------------
# Step 4: Save feature DataFrame
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Save the feature-engineered DataFrame to parquet with deterministic ordering.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame (must include customerID for sorting).
    output_path:
        Destination path. Defaults to PROCESSED_DIR/customer_features.parquet.
    """
    path = output_path if output_path is not None else FEATURES_PARQUET
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out = df.copy()
    if "customerID" in out.columns:
        out = out.sort_values("customerID").reset_index(drop=True)

    out.to_parquet(path, index=False, engine="pyarrow")
    print(f"[Step 4] Saved customer features to: {path}  ({len(out):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Feature Engineering — Model 3: Churn Propensity")
    print("=" * 60)

    print(f"\n[Load] Reading cleaned data from: {CLEAN_PARQUET}")
    df_clean = pd.read_parquet(CLEAN_PARQUET, engine="pyarrow")
    print(f"  Loaded {len(df_clean):,} rows, {df_clean.shape[1]} columns.")

    df_features = compute_customer_features(df_clean)

    save_features(df_features)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Input rows:           {len(df_clean):,}")
    print(f"  Output rows:          {len(df_features):,}")
    print(f"  Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  Output:               {FEATURES_PARQUET}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
