"""
Feature Engineering: Phase 3
- Computes per-customer behavioral features from transactions_clean.csv
- Computes RFM segments
- Builds event_features.csv (event-level training dataset with send_datetime)
- Builds model_features.csv (customer × 168 slots for inference/ranking)
- Saves customer_features.csv

Usage:
    python src/feature_engineering.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")
SYNTHETIC_CSV = os.path.join(BASE_DIR, "data", "synthetic", "campaign_events.csv")
CUSTOMER_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
MODEL_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "model_features.csv")
EVENT_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "event_features.csv")

# ---------------------------------------------------------------------------
# Module-level constant — industry open rate by send hour (imported by API)
# ---------------------------------------------------------------------------
INDUSTRY_OPEN_RATE_BY_HOUR = {
    0: 0.17,
    1: 0.15,
    2: 0.13,
    3: 0.12,
    4: 0.12,
    5: 0.14,
    6: 0.19,
    7: 0.24,
    8: 0.28,
    9: 0.32,
    10: 0.34,
    11: 0.33,
    12: 0.31,
    13: 0.29,
    14: 0.28,
    15: 0.27,
    16: 0.26,
    17: 0.25,
    18: 0.28,
    19: 0.27,
    20: 0.26,
    21: 0.24,
    22: 0.22,
    23: 0.19,
}

CUSTOMER_FEATURE_COLUMNS = [
    "modal_purchase_hour",
    "modal_purchase_dow",
    "purchase_hour_entropy",
    "avg_daily_txn_count",
    "recency_days",
    "frequency",
    "monetary_total",
    "tenure_days",
    "country_uk",
    "unique_products",
    "cancellation_rate",
]

SLOT_FEATURE_COLUMNS = [
    "send_hour",
    "send_dow",
    "is_weekend",
    "is_business_hours",
    "hour_delta_from_modal",
    "dow_match",
    "industry_open_rate_by_hour",
]

INTERACTION_FEATURE_COLUMNS = ["hour_x_entropy", "recency_x_frequency"]
FEATURE_COLUMNS = CUSTOMER_FEATURE_COLUMNS + SLOT_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Step 1: Compute customer behavioral features
# ---------------------------------------------------------------------------

def compute_customer_features(transactions: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    print("[Step 1] Computing customer behavioral features...")

    df = transactions.copy()
    df["_hour"] = df["InvoiceDate"].dt.hour
    df["_dow"] = df["InvoiceDate"].dt.dayofweek
    df["_date"] = df["InvoiceDate"].dt.normalize()
    df["_revenue"] = df["Quantity"] * df["Price"]

    modal_hour = (
        df.groupby("CustomerID")["_hour"].agg(lambda s: int(s.mode().iloc[0])).rename("modal_purchase_hour")
    )
    modal_dow = (
        df.groupby("CustomerID")["_dow"].agg(lambda s: int(s.mode().iloc[0])).rename("modal_purchase_dow")
    )

    def _hour_entropy(s: pd.Series) -> float:
        counts = s.value_counts().values.astype(float)
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        raw_h = float(scipy_entropy(probs, base=2))
        return raw_h / np.log2(24)

    hour_entropy = df.groupby("CustomerID")["_hour"].apply(_hour_entropy).rename("purchase_hour_entropy")

    rfm_base = df.groupby("CustomerID").agg(
        _last_date=("_date", "max"),
        _first_date=("_date", "min"),
        frequency=("_date", "nunique"),
        monetary_total=("_revenue", "sum"),
        unique_products=("StockCode", "nunique"),
    )

    rfm_base["recency_days"] = ((reference_date.normalize() - rfm_base["_last_date"]).dt.days).astype(int)
    rfm_base["tenure_days"] = ((rfm_base["_last_date"] - rfm_base["_first_date"]).dt.days).astype(int)

    txn_counts = df.groupby("CustomerID").size().rename("_total_txns")
    rfm_base = rfm_base.join(txn_counts)
    rfm_base["avg_daily_txn_count"] = rfm_base["_total_txns"] / rfm_base["tenure_days"].clip(lower=1)

    # Training/serving scope in v2 is UK-only.
    rfm_base["country_uk"] = 1
    # Cancellations removed in cleaning, so residual cancellation_rate is 0 in current scope.
    rfm_base["cancellation_rate"] = 0.0

    customer_df = (
        rfm_base.drop(columns=["_last_date", "_first_date", "_total_txns"])
        .join(modal_hour)
        .join(modal_dow)
        .join(hour_entropy)
        .reset_index()
        .rename(columns={"CustomerID": "customer_id"})
    )

    customer_df = customer_df[["customer_id", *CUSTOMER_FEATURE_COLUMNS]]
    customer_df["customer_id"] = customer_df["customer_id"].astype(np.int64)

    print(f"  Customer features shape: {customer_df.shape}")
    print(f"  Unique customers: {customer_df['customer_id'].nunique():,}")
    return customer_df


# ---------------------------------------------------------------------------
# Step 2: RFM segments
# ---------------------------------------------------------------------------

def compute_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    print("[Step 2] Computing RFM segments...")
    out = df.copy()

    r_score = pd.qcut(out["recency_days"], q=4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    f_score = pd.qcut(out["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    m_score = pd.qcut(out["monetary_total"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    out["R_score"] = r_score
    out["F_score"] = f_score
    out["M_score"] = m_score

    conditions = [
        (out["R_score"] == 4) & (out["F_score"] >= 3),
        (out["F_score"] >= 3) & (out["R_score"] >= 2),
        (out["R_score"] <= 2) & (out["F_score"] >= 2),
        (out["R_score"] == 1) & (out["F_score"] <= 2),
    ]
    choices = ["Champions", "Loyal", "At Risk", "Hibernating"]
    out["rfm_segment"] = np.select(conditions, choices, default="Other")

    out = out.drop(columns=["R_score", "F_score", "M_score"])
    print(f"  RFM segment distribution:\n{out['rfm_segment'].value_counts().to_string()}")
    return out


# ---------------------------------------------------------------------------
# Step 3: Shared slot-level feature logic
# ---------------------------------------------------------------------------

def add_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["send_hour"] = out["send_hour"].astype(np.int16)
    out["send_dow"] = out["send_dow"].astype(np.int8)
    out["is_weekend"] = (out["send_dow"] >= 5).astype(np.int8)
    out["is_business_hours"] = ((out["send_hour"] >= 9) & (out["send_hour"] <= 17)).astype(np.int8)

    delta = (out["send_hour"] - out["modal_purchase_hour"]).abs()
    out["hour_delta_from_modal"] = np.minimum(delta, 24 - delta).astype(np.int16)
    out["dow_match"] = (out["send_dow"] == out["modal_purchase_dow"]).astype(np.int8)
    out["industry_open_rate_by_hour"] = out["send_hour"].map(INDUSTRY_OPEN_RATE_BY_HOUR).astype(float)
    out["hour_x_entropy"] = out["send_hour"] * out["purchase_hour_entropy"]
    out["recency_x_frequency"] = out["recency_days"] * out["frequency"]
    return out


def build_168_grid(profile: pd.Series | dict) -> pd.DataFrame:
    """Build a 168-slot (7 days × 24 hours) feature grid from a customer profile.

    Accepts either a pd.Series or a dict with keys from CUSTOMER_FEATURE_COLUMNS.
    Returns a DataFrame with all FEATURE_COLUMNS ready for model scoring.
    """
    slots = pd.DataFrame(
        [(h, d) for d in range(7) for h in range(24)],
        columns=["send_hour", "send_dow"],
    )
    for col in CUSTOMER_FEATURE_COLUMNS:
        slots[col] = profile[col] if not isinstance(profile, dict) else profile.get(col)

    return add_slot_features(slots)


# ---------------------------------------------------------------------------
# Step 4: Build inference grid (customer × 168)
# ---------------------------------------------------------------------------

def build_model_features(customer_feats: pd.DataFrame) -> pd.DataFrame:
    print("[Step 3] Building model_features (customer × 168 slots)...")

    hours = np.arange(24, dtype=np.int16)
    dows = np.arange(7, dtype=np.int8)
    slots = pd.DataFrame([(h, d) for d in dows for h in hours], columns=["send_hour", "send_dow"])

    cust = customer_feats.copy()
    cust["_key"] = 1
    slots["_key"] = 1
    model_df = cust.merge(slots, on="_key").drop(columns="_key")

    model_df = add_slot_features(model_df)
    model_df = model_df[["customer_id", *FEATURE_COLUMNS]]
    model_df = model_df.sort_values(["customer_id", "send_dow", "send_hour"]).reset_index(drop=True)

    print(f"  Final model_features shape: {model_df.shape}")
    return model_df


# ---------------------------------------------------------------------------
# Step 5: Build event-level training set
# ---------------------------------------------------------------------------

def build_event_features(customer_feats: pd.DataFrame, campaign_events: pd.DataFrame) -> pd.DataFrame:
    print("[Step 4] Building event_features (event-level training dataset)...")

    events = campaign_events.copy()
    events["send_datetime"] = pd.to_datetime(events["send_datetime"])
    events["customer_id"] = events["customer_id"].astype(np.int64)

    event_df = events.merge(customer_feats, on="customer_id", how="inner")
    event_df = add_slot_features(event_df)

    keep_cols = ["customer_id", "send_datetime", *FEATURE_COLUMNS, "opened"]
    event_df = event_df[keep_cols]
    event_df["opened"] = event_df["opened"].astype(np.int8)
    event_df = event_df.sort_values(["send_datetime", "customer_id"]).reset_index(drop=True)

    print(f"  Final event_features shape: {event_df.shape}")
    return event_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(model_df: pd.DataFrame, customer_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    print("\n[Tests] Running assertions...")

    # customer_features
    assert len(customer_df) >= 3_500, f"customer_features too few rows: {len(customer_df):,}"
    assert customer_df["customer_id"].dtype == np.int64, "customer_id must be int64"
    assert "rfm_segment" in customer_df.columns, "rfm_segment missing"
    assert customer_df["rfm_segment"].notna().all(), "rfm_segment has nulls"

    # model_features: inference grid only (20 features, no target)
    n_customers = model_df["customer_id"].nunique()
    expected = n_customers * 168
    assert expected * 0.9 <= len(model_df) <= expected * 1.1, "model_features rows out of range"
    assert list(model_df.columns) == ["customer_id", *FEATURE_COLUMNS], "model_features schema mismatch"

    numeric_cols_model = model_df.select_dtypes(include="number").columns.tolist()
    assert model_df[numeric_cols_model].isna().sum().sum() == 0, "NaNs in model_features numeric cols"
    assert model_df["purchase_hour_entropy"].between(0, 1).all(), "purchase_hour_entropy outside [0,1]"

    # event_features: training set with timestamp + target
    assert "send_datetime" in event_df.columns, "event_features must include send_datetime"
    assert list(event_df.columns) == ["customer_id", "send_datetime", *FEATURE_COLUMNS, "opened"], "event_features schema mismatch"
    assert event_df["opened"].isin([0, 1]).all(), "opened not binary"
    assert event_df["send_datetime"].is_monotonic_increasing, "event_features must be sorted by send_datetime"

    # leakage guard: non-feature columns are only identifiers and allowed target/timestamp.
    allowed_non_features = {"customer_id", "send_datetime", "opened"}
    actual_non_features = set(event_df.columns) - set(FEATURE_COLUMNS)
    assert actual_non_features == allowed_non_features, f"Unexpected non-feature columns: {actual_non_features}"

    print("[PASS] All assertions passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Feature Engineering — Phase 3")
    print("=" * 60)

    print("\n[Load] Reading transactions_clean.csv...")
    transactions = pd.read_csv(PROCESSED_CSV)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    print(f"  Loaded {len(transactions):,} rows, {transactions['CustomerID'].nunique():,} unique customers.")

    print("[Load] Reading campaign_events.csv...")
    campaign_events = pd.read_csv(SYNTHETIC_CSV)
    campaign_events["send_datetime"] = pd.to_datetime(campaign_events["send_datetime"])
    print(f"  Loaded {len(campaign_events):,} rows, {campaign_events['customer_id'].nunique():,} unique customers.")

    reference_date = transactions["InvoiceDate"].max()
    print(f"  Reference date: {reference_date}")

    customer_feats = compute_customer_features(transactions, reference_date)
    customer_feats = compute_rfm_segments(customer_feats)

    os.makedirs(os.path.dirname(CUSTOMER_FEATURES_CSV), exist_ok=True)
    customer_feats.to_csv(CUSTOMER_FEATURES_CSV, index=False)
    print(f"\n[Save] customer_features.csv -> {CUSTOMER_FEATURES_CSV}")
    print(f"  Shape: {customer_feats.shape} (rows={len(customer_feats):,}, cols={customer_feats.shape[1]})")

    model_feats = build_model_features(customer_feats)
    model_feats.to_csv(MODEL_FEATURES_CSV, index=False)
    print(f"\n[Save] model_features.csv -> {MODEL_FEATURES_CSV}")
    print(f"  Shape: {model_feats.shape} (rows={len(model_feats):,}, cols={model_feats.shape[1]})")

    event_feats = build_event_features(customer_feats, campaign_events)
    event_feats.to_csv(EVENT_FEATURES_CSV, index=False)
    print(f"\n[Save] event_features.csv -> {EVENT_FEATURES_CSV}")
    print(f"  Shape: {event_feats.shape} (rows={len(event_feats):,}, cols={event_feats.shape[1]})")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  customer_features rows : {len(customer_feats):,}")
    print(f"  model_features rows    : {len(model_feats):,}")
    print(f"  event_features rows    : {len(event_feats):,}")
    print(f"  feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  event open rate        : {event_feats['opened'].mean():.4f}")

    run_tests(model_feats, customer_feats, event_feats)

    print("\nDone.")


if __name__ == "__main__":
    main()
