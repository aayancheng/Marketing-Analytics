"""
Feature Engineering: Phase 3
- Computes 11 customer behavioral features from transactions_clean.csv
- Computes RFM segments
- Builds model_features.csv (one row per customer × send slot, 168 slots)
- Saves customer_features.csv (one row per customer)

Usage:
    python src/feature_engineering.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV       = os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")
SYNTHETIC_CSV       = os.path.join(BASE_DIR, "data", "synthetic", "campaign_events.csv")
CUSTOMER_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
MODEL_FEATURES_CSV  = os.path.join(BASE_DIR, "data", "processed", "model_features.csv")

# ---------------------------------------------------------------------------
# Module-level constant — industry open rate by send hour (imported by API)
# ---------------------------------------------------------------------------
INDUSTRY_OPEN_RATE_BY_HOUR = {
     0: 0.17,  1: 0.15,  2: 0.13,  3: 0.12,  4: 0.12,  5: 0.14,
     6: 0.19,  7: 0.24,  8: 0.28,  9: 0.32, 10: 0.34, 11: 0.33,
    12: 0.31, 13: 0.29, 14: 0.28, 15: 0.27, 16: 0.26, 17: 0.25,
    18: 0.28, 19: 0.27, 20: 0.26, 21: 0.24, 22: 0.22, 23: 0.19,
}


# ---------------------------------------------------------------------------
# Step 1: Compute customer behavioral features
# ---------------------------------------------------------------------------

def compute_customer_features(transactions: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute 11 per-customer behavioral features from the cleaned transactions.

    Returns a DataFrame with one row per CustomerID and columns:
        customer_id, modal_purchase_hour, modal_purchase_dow,
        purchase_hour_entropy, avg_daily_txn_count, recency_days,
        frequency, monetary_total, tenure_days, country_uk,
        unique_products, cancellation_rate
    """
    print("[Step 1] Computing customer behavioral features...")

    df = transactions.copy()

    # Derive hour and day-of-week from InvoiceDate
    df["_hour"] = df["InvoiceDate"].dt.hour
    df["_dow"]  = df["InvoiceDate"].dt.dayofweek   # 0=Mon, 6=Sun
    df["_date"] = df["InvoiceDate"].dt.normalize()  # date-only (midnight timestamp)
    df["_revenue"] = df["Quantity"] * df["Price"]

    # ---- modal_purchase_hour ------------------------------------------------
    modal_hour = (
        df.groupby("CustomerID")["_hour"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .rename("modal_purchase_hour")
    )

    # ---- modal_purchase_dow -------------------------------------------------
    modal_dow = (
        df.groupby("CustomerID")["_dow"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .rename("modal_purchase_dow")
    )

    # ---- purchase_hour_entropy ----------------------------------------------
    # Shannon entropy of per-hour purchase counts, normalised by log2(24)
    def _hour_entropy(s: pd.Series) -> float:
        counts = s.value_counts().values.astype(float)
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        # scipy entropy returns natural-log entropy; divide by log(24) for base-24,
        # then convert to base-2 by multiplying by log2(e) / log(e) — simpler:
        # H = -sum(p * log2(p))
        # normalised = H / log2(24)
        raw_h = float(scipy_entropy(probs, base=2))
        return raw_h / np.log2(24)

    hour_entropy = (
        df.groupby("CustomerID")["_hour"]
        .apply(_hour_entropy)
        .rename("purchase_hour_entropy")
    )

    # ---- RFM base features --------------------------------------------------
    rfm_base = df.groupby("CustomerID").agg(
        _last_date  = ("_date",   "max"),
        _first_date = ("_date",   "min"),
        frequency   = ("_date",   "nunique"),     # distinct invoice dates
        monetary_total = ("_revenue", "sum"),
        unique_products = ("StockCode", "nunique"),
    )

    rfm_base["recency_days"] = (
        (reference_date.normalize() - rfm_base["_last_date"])
        .dt.days
        .astype(int)
    )
    rfm_base["tenure_days"] = (
        (rfm_base["_last_date"] - rfm_base["_first_date"])
        .dt.days
        .astype(int)
    )

    # ---- avg_daily_txn_count ------------------------------------------------
    # Total transaction rows / tenure_days (use 1 if tenure_days == 0)
    txn_counts = df.groupby("CustomerID").size().rename("_total_txns")
    rfm_base = rfm_base.join(txn_counts)
    rfm_base["avg_daily_txn_count"] = rfm_base["_total_txns"] / rfm_base["tenure_days"].clip(lower=1)

    # ---- country_uk ---------------------------------------------------------
    # All customers in cleaned dataset are UK (pipeline filtered), so always 1
    rfm_base["country_uk"] = 1

    # ---- cancellation_rate --------------------------------------------------
    # Cancellations were removed during pipeline cleaning; set to 0.0 for all
    rfm_base["cancellation_rate"] = 0.0

    # ---- Assemble -----------------------------------------------------------
    customer_df = (
        rfm_base
        .drop(columns=["_last_date", "_first_date", "_total_txns"])
        .join(modal_hour)
        .join(modal_dow)
        .join(hour_entropy)
        .reset_index()
        .rename(columns={"CustomerID": "customer_id"})
    )

    # Reorder columns
    ordered_cols = [
        "customer_id",
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
    customer_df = customer_df[ordered_cols]

    print(f"  Customer features shape: {customer_df.shape}")
    print(f"  Unique customers: {customer_df['customer_id'].nunique():,}")
    return customer_df


# ---------------------------------------------------------------------------
# Step 2: RFM segment computation
# ---------------------------------------------------------------------------

def compute_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'rfm_segment' column to the customer features DataFrame.

    Segments (first match wins):
        Champions:   R_score == 4 AND F_score >= 3
        Loyal:       F_score >= 3 AND R_score >= 2
        At Risk:     R_score <= 2 AND F_score >= 2
        Hibernating: R_score == 1 AND F_score <= 2
        Other:       all remaining
    """
    print("[Step 2] Computing RFM segments...")

    df = df.copy()

    # R_score: smallest recency_days (most recent) -> label 4
    r_score = pd.qcut(
        df["recency_days"], q=4, labels=[4, 3, 2, 1], duplicates="drop"
    ).astype(int)

    # F_score: rank first to handle ties, then qcut
    f_score = pd.qcut(
        df["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop"
    ).astype(int)

    # M_score: rank first to handle ties, then qcut
    m_score = pd.qcut(
        df["monetary_total"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop"
    ).astype(int)

    df["R_score"] = r_score
    df["F_score"] = f_score
    df["M_score"] = m_score

    # Segment assignment (first match wins)
    conditions = [
        (df["R_score"] == 4) & (df["F_score"] >= 3),
        (df["F_score"] >= 3) & (df["R_score"] >= 2),
        (df["R_score"] <= 2) & (df["F_score"] >= 2),
        (df["R_score"] == 1) & (df["F_score"] <= 2),
    ]
    choices = ["Champions", "Loyal", "At Risk", "Hibernating"]
    df["rfm_segment"] = np.select(conditions, choices, default="Other")

    # Drop helper score columns before saving
    df = df.drop(columns=["R_score", "F_score", "M_score"])

    segment_counts = df["rfm_segment"].value_counts()
    print(f"  RFM segment distribution:\n{segment_counts.to_string()}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Build model features (customer × 168 send slots)
# ---------------------------------------------------------------------------

def build_model_features(
    customer_feats: pd.DataFrame,
    campaign_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-join customer features with all 168 send slots (7 DOW × 24 hours),
    compute slot-level and interaction features, then join the `opened` target
    from campaign_events.

    Returns a DataFrame with 22 columns:
        customer_id  (identifier)
        11 customer features
        7 slot features
        2 interaction features
        opened  (target)
    """
    print("[Step 3] Building model features (customer × 168 send slots)...")

    # ---- Build the 168 send slots -------------------------------------------
    hours = np.arange(24, dtype=np.int32)
    dows  = np.arange(7,  dtype=np.int32)
    # Create all combinations via a cross join
    slots = pd.DataFrame(
        [(h, d) for d in dows for h in hours],
        columns=["send_hour", "send_dow"],
    )
    slots["send_hour"] = slots["send_hour"].astype(np.int32)
    slots["send_dow"]  = slots["send_dow"].astype(np.int32)

    # Slot-level features (no customer dependency yet)
    slots["is_weekend"]         = (slots["send_dow"] >= 5).astype(np.int32)
    slots["is_business_hours"]  = (
        (slots["send_hour"] >= 9) & (slots["send_hour"] <= 17)
    ).astype(np.int32)
    slots["industry_open_rate_by_hour"] = slots["send_hour"].map(INDUSTRY_OPEN_RATE_BY_HOUR)

    # ---- Cross join: customers × slots --------------------------------------
    # Add a temporary key column for a many-to-many merge
    cust = customer_feats.copy()
    cust["_key"] = 1
    slots["_key"] = 1
    model_df = cust.merge(slots, on="_key").drop(columns="_key")

    print(f"  Cross-joined shape: {model_df.shape}  ({model_df['customer_id'].nunique():,} customers × {len(slots)} slots)")

    # ---- Customer-slot features ---------------------------------------------
    # hour_delta_from_modal: circular distance, wrap at 24
    delta = (model_df["send_hour"] - model_df["modal_purchase_hour"]).abs()
    model_df["hour_delta_from_modal"] = np.minimum(delta, 24 - delta).astype(np.int32)

    # dow_match
    model_df["dow_match"] = (
        model_df["send_dow"] == model_df["modal_purchase_dow"]
    ).astype(np.int32)

    # ---- Interaction features -----------------------------------------------
    model_df["hour_x_entropy"]      = model_df["send_hour"] * model_df["purchase_hour_entropy"]
    model_df["recency_x_frequency"] = model_df["recency_days"] * model_df["frequency"]

    # ---- Compute `opened` target from campaign_events -----------------------
    print("  Computing opened target from campaign_events...")

    # Mean open rate per (customer_id, send_hour, send_dow)
    open_agg = (
        campaign_events
        .groupby(["customer_id", "send_hour", "send_dow"])["opened"]
        .mean()
        .reset_index()
        .rename(columns={"opened": "opened"})
    )

    model_df = model_df.merge(
        open_agg,
        on=["customer_id", "send_hour", "send_dow"],
        how="left",
    )
    # Default to 0.0 where no campaign event data exists
    model_df["opened"] = model_df["opened"].fillna(0.0)

    # ---- Column ordering (22 columns total) ---------------------------------
    col_order = [
        "customer_id",
        # 11 customer behavioral features
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
        # 7 slot features
        "send_hour",
        "send_dow",
        "is_weekend",
        "is_business_hours",
        "hour_delta_from_modal",
        "dow_match",
        "industry_open_rate_by_hour",
        # 2 interaction features
        "hour_x_entropy",
        "recency_x_frequency",
        # target
        "opened",
    ]
    model_df = model_df[col_order]

    print(f"  Final model_features shape: {model_df.shape}")
    return model_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(model_df: pd.DataFrame, customer_df: pd.DataFrame) -> None:
    """Run assertions to verify both output DataFrames."""
    print("\n[Tests] Running assertions...")

    # ---- model_features checks ----------------------------------------------
    n = len(model_df)
    n_customers = model_df["customer_id"].nunique()
    # Expected: n_customers × 168 slots; allow ±10% tolerance
    expected = n_customers * 168
    assert expected * 0.9 <= n <= expected * 1.1, \
        f"model_features rows out of range: {n:,} (expected ~{expected:,} = {n_customers}×168)"

    assert model_df["customer_id"].nunique() >= 3_500, \
        f"Too few unique customers in model_features: {model_df['customer_id'].nunique():,}"

    feature_cols = [c for c in model_df.columns if c not in ("customer_id", "opened")]
    assert len(feature_cols) == 20, \
        f"Expected 20 feature cols, got {len(feature_cols)}: {feature_cols}"

    numeric_cols = model_df.select_dtypes(include="number").columns.tolist()
    nan_count = model_df[numeric_cols].isna().sum().sum()
    assert nan_count == 0, \
        f"NaN values found in numeric columns: {model_df[numeric_cols].isna().sum()[model_df[numeric_cols].isna().sum() > 0]}"

    assert (model_df["purchase_hour_entropy"] >= 0).all(), \
        "purchase_hour_entropy has negative values"
    assert (model_df["purchase_hour_entropy"] <= 1).all(), \
        "purchase_hour_entropy exceeds 1.0"

    # ---- customer_features checks -------------------------------------------
    assert len(customer_df) >= 3_500, \
        f"customer_features has too few rows: {len(customer_df):,}"

    assert "rfm_segment" in customer_df.columns, \
        "rfm_segment column missing from customer_features"

    assert customer_df["rfm_segment"].notna().all(), \
        "rfm_segment has NaN values"

    print("[PASS] All assertions passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Feature Engineering — Phase 3")
    print("=" * 60)

    # 1. Load data
    print("\n[Load] Reading transactions_clean.csv...")
    transactions = pd.read_csv(PROCESSED_CSV)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    print(f"  Loaded {len(transactions):,} rows, {transactions['CustomerID'].nunique():,} unique customers.")

    print("[Load] Reading campaign_events.csv...")
    campaign_events = pd.read_csv(SYNTHETIC_CSV)
    print(f"  Loaded {len(campaign_events):,} rows, {campaign_events['customer_id'].nunique():,} unique customers.")

    # Deterministic reference date derived from data (never use datetime.now())
    reference_date = transactions["InvoiceDate"].max()
    print(f"  Reference date: {reference_date}")

    # 2. Compute customer behavioral features
    customer_feats = compute_customer_features(transactions, reference_date)

    # 3. Compute RFM segments
    customer_feats = compute_rfm_segments(customer_feats)

    # 4. Save customer_features.csv
    os.makedirs(os.path.dirname(CUSTOMER_FEATURES_CSV), exist_ok=True)
    customer_feats.to_csv(CUSTOMER_FEATURES_CSV, index=False)
    print(f"\n[Save] customer_features.csv -> {CUSTOMER_FEATURES_CSV}")
    print(f"  Shape: {customer_feats.shape}  (rows={len(customer_feats):,}, cols={customer_feats.shape[1]})")

    # 5. Build model features
    model_feats = build_model_features(customer_feats, campaign_events)

    # 6. Save model_features.csv
    os.makedirs(os.path.dirname(MODEL_FEATURES_CSV), exist_ok=True)
    model_feats.to_csv(MODEL_FEATURES_CSV, index=False)
    print(f"\n[Save] model_features.csv -> {MODEL_FEATURES_CSV}")
    print(f"  Shape: {model_feats.shape}  (rows={len(model_feats):,}, cols={model_feats.shape[1]})")

    # 7. Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  customer_features rows : {len(customer_feats):,}")
    print(f"  customer_features cols : {customer_feats.shape[1]}")
    print(f"  model_features rows    : {len(model_feats):,}")
    print(f"  model_features cols    : {model_feats.shape[1]}")
    print(f"  Unique customers       : {model_feats['customer_id'].nunique():,}")
    print(f"  Send slots             : {168}")
    print(f"  Overall open rate      : {model_feats['opened'].mean():.4f}")
    feature_cols = [c for c in model_feats.columns if c not in ("customer_id", "opened")]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"  RFM segments:\n{customer_feats['rfm_segment'].value_counts().to_string()}")

    # 8. Run tests
    run_tests(model_feats, customer_feats)

    print("\nDone.")


if __name__ == "__main__":
    main()
