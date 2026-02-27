"""
Data Pipeline: Phase 2
- Step 1: Clean UCI Online Retail II transactions
- Step 2: Synthesize email campaign events
- Adds deterministic sorting and schema contracts for outputs

Usage:
    python src/data_pipeline.py
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from scipy.stats import truncnorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "online_retail_ii.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")
SYNTHETIC_CSV = os.path.join(BASE_DIR, "data", "synthetic", "campaign_events.csv")

# Non-product StockCodes to exclude
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "ADJUST", "ADJUST2", "DOT"}


# ---------------------------------------------------------------------------
# Step 1: Clean transactions
# ---------------------------------------------------------------------------

def clean_transactions(raw_path: str, output_path: str) -> pd.DataFrame:
    """Load, clean, and save the UCI Online Retail II transactions."""
    print("[Step 1] Loading raw transactions...")
    df = pd.read_csv(raw_path, dtype={"Invoice": str, "StockCode": str})
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    if "UnitPrice" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"UnitPrice": "Price"})
        print("  Renamed 'UnitPrice' to 'Price'.")

    before = len(df)
    df = df.dropna(subset=["Customer ID"])
    print(f"  Dropped {before - len(df):,} rows with null Customer ID. Remaining: {len(df):,}")

    before = len(df)
    cancellation_mask = df["Invoice"].astype(str).str.startswith("C") | (df["Quantity"] < 0)
    df = df[~cancellation_mask].copy()
    print(f"  Dropped {before - len(df):,} cancellation rows. Remaining: {len(df):,}")

    before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)].copy()
    df = df[~df["StockCode"].str.startswith("TEST", na=False)].copy()
    print(f"  Dropped {before - len(df):,} non-product rows (codes + TEST* prefix). Remaining: {len(df):,}")

    before = len(df)
    df = df[df["Country"] == "United Kingdom"].copy()
    print(f"  Dropped {before - len(df):,} non-UK rows. Remaining: {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset=["Invoice", "StockCode", "Customer ID"])
    print(f"  Dropped {before - len(df):,} duplicate rows. Remaining: {len(df):,}")

    df = df.rename(columns={"Customer ID": "CustomerID"})

    float_ids = df["CustomerID"].astype(float)
    if (float_ids % 1 != 0).any():
        raise ValueError("Non-integer Customer ID values detected after nulls dropped.")
    df["CustomerID"] = float_ids.astype(np.int64)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Deterministic ordering before persistence
    df = df.sort_values(["CustomerID", "InvoiceDate", "Invoice", "StockCode"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved cleaned transactions to: {output_path}")
    print(f"  Final cleaned shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Synthesize campaign events
# ---------------------------------------------------------------------------

def synthesize_campaign_events(transactions: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Generate synthetic email campaign events from cleaned transactions."""
    print("\n[Step 2] Synthesizing campaign events...")

    rng = np.random.default_rng(42)

    txn_counts = transactions.groupby("CustomerID").size()
    eligible_customers = txn_counts[txn_counts >= 5].index
    print(f"  Eligible customers (>= 5 txns): {len(eligible_customers):,}")

    customer_data = transactions[transactions["CustomerID"].isin(eligible_customers)].copy()

    campaign_start = pd.Timestamp("2009-12-01")
    campaign_end = pd.Timestamp("2011-12-31")

    all_events: list[pd.DataFrame] = []
    global_campaign_idx = 0

    for customer_id, customer_txns in customer_data.groupby("CustomerID"):
        customer_txns = customer_txns.copy()
        customer_txns["InvoiceDate"] = pd.to_datetime(customer_txns["InvoiceDate"])
        purchase_hours = customer_txns["InvoiceDate"].dt.hour
        modal_hour = int(purchase_hours.mode().iloc[0])

        first_month = customer_txns["InvoiceDate"].dt.to_period("M").min()
        last_month = customer_txns["InvoiceDate"].dt.to_period("M").max()
        n_active_months = (last_month - first_month).n + 1

        n_emails = int(rng.poisson(lam=2 * n_active_months))
        if n_emails == 0:
            continue

        a = (0 - modal_hour) / 3.0
        b = (23 - modal_hour) / 3.0
        rng_int = int(rng.integers(0, 2**32))
        np_rs = np.random.RandomState(rng_int)
        send_hours_raw = truncnorm.rvs(
            a,
            b,
            loc=modal_hour,
            scale=3,
            size=n_emails,
            random_state=np_rs,
        )
        send_hours = np.round(send_hours_raw).astype(int).clip(0, 23)

        dow_counts = customer_txns["InvoiceDate"].dt.dayofweek.value_counts()
        dow_probs = dow_counts.reindex(range(7), fill_value=0).values.astype(float)
        if dow_probs.sum() == 0:
            dow_probs = np.ones(7)
        dow_probs = dow_probs / dow_probs.sum()
        send_dows = rng.choice(7, size=n_emails, p=dow_probs)

        total_campaign_days = (campaign_end - campaign_start).days
        random_day_offsets = rng.integers(0, total_campaign_days + 1, size=n_emails)
        base_dates = pd.to_datetime([campaign_start + pd.Timedelta(days=int(d)) for d in random_day_offsets])

        send_datetimes = []
        for i in range(n_emails):
            base_date = base_dates[i]
            target_dow = int(send_dows[i])
            current_dow = base_date.dayofweek
            delta_days = (target_dow - current_dow) % 7
            candidate_date = base_date + pd.Timedelta(days=delta_days)
            if candidate_date > campaign_end:
                candidate_date = base_date - pd.Timedelta(days=(7 - delta_days))
                if candidate_date < campaign_start:
                    candidate_date = base_date
            send_dt = candidate_date.replace(hour=int(send_hours[i]), minute=0, second=0, microsecond=0)
            send_datetimes.append(send_dt)

        send_datetimes = pd.DatetimeIndex(send_datetimes)

        hour_delta = np.abs(send_hours - modal_hour)
        time_alignment_score = np.where(hour_delta <= 2, 1.0, 1.0 - (hour_delta - 2) * (0.6 / 10))
        time_alignment_score = np.clip(time_alignment_score, 0.4, 1.0)

        noise = rng.normal(0, 0.05, size=n_emails)
        p_open = np.clip(0.25 * time_alignment_score + noise, 0, 1)

        opened = (rng.random(n_emails) < p_open).astype(int)
        clicked = np.zeros(n_emails, dtype=int)
        clicked[opened == 1] = (rng.random(opened.sum()) < 0.35).astype(int)
        purchased = np.zeros(n_emails, dtype=int)
        n_clicked = clicked.sum()
        if n_clicked > 0:
            purchased[clicked == 1] = (rng.random(n_clicked) < 0.12).astype(int)

        campaign_ids = [f"camp_{global_campaign_idx + i:06d}" for i in range(n_emails)]
        global_campaign_idx += n_emails

        events = pd.DataFrame(
            {
                "customer_id": np.int64(customer_id),
                "campaign_id": campaign_ids,
                "send_datetime": send_datetimes,
                "send_hour": send_hours.astype(np.int16),
                "send_dow": send_dows.astype(np.int8),
                "opened": opened.astype(np.int8),
                "clicked": clicked.astype(np.int8),
                "purchased": purchased.astype(np.int8),
                "channel": "email",
            }
        )
        all_events.append(events)

    print(f"  Processed {len(eligible_customers):,} eligible customers.")

    df_events = pd.concat(all_events, ignore_index=True)
    print(f"  Total campaign events generated: {len(df_events):,}")

    # Deterministic ordering before persistence
    df_events = df_events.sort_values(["send_datetime", "customer_id", "campaign_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_events.to_csv(output_path, index=False)
    print(f"  Saved campaign events to: {output_path}")

    return df_events


# ---------------------------------------------------------------------------
# Contracts / Assertions
# ---------------------------------------------------------------------------

def run_tests(df_transactions: pd.DataFrame, df_events: pd.DataFrame) -> None:
    """Run assertions to verify both pipeline outputs."""
    print("\n[Tests] Running assertions on outputs...")

    # transactions_clean contracts
    assert df_transactions["CustomerID"].dtype == np.int64, "CustomerID must be int64 in transactions_clean.csv"
    assert df_transactions["CustomerID"].isna().sum() == 0, "CustomerID has nulls"
    assert (~df_transactions["StockCode"].isin(NON_PRODUCT_CODES)).all(), "Non-product stock codes present"
    assert (~df_transactions["StockCode"].str.startswith("TEST", na=False)).all(), "TEST* stock codes present"
    assert (~df_transactions["Invoice"].astype(str).str.startswith("C")).all(), "Cancelled invoices present"
    assert (df_transactions["Quantity"] >= 0).all(), "Negative quantities present"

    # campaign_events contracts
    n_rows = len(df_events)
    assert 80_000 <= n_rows <= 120_000, f"Row count {n_rows:,} not in [80k, 120k]"

    expected_cols = [
        "customer_id",
        "campaign_id",
        "send_datetime",
        "send_hour",
        "send_dow",
        "opened",
        "clicked",
        "purchased",
        "channel",
    ]
    assert list(df_events.columns) == expected_cols, "campaign_events schema mismatch"

    assert pd.api.types.is_integer_dtype(df_events["customer_id"]), "customer_id must be integer typed"
    for col in ["customer_id", "send_datetime", "send_hour", "send_dow", "opened", "clicked", "purchased"]:
        assert df_events[col].isna().sum() == 0, f"{col} has nulls"

    assert set(df_events["opened"].unique()).issubset({0, 1}), "opened must be binary"
    assert set(df_events["clicked"].unique()).issubset({0, 1}), "clicked must be binary"
    assert set(df_events["purchased"].unique()).issubset({0, 1}), "purchased must be binary"
    assert ((df_events["opened"] == 0) & (df_events["clicked"] == 1)).sum() == 0, "clicked without opened"
    assert ((df_events["clicked"] == 0) & (df_events["purchased"] == 1)).sum() == 0, "purchased without clicked"

    assert df_events["campaign_id"].is_unique, "campaign_id must be unique"
    assert df_events["channel"].eq("email").all(), "channel must be email"
    assert df_events["send_hour"].between(0, 23).all(), "send_hour out of range"
    assert df_events["send_dow"].between(0, 6).all(), "send_dow out of range"

    open_rate = df_events["opened"].mean()
    assert 0.22 <= open_rate <= 0.28, f"Open rate {open_rate:.4f} not in [0.22, 0.28]"

    print(f"  [PASS] transactions_clean rows: {len(df_transactions):,}")
    print(f"  [PASS] campaign_events rows: {len(df_events):,}")
    print(f"  [PASS] open rate: {open_rate:.4f} ({open_rate*100:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Data Pipeline — Phase 2")
    print("=" * 60)

    transactions = clean_transactions(RAW_CSV, PROCESSED_CSV)
    df_events = synthesize_campaign_events(transactions, SYNTHETIC_CSV)

    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"  Campaign events rows:     {len(df_events):,}")
    print(f"  Open rate:                {df_events['opened'].mean():.4f} ({df_events['opened'].mean()*100:.2f}%)")
    print(f"  Click rate:               {df_events['clicked'].mean():.4f} ({df_events['clicked'].mean()*100:.2f}%)")
    print(f"  Purchase rate:            {df_events['purchased'].mean():.4f} ({df_events['purchased'].mean()*100:.2f}%)")
    print(f"  Unique customers:         {df_events['customer_id'].nunique():,}")
    print(f"  Output: {PROCESSED_CSV}")
    print(f"  Output: {SYNTHETIC_CSV}")

    run_tests(transactions, df_events)

    print("\nDone.")


if __name__ == "__main__":
    main()
