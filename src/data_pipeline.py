"""
Data Pipeline: Phase 2
- Step 1: Clean UCI Online Retail II transactions
- Step 2: Synthesize email campaign events

Usage:
    python src/data_pipeline.py
"""

import os
import sys
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
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES"}


# ---------------------------------------------------------------------------
# Step 1: Clean transactions
# ---------------------------------------------------------------------------

def clean_transactions(raw_path: str, output_path: str) -> pd.DataFrame:
    """Load, clean, and save the UCI Online Retail II transactions."""
    print("[Step 1] Loading raw transactions...")
    df = pd.read_csv(raw_path, dtype={"Invoice": str, "StockCode": str})
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Handle both Price and UnitPrice column names
    if "UnitPrice" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"UnitPrice": "Price"})
        print("  Renamed 'UnitPrice' to 'Price'.")

    # Drop rows where Customer ID is null
    before = len(df)
    df = df.dropna(subset=["Customer ID"])
    print(f"  Dropped {before - len(df):,} rows with null Customer ID. Remaining: {len(df):,}")

    # Exclude EITHER cancelled invoices (prefix 'C') OR returns/adjustments (Quantity < 0)
    # Both are independent exclusion reasons per the data brief
    before = len(df)
    cancellation_mask = df["Invoice"].astype(str).str.startswith("C") | (df["Quantity"] < 0)
    df = df[~cancellation_mask].copy()
    print(f"  Dropped {before - len(df):,} cancellation rows. Remaining: {len(df):,}")

    # Filter non-product StockCodes
    before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)].copy()
    print(f"  Dropped {before - len(df):,} non-product rows. Remaining: {len(df):,}")

    # Filter UK-only customers
    before = len(df)
    df = df[df["Country"] == "United Kingdom"].copy()
    print(f"  Dropped {before - len(df):,} non-UK rows. Remaining: {len(df):,}")

    # Deduplicate on (Invoice, StockCode, Customer ID)
    before = len(df)
    df = df.drop_duplicates(subset=["Invoice", "StockCode", "Customer ID"])
    print(f"  Dropped {before - len(df):,} duplicate rows. Remaining: {len(df):,}")

    # Rename 'Customer ID' to 'CustomerID'
    df = df.rename(columns={"Customer ID": "CustomerID"})

    # Ensure CustomerID is stored as int then string for clean IDs
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Parse InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Save processed file
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

    # Eligible customers: >= 5 transactions
    txn_counts = transactions.groupby("CustomerID").size()
    eligible_customers = txn_counts[txn_counts >= 5].index
    print(f"  Eligible customers (>= 5 txns): {len(eligible_customers):,}")

    customer_data = transactions[transactions["CustomerID"].isin(eligible_customers)].copy()

    # Campaign period: Dec 2010 - Dec 2011
    campaign_start = pd.Timestamp("2010-12-01")
    campaign_end = pd.Timestamp("2011-12-31")

    all_events = []
    global_campaign_idx = 0

    for customer_id, customer_txns in customer_data.groupby("CustomerID"):
        # Modal purchase hour
        customer_txns = customer_txns.copy()
        customer_txns["InvoiceDate"] = pd.to_datetime(customer_txns["InvoiceDate"])
        purchase_hours = customer_txns["InvoiceDate"].dt.hour
        modal_hour = int(purchase_hours.mode().iloc[0])

        # Active months
        first_month = customer_txns["InvoiceDate"].dt.to_period("M").min()
        last_month = customer_txns["InvoiceDate"].dt.to_period("M").max()
        n_active_months = (last_month - first_month).n + 1

        # N emails ~ Poisson(lambda=2) per active month
        n_emails = int(rng.poisson(lam=2 * n_active_months))
        if n_emails == 0:
            continue

        # Send hours using truncnorm centered on modal_hour ± 3h, clipped 0-23
        a = (0 - modal_hour) / 3.0
        b = (23 - modal_hour) / 3.0
        # Use a fresh integer seed derived from global_campaign_idx for reproducibility
        rng_int = int(rng.integers(0, 2**31))
        send_hours_raw = truncnorm.rvs(
            a, b,
            loc=modal_hour,
            scale=3,
            size=n_emails,
            random_state=rng_int,
        )
        send_hours = np.round(send_hours_raw).astype(int).clip(0, 23)

        # Day of week weighted sampling
        dow_counts = customer_txns["InvoiceDate"].dt.dayofweek.value_counts()
        dow_probs = dow_counts.reindex(range(7), fill_value=0).values.astype(float)
        if dow_probs.sum() == 0:
            dow_probs = np.ones(7)
        dow_probs = dow_probs / dow_probs.sum()
        send_dows = rng.choice(7, size=n_emails, p=dow_probs)

        # Build send datetimes within the campaign period
        # Generate one random date per email in the campaign range
        total_campaign_days = (campaign_end - campaign_start).days
        random_day_offsets = rng.integers(0, total_campaign_days + 1, size=n_emails)
        base_dates = pd.to_datetime([campaign_start + pd.Timedelta(days=int(d)) for d in random_day_offsets])

        # Adjust to nearest correct day-of-week (within the same week)
        send_datetimes = []
        for i in range(n_emails):
            base_date = base_dates[i]
            target_dow = int(send_dows[i])
            current_dow = base_date.dayofweek
            delta_days = (target_dow - current_dow) % 7
            # Keep the date within campaign bounds
            candidate_date = base_date + pd.Timedelta(days=delta_days)
            if candidate_date > campaign_end:
                candidate_date = base_date - pd.Timedelta(days=(7 - delta_days))
            send_dt = candidate_date.replace(hour=int(send_hours[i]), minute=0, second=0, microsecond=0)
            send_datetimes.append(send_dt)

        send_datetimes = pd.DatetimeIndex(send_datetimes)

        # Time alignment score for open probability
        hour_delta = np.abs(send_hours - modal_hour)
        time_alignment_score = np.where(
            hour_delta <= 2,
            1.0,
            1.0 - (hour_delta - 2) * (0.6 / 10)
        )
        time_alignment_score = np.clip(time_alignment_score, 0.4, 1.0)

        # Add Gaussian noise to prevent perfect signal
        noise = rng.normal(0, 0.05, size=n_emails)
        p_open = np.clip(0.25 * time_alignment_score + noise, 0, 1)

        # Draw opened, clicked, purchased
        opened = (rng.random(n_emails) < p_open).astype(int)
        clicked = np.zeros(n_emails, dtype=int)
        clicked[opened == 1] = (rng.random(opened.sum()) < 0.35).astype(int)
        purchased = np.zeros(n_emails, dtype=int)
        n_clicked = clicked.sum()
        if n_clicked > 0:
            purchased[clicked == 1] = (rng.random(n_clicked) < 0.12).astype(int)

        # Build campaign IDs for this customer's emails
        campaign_ids = [
            f"camp_{global_campaign_idx + i:06d}" for i in range(n_emails)
        ]
        global_campaign_idx += n_emails

        # Create events dataframe for this customer
        events = pd.DataFrame({
            "customer_id": customer_id,
            "campaign_id": campaign_ids,
            "send_datetime": send_datetimes,
            "send_hour": send_hours,
            "send_dow": send_dows.astype(int),
            "opened": opened,
            "clicked": clicked,
            "purchased": purchased,
            "channel": "email",
        })
        all_events.append(events)

    print(f"  Processed {len(eligible_customers):,} eligible customers.")

    # Concatenate all events
    df_events = pd.concat(all_events, ignore_index=True)
    print(f"  Total campaign events generated: {len(df_events):,}")

    # Save synthetic events
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_events.to_csv(output_path, index=False)
    print(f"  Saved campaign events to: {output_path}")

    return df_events


# ---------------------------------------------------------------------------
# Tests / Assertions
# ---------------------------------------------------------------------------

def run_tests(df_events: pd.DataFrame) -> None:
    """Run assertions to verify the campaign events dataset."""
    print("\n[Tests] Running assertions on campaign_events...")

    # Row count
    n_rows = len(df_events)
    assert 80_000 <= n_rows <= 120_000, (
        f"Row count {n_rows:,} not in [80k, 120k]"
    )
    print(f"  [PASS] Row count: {n_rows:,}")

    # Column schema
    expected_cols = [
        "customer_id", "campaign_id", "send_datetime", "send_hour",
        "send_dow", "opened", "clicked", "purchased", "channel",
    ]
    actual_cols = list(df_events.columns)
    assert actual_cols == expected_cols, (
        f"Column mismatch.\n  Expected: {expected_cols}\n  Got:      {actual_cols}"
    )
    print(f"  [PASS] Column schema matches.")

    # No nulls in key columns
    key_cols = ["customer_id", "opened", "clicked", "purchased", "send_hour", "send_dow"]
    for col in key_cols:
        null_count = df_events[col].isna().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} nulls."
    print(f"  [PASS] No nulls in key columns.")

    # customer_id no nulls (explicit)
    assert df_events["customer_id"].isna().sum() == 0, "customer_id has nulls."
    print(f"  [PASS] customer_id has no nulls.")

    # opened dtype and values
    assert df_events["opened"].dtype in [np.int64, np.int32, int], (
        f"opened dtype is {df_events['opened'].dtype}, expected int."
    )
    assert set(df_events["opened"].unique()).issubset({0, 1}), (
        f"opened has values outside {{0,1}}: {df_events['opened'].unique()}"
    )
    print(f"  [PASS] opened dtype=int, values in {{0,1}}.")

    # clicked: rows where opened==0 must have clicked==0
    bad_clicked = df_events[(df_events["opened"] == 0) & (df_events["clicked"] == 1)]
    assert len(bad_clicked) == 0, (
        f"Found {len(bad_clicked)} rows where opened==0 but clicked==1."
    )
    print(f"  [PASS] clicked is 0 whenever opened==0.")

    # purchased: rows where clicked==0 must have purchased==0
    bad_purchased = df_events[(df_events["clicked"] == 0) & (df_events["purchased"] == 1)]
    assert len(bad_purchased) == 0, (
        f"Found {len(bad_purchased)} rows where clicked==0 but purchased==1."
    )
    print(f"  [PASS] purchased is 0 whenever clicked==0.")

    # Open rate
    open_rate = df_events["opened"].mean()
    print(f"  Open rate: {open_rate:.4f} ({open_rate*100:.2f}%)")
    assert 0.22 <= open_rate <= 0.28, (
        f"Open rate {open_rate:.4f} not in [0.22, 0.28]."
    )
    print(f"  [PASS] Open rate in [22%, 28%].")

    print("\n[Tests] All assertions passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Data Pipeline — Phase 2")
    print("=" * 60)

    # Step 1: Clean transactions
    transactions = clean_transactions(RAW_CSV, PROCESSED_CSV)

    # Step 2: Synthesize campaign events
    df_events = synthesize_campaign_events(transactions, SYNTHETIC_CSV)

    # Print final stats
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"  Campaign events rows:     {len(df_events):,}")
    open_rate = df_events["opened"].mean()
    click_rate = df_events["clicked"].mean()
    purchase_rate = df_events["purchased"].mean()
    unique_customers = df_events["customer_id"].nunique()
    print(f"  Open rate:                {open_rate:.4f} ({open_rate*100:.2f}%)")
    print(f"  Click rate:               {click_rate:.4f} ({click_rate*100:.2f}%)")
    print(f"  Purchase rate:            {purchase_rate:.4f} ({purchase_rate*100:.2f}%)")
    print(f"  Unique customers:         {unique_customers:,}")
    print(f"  Output: {PROCESSED_CSV}")
    print(f"  Output: {SYNTHETIC_CSV}")

    # Run tests
    run_tests(df_events)

    print("\nDone.")


if __name__ == "__main__":
    main()
