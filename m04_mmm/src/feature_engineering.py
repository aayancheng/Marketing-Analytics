"""
Feature engineering for the Marketing Mix Model.

The raw data is already weekly aggregates, so engineering is minimal:
- Add a normalised linear trend column ``t``
- Add Fourier seasonality terms (yearly, period = 52 weeks)
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants (shared with API / training code)
# ---------------------------------------------------------------------------

CHANNEL_COLUMNS = [
    "tv_spend",
    "ooh_spend",
    "print_spend",
    "facebook_spend",
    "search_spend",
]

CONTROL_COLUMNS = ["competitor_index", "event_flag", "t"]

TARGET_COLUMN = "revenue"


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend and Fourier seasonality columns to *df*.

    New columns added:
        t      - linear trend normalised to [0, 1]
        sin_1  - sin(2 pi t_idx / 52)
        cos_1  - cos(2 pi t_idx / 52)
        sin_2  - sin(4 pi t_idx / 52)
        cos_2  - cos(4 pi t_idx / 52)

    Returns a copy; the original DataFrame is not mutated.
    """
    df = df.copy()
    n = len(df)
    t_idx = np.arange(n, dtype=np.float64)

    # Normalised trend
    df["t"] = t_idx / max(n - 1, 1)

    # Fourier terms — yearly cycle (period = 52 weeks)
    df["sin_1"] = np.sin(2 * np.pi * t_idx / 52)
    df["cos_1"] = np.cos(2 * np.pi * t_idx / 52)
    df["sin_2"] = np.sin(4 * np.pi * t_idx / 52)
    df["cos_2"] = np.cos(4 * np.pi * t_idx / 52)

    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run() -> pd.DataFrame:
    """Load raw synthetic data, add features, save to processed directory.

    Reads:  data/synthetic/mmm_weekly_data.csv
    Writes: data/processed/mmm_features.csv
    """
    input_path = os.path.join(_PROJECT_ROOT, "data", "synthetic", "mmm_weekly_data.csv")
    df = pd.read_csv(input_path, parse_dates=["date_week"])

    df = add_features(df)

    output_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mmm_features.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows with {len(df.columns)} columns to {output_path}")
    return df


if __name__ == "__main__":
    run()
