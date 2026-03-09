"""
Synthetic data generator for Marketing Mix Model (MMM).

Generates 208 weeks of marketing mix data with KNOWN TRUE PARAMETERS
so that parameter recovery can be validated against ground truth.
"""

import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TRUE PARAMETERS (module-level constants for parameter-recovery validation)
# ---------------------------------------------------------------------------

CHANNELS = ["tv", "ooh", "print", "facebook", "search"]
CHANNEL_SPEND_COLS = [f"{c}_spend" for c in CHANNELS]

TRUE_ADSTOCK_ALPHAS = {
    "tv": 0.70,
    "ooh": 0.50,
    "print": 0.30,
    "facebook": 0.20,
    "search": 0.10,
}

TRUE_SATURATION_LAMBDAS = {
    "tv": 0.50,
    "ooh": 0.80,
    "print": 1.00,
    "facebook": 0.30,
    "search": 0.40,
}

TRUE_BETAS = {
    "tv": 2.5, "ooh": 1.2, "print": 0.8, "facebook": 1.8, "search": 2.0}

TRUE_INTERCEPT = 5.0
TRUE_TREND_COEF = 0.005
TRUE_COMPETITOR_COEF = -0.3
TRUE_EVENT_COEF = 0.15
TRUE_NOISE_SIGMA = 0.05

# Fourier seasonality coefficients (yearly, period = 52 weeks)
TRUE_FOURIER_COEFFS = {
    "sin1": 0.15,
    "cos1": -0.10,
    "sin2": 0.05,
    "cos2": 0.03,
}

# Target mean spend per channel (EUR / week) used when rescaling
_CHANNEL_MEAN_SPEND = {
    "tv": 20_000,
    "ooh": 10_000,
    "print": 7_000,
    "facebook": 15_000,
    "search": 12_000,
}

# Seasonal amplitude per channel (higher = more seasonal variation)
_CHANNEL_SEASONALITY = {
    "tv": 0.40,       # TV heavier in Q4
    "ooh": 0.25,
    "print": 0.20,
    "facebook": 0.15,
    "search": 0.05,   # search is near-constant year-round
}

# Q4 peak week offset (week-of-year where seasonal peak occurs)
_CHANNEL_PEAK_WEEK = {
    "tv": 48,       # late November
    "ooh": 30,      # summer
    "print": 46,
    "facebook": 47,
    "search": 10,   # mild spring bump
}


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Apply geometric (carry-over) adstock to a 1-D spend series.

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw spend series.
    alpha : float in [0, 1)
        Retention / decay rate.  Higher = longer carry-over.
    l_max : int
        Maximum lag window in weeks.

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked series.
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.array([alpha ** i for i in range(l_max)])
    weights = weights / weights.sum()

    # Pad the beginning so output length == input length
    padded = np.concatenate([np.zeros(l_max - 1), x])
    adstocked = np.convolve(padded, weights, mode="valid")
    return adstocked[: len(x)]


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Logistic (Hill-like) saturation: x / (x + lam).

    Parameters
    ----------
    x : array-like
        Non-negative input (e.g. adstocked spend).
    lam : float > 0
        Half-saturation point.

    Returns
    -------
    np.ndarray
        Saturated values in [0, 1).
    """
    x = np.asarray(x, dtype=np.float64)
    return x / (x + lam)


# ---------------------------------------------------------------------------
# Main data generator
# ---------------------------------------------------------------------------

def generate_mmm_data(
    start_date: str = "2015-11-23",
    periods: int = 208,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic weekly marketing-mix dataset.

    Returns a DataFrame with columns:
        date_week, revenue, tv_spend, ooh_spend, print_spend,
        facebook_spend, search_spend, newsletter_sends,
        competitor_index, event_flag
    """
    rng = np.random.default_rng(seed)

    # --- 1. Date index -------------------------------------------------------
    dates = pd.date_range(start=start_date, periods=periods, freq="W-MON")
    n = len(dates)
    t_norm = np.linspace(0, 1, n)  # normalised time index

    # --- 2. Channel spends ---------------------------------------------------
    spend = {}
    for ch in CHANNELS:
        # Log-normal base
        raw = rng.lognormal(mean=0.0, sigma=0.6, size=n)

        # Seasonal modulation: cosine wave peaking at _CHANNEL_PEAK_WEEK
        week_of_year = np.array([d.isocalendar()[1] for d in dates])
        peak = _CHANNEL_PEAK_WEEK[ch]
        seasonal = 1.0 + _CHANNEL_SEASONALITY[ch] * np.cos(
            2 * np.pi * (week_of_year - peak) / 52
        )
        raw = raw * seasonal

        # Rescale to target mean spend
        raw = raw / raw.mean() * _CHANNEL_MEAN_SPEND[ch]

        # Ensure non-negative and round to whole euros
        spend[ch] = np.maximum(raw, 0.0).round(0)

    # --- 3. Adstock + saturation -> channel contributions --------------------
    channel_contributions = np.zeros(n)
    for ch in CHANNELS:
        adstocked = geometric_adstock(spend[ch], TRUE_ADSTOCK_ALPHAS[ch])
        saturated = logistic_saturation(adstocked, TRUE_SATURATION_LAMBDAS[ch])
        # Scale to [0, 1] via MaxAbs
        max_val = saturated.max()
        if max_val > 0:
            saturated = saturated / max_val
        channel_contributions += TRUE_BETAS[ch] * saturated

    # --- 4. Base = intercept + trend ------------------------------------------
    base = TRUE_INTERCEPT + TRUE_TREND_COEF * t_norm

    # --- 5. Fourier seasonality (period = 52 weeks) --------------------------
    t_idx = np.arange(n, dtype=np.float64)
    sin1 = np.sin(2 * np.pi * t_idx / 52)
    cos1 = np.cos(2 * np.pi * t_idx / 52)
    sin2 = np.sin(4 * np.pi * t_idx / 52)
    cos2 = np.cos(4 * np.pi * t_idx / 52)
    seasonality = (
        TRUE_FOURIER_COEFFS["sin1"] * sin1
        + TRUE_FOURIER_COEFFS["cos1"] * cos1
        + TRUE_FOURIER_COEFFS["sin2"] * sin2
        + TRUE_FOURIER_COEFFS["cos2"] * cos2
    )

    # --- 6. Control variables ------------------------------------------------
    # Competitor index: AR(1), mean ~1.0
    competitor = np.empty(n)
    competitor[0] = 1.0
    for i in range(1, n):
        competitor[i] = 0.7 * competitor[i - 1] + 0.3 * 1.0 + rng.normal(0, 0.08)

    # Event flag: ~10% of weeks
    event_flag = rng.binomial(1, 0.10, size=n).astype(float)

    # Newsletter sends: loosely correlated with total channel spend
    total_spend = sum(spend[ch] for ch in CHANNELS)
    newsletter_base = 5000 + 0.05 * total_spend + rng.normal(0, 500, size=n)
    newsletter_sends = np.maximum(newsletter_base, 0).round(0)

    # --- 7. Revenue (log-space) ----------------------------------------------
    log_revenue = (
        base
        + seasonality
        + channel_contributions
        + TRUE_COMPETITOR_COEF * (competitor - 1.0)
        + TRUE_EVENT_COEF * event_flag
        + rng.normal(0, TRUE_NOISE_SIGMA, size=n)
    )
    # Exponentiate and scale to realistic EUR (~200k-300k / week)
    revenue_raw = np.exp(log_revenue)
    scale_factor = 250_000 / revenue_raw.mean()
    revenue = (revenue_raw * scale_factor).round(2)

    # --- 8. Assemble DataFrame -----------------------------------------------
    df = pd.DataFrame(
        {
            "date_week": dates,
            "revenue": revenue,
            "tv_spend": spend["tv"],
            "ooh_spend": spend["ooh"],
            "print_spend": spend["print"],
            "facebook_spend": spend["facebook"],
            "search_spend": spend["search"],
            "newsletter_sends": newsletter_sends,
            "competitor_index": competitor.round(4),
            "event_flag": event_flag.astype(int),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save generated data and true parameters to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "mmm_weekly_data.csv")
    df.to_csv(csv_path, index=False)

    params = {
        "channels": CHANNELS,
        "adstock_alphas": TRUE_ADSTOCK_ALPHAS,
        "saturation_lambdas": TRUE_SATURATION_LAMBDAS,
        "betas": TRUE_BETAS,
        "intercept": TRUE_INTERCEPT,
        "trend_coef": TRUE_TREND_COEF,
        "competitor_coef": TRUE_COMPETITOR_COEF,
        "event_coef": TRUE_EVENT_COEF,
        "noise_sigma": TRUE_NOISE_SIGMA,
        "fourier_coeffs": TRUE_FOURIER_COEFFS,
    }
    json_path = os.path.join(output_dir, "true_params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = generate_mmm_data()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic")
    save_data(data, out)
    print(f"Generated {len(data)} rows -> {out}")
