# Data Dictionary -- Marketing Mix Model (m04_mmm)

This document describes every dataset used in the Marketing Mix Model pipeline, including raw inputs, engineered features, ground-truth parameters, and the train/test split convention.

---

## Raw Data

**File:** `data/synthetic/mmm_weekly_data.csv`

This is the primary synthetic dataset containing 208 weeks of marketing spend and revenue observations. It simulates a single brand operating across five paid media channels, one organic channel, and two external control variables.

| Column | Type | Description | Role |
|--------|------|-------------|------|
| `date_week` | date | Week start date (YYYY-MM-DD) | Date index |
| `revenue` | float | Weekly revenue (EUR) | Target variable |
| `tv_spend` | float | TV advertising spend (EUR/week) | Paid media spend |
| `ooh_spend` | float | Out-of-home (billboard) spend (EUR/week) | Paid media spend |
| `print_spend` | float | Print advertising spend (EUR/week) | Paid media spend |
| `facebook_spend` | float | Facebook advertising spend (EUR/week) | Paid media spend |
| `search_spend` | float | Paid search spend (EUR/week) | Paid media spend |
| `newsletter_sends` | float | Newsletter email volume | Organic media |
| `competitor_index` | float | Competitor sales index (AR(1) process, mean=1.0) | Control variable |
| `event_flag` | int | Promotional event indicator (0/1, approximately 10% of weeks) | Control variable |

### Column Notes

- **revenue**: The dependent variable the model aims to decompose into base, media, and control contributions.
- **tv_spend, ooh_spend, print_spend, facebook_spend, search_spend**: Each channel's weekly investment in EUR. These are the primary media variables subject to adstock transformation and saturation modeling.
- **newsletter_sends**: Volume of newsletters dispatched per week. Treated as an organic (non-paid) media input.
- **competitor_index**: A synthetic AR(1) time series centered at 1.0 representing relative competitor activity. Values above 1.0 indicate above-average competitor pressure.
- **event_flag**: Binary indicator for promotional events (sales, product launches, etc.). Approximately 10% of weeks are flagged.

---

## Engineered Features

**File:** `data/processed/mmm_features.csv`

These features are derived during the feature engineering step and appended to the raw data for model training. They capture trend and seasonality components.

| Column | Type | Description |
|--------|------|-------------|
| `t` | float | Linear trend, normalized from 0 (first week) to 1 (last week) |
| `sin_1` | float | Fourier term: sin(2 * pi * week_number / 52) |
| `cos_1` | float | Fourier term: cos(2 * pi * week_number / 52) |
| `sin_2` | float | Fourier term: sin(4 * pi * week_number / 52) |
| `cos_2` | float | Fourier term: cos(4 * pi * week_number / 52) |

### Feature Notes

- **t (linear trend)**: Captures long-run growth or decline in baseline revenue over the observation window.
- **Fourier terms (sin_1, cos_1, sin_2, cos_2)**: Two pairs of sine/cosine terms at annual (52-week) frequency. The first pair captures the fundamental annual cycle; the second pair captures a semi-annual harmonic. Together they model seasonality without requiring dummy variables for each week of the year.

---

## True Parameters

**File:** `data/synthetic/true_params.json`

Because the dataset is synthetic, the exact data-generating process is known. This JSON file stores the ground-truth parameters used to simulate the data:

- **Adstock alphas** -- The geometric decay rate for each media channel's carryover effect. A higher alpha means the effect of a spend pulse persists longer.
- **Saturation lambdas** -- The Hill/saturation curve shape parameter for each channel. Controls how quickly incremental spend yields diminishing returns.
- **Betas** -- The true regression coefficients (contribution weights) for each channel, control variable, and intercept term.

This file is critical for **parameter recovery validation**: after fitting the model, recovered adstock, saturation, and beta estimates can be compared directly against these ground-truth values to assess model accuracy and identifiability.

---

## Train/Test Split

The 208-week dataset is split chronologically (no shuffling) to respect the time-series nature of the data.

| Split | Weeks | Date Range | Proportion |
|-------|-------|------------|------------|
| Training | 1--156 | 2015-11-23 to 2018-11-30 | 75% |
| Test | 157--208 | 2018-12-01 to 2019-11-11 | 25% |

The split is applied after feature engineering so that trend normalization and Fourier terms are computed on the full timeline, but model coefficients are estimated on the training set only. Out-of-sample metrics (MAPE, R-squared) are reported on the test set to guard against overfitting.
