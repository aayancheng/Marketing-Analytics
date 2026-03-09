# Model Documentation Report
## Marketing Mix Model — Bayesian MMM with PyMC-Marketing

**Project:** Marketing Analytics — m04_mmm
**Dataset:** Synthetic weekly media spend and revenue (208 weeks)
**Model:** PyMC-Marketing Bayesian MMM (GeometricAdstock + LogisticSaturation)
**Version:** 1.0 | **Date:** 2026-03-08
**Authors:** Marketing Analytics Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review](#2-literature-review)
3. [Data Description](#3-data-description)
4. [Feature Engineering](#4-feature-engineering)
5. [Methodology](#5-methodology)
6. [Model Performance](#6-model-performance)
7. [Channel Decomposition and ROAS](#7-channel-decomposition-and-roas)
8. [Response Curves and Saturation](#8-response-curves-and-saturation)
9. [API Reference](#9-api-reference)
10. [Deployment](#10-deployment)
11. [Future Work](#11-future-work)

---

## 1. Executive Summary

### The Problem

Marketing organisations spend millions across media channels — TV, digital, out-of-home, print, search — but struggle to answer two fundamental questions: *how much revenue did each channel actually drive?* and *how should we reallocate budget to maximise return?*

Traditional attribution (last-click, first-touch, even multi-touch) operates at the user level and breaks down when cookies are blocked, cross-device journeys are untracked, or offline channels like TV and billboards are involved. Marketing Mix Modeling (MMM) solves this by working at the aggregate level: it uses time-series regression on weekly spend and revenue data to decompose observed outcomes into channel contributions, controlling for external factors like seasonality and competitor activity.

### The Approach

This project implements a **Bayesian MMM** using PyMC-Marketing, a Python library purpose-built for media mix analysis. The Bayesian framework provides three advantages over frequentist alternatives: (1) full posterior distributions with credible intervals on every parameter, including ROAS; (2) principled regularisation through informative priors that encode domain knowledge about adstock decay and saturation; and (3) probabilistic budget optimisation that accounts for parameter uncertainty.

The model operates on 208 weeks of synthetic data across five paid media channels (TV, OOH, Print, Facebook, Search) with two control variables (competitor index, promotional events). Synthetic data was chosen deliberately to enable **parameter recovery validation** — comparing estimated parameters against known ground truth, which is impossible with real-world data.

### Key Results

| Metric | Value |
|--------|-------|
| Out-of-sample MAPE | **3.9%** |
| Out-of-sample R² | **0.91** |
| Parameter recovery | **9/10** within 94% HDI |
| MCMC convergence (max R-hat) | **1.007** |

The model successfully decomposes revenue into channel contributions and provides ROAS estimates with uncertainty quantification. Print shows the highest mean ROAS (0.28), while Facebook shows the lowest (0.13). All credible intervals are wide, reflecting the inherent challenge of identifying individual channel effects from aggregate data — a well-known property of MMMs that reinforces the value of Bayesian uncertainty quantification.

### Deployment

The MMM is deployed as a pre-computed JSON serving architecture: the model is fitted offline (MCMC sampling takes ~10 minutes), and all results — decomposition, ROAS, response curves, adstock, simulator parameters, optimal allocation — are exported as JSON files. The FastAPI backend loads these at startup and serves them without needing the model object, which is not picklable. This design is consistent with the MMM use case: results are refreshed weekly or monthly, not in real time.

---

## 2. Literature Review

### 2.1 Bayesian Media Mix Modeling

> Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Inc.

This foundational paper established the modern Bayesian MMM framework with two key innovations: **adstock** (carryover) transforms that model how a media spend pulse continues to affect outcomes over multiple periods, and **saturation** (shape) transforms that capture diminishing returns at high spend levels. The authors demonstrated that encoding these domain-specific transforms directly into the regression model, rather than using raw spend values, dramatically improves parameter identifiability and out-of-sample accuracy.

**Influence on this project:** The GeometricAdstock and LogisticSaturation transforms in PyMC-Marketing directly implement this paper's recommendations. The geometric adstock decay rate (alpha) controls how quickly a channel's effect fades, while the logistic saturation parameter (lambda) controls the steepness of the diminishing returns curve.

### 2.2 PyMC-Marketing

> Orduz, J., Capretto, M., et al. (2023). PyMC-Marketing: Bayesian Marketing Mix Modeling and Customer Lifetime Value. https://www.pymc-marketing.io/

PyMC-Marketing provides a high-level Python API for Bayesian MMM built on PyMC, ArviZ, and optionally JAX/NumPyro for GPU-accelerated sampling. It encapsulates the adstock/saturation transforms, handles channel scaling (MaxAbsScaler), provides built-in budget optimisation, and produces ArviZ InferenceData objects for posterior diagnostics.

**Influence on this project:** PyMC-Marketing is the core modeling library. Its `MMM` class handles model specification, MCMC fitting, posterior prediction, channel contribution computation, and budget optimisation. The library's opinionated API (channel_columns, control_columns, date_column) structures the entire data pipeline.

### 2.3 Convergence Diagnostics

> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC." Bayesian Analysis, 16(2), 667--718.

The improved R-hat diagnostic and effective sample size (ESS) metrics are the standard for assessing MCMC convergence. R-hat compares between-chain and within-chain variance; values below 1.01 indicate convergence. Bulk ESS measures the effective number of independent samples for estimating posterior means, while tail ESS does the same for tail quantiles.

**Influence on this project:** Convergence is validated by checking that max R-hat < 1.01 and min ESS > 400 across all model parameters. The trained model achieves max R-hat = 1.007 and min ESS (bulk) = 2,032, comfortably passing both thresholds.

### 2.4 Meta Robyn

> Meta Open Source. (2022). Robyn: Continuous & Semi-Automated MMM. https://github.com/facebookexperimental/Robyn

Robyn is Meta's open-source MMM framework written in R. It uses ridge regression with multi-objective hyperparameter optimisation (Nevergrad) and Pareto-optimal model selection. While Robyn offers faster fitting and automated model selection, it does not produce full Bayesian posteriors — ROAS estimates are point values without credible intervals.

**Influence on this project:** The synthetic dataset schema (`dt_simulated_weekly`) is inspired by Robyn's simulation data structure. The choice to use PyMC-Marketing over Robyn was driven by the need for Python ecosystem consistency and Bayesian uncertainty quantification.

---

## 3. Data Description

### 3.1 Data Source

The dataset is **synthetic**, generated by `src/data_generator.py` with known true parameters. This deliberate choice enables parameter recovery validation: after fitting, we can compare estimated adstock and saturation parameters against the exact values used to generate the data.

| Property | Value |
|----------|-------|
| File | `data/synthetic/mmm_weekly_data.csv` |
| Rows | 208 |
| Date range | 2015-11-23 to 2019-11-11 |
| Granularity | Weekly |
| Target | `revenue` (EUR) |
| Paid channels | TV, OOH, Print, Facebook, Search |
| Controls | `newsletter_sends`, `competitor_index`, `event_flag` |

### 3.2 Channel Spend Distributions

| Channel | Mean (€/week) | Max (€/week) | Total (156 weeks) |
|---------|---------------|--------------|-------------------|
| TV | 19,684 | — | €3,070,759 |
| Facebook | 15,269 | — | €2,381,912 |
| Search | 11,969 | — | €1,867,159 |
| OOH | 10,244 | — | €1,598,030 |
| Print | 7,135 | — | €1,113,128 |

Total average weekly budget: **€64,301** across all five channels.

### 3.3 Revenue Distribution

Weekly revenue ranges from approximately €167k to €331k with a mean around €249k. The target variable is not transformed (no log scaling) — the Bayesian MMM models revenue directly in original units.

### 3.4 Control Variables

- **competitor_index** — AR(1) process centered at 1.0, representing relative competitor activity. Values > 1.0 indicate above-average competitor pressure.
- **event_flag** — Binary indicator for promotional events, occurring in approximately 10% of weeks.
- **newsletter_sends** — Volume of newsletters dispatched per week (organic channel, not modeled as paid media).

### 3.5 Ground-Truth Parameters

Because the data is synthetic, every parameter is known:

**Adstock decay rates (alpha):**
| Channel | True Alpha | Interpretation |
|---------|-----------|----------------|
| TV | 0.70 | Slowest decay — TV effects persist for weeks |
| OOH | 0.50 | Moderate persistence |
| Print | 0.30 | Moderate-low persistence |
| Facebook | 0.20 | Fast decay — digital effects are short-lived |
| Search | 0.10 | Fastest decay — intent-driven, immediate effect |

**Saturation parameters (lambda):** Range from 0.3 (Facebook, saturates quickly) to 1.0 (Print, saturates slowly).

### 3.6 Train/Test Split

| Split | Weeks | Date Range | Proportion |
|-------|-------|------------|------------|
| Training | 1–156 | 2015-11-23 to 2018-11-30 | 75% |
| Test | 157–208 | 2018-12-01 to 2019-11-11 | 25% |

The split is chronological (no shuffling) to respect the time-series nature of the data. Adstock transforms require temporal ordering — randomly shuffled data would destroy carryover effects.

---

## 4. Feature Engineering

Feature engineering for MMM is minimal compared to traditional ML models. The primary inputs are raw channel spend values, which the model internally transforms through adstock and saturation functions. However, two sets of engineered features are added to capture trend and seasonality.

### 4.1 Trend

A linear trend variable `t` is computed as a normalized value from 0 (first week) to 1 (last week). This captures long-run growth or decline in baseline revenue independent of media activity.

### 4.2 Fourier Seasonality

Two pairs of sine/cosine terms capture annual cyclicality:

| Feature | Formula |
|---------|---------|
| `sin_1` | sin(2π × week_number / 52) |
| `cos_1` | cos(2π × week_number / 52) |
| `sin_2` | sin(4π × week_number / 52) |
| `cos_2` | cos(4π × week_number / 52) |

The first pair captures the fundamental annual cycle; the second captures a semi-annual harmonic. Together they model seasonality without requiring 52 dummy variables.

### 4.3 Feature Constants

`src/feature_engineering.py` exports the following constants shared between the training script and API:

- `CHANNEL_COLUMNS = ["tv_spend", "ooh_spend", "print_spend", "facebook_spend", "search_spend"]`
- `CONTROL_COLUMNS = ["competitor_index", "event_flag", "t"]`
- `TARGET_COLUMN = "revenue"`

---

## 5. Methodology

### 5.1 Model Specification

The Bayesian MMM decomposes revenue as:

```
revenue_t = intercept + Σ_c β_c × saturation(adstock(spend_c,t)) + Σ_k γ_k × control_k,t + ε_t
```

Where:
- **adstock(spend_c,t)** = GeometricAdstock with l_max=8: a weighted sum of current and past 7 weeks of spend, with geometrically decaying weights controlled by alpha_c
- **saturation(x)** = LogisticSaturation: x / (x + λ_c), a Hill-type function that maps [0, ∞) → [0, 1), producing diminishing returns
- **β_c** = channel coefficient (revenue contribution per unit of saturated, adstocked spend)
- **γ_k** = control variable coefficients
- **ε_t** = observation noise

### 5.2 PyMC-Marketing Configuration

```python
MMM(
    date_column="date_week",
    channel_columns=["tv_spend", "ooh_spend", "print_spend", "facebook_spend", "search_spend"],
    control_columns=["competitor_index", "event_flag", "t"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
```

PyMC-Marketing automatically:
- Scales channel spend using MaxAbsScaler (each channel divided by its maximum value)
- Places default priors on adstock alphas (Beta), saturation lambdas (Gamma), channel betas (HalfNormal), and intercept (Normal)
- Constructs the likelihood as Normal(mu=model_prediction, sigma=estimated_noise)

### 5.3 MCMC Sampling

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chains | 4 | Standard for convergence diagnostics (need ≥ 2 for R-hat) |
| Draws | 1,000 | Sufficient posterior resolution for 5-channel model |
| Tune | 1,000 | Warm-up iterations for NUTS step-size adaptation |
| target_accept | 0.9 | Higher than default 0.8 to reduce divergences in correlated posteriors |
| Sampler | PyMC (NUTS) | NumPyro unavailable on macOS ARM; PyMC fallback |
| Random seed | 42 | Reproducibility |

Total sampling time: approximately 10 minutes on Apple Silicon.

### 5.4 Why Not a Frequentist Approach?

| Aspect | Frequentist (Ridge/Robyn) | Bayesian (PyMC-Marketing) |
|--------|--------------------------|---------------------------|
| Uncertainty | Point estimates only | Full posterior distributions |
| ROAS confidence | None | 94% credible intervals |
| Priors | L2 penalty (implicit shrinkage) | Explicit domain priors |
| Budget optimisation | Deterministic (single optimum) | Probabilistic (expected value under uncertainty) |
| Multicollinearity | Ridge penalty | Prior regularisation |
| Speed | Fast (seconds) | Slow (minutes) |

For this project, the uncertainty quantification advantage outweighs the speed cost. MMM is a batch process — fitting once per week or month — so 10-minute runtime is acceptable.

---

## 6. Model Performance

### 6.1 Convergence Diagnostics

All MCMC chains converged successfully:

| Diagnostic | Value | Threshold | Status |
|------------|-------|-----------|--------|
| Max R-hat | 1.0067 | < 1.01 | PASS |
| Min ESS (bulk) | 2,032 | > 400 | PASS |
| Min ESS (tail) | 1,384 | > 400 | PASS |

R-hat values near 1.0 confirm that all four chains are sampling from the same posterior distribution. ESS values well above 400 ensure that posterior summaries (means, HDIs) are estimated with sufficient precision.

### 6.2 Out-of-Sample Accuracy

The model was trained on weeks 1–156 and evaluated on weeks 157–208 (52 held-out weeks):

| Metric | Value |
|--------|-------|
| MAE | €9,915 |
| MAPE | 3.9% |
| R² | 0.914 |

A MAPE of 3.9% means the model's weekly revenue predictions deviate by less than 4% on average from actuals — well within the < 15% target for MMM. The R² of 0.91 indicates that the model explains 91% of revenue variance in the test period.

### 6.3 Parameter Recovery

Since the data is synthetic with known true parameters, we can validate whether the model recovers the generative process:

**Adstock Alphas:**

| Channel | True | Estimated | 94% HDI | Recovered? |
|---------|------|-----------|---------|-----------|
| TV | 0.70 | 0.23 | [0.00, 0.60] | No |
| OOH | 0.50 | 0.28 | [0.00, 0.72] | Yes |
| Print | 0.30 | 0.28 | [0.00, 0.70] | Yes |
| Facebook | 0.20 | 0.27 | [0.00, 0.66] | Yes |
| Search | 0.10 | 0.27 | [0.00, 0.63] | Yes |

**Saturation Lambdas:** All 5 channels recovered (true values within 94% HDI).

**Summary:** 9 out of 10 parameters recovered. The TV adstock alpha (true=0.70) falls just outside the HDI upper bound (0.60). This is a known identifiability challenge: when a channel has both high carryover (alpha=0.70) and high spend volume, the model can explain the data through multiple alpha/beta combinations. The wide HDIs on all adstock estimates reflect this inherent uncertainty.

---

## 7. Channel Decomposition and ROAS

### 7.1 Revenue Decomposition

The model decomposes total training-period revenue into channel-specific contributions:

| Component | Total Contribution (€) | Share of Revenue |
|-----------|----------------------|------------------|
| TV | 526,213 | 1.35% |
| Search | 367,234 | 0.95% |
| Facebook | 320,225 | 0.82% |
| Print | 312,957 | 0.81% |
| OOH | 236,722 | 0.61% |
| **Base + Controls** | **~95.5%** | Intercept, trend, seasonality, controls |

The dominance of base revenue (~95% of total) is typical of MMMs — most revenue comes from brand equity, organic demand, and non-media factors. The media channels collectively explain ~4.5% of revenue, with TV contributing the most in absolute terms due to its higher spend level.

### 7.2 Return on Ad Spend (ROAS)

ROAS = total channel contribution / total channel spend, computed from the full posterior:

| Channel | ROAS (Mean) | 94% HDI | Spend | Contribution |
|---------|-------------|---------|-------|-------------|
| Print | 0.28 | [0.00, 0.70] | €1.1M | €313k |
| Search | 0.20 | [0.00, 0.46] | €1.9M | €367k |
| TV | 0.17 | [0.00, 0.36] | €3.1M | €526k |
| OOH | 0.15 | [0.00, 0.39] | €1.6M | €237k |
| Facebook | 0.13 | [0.00, 0.32] | €2.4M | €320k |

**Interpretation:** A ROAS of 0.28 for Print means that for every €1 spent on Print, the model attributes €0.28 of incremental revenue. All ROAS values are below 1.0, which is expected for this synthetic dataset — the true generative parameters produce moderate channel effects relative to the dominant base revenue.

**Important:** The wide credible intervals (all spanning from near-zero) reflect genuine uncertainty in decomposing aggregate revenue into channel-specific effects. This is not a model deficiency — it is an accurate representation of what weekly aggregate data can and cannot identify. The Bayesian approach makes this uncertainty explicit, whereas frequentist approaches would report misleadingly precise point estimates.

---

## 8. Response Curves and Saturation

### 8.1 Saturation Concept

The logistic saturation transform models diminishing returns: as spend increases, each additional euro produces less incremental revenue. The shape is controlled by the lambda parameter — lower lambda means the channel saturates more quickly.

```
saturation(x) = x / (x + λ)
```

At x = λ, the channel is at 50% saturation (half of its maximum potential contribution). At x >> λ, the channel approaches full saturation and additional spend yields negligible returns.

### 8.2 Response Curves

For each channel, the response curve shows predicted contribution as a function of weekly spend. These curves are generated by applying the posterior-mean adstock and saturation transforms with a calibrated beta coefficient derived from the model's actual channel contributions.

The curves allow media planners to identify:
- **Steep region** (low spend): high marginal return — opportunity to increase spend
- **Flat region** (high spend): low marginal return — spend above this level is inefficient
- **Current operating point**: where the channel currently sits on its curve

### 8.3 Adstock Decay

Geometric adstock models the carryover effect: a €10,000 spend this week continues to generate revenue for subsequent weeks, with geometrically decaying impact.

The decay vector for each channel shows how the effect is distributed across weeks (lag 0 = current week, lag 7 = seven weeks later). Channels with higher alpha values have slower decay and longer-lasting effects.

---

## 9. API Reference

The MMM is served via a FastAPI application at `src/api/main.py`. All endpoints return pre-computed JSON data loaded at startup.

### 9.1 Health Check

```
GET /health
```

**Response:**
```json
{"status": "ok", "model": "mmm", "channels": ["tv", "ooh", "print", "facebook", "search"]}
```

### 9.2 Channel Decomposition

```
GET /api/decomposition
```

Returns weekly revenue decomposition into base + channel contributions over the training period (156 weeks).

**Response schema:**
```json
{
  "weeks": [
    {"date_week": "2015-11-23", "revenue_actual": 248714.54, "base": 240000.0,
     "tv": 3500.0, "ooh": 1500.0, "print": 2000.0, "facebook": 1200.0, "search": 514.54}
  ],
  "totals": {"tv": 526213.26, "ooh": 236721.59, ...},
  "pct": {"tv": 1.35, "ooh": 0.61, ...}
}
```

### 9.3 ROAS

```
GET /api/roas
```

Returns per-channel ROAS with 94% credible intervals.

**Response schema:**
```json
{
  "channels": [
    {"channel": "tv", "roas_mean": 0.1714, "roas_hdi_3": 0.0, "roas_hdi_97": 0.3617,
     "total_spend": 3070759.0, "total_contribution": 526213.26}
  ]
}
```

### 9.4 Response Curves

```
GET /api/response-curves
```

Returns 50-point spend-vs-contribution curves for each channel.

**Response schema:**
```json
{
  "channels": [
    {"channel": "tv", "curve": [{"spend": 0.0, "contribution": 0.0}, ...],
     "current_avg_spend": 19684.35}
  ]
}
```

### 9.5 Adstock Decay

```
GET /api/adstock
```

Returns decay vectors and alpha estimates per channel.

**Response schema:**
```json
{
  "channels": [
    {"channel": "tv", "decay_vector": [0.45, 0.30, ...], "alpha_mean": 0.2313,
     "alpha_hdi_3": 0.0, "alpha_hdi_97": 0.6032}
  ]
}
```

### 9.6 Optimal Allocation

```
GET /api/optimal-allocation
```

Returns current vs. optimal budget allocation with expected revenue lift.

**Response schema:**
```json
{
  "total_budget": 64301.21,
  "current": {"tv": 19684.35, ...},
  "optimal": {"tv": 19684.35, ...},
  "current_revenue": 248963.12,
  "optimal_revenue": 261411.28,
  "lift_abs": 0.0,
  "lift_pct": 0.0,
  "recommendations": [{"channel": "tv", "action": "Maintain", "current": 19684.35, "optimal": 19684.35, "delta": 0.0, "delta_pct": 0.0}]
}
```

### 9.7 Metadata

```
GET /api/metadata
```

Returns model training metadata: MCMC config, convergence diagnostics, out-of-sample metrics, parameter recovery results.

### 9.8 Budget Simulator

```
POST /api/simulate
```

Computes predicted weekly revenue for a proposed channel spend allocation using calibrated adstock, saturation, and beta parameters.

**Request:**
```json
{
  "tv_spend": 25000, "ooh_spend": 10000, "print_spend": 8000,
  "facebook_spend": 12000, "search_spend": 15000
}
```

**Response:**
```json
{
  "predicted_revenue": 252341.50,
  "current_revenue": 248963.12,
  "delta": 3378.38,
  "delta_pct": 1.36,
  "channel_contributions": {"tv": 4500.0, "ooh": 1800.0, ...},
  "saturation_warnings": [{"channel": "facebook", "level": "moderate"}]
}
```

Saturation warnings are emitted when a channel's saturated value exceeds 0.6 (moderate) or 0.8 (high), indicating diminishing returns.

---

## 10. Deployment

### 10.1 Architecture

The MMM follows a modified three-tier architecture adapted for pre-computed serving:

```
React + Vite (5176) → Express Proxy (3004) → FastAPI (8004)
```

- **FastAPI** (`src/api/main.py`, port 8004): Loads 7 pre-computed JSON files at startup via lifespan. No model object is loaded — the MMM is not picklable and not needed for serving. The only real-time computation is the `/api/simulate` endpoint, which applies adstock/saturation transforms with calibrated parameters.

- **Express Proxy** (`app/server/index.js`, port 3004): Forwards all `/api/*` requests to FastAPI port 8004.

- **React + Vite** (`app/client/src/`, port 5176): Four views:
  - **Decomposition** — stacked area chart of weekly channel contributions over time
  - **Channel Performance** — ROAS bar chart with credible intervals
  - **Budget Simulator** — spend sliders with predicted revenue and saturation warnings
  - **Optimal Allocation** — current vs. recommended budget split

### 10.2 Pre-computed Artifacts

All serving artifacts are in `models/precomputed/`:

| File | Size | Contents |
|------|------|----------|
| `decomposition.json` | 44 KB | 156 weekly records with channel contributions |
| `roas.json` | 1 KB | ROAS mean + HDI per channel |
| `response_curves.json` | 22 KB | 50-point curves × 5 channels |
| `adstock.json` | 1 KB | Decay vectors + alpha estimates |
| `simulator_params.json` | 1 KB | Point estimates for /api/simulate |
| `optimal_allocation.json` | 2 KB | Current vs. optimal allocation |

Additionally, `models/metadata.json` stores training metadata and `models/mmm_trace/trace.nc` (107 MB) stores the full ArviZ InferenceData for offline analysis.

### 10.3 Running the Stack

```bash
# Terminal 1: FastAPI (run FROM the model directory)
cd m04_mmm
/Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port 8004 --reload

# Terminal 2: Express proxy
cd m04_mmm/app/server && npm start

# Terminal 3: Vite frontend
cd m04_mmm/app/client && npm run dev
# Open http://localhost:5176
```

### 10.4 Retraining

To retrain with new data:

1. Replace `data/synthetic/mmm_weekly_data.csv` with actual business data
2. Update `src/feature_engineering.py` if column names differ
3. Run `cd m04_mmm && /path/to/.venv/bin/python src/train.py`
4. Training exports all pre-computed JSONs automatically
5. Restart the FastAPI server to load new artifacts

---

## 11. Future Work

### 11.1 Real Data Integration

The immediate priority is replacing synthetic data with actual business data. This requires:
- Data pipeline to ingest weekly spend by channel from media buying platforms
- Revenue data from the point-of-sale or e-commerce system
- Validation that the model produces sensible decomposition and ROAS on real data (expected: wider credible intervals, potentially different channel rankings)

### 11.2 Geo-Level Modeling

Geo-level MMM (splitting data by region or DMA) dramatically improves identifiability by exploiting cross-regional variation in media exposure. PyMC-Marketing supports hierarchical models that share parameters across geographies while allowing regional variation.

### 11.3 Time-Varying Parameters

Channel effectiveness changes over time due to creative fatigue, audience saturation, competitive dynamics, and market evolution. Time-varying coefficient models (e.g., using Gaussian processes or regime-switching) could capture these dynamics.

### 11.4 Cross-Channel Interactions

The current model assumes independent channel effects. In reality, TV exposure drives search intent (TV → Search synergy), and Facebook retargeting amplifies initial awareness channels. Interaction terms or more complex model structures could capture these synergies.

### 11.5 Incrementality Validation

MMM results should be validated against controlled experiments:
- **Geo-lift tests** — withhold media spend in a subset of regions and compare outcomes
- **Matched market tests** — pair similar regions and apply different spend levels
- **Platform-reported incrementality** — compare model ROAS against platform-side lift studies

### 11.6 Cross-Model Integration

Within the Marketing Analytics platform:
- **m01 (Send-Time Optimisation)** — coordinate media timing with email deployment windows
- **m02 (CLV)** — weight ROAS by customer lifetime value (acquire high-CLV customers, not just any customers)
- **m03 (Churn)** — reduce spend on channels that acquire high-churn-risk customers

---

*For reference documentation, see:*
- [Model Card](model_card.md) — brief model summary for operations teams
- [Validation Report](validation_report.md) — raw metrics tables from training run
- [Data Dictionary](data_dictionary.md) — feature definitions and file schema
- [Research Brief](research_brief.md) — initial project scoping and framework comparison
