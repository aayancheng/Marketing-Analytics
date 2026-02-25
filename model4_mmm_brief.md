# Project Brief: Model 4 — Marketing Mix Modeling (MMM)
**Version:** 1.0 | **Stack:** Python (ML) + React/Node.js (App) | **License:** MIT (Robyn dataset)

---

## 1. Project Overview

Build a Bayesian Marketing Mix Model that quantifies how much each marketing channel (TV, out-of-home, print, Facebook, paid search, newsletter) contributed to weekly revenue — and uses that attribution to recommend an optimal budget allocation. The model uses PyMC-Marketing (Python-native Bayesian MMM) as the primary framework, with Meta's Robyn simulated weekly dataset (`dt_simulated_weekly`) as the data source.

This is the most statistically sophisticated of the five projects and demonstrates Claude Code's ability to handle Bayesian time-series modeling, adstock transformations, diminishing returns curves, and budget optimization in a single autonomous build.

The project delivers:
1. A fitted Bayesian MMM via PyMC-Marketing with geometric adstock and logistic saturation
2. Channel contribution decomposition and ROAS estimates per channel
3. A budget optimization output (optimal spend allocation)
4. A React/Node.js SPA with channel contribution charts, response curves, and a budget simulator
5. Full methodology documentation

**Business framing:** A German-market retailer (per dataset country context) wants to understand which of its five paid media channels is driving revenue, where spend is saturating, and how to reallocate a fixed quarterly budget to maximize total revenue.

---

## 2. Dataset

### 2a. Primary Dataset — Robyn `dt_simulated_weekly`
| Property | Value |
|---|---|
| **Source** | Meta's Robyn R package (built-in simulated dataset) |
| **Access (R)** | `data("dt_simulated_weekly")` after `library(Robyn)` |
| **Access (Python)** | Export from R or use PyMC-Marketing's own synthetic data generator |
| **Preferred approach** | Generate with PyMC-Marketing's `data_generator()` function — full control over parameters |
| **Frequency** | Weekly |
| **Date range** | 2015-11-23 to 2019-11-11 (~208 weeks / 4 years) |
| **Rows** | ~208 weekly observations |
| **License** | MIT (Robyn), free to use |

### 2b. Robyn `dt_simulated_weekly` Column Schema
| Column | Type | Description | Variable Role |
|---|---|---|---|
| `DATE` | date | Week start date (YYYY-MM-DD) | Date index |
| `revenue` | float | **TARGET** — weekly revenue (EUR) | Dependent variable |
| `tv_S` | float | TV spend (EUR) | Paid media spend |
| `ooh_S` | float | Out-of-home (billboard) spend (EUR) | Paid media spend |
| `print_S` | float | Print advertising spend (EUR) | Paid media spend |
| `facebook_S` | float | Facebook advertising spend (EUR) | Paid media spend |
| `search_S` | float | Paid search spend (EUR) | Paid media spend |
| `facebook_I` | float | Facebook impressions | Paid media exposure metric |
| `search_clicks_P` | float | Paid search clicks | Paid media exposure metric |
| `newsletter` | float | Newsletter sends (volume) | Organic media variable |
| `competitor_sales_B` | float | Competitor sales index | Context/control variable |
| `events` | string/factor | Promotional event indicator | Context variable (categorical) |

### 2c. PyMC-Marketing Synthetic Data Generator (Recommended Alternative)
Since the Robyn dataset requires R to extract, Claude Code should use PyMC-Marketing's built-in generator for full Python-native workflow:

```python
import pandas as pd
import numpy as np
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from sklearn.preprocessing import MaxAbsScaler

def data_generator(
    start_date="2015-11-23",
    periods=208,              # 4 years weekly
    channels=["tv", "ooh", "print", "facebook", "search"],
    spend_scalar=[0.3, 0.15, 0.1, 0.25, 0.2],      # share of total spend
    adstock_alphas=[0.7, 0.5, 0.3, 0.2, 0.1],       # decay rates per channel
    saturation_lamdas=[0.5, 0.8, 1.0, 0.3, 0.4],   # saturation steepness
    betas=[2.5, 1.2, 0.8, 1.8, 2.0],               # channel effect sizes
    freq="W"
):
```

**Generated dataset schema (Python-native):**
| Column | Description |
|---|---|
| `date_week` | Weekly date index |
| `y` | Revenue (target, log-normally distributed with trend + seasonality) |
| `tv_spend` | TV channel spend |
| `ooh_spend` | OOH channel spend |
| `print_spend` | Print channel spend |
| `facebook_spend` | Facebook channel spend |
| `search_spend` | Search channel spend |
| `newsletter_sends` | Organic newsletter volume |
| `competitor_index` | External competitor sales index |
| `event_flag` | Binary promotional event flag |

**True parameters to use for generation (enables parameter recovery validation):**
```python
TRUE_ADSTOCK_ALPHAS = {
    'tv': 0.70,       # longest carryover — TV brand awareness
    'ooh': 0.50,      # medium carryover
    'print': 0.30,    # shorter carryover
    'facebook': 0.20, # short carryover — direct response
    'search': 0.10    # near-instantaneous — high intent channel
}

TRUE_SATURATION_LAMDAS = {
    'tv': 0.50,       # moderate saturation
    'ooh': 0.80,      # higher saturation (limited inventory)
    'print': 1.00,    # highly saturated
    'facebook': 0.30, # less saturated (broad audience)
    'search': 0.40    # moderate saturation
}
```

### 2d. Train/Test Split Strategy (Time-Based)
```
Training window:   2015-11-23 → 2018-11-30  (~156 weeks, 75%)
Test window:       2018-12-01 → 2019-11-11  (~52 weeks, 25%)
```
Hold-out test set used only for out-of-sample revenue prediction accuracy (MAPE). All adstock/saturation parameters are estimated on training data only.

---

## 3. Modeling Target

| Property | Value |
|---|---|
| **Dependent variable** | `revenue` — weekly revenue (EUR) |
| **Model objective** | Decompose revenue into: base (intercept + trend + seasonality) + channel contributions + context effects |
| **Key outputs** | Channel-level contribution (£/week), ROAS per channel, response curves, optimal budget allocation |
| **Prediction task** | Revenue attribution (not forecasting) — though out-of-sample MAPE is reported as a secondary quality check |

---

## 4. Model Specification

### 4a. Bayesian MMM Structure (PyMC-Marketing)

```
revenue_t = intercept
           + trend_t
           + seasonality_t (Fourier terms, yearly_seasonality=2)
           + Σ_c [ beta_c * saturation(adstock(spend_ct)) ]
           + gamma_competitor * competitor_index_t
           + gamma_event * event_flag_t
           + ε_t
```

Where for each channel c:
- `adstock(spend_ct)` = Geometric Adstock with decay parameter `alpha_c` and max lag `l_max=8` weeks
- `saturation(x)` = Logistic Saturation: `x / (x + lambda_c)`

### 4b. PyMC-Marketing Configuration
```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior
import numpy as np

# Prior configuration (weakly informative)
model_config = {
    "intercept": Prior("Normal", mu=0.5, sigma=0.2),
    "saturation_beta": Prior("HalfNormal", sigma=prior_sigma),  # prior_sigma from spend shares
    "gamma_control": Prior("Normal", mu=0, sigma=0.05),
    "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
    "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
}

mmm = MMM(
    model_config=model_config,
    date_column="date_week",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    channel_columns=["tv_spend", "ooh_spend", "print_spend", "facebook_spend", "search_spend"],
    control_columns=["competitor_index", "event_flag", "t"],  # t = linear trend
    yearly_seasonality=2,
)
```

### 4c. MCMC Sampler Configuration
```python
mmm.fit(
    X=X_train,
    y=y_train,
    target_accept=0.9,
    chains=4,
    draws=2000,      # reduce to 1000 for faster iteration
    tune=1000,
    nuts_sampler="numpyro",  # faster than pymc default; fallback to "pymc"
    random_seed=42,
)
```

**Estimated fit time:** 5–15 minutes on a standard CPU (4 chains × 2,000 draws). Use `draws=500` for rapid prototyping.

### 4d. Key Outputs to Extract Post-Fitting
| Output | How to Extract |
|---|---|
| Channel contributions (mean + HDI) | `mmm.plot_channel_contributions()` or `mmm.get_channel_contributions_ref()` |
| ROAS per channel | `sum(channel_contribution) / sum(channel_spend)` per channel |
| Adstock alpha posterior | `mmm.fit_result["adstock_alpha"].mean(dim=["chain","draw"])` |
| Saturation lambda posterior | `mmm.fit_result["saturation_lam"].mean(dim=["chain","draw"])` |
| Response curves | `mmm.plot_direct_contribution_curves()` |
| Out-of-sample MAPE | `mmm.predict(X_test)` → compare to y_test |

### 4e. Budget Optimization
After fitting, run PyMC-Marketing's budget allocator:

```python
# Find optimal allocation for a fixed total budget
total_budget = X_train[channel_columns].sum().sum() / len(X_train)  # avg weekly budget

optimal_budget = mmm.optimize_budget(
    budget=total_budget,
    num_periods=52,        # 1 year forward
    response_variable="revenue",
)
# Returns: optimal spend per channel, expected revenue lift vs. current allocation
```

### 4f. Model Validation
| Check | Method |
|---|---|
| Prior predictive check | `mmm.sample_prior_predictive()` → verify priors produce plausible revenue range |
| MCMC convergence | R-hat < 1.01 for all parameters, ESS > 400 |
| Parameter recovery | Compare estimated adstock alphas to true values used in data generation |
| Out-of-sample accuracy | MAPE on held-out 52 weeks (target: < 15%) |
| Decomposition sanity | Base + all channel contributions should sum to ~actual revenue |

---

## 5. App Requirements — React/Node.js SPA

### 5a. Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS |
| Backend API | Node.js + Express |
| Charts | Recharts (area chart, bar chart, line chart) |
| ML serving | Pre-computed results served as JSON (MMM fit too slow for real-time inference) |
| Model artifact | Fitted `mmm` object serialized via `arviz` + numpy arrays for contribution lookup |

### 5b. App Layout — Four Panels

**Panel 1 — Revenue Decomposition (Waterfall/Area Chart)**
- Stacked area chart: weekly revenue broken into base + TV + OOH + Print + Facebook + Search + Newsletter + Other
- Date range filter (slider to zoom in on any time window)
- Pie chart: % revenue attributed to each channel over full period
- KPI row: Total Revenue | Base Revenue % | Paid Media Revenue % | Top Channel

**Panel 2 — Channel Performance (ROAS + Response Curves)**
- Bar chart: ROAS by channel with 95% credible interval error bars
- Response curve panel: one S-curve per channel showing marginal return vs. spend (generated from logistic saturation parameters)
- Adstock decay chart: visualize how each channel's effect decays over 8-week window
- Highlight: "Most efficient channel" (highest ROAS) vs. "Most saturated channel" (flattest curve)

**Panel 3 — Budget Simulator (What-If Sliders)**
This is the highest-value panel for the business audience:

| Slider | Range | Default |
|---|---|---|
| TV Spend (weekly EUR) | €0 – €50,000 | Current avg |
| OOH Spend (weekly EUR) | €0 – €30,000 | Current avg |
| Print Spend (weekly EUR) | €0 – €20,000 | Current avg |
| Facebook Spend (weekly EUR) | €0 – €40,000 | Current avg |
| Search Spend (weekly EUR) | €0 – €40,000 | Current avg |
| Total Budget Lock (toggle) | Lock/Unlock | Locked |

- When "Total Budget Lock" is ON: adjusting one channel auto-scales others proportionally
- Output: Predicted revenue at proposed allocation vs. current allocation
- Revenue uplift indicator: "+€X,XXX/week (+X%)"
- Warning indicator when a channel enters saturation zone (spend > 80% of saturation point)

**Panel 4 — Optimal Allocation Recommendation**
- Side-by-side bar chart: Current allocation vs. Optimal allocation (from `optimize_budget()`)
- Expected revenue lift from reallocation: "Reallocating budget as recommended could lift weekly revenue by +€X,XXX"
- Channel-level recommendation table: Increase / Decrease / Maintain per channel with magnitude

### 5c. API Endpoints
```
GET  /api/decomposition         → weekly revenue decomposition by channel (pre-computed)
GET  /api/roas                  → ROAS + credible intervals per channel
GET  /api/response-curves       → saturation curve data per channel (x=spend, y=contribution)
GET  /api/adstock               → adstock decay vectors per channel
POST /api/simulate              → body: {channel_spends} → predicted revenue
GET  /api/optimal-allocation    → pre-computed optimal budget allocation
```

**Note:** MMM inference is pre-computed at model training time. The API serves pre-computed numpy arrays and lookup tables, not live MCMC — this ensures sub-100ms response times in the app.

---

## 6. Model Documentation Requirements

| Document | Content |
|---|---|
| **Model Card** | Purpose, intended users (CMO, media planning team), dataset provenance (Robyn simulated — flag clearly), performance metrics (MAPE, R²), known limitations (observational data — cannot prove causality, holiday effects not modeled in v1) |
| **Methodology** | Bayesian MMM theory, adstock explanation with decay curve diagrams, saturation/diminishing returns explanation, how ROAS is computed from posteriors, MCMC convergence diagnostics |
| **Parameter Recovery Report** | Table: true adstock alphas vs. estimated posteriors (mean + 95% HDI); true saturation lambdas vs. estimated — demonstrates model correctness |
| **Budget Optimization Methodology** | How `optimize_budget()` works (gradient-based optimization on response curves), assumptions, and limitations |
| **Format** | Markdown in `/docs` + rendered in app's Methodology tab |

---

## 7. Constraints and Assumptions

- Simulated dataset — explicitly flag throughout documentation that results are from synthetic data; real-world MMM requires 2+ years of actual marketing + revenue data
- Weekly granularity only (daily data is too noisy for this dataset)
- MCMC sampling is computationally intensive — use `numpyro` sampler for speed; Claude Code should use `draws=500` for development, `draws=2000` for final fit
- Budget optimization assumes stationarity of adstock/saturation parameters — real-world budgets require re-fitting the model on fresh data periodically
- Robyn's `dt_simulated_weekly` uses EUR currency and German market context (DE holidays)
- PyMC-Marketing version: pin to `pymc-marketing>=0.11.0`; API changed significantly in earlier versions

---

## 8. Deliverables

```
project/
├── data/
│   └── generated/mmm_weekly_data.csv   # PyMC-Marketing generated dataset
├── notebooks/
│   └── 01_mmm_exploration.ipynb
├── src/
│   ├── data_generator.py               # Synthetic data generation script
│   ├── train_mmm.py                    # PyMC-Marketing model fit + export results
│   ├── budget_optimizer.py             # Budget allocation optimization
│   └── api/main.py                     # Node.js API serves pre-computed results
├── app/
│   ├── server/                         # Node.js/Express
│   └── client/                         # React + Tailwind
├── models/
│   ├── mmm_trace/                      # ArviZ InferenceData (fitted MCMC trace)
│   └── precomputed/
│       ├── decomposition.json          # Weekly channel contributions
│       ├── roas.json                   # ROAS + credible intervals
│       ├── response_curves.json        # Spend → contribution lookup
│       └── optimal_allocation.json     # Budget optimizer output
├── docs/
│   ├── model_card.md
│   ├── methodology.md
│   ├── parameter_recovery.md
│   └── budget_optimization.md
├── requirements.txt
├── package.json
└── README.md
```

---

## 9. Getting Started Commands for Claude Code

```bash
# Install Python deps — note: arviz + pymc-marketing have heavy dependencies
pip install pymc-marketing>=0.11.0 arviz numpyro jax jaxlib numpy pandas scikit-learn fastapi uvicorn joblib matplotlib

# If numpyro install fails, fallback to pymc sampler (slower but works everywhere)
pip install pymc-marketing arviz numpy pandas scikit-learn fastapi uvicorn

# Install Node + React
npm install express cors axios
npx create-react-app app/client
cd app/client && npm install recharts tailwindcss lucide-react
```

**Important for Claude Code:** PyMC-Marketing's MMM fit is slow. Run data generation and model training as separate scripts that save output to JSON. The React app should only ever read pre-computed JSON — never trigger live MCMC from the frontend.

---

## 10. Key References

- PyMC-Marketing documentation: https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html
- End-to-end case study: https://www.pymc-marketing.io/en/0.11.0/notebooks/mmm/mmm_case_study.html
- Robyn (Meta): https://github.com/facebookexperimental/Robyn
- Core paper: Jin, Y. et al. (2017). "Bayesian methods for media mix modeling with carryover and shape effects." Google Inc.
- Robyn simulated dataset columns confirmed via: https://facebookexperimental.github.io/Robyn/docs/features/
