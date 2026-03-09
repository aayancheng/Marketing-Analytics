# Research Brief: Marketing Mix Modeling (m04_mmm)

**Date:** 2026-03-06
**Status:** Planning
**Framework:** PyMC-Marketing (Bayesian)

---

## Domain Overview

Marketing Mix Modeling (MMM) is a statistical approach that measures the incremental impact of marketing channels on a business outcome -- in this case, weekly revenue. Unlike attribution models that rely on user-level tracking, MMM operates on aggregate time-series data and is resilient to cookie deprecation and privacy restrictions.

**Business value.** MMM answers two core questions: (1) how much did each channel contribute to revenue? and (2) how should budget be reallocated to maximise return? These answers feed directly into media planning and CFO-level investment decisions.

**Industry context.** The modelled entity is a German-market retailer investing across five paid media channels (TV, Out-of-Home, Print, Facebook, Search) with an owned newsletter channel and exposure to competitor activity and promotional events.

**Why Bayesian.** A Bayesian formulation provides several advantages over frequentist alternatives:

- **Uncertainty quantification** -- every parameter estimate (including ROAS) comes with credible intervals, not just point estimates.
- **Prior incorporation** -- domain knowledge about adstock decay rates and saturation curves can be encoded as informative priors.
- **Multicollinearity handling** -- regularisation through priors is more principled than ridge/lasso penalties when channels are correlated in spend timing.
- **Full posterior** -- enables probabilistic budget optimisation (optimise expected revenue subject to a risk constraint).

---

## KPIs and Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Out-of-sample MAPE | < 15% on held-out 52 weeks | Demonstrates predictive accuracy on unseen periods |
| MCMC convergence (R-hat) | < 1.01 for all parameters | Standard diagnostic; values above 1.01 indicate non-convergence |
| Effective sample size (ESS) | > 400 for all parameters | Ensures posterior estimates are reliable |
| Parameter recovery | True adstock/saturation values within 95% HDI | Validates that the model can recover known generative parameters |
| Decomposition sanity | Channel contributions + base ~= actual revenue | Ensures the decomposition is internally consistent and exhaustive |

---

## Data Source

The dataset is **synthetic**, generated with known true parameters. This is a deliberate choice: it enables parameter recovery validation, which is impossible with real-world data where ground truth is unknown.

| Property | Value |
|----------|-------|
| Observations | 208 weeks |
| Date range | 2015-11-23 to 2019-11-11 |
| Granularity | Weekly |
| Target variable | Revenue (EUR) |
| Paid channels | TV, OOH (Out-of-Home), Print, Facebook, Search |
| Controls | Newsletter sends, competitor sales index, promotional event flag |
| Schema inspiration | Meta Robyn `dt_simulated_weekly` |

Each paid channel has a known true adstock decay rate and saturation curve, allowing posterior recovery checks against ground truth.

---

## Model Candidates

| Framework | Type | Pros | Cons |
|-----------|------|------|------|
| PyMC-Marketing | Bayesian (Python) | Native Python, full posterior, built-in adstock/saturation transforms, budget optimiser | Heavy dependencies (PyMC/JAX), slower fitting |
| Meta Robyn | Frequentist/Ridge (R) | Industry standard, fast, multi-objective optimisation | R-only, no true posteriors, harder to integrate into Python stack |
| Google LightweightMMM | Bayesian (Python/JAX) | Google-backed, JAX-native | Deprecated in favour of Meridian, limited maintenance |
| Meridian (Google) | Bayesian (Python) | Google's successor to LightweightMMM | Very new, limited documentation and community support |

**Decision: PyMC-Marketing.** It is Python-native (consistent with the rest of this monorepo), produces full Bayesian posteriors with uncertainty quantification, includes a built-in budget optimiser, has a well-documented API, and is under active development. The slower fitting time is acceptable because MMM is a batch process, not a real-time scoring model.

---

## Recommended Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Model | PyMC-Marketing MMM | GeometricAdstock + LogisticSaturation transforms |
| Sampler | numpyro (JAX-based) | Fast NUTS sampling; pymc fallback if JAX unavailable |
| Diagnostics | ArviZ | Posterior plots, trace plots, R-hat, ESS, LOO-CV |
| Serving | FastAPI with pre-computed JSON | MMM is too slow for real-time inference; results are computed offline and served as static JSON payloads |
| Frontend | React + Vite + TailwindCSS | Four views (see below) |
| Charts | Recharts | Consistent with m01--m03 |

**Frontend views (4 instead of standard 3):**

1. **Decomposition** -- stacked area chart showing weekly revenue decomposed into base + channel contributions over time.
2. **Channel Performance** -- bar chart of channel-level ROAS with credible intervals; cost vs. contribution scatter.
3. **Budget Simulator** -- sliders to adjust channel spend; shows predicted revenue change with uncertainty bands.
4. **Optimal Allocation** -- displays the optimiser's recommended budget split versus current allocation, with expected revenue uplift.

---

## Reference Patterns

This model diverges from the m01--m03 pattern in several important ways:

- **No per-entity scoring.** MMM is an aggregate time-series model. There are no individual customers or transactions to look up.
- **No SHAP values.** Channel contribution is derived directly from the Bayesian decomposition (posterior mean of each channel's transformed spend multiplied by its coefficient). SHAP is unnecessary and inapplicable here.
- **Pre-computed serving.** The FastAPI layer loads pre-computed JSON results at startup rather than running live model inference. Endpoints serve decomposition data, channel summaries, and optimisation results.
- **Budget optimiser.** PyMC-Marketing's `optimize_budget` method is run offline. Results are serialised and served via the API.

---

## Key References

1. Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Inc.
2. PyMC-Marketing documentation: MMM module and budget optimisation case studies. https://www.pymc-marketing.io/
3. Meta Robyn open-source MMM framework (dataset schema inspiration). https://github.com/facebookexperimental/Robyn
4. Vehtari, A., Gelman, A., and Gabry, J. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." Statistics and Computing, 27(5), 1413--1432.
