# Model Card -- Marketing Mix Model (MMM)

**Project:** Marketing Analytics -- m04_mmm
**Version:** 1.0 | **Date:** 2026-03-08
**Model:** Bayesian Marketing Mix Model (PyMC-Marketing)

---

## Purpose

Quantify the incremental revenue contribution of each paid media channel (TV, Out-of-Home, Print, Facebook, Search) and provide actionable budget reallocation recommendations to maximise return on advertising spend. The model decomposes weekly revenue into base demand, channel-specific media effects (after accounting for adstock carryover and saturation), seasonal patterns, and control variables (competitor activity, promotional events).

## Intended Users

- **CMOs and marketing leadership** -- understand which channels drive revenue and where to shift budget for maximum impact
- **Media planners and buyers** -- use channel-level ROAS estimates and saturation curves to inform weekly allocation decisions
- **Finance and planning** -- quantify the revenue contribution attributable to marketing spend versus organic demand
- **Analytics teams** -- validate media effectiveness hypotheses with full posterior uncertainty quantification

## Training Data

- **Source:** Synthetic dataset with known ground-truth parameters, inspired by Meta Robyn `dt_simulated_weekly`
- **Observations:** 208 weeks (2015-11-23 to 2019-11-11)
- **Granularity:** Weekly
- **Train/test split:** 156 weeks training (75%) / 52 weeks test (25%), chronological
- **Target variable:** Revenue (EUR per week)
- **Media channels:** TV, OOH (Out-of-Home), Print, Facebook, Search
- **Controls:** Competitor sales index (AR(1) process), promotional event flag, linear trend, Fourier seasonality (2 harmonic pairs)

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Out-of-sample MAPE | 3.9% | < 15% |
| Out-of-sample R-squared | 0.91 | -- |
| Out-of-sample MAE | 9,915 EUR | -- |
| Max R-hat (convergence) | 1.0067 | < 1.01 |
| Min ESS (bulk) | 2,032 | > 400 |
| Min ESS (tail) | 1,384 | > 400 |
| Parameter recovery (in 95% HDI) | 9/10 parameters | all in HDI |

## Key Outputs

- **Channel decomposition** -- weekly revenue attributed to each channel plus base demand, as a stacked time series
- **ROAS estimates** -- return on ad spend per channel with 94% credible intervals from the full posterior
- **Response curves** -- spend-vs-contribution curves showing diminishing returns and current saturation levels
- **Adstock decay vectors** -- estimated carryover profiles showing how long each channel's effect persists
- **Budget optimisation** -- recommended reallocation of the current total weekly budget across channels to maximise predicted revenue

## Known Limitations

- **Synthetic data.** The dataset is generated with known parameters for validation purposes. Results demonstrate model capability but do not reflect real market dynamics. Production deployment requires retraining on actual business data.
- **Weekly granularity.** The model cannot capture within-week effects (e.g., day-of-week patterns, hourly campaign pacing). Daily or sub-daily optimisation is out of scope.
- **No interaction effects.** The model assumes independent channel effects. Synergies between channels (e.g., TV driving search volume) are not modelled. This may understate the true contribution of channels that operate primarily through cross-channel amplification.
- **Static coefficients.** Channel effectiveness is assumed constant over the 4-year observation window. Time-varying parameters (e.g., creative fatigue, market entry of new competitors) are not captured.
- **Single-market scope.** The model covers one brand in one market. Multi-market or multi-product extensions would require separate models or hierarchical structures.
- **TV adstock recovery.** The true TV adstock alpha (0.70) falls just outside the 95% HDI upper bound (0.60), suggesting some identifiability challenges for high-carryover channels. All other 9 parameters are successfully recovered.

## Ethical Considerations

- **Budget allocation bias.** The model's recommendations may systematically favour channels with cleaner measurement signals (e.g., digital search) over channels with longer, harder-to-measure effects (e.g., brand TV). Users should apply domain judgment when interpreting relative ROAS estimates.
- **Over-reliance risk.** MMM is one input to media planning, not a replacement for strategic judgment. The model cannot capture qualitative factors such as brand building, creative quality, or competitive positioning.
- **Synthetic validation only.** Deploying this model to make real budget decisions without retraining on actual data would be inappropriate. The current results demonstrate methodology, not market truth.
- **No individual-level data.** The model operates on aggregate weekly data and does not process or store any personally identifiable information.
