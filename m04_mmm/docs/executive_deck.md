---
marp: true
theme: default
paginate: true
header: "**Marketing Mix Model** | Marketing Analytics"
footer: "m04_mmm | Confidential"
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  section.lead h1 {
    font-size: 2.5em;
  }
  table {
    font-size: 0.85em;
  }
  .metric {
    font-size: 3em;
    font-weight: bold;
    color: #2563eb;
    text-align: center;
    margin: 0.5em 0;
  }
---

<!-- _class: lead -->

# Marketing Mix Model

### Measuring channel ROI and optimising media spend with Bayesian analytics

**Marketing Analytics | m04_mmm**
March 2026

---

## Executive Summary

> **Every euro of media spend now comes with a credible ROI estimate and a diminishing-returns curve — enabling data-driven budget reallocation across five channels.**

| | Without Model | With Model |
|--|---------------|------------|
| **Channel attribution** | Last-click or gut feel | Bayesian decomposition with uncertainty |
| **ROAS confidence** | Point estimates, no intervals | 94% credible intervals per channel |
| **Budget decisions** | "Spend what we spent last year" | Optimised allocation with saturation awareness |
| **Carryover effects** | Ignored | Adstock decay modelled per channel |
| **Diminishing returns** | Unknown | Response curves per channel |

---

## The Problem: Flying Blind on Media Spend

Marketing teams spend **€64k per week** across five channels — but cannot answer basic questions:

- **Which channels are working?** Last-click attribution credits Search for conversions that TV awareness created
- **Are we overspending?** Every channel has diminishing returns, but without saturation curves, there is no way to know where the plateau begins
- **What is the carryover?** A TV ad seen this week may drive a purchase next week — but that revenue is attributed to whatever was clicked at checkout

Without a Marketing Mix Model, budget decisions are based on **historical precedent** ("spend what we spent last year") rather than **measured return**.

> The cost of inaction is not just wasted spend — it is the opportunity cost of misallocated budget across channels.

---

## The Solution: Bayesian Marketing Mix Model

A **Bayesian MMM** decomposes weekly revenue into the contribution of each media channel, after accounting for:

- **Adstock (carryover)** — TV ad effects persist for weeks; search effects are immediate
- **Saturation (diminishing returns)** — each additional euro produces less incremental revenue
- **Controls** — competitor activity, promotional events, seasonality, and trend

**Five channels modelled:** TV, Out-of-Home, Print, Facebook, Search

**Key advantage over frequentist approaches:** Every estimate comes with a **credible interval**, not just a point value. When the model says "Print ROAS is 0.28 [0.00, 0.70]", the width of the interval tells us how confident we should be.

---

## Dataset Overview

| Property | Detail |
|----------|--------|
| **Observations** | 208 weeks (Nov 2015 – Nov 2019) |
| **Granularity** | Weekly |
| **Total weekly budget** | €64,301 across 5 channels |
| **Revenue range** | €167k – €331k per week |
| **Train / Test split** | 156 weeks / 52 weeks (chronological) |
| **Data type** | Synthetic with known true parameters |

**Why synthetic?** Known ground truth enables **parameter recovery validation** — we can verify the model recovers the true channel effects. This is impossible with real-world data where the true parameters are unknown.

**Channels by average weekly spend:**
TV (€19.7k) > Facebook (€15.3k) > Search (€12.0k) > OOH (€10.2k) > Print (€7.1k)

---

## Methodology: Bayesian Framework

```
Revenue = Base Demand + Σ β × saturation(adstock(spend)) + Controls + Noise
```

**Three-stage channel transform:**

1. **Scale** — normalise spend to [0, 1] range (MaxAbsScaler)
2. **Adstock** — geometric decay over 8 weeks: models carryover effects
3. **Saturation** — logistic function: models diminishing returns

**MCMC sampling:** 4 chains × 1,000 draws with PyMC-Marketing
**Convergence:** R-hat < 1.01, ESS > 2,000 for all parameters

The Bayesian approach provides **full posterior distributions** — not just "Print ROAS = 0.28" but "Print ROAS = 0.28, and we are 94% confident it is between 0.00 and 0.70."

---

## Key Findings: Channel ROAS Rankings

| Rank | Channel | ROAS (Mean) | 94% HDI | Total Contribution |
|------|---------|-------------|---------|-------------------|
| 1 | **Print** | **0.28** | [0.00, 0.70] | €313k |
| 2 | **Search** | **0.20** | [0.00, 0.46] | €367k |
| 3 | **TV** | **0.17** | [0.00, 0.36] | €526k |
| 4 | **OOH** | **0.15** | [0.00, 0.39] | €237k |
| 5 | **Facebook** | **0.13** | [0.00, 0.32] | €320k |

**Business interpretation:**
- **Print** delivers the highest return per euro spent, despite having the lowest total spend
- **TV** has the highest absolute contribution (€526k) due to its larger budget, but its per-euro efficiency ranks third
- **Facebook** shows the lowest ROAS — each euro returns only €0.13 in attributable revenue
- Wide credible intervals are expected: aggregate weekly data has inherent identification limits

---

## Model Performance

<div class="metric">MAPE: 3.9% &nbsp; | &nbsp; R²: 0.91</div>

The model predicts weekly revenue within **4% of actuals** on 52 held-out weeks.

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Out-of-sample MAPE | 3.9% | < 15% | PASS |
| Out-of-sample R² | 0.914 | > 0.70 | PASS |
| Out-of-sample MAE | €9,915 | — | — |
| Max R-hat (convergence) | 1.007 | < 1.01 | PASS |
| Min ESS (bulk) | 2,032 | > 400 | PASS |
| Parameter recovery | 9/10 in 94% HDI | All | 1 miss (TV adstock) |

**Parameter recovery:** The model recovers 9 of 10 ground-truth parameters within their credible intervals. Only TV's adstock alpha (true=0.70) is missed, reflecting an identifiability challenge for high-carryover channels.

---

## Channel Decomposition: Where Does Revenue Come From?

The model decomposes total revenue into base demand and channel contributions:

| Component | Share |
|-----------|-------|
| **Base + Controls** | ~95.5% |
| **TV** | 1.35% |
| **Search** | 0.95% |
| **Facebook** | 0.82% |
| **Print** | 0.81% |
| **OOH** | 0.61% |

**Base dominance (~95%)** is typical of MMMs — most revenue comes from brand equity, organic demand, repeat purchasing, and non-media factors. The five paid channels collectively explain **~4.5%** of revenue.

This does not mean media spend is unimportant: that 4.5% represents **€1.76 million** in incremental revenue over the training period, generated by **€10 million** in total media spend.

---

## ROI Estimate: The Value of Measurement

**Current total media spend:** €64,301 per week (~€3.3M annually)

| Scenario | Mechanism | Annual Impact |
|----------|-----------|---------------|
| Reallocate 10% from lowest-ROAS to highest-ROAS | Shift €335k/year from Facebook to Print/Search | +€20k–€50k revenue |
| Identify saturation points | Reduce overspend on saturated channels | Save €100k–€200k in wasted spend |
| Carryover-aware planning | Time campaigns for maximum cumulative effect | +5–10% campaign efficiency |
| **Net estimated value of model** | | **€150k–€350k annually** |

The model pays for itself by replacing guesswork with measured return. Even modest reallocation — shifting spend from the bottom-ROAS channel to the top — yields measurable improvement.

---

## Response Curves: Diminishing Returns

Every channel follows a saturation curve — each additional euro produces less incremental revenue than the last.

**What the response curves tell us:**

- **Steep region** (low spend): high marginal return → opportunity to increase
- **Flat region** (high spend): low marginal return → overspending territory
- **Inflection point**: where the curve bends → optimal operating range

**Saturation warnings** in the Budget Simulator flag channels operating in the "moderate" (>60% saturated) or "high" (>80% saturated) zones, alerting media planners before spend crosses into diminishing-returns territory.

---

## The Application: Four Interactive Views

### Decomposition
Stacked area chart showing weekly revenue decomposed into base + channel contributions over time. Identify seasonal patterns and channel contribution trends.

### Channel Performance
ROAS bar chart with 94% credible intervals, plus spend vs. contribution comparison. Quickly identify which channels deliver the most efficient return.

### Budget Simulator
Adjust channel spend via sliders and see predicted revenue update in real time. Saturation warnings (amber/red badges) flag diminishing-returns zones. Budget lock toggle ensures total spend is preserved when adjusting individual channels.

### Optimal Allocation
Side-by-side comparison of current vs. optimised budget allocation with expected revenue uplift and channel-level reallocation recommendations.

---

## Budget Optimisation Recommendations

Based on ROAS rankings and saturation analysis:

| Channel | Current (€/wk) | Action | Rationale |
|---------|----------------|--------|-----------|
| **Print** | €7,135 | Increase | Highest ROAS (0.28), lowest current spend, room on saturation curve |
| **Search** | €11,969 | Increase | Second-highest ROAS (0.20), intent-driven with fast payback |
| **TV** | €19,684 | Maintain | Highest absolute contribution; moderate ROAS but strong brand effect |
| **OOH** | €10,244 | Monitor | Fourth in ROAS; evaluate creative effectiveness before changing |
| **Facebook** | €15,269 | Decrease | Lowest ROAS (0.13); reallocate savings to Print and Search |

**Constraint:** Total weekly budget held constant at €64,301. Recommendations shift budget between channels, not overall spend level.

---

## Next Steps

### Immediate (Weeks 1–2)
1. Deploy the application to the internal analytics platform
2. Train marketing and media planning teams on the four views
3. Integrate ROAS estimates into the quarterly media planning process

### Near-term (Months 1–3)
4. **Replace synthetic data with real business data** — retrain the model on actual weekly spend and revenue
5. Run **geo-lift validation tests** — withhold spend in test regions to validate ROAS estimates against experimental evidence
6. Build automated weekly refresh pipeline

### Future
7. Add **time-varying parameters** to capture creative fatigue and seasonal shifts in channel effectiveness
8. Model **cross-channel interactions** (TV → Search synergy)
9. Integrate with **m02 CLV** — weight ROAS by customer lifetime value
10. Integrate with **m03 Churn** — penalise channels that acquire high-churn customers

---

<!-- _class: lead -->

# Thank You

### Key Takeaway

> The Marketing Mix Model transforms media budget allocation from **gut-driven** to **evidence-based** — providing ROAS estimates with uncertainty quantification, response curves with saturation warnings, and optimised budget recommendations across all five channels.

**Model:** Bayesian MMM (PyMC-Marketing) | **MAPE:** 3.9% | **R²:** 0.91

**Access:** `http://localhost:5176` | **Docs:** `m04_mmm/docs/`
