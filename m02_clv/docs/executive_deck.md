---
marp: true
theme: default
paginate: true
header: "**Customer Lifetime Value** | Marketing Analytics"
footer: "m02_clv | Confidential"
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
---

<!-- _class: lead -->

# Customer Lifetime Value Prediction

### Turning transaction history into forward-looking customer scores

**Marketing Analytics | m02_clv**
February 2026

---

## The Problem: One-Size-Fits-All Marketing

Most marketing budgets are spread uniformly across the customer base. This is expensive and ineffective.

- **Top 10% of customers** generate over **50% of revenue** in most retail businesses
- Acquiring a new customer costs **5-7x more** than retaining an existing one
- A **5% improvement** in retention can yield up to **95% profit lift**

Without customer-level value predictions, teams overspend on low-value accounts and underinvest in their most profitable relationships.

> "Treating all customers equally means treating your best customers poorly."

---

## The Solution: Predict Every Customer's 12-Month Value

We built a machine learning model that scores each customer with a **predicted 12-month CLV in pounds sterling**.

**What it does:**
- Assigns a forward-looking revenue estimate to every customer
- Segments customers into five actionable tiers (Champions through Dormant)
- Identifies the behavioural drivers behind each customer's predicted value
- Provides a simulator to explore "what-if" scenarios

**What it enables:**
- Proportional marketing spend aligned to predicted value
- Early identification of high-potential "Growing" customers
- Data-driven retention budgets for Champions

---

## Data Foundation: UCI Online Retail II

| Attribute | Detail |
|-----------|--------|
| **Source** | UCI Machine Learning Repository (CC BY 4.0) |
| **Business** | UK-based wholesale/gifting e-commerce retailer |
| **Volume** | ~1,067,000 transactions across 4,372 repeat customers |
| **Time span** | December 2009 -- December 2011 (2 full years) |
| **Geography** | Predominantly UK (89%), with EU and other markets |

**Temporal split design:**
- **Observation window** (features): Dec 2009 -- Nov 2010
- **Prediction window** (target): Dec 2010 -- Dec 2011
- This mirrors a real deployment: learn from the past year, predict the next

---

## Methodology: RFM Features + LightGBM

```
Transaction logs                   22 Engineered              LightGBM            Customer-level
  (1M+ rows)        ──────>        Features        ──────>   Regressor   ──────>   CLV score
                                                            (log1p target)          in GBP
```

**Feature groups (22 features):**
- **RFM core**: recency, frequency, monetary total/avg/max
- **Behavioural**: unique products, cancellation rate, avg quantity per item
- **Temporal**: tenure, purchase velocity, inter-purchase variability
- **Contextual**: UK flag, Q4 purchases, weekend/evening ratios
- **Composite**: RFM combined score (recency + frequency + monetary quintiles)

**Target**: `clv_12m` = total revenue in the 12-month prediction window, log-transformed for training

---

## What Drives Customer Value?

Top 10 features ranked by SHAP importance (mean absolute contribution to predictions):

| Rank | Feature | Impact |
|-----:|---------|--------|
| 1 | **Monetary total** (historical spend) | Very high |
| 2 | **Recency** (days since last purchase) | High |
| 3 | **RFM combined score** | High |
| 4 | **Monetary max** (largest single order) | Medium-high |
| 5 | **Unique products** purchased | Medium-high |
| 6 | **Monetary average** (avg order value) | Medium |
| 7 | **Purchase velocity** (orders per month) | Medium |
| 8 | **Avg quantity per item** | Medium |
| 9 | **Tenure** (days since first purchase) | Medium |
| 10 | **Inter-purchase variability** | Moderate |

**Key insight**: Past spending behaviour is the strongest predictor. Customers who spend more, buy more frequently, and purchase a wider range of products are predicted to deliver higher future value.

---

## Model Performance: Business Terms

| Metric | LightGBM | Naive Mean | Improvement |
|--------|----------|------------|-------------|
| **MAE** | **£1,228** | £2,551 | 52% lower error |
| **Spearman rank correlation** | **0.60** | n/a | Strong ordering |
| **MAPE** | **79%** | 395% | 5x better |
| **Top-decile lift** | **5.33x** | 1.0x | Top 10% captures 53% of value |

**What this means in practice:**
- The model correctly **ranks** customers by value (Spearman 0.60)
- Targeting the **top predicted decile** captures **5.3x** its proportional share of actual revenue
- Average prediction error of £1,228 on a customer base with mean CLV of £2,397

> The model also outperforms the classical BG/NBD + Gamma-Gamma probabilistic baseline (MAE £1,426) by 14%.

---

## The Application: Three Interactive Views

The CLV application provides a **React + Vite** frontend backed by **FastAPI** model serving.

### Customer Lookup
Search any customer by ID. View their profile, predicted CLV in pounds, value tier assignment, and the SHAP waterfall chart showing which factors drove their individual prediction.

### CLV Simulator
Adjust behavioural inputs (recency, frequency, average order value, purchase velocity, cancellation rate) via sliders and see the predicted CLV update in real time. Enables "what-if" scenario planning.

### Portfolio Explorer
Browse the full customer portfolio by segment. Interactive scatter plot of recency vs. CLV, filterable by tier. Paginated data table with sort and search.

---

## CLV Simulator: Scenario Planning

The What-If Simulator lets business users explore how changes in customer behaviour affect predicted value.

**Example scenarios:**

| Scenario | Lever | CLV Impact |
|----------|-------|------------|
| Re-engage a lapsed customer | Reduce recency from 300 to 30 days | Significant increase |
| Upsell campaign | Increase avg order value by 50% | Moderate increase |
| Cross-sell initiative | Double unique products purchased | Moderate increase |
| Reduce cancellations | Drop cancellation rate to 0% | Small positive lift |

**Use case**: Before launching a retention campaign, simulate the expected CLV uplift to justify the spend.

---

## Segment: Champions (Top 10%)

**Profile**: Highest predicted CLV. Frequent, recent, high-spending wholesale accounts.

| Characteristic | Typical Range |
|----------------|---------------|
| Predicted CLV | > 90th percentile |
| Purchase frequency | Weekly or bi-weekly |
| Monetary total | Well above average |

**Recommendations:**
- Assign dedicated account managers
- Offer exclusive early access to new product lines
- Priority fulfilment and premium service tiers
- Monitor closely for any decline in purchase velocity (early churn signal)

---

## Segment: High Value (75th--90th Percentile)

**Profile**: Strong, consistent buyers just below Champion status. Many are one behavioural shift away from the top tier.

**Recommendations:**
- Targeted upsell campaigns to increase average order value
- Loyalty programme enrolment with tiered rewards
- Personalised product recommendations based on purchase history
- Quarterly business reviews to deepen the relationship

**Promotion potential**: A 20% increase in purchase frequency would move many High Value customers into the Champion tier.

---

## Segment: Growing (40th--75th Percentile)

**Profile**: Mid-range customers with moderate frequency and spend. This is the largest segment and the primary growth opportunity.

**Recommendations:**
- Cross-sell bundles to expand product breadth (unique_products is a top-5 driver)
- Time-limited offers to increase purchase velocity
- Educational content about product range to drive discovery
- Automated re-engagement emails at optimal intervals

**Why this segment matters**: Small per-customer lifts here translate to large aggregate revenue gains due to segment size.

---

## Segment: Occasional & Dormant (Below 40th Percentile)

### Occasional (10th--40th Percentile)
- Infrequent buyers with low monetary totals
- **Strategy**: Low-cost automated nurture campaigns; focus on converting to Growing
- Promotional codes with minimum order thresholds to increase basket size

### Dormant (Below 10th Percentile)
- High recency (long time since last purchase), minimal historical engagement
- **Strategy**: Win-back campaign with strong incentive, but cap spend
- Cold-start customers (< 2 purchases) receive the median CLV estimate (£622)
- If win-back fails after 2 attempts, deprioritise to avoid wasted spend

---

## ROI Estimate: Value of Targeted Marketing

**Conservative scenario** based on redirecting marketing budget proportionally to predicted CLV:

| Lever | Assumption | Annual Impact |
|-------|------------|---------------|
| Reduce spend on Dormant segment | Cut 50% of spend on bottom 10% | Save 5% of total budget |
| Reinvest in Champions + High Value | Targeted retention reduces churn by 3% | +£18k incremental revenue |
| Growing segment upsell | 10% of Growing move to High Value | +£12k incremental revenue |
| **Net estimated uplift** | | **+£30k or 3-5% revenue lift** |

The model's **5.33x top-decile lift** means every pound spent on the highest-predicted customers is over 5x more efficient than uniform allocation.

---

## Next Steps

### Immediate (Weeks 1--2)
- Deploy the application to the internal analytics platform
- Train marketing team on Lookup, Simulator, and Portfolio views
- Integrate CLV scores into the CRM for campaign targeting

### Near-term (Months 1--3)
- A/B test segment-specific campaigns (Champions retention vs. Growing upsell)
- Connect to live transaction feed for monthly CLV score refresh
- Build automated alerts for customers crossing segment boundaries

### Future
- Combine with **m03 churn model** for joint retention/value scoring
- Extend to non-UK markets as transaction volume grows
- Evaluate deep learning (sequence models) once per-customer history depth increases
