# Project Brief: Model 5 — Next Best Offer / Product Recommendation
**Version:** 1.0 | **Stack:** Python (ML) + React/Node.js (App) | **License:** CC0 (Retailrocket dataset)

---

## 1. Project Overview

Build a Next Best Offer engine that predicts which product category a customer is most likely to purchase next, using a two-stage approach: collaborative filtering to build customer-product affinity scores, plus a LightGBM ranker to re-rank candidates using behavioral features. The model uses the Retailrocket eCommerce dataset — the best public dataset for implicit feedback recommendation systems, with real-world clickstream, add-to-cart, and transaction events.

The project delivers:
1. A two-stage recommendation pipeline (candidate generation + LightGBM re-ranking)
2. A React/Node.js SPA with a personalized offer explorer, customer affinity map, and real-time "next best offer" surfacing with confidence scores
3. Full model card and documentation

**Business framing:** An e-commerce retailer wants to surface the right product to each customer at the right moment — on the website homepage, in post-purchase emails, or during cart checkout — to increase average order value and repeat purchase rate. Unlike a generic bestseller list, each recommendation is personalized to the individual customer's behavior history.

---

## 2. Dataset

### 2a. Primary Dataset — Retailrocket eCommerce Dataset
| Property | Value |
|---|---|
| **Source** | Kaggle / Retailrocket |
| **Kaggle URL** | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset |
| **License** | CC0 Public Domain |
| **Collection period** | ~4.5 months (real-world e-commerce website) |
| **Files** | `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv` |
| **Hashing note** | All values are hashed due to confidentiality — visitor IDs, item IDs, and property values are integers, not human-readable names |

### 2b. File Schemas

**events.csv — User Interaction Log (PRIMARY)**
| Column | Type | Description | Notes |
|---|---|---|---|
| `timestamp` | integer | Unix timestamp (ms) | Convert to datetime |
| `visitorid` | integer | Unique visitor/user ID | ~1.4M unique visitors |
| `event` | string | Interaction type | `view`, `addtocart`, `transaction` |
| `itemid` | integer | Product item ID | ~235K unique items |
| `transactionid` | float (nullable) | Transaction group ID | Only populated for `transaction` events; NaN otherwise |

**Raw event counts:**
| Event Type | Count | % |
|---|---|---|
| `view` | 2,664,312 | 96.6% |
| `addtocart` | 69,332 | 2.5% |
| `transaction` | 22,457 | 0.8% |
| **Total** | **2,756,101** | |
| Unique visitors | 1,407,580 | |
| Unique items | ~235,061 | |

**item_properties_part1.csv + item_properties_part2.csv — Item Attributes**
| Column | Type | Description |
|---|---|---|
| `timestamp` | integer | When property was recorded |
| `itemid` | integer | Item ID (joins to events.csv) |
| `property` | string | Property name (hashed) — e.g., `categoryid`, `available`, `790` (price proxy) |
| `value` | string | Property value (hashed for most; numeric for some) |

**Key item properties to extract:**
- `categoryid` — item's category (numeric, maps to category_tree.csv)
- `available` — item availability flag (1 = in stock)
- Property `790` — commonly identified as a price-proxy field in community analyses

**category_tree.csv — Category Hierarchy**
| Column | Type | Description |
|---|---|---|
| `categoryid` | integer | Category ID |
| `parentid` | float | Parent category ID (NaN = root) |

### 2c. Data Scope Decisions (Important for Claude Code)

**Filter to active users only** — the full dataset has 1.4M visitors but most have only 1–2 views. For meaningful recommendations, filter to visitors with:
- At least **5 view events** OR at least **1 addtocart or transaction event**
- This yields ~50,000–80,000 actionable users

**Use most recent 90-day window** — item properties change over time (availability, price). Extract the most recent property snapshot per item.

**Transaction focus** — for the recommendation target, use `transaction` events as positive signals, `addtocart` as soft positives, and `view` as implicit feedback.

---

## 3. Modeling Target

| Property | Value |
|---|---|
| **Recommendation task** | Given a customer's interaction history, predict the top-5 items most likely to be purchased next |
| **Implicit feedback** | No explicit ratings — use event type as proxy: transaction (weight=3), addtocart (weight=2), view (weight=1) |
| **Target** | Binary per (user, item) pair: 1 if user purchased item in held-out period, 0 otherwise |
| **Output** | Ranked list of top-5 recommended items per customer, with confidence scores and category labels |
| **Cold start** | Visitors with < 5 events → fallback to global bestsellers (top 20 by transaction count in past 30 days) |

---

## 4. Modeling Pipeline — Two-Stage Approach

### Stage 1: Candidate Generation (Collaborative Filtering)

**Method:** Item-based collaborative filtering using implicit feedback signals.

```python
# Approach A: Matrix Factorization with implicit library (ALS)
import implicit
from scipy.sparse import csr_matrix

# Build user-item interaction matrix
# Rows = visitors, Columns = items, Values = weighted event score
# transaction=3, addtocart=2, view=1

model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=50,
    random_state=42
)
model.fit(item_user_matrix)  # note: implicit takes item × user

# Generate top-50 candidates per user
user_items, scores = model.recommend(
    userid=user_id,
    user_items=user_item_matrix[user_id],
    N=50,                    # top-50 candidates for re-ranking
    filter_already_liked=True
)
```

**Why ALS over pure collaborative filtering?**
- Scales to 1.4M users efficiently
- Handles implicit feedback natively
- `implicit` library is the standard for this class of problem

### Stage 2: Re-ranking (LightGBM Ranker)

After generating 50 candidates per user, re-rank using a gradient boosted ranker that incorporates behavioral features:

**LightGBM Ranker configuration:**
```python
import lightgbm as lgb

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[5, 10],
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    n_estimators=300,
    random_state=42
)
```

### 4a. Feature Engineering for Re-Ranker

**User-level features (from interaction history):**
| Feature | Description |
|---|---|
| `user_total_views` | Total view events |
| `user_total_addtocarts` | Total add-to-cart events |
| `user_total_transactions` | Total purchase events |
| `user_transaction_rate` | `transactions / views` — purchase intent signal |
| `user_days_active` | Span of activity in days |
| `user_favorite_category` | Most-viewed category ID |
| `user_recency_hours` | Hours since last event |
| `user_avg_session_items` | Average items viewed per session |

**Item-level features (from item properties):**
| Feature | Description |
|---|---|
| `item_total_views_global` | Global view count (popularity) |
| `item_total_transactions_global` | Global transaction count |
| `item_conversion_rate` | `transactions / views` globally |
| `item_category_id` | Encoded category (integer) |
| `item_is_available` | Availability flag |
| `item_category_depth` | Depth in category tree (1=root category, 3+=subcategory) |
| `item_recency_days` | Days since last global view |

**User × Item interaction features:**
| Feature | Description |
|---|---|
| `user_item_views` | How many times this user viewed this item |
| `user_item_addtocarts` | Times user added this item to cart |
| `user_category_affinity` | User's view/transaction ratio for this item's category |
| `als_score` | Stage 1 ALS affinity score for this (user, item) pair |
| `category_match` | Binary: item category == user's favorite category |
| `item_in_user_cart` | Binary: item currently in user's cart |

### 4b. Train/Test Split (Temporal)
```
Train: events from days 1–120 (first ~4 months)
Test:  events from days 121–135 (final ~2 weeks)
Positive labels: items purchased in test window by users seen in train window
```

### 4c. Evaluation Metrics
| Metric | Description | Target |
|---|---|---|
| `NDCG@5` | Normalized Discounted Cumulative Gain at top-5 | > 0.15 |
| `Recall@5` | % of actual purchases in top-5 recommendations | > 0.10 |
| `Precision@5` | % of top-5 recommendations that were purchased | > 0.05 |
| `Hit Rate@5` | % of users for whom ≥ 1 top-5 item was purchased | > 0.20 |

**Note:** These targets are calibrated for the Retailrocket dataset's sparsity (most users have very few transactions). The ALS baseline on this dataset typically achieves Hit Rate@10 ~0.18–0.25; adding LightGBM re-ranking improves by ~15–20%.

---

## 5. App Requirements — React/Node.js SPA

### 5a. Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS |
| Backend API | Node.js + Express |
| Charts | Recharts (heatmap, bar chart, network-like scatter) |
| ML serving | Python FastAPI microservice (ALS recommend + LightGBM predict) |
| Model artifacts | `als_model.npz` (ALS factors), `lgbm_ranker.pkl`, `item_metadata.json` |

### 5b. App Layout — Three Panels

**Panel 1 — Customer Offer Explorer**
- Input: VisitorID lookup field
- Customer behavioral summary card:
  - Total events | Transactions | Days active | Top category
  - Activity timeline: mini sparkline of event volume over time
  - Behavioral tag: "Power Buyer" / "Browser" / "Cart Abandoner" / "Occasional Buyer"
- Top 5 Recommended Items panel:
  - Item card for each: Item ID, Category, Confidence score (%), "Why recommended?" tooltip (top SHAP features)
  - Offer type badge: "Replenishment" (bought before) / "Discovery" (new category) / "Upsell" (same category, higher-value)
  - CTA button: "Show Similar Items"

**Panel 2 — What-If Preference Simulator**
Simulate recommendations for a hypothetical customer profile (no real ID needed):

| Input | Control |
|---|---|
| Preferred category | Dropdown of top-10 categories by transaction volume |
| Purchase frequency | Slider: 1–20 lifetime transactions |
| Recency | Slider: last active 1–30 days ago |
| ALS affinity threshold | Slider: minimum candidate score (0.1–0.9) — controls discovery vs. safe bets |

- Real-time API call → generates top-5 recommendations for hypothetical profile
- Toggle: "Show bestsellers" vs. "Show personalized" — side-by-side comparison
- Discovery score: % of recommendations that are outside user's typical categories (novelty indicator)

**Panel 3 — Catalog & Popularity Explorer**
- Scatter plot: all items plotted by view count (x) vs. conversion rate (y), sized by transaction count
  - Quadrants: "Hidden Gems" (low views, high conversion) | "Stars" (high views, high conversion) | "Browsers' Picks" (high views, low conversion) | "Dead Stock" (low both)
- Category heatmap: category × event type (view / addtocart / transaction) — shows which categories drive the most purchases
- Top 20 items table: ranked by transaction count with view count, conversion rate, and category
- Filter: by category, availability, transaction count range

### 5c. API Endpoints
```
GET  /api/customer/:visitorid             → interaction summary + top-5 recommendations + SHAP
POST /api/recommend                       → body: {user_features} → top-5 recs for hypothetical profile
GET  /api/items/popular                   → top items by transaction count with metadata
GET  /api/items/:itemid                   → item metadata + interaction stats
GET  /api/catalog                         → all items with view/transaction/conversion metrics
GET  /api/categories                      → category tree with aggregate stats
GET  /api/customers/search?q=             → visitor ID search (active users only)
```

---

## 6. Model Documentation Requirements

| Document | Content |
|---|---|
| **Model Card** | Purpose, intended users (e-commerce merchandising team, personalization engine), training data (Retailrocket, 4.5 months, hashed IDs), performance metrics (NDCG@5, Hit Rate@5), limitations (cold start, hash anonymization prevents content-based signals, dataset sparsity), ethical note (no demographic targeting possible due to hashed IDs) |
| **Data Dictionary** | All raw columns from events.csv + item_properties, event weight schema, derived features list |
| **Two-Stage Architecture** | Diagram + explanation of why candidate generation + re-ranking outperforms single-stage approaches; ALS vs. collaborative filtering trade-offs |
| **Cold Start Strategy** | How the system handles new visitors (< 5 events) with popularity-based fallback; graduated personalization as events accumulate |
| **Evaluation Report** | NDCG@5, Recall@5, Hit Rate@5 on test window; comparison to ALS-only baseline; comparison to global bestseller baseline |
| **Format** | Markdown in `/docs` + rendered in app About/Architecture tab |

---

## 7. Constraints and Assumptions

- All item and visitor IDs are hashed integers — no human-readable product names or customer names are available; the app must display category names and item IDs as identifiers
- The dataset has extreme sparsity: ~97% of interactions are views, only 0.8% are purchases — models are optimized for this ratio
- Cold start customers (< 5 events): serve global top-20 bestsellers; flag in API response with `cold_start: true`
- `implicit` library uses item × user matrix convention (not user × item); be careful with matrix transposition
- ALS factors are stored in memory at API startup for fast inference — full user × item matrix is not stored (too large)
- LightGBM ranker requires `group` array matching query groups for `lambdarank` objective — each user's candidates form one group
- Random seed: 42 for all stochastic operations (ALS, LightGBM, train/test split)
- Item properties dataset has two parts (`_part1`, `_part2`) that must be concatenated and deduplicated on `(itemid, property)` keeping most recent timestamp

---

## 8. Deliverables

```
project/
├── data/
│   └── raw/
│       ├── events.csv
│       ├── item_properties_part1.csv
│       ├── item_properties_part2.csv
│       └── category_tree.csv
├── notebooks/
│   └── 01_eda_retailrocket.ipynb
├── src/
│   ├── data_pipeline.py              # Event processing + interaction matrix build
│   ├── feature_engineering.py        # User + item + user×item features
│   ├── stage1_als.py                 # ALS candidate generation
│   ├── stage2_ranker.py              # LightGBM re-ranker
│   └── api/main.py                   # FastAPI serving
├── app/
│   ├── server/                       # Node.js/Express
│   └── client/                       # React + Tailwind
├── models/
│   ├── als_model.npz                 # ALS user + item factors
│   ├── lgbm_ranker.pkl               # Fitted LightGBM ranker
│   └── item_metadata.json            # Category + popularity lookup
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   ├── architecture.md
│   ├── cold_start_strategy.md
│   └── evaluation_report.md
├── requirements.txt
├── package.json
└── README.md
```

---

## 9. Getting Started Commands for Claude Code

```bash
# Download dataset (requires Kaggle CLI)
pip install kaggle
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/ --unzip

# Install Python deps
pip install implicit lightgbm shap pandas numpy scipy scikit-learn fastapi uvicorn joblib

# implicit requires a C compiler — if on Linux use:
apt-get install -y build-essential  # or pip install implicit --no-binary implicit

# Install Node + React
npm install express cors axios
npx create-react-app app/client
cd app/client && npm install recharts tailwindcss lucide-react
```

**Important for Claude Code:** The full events.csv is 2.75M rows — processing is fast but the user-item matrix construction can use significant memory. Use `scipy.sparse.csr_matrix` throughout; never convert to dense arrays. The ALS model should be fit on the filtered active-user subset (~50–80K users), not all 1.4M visitors.

---

## 10. Key References

- Dataset: Retailrocket Recommender System Dataset. https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- `implicit` library: https://implicit.readthedocs.io/en/latest/als.html
- Community analysis of dataset structure: https://github.com/caserec/Datasets-for-Recommender-Systems/blob/master/Processed%20Datasets/RetailrocketEcommerce/README.md
- LightGBM LambdaRank: https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective
- Two-stage recommender pattern reference: Covington, P. et al. (2016). "Deep Neural Networks for YouTube Recommendations." RecSys 2016.
