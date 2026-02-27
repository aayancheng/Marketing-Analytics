"""
FastAPI backend for m02_clv (Customer Lifetime Value).

Run from the model directory:
    cd m02_clv
    /Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port 8002 --reload
"""
from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# --- sys.path fix (required for module resolution) ---
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from src.api.schemas import (
    CustomerDetailResponse,
    CustomerProfile,
    ErrorResponse,
    PaginatedResponse,
    PredictRequest,
    PredictResponse,
    PortfolioItem,
    SegmentSummary,
    ShapContribution,
)
from src.feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_clv.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "processed", "clv_labels.csv")

# ---------------------------------------------------------------------------
# CLV segment definitions (percentile thresholds)
# ---------------------------------------------------------------------------

SEGMENT_THRESHOLDS = [
    (90, "Champions"),
    (75, "High Value"),
    (40, "Growing"),
    (10, "Occasional"),
    (0, "Dormant"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _predict_clv(model: Any, feature_row: pd.DataFrame) -> float:
    """Predict CLV for a single-row DataFrame. Model outputs log1p-scale; invert with expm1."""
    log_pred = model.predict(feature_row[FEATURE_COLUMNS])[0]
    clv = float(np.expm1(log_pred))
    return max(clv, 0.0)


def _predict_clv_batch(model: Any, features_df: pd.DataFrame) -> np.ndarray:
    """Predict CLV for all rows. Returns array of GBP values clipped at 0."""
    log_preds = model.predict(features_df[FEATURE_COLUMNS])
    clv_values = np.expm1(log_preds)
    return np.clip(clv_values, 0.0, None)


def _assign_segment(predicted_clv: float, percentiles: dict[int, float]) -> str:
    """Assign CLV segment label based on precomputed percentile thresholds."""
    for pct, label in SEGMENT_THRESHOLDS:
        if pct == 0:
            return label
        if predicted_clv >= percentiles[pct]:
            return label
    return "Dormant"


def _percentile_rank(predicted_clv: float, all_clv_values: np.ndarray) -> float:
    """Compute percentile rank of a CLV value within the full distribution."""
    rank = float(np.searchsorted(np.sort(all_clv_values), predicted_clv, side="right"))
    return round(rank / len(all_clv_values) * 100, 1)


def _shap_top5(explainer: Any, row: pd.DataFrame) -> list[ShapContribution]:
    """Compute SHAP values for a single-row DataFrame and return top-5 contributions."""
    x = row[FEATURE_COLUMNS].copy()
    shap_vals = explainer.shap_values(x)

    # Handle list (older SHAP) vs ndarray (newer SHAP) output
    if isinstance(shap_vals, list):
        sv = np.asarray(shap_vals[0])[0]
    else:
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:
            sv = arr[0, :, 0]
        else:
            sv = arr[0]

    contrib = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": [float(x.iloc[0][f]) for f in FEATURE_COLUMNS],
            "contribution": sv.astype(float),
        }
    )
    top = contrib.reindex(
        contrib["contribution"].abs().sort_values(ascending=False).index
    ).head(5)

    return [
        ShapContribution(
            feature=r["feature"],
            value=float(r["value"]),
            contribution=float(r["contribution"]),
        )
        for _, r in top.iterrows()
    ]


def _parse_customer_id_or_422(customer_id_str: str) -> int:
    """Parse customer_id string to int; raise 422 if not a valid integer."""
    try:
        cid = int(customer_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="invalid_customer_id",
                message="customer_id must be a valid integer.",
            ).model_dump(),
        )
    return cid


# ---------------------------------------------------------------------------
# Lifespan: load all artifacts and score customers at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts
    app.state.model = joblib.load(MODEL_PATH)
    app.state.explainer = joblib.load(EXPLAINER_PATH)

    with open(METADATA_PATH, "r") as fh:
        app.state.metadata = json.load(fh)

    # Load customer features and labels
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    # Score ALL customers at startup
    clv_predictions = _predict_clv_batch(app.state.model, features_df)
    features_df["predicted_clv"] = clv_predictions

    # Compute percentile thresholds from scored predictions
    percentiles: dict[int, float] = {}
    for pct, _label in SEGMENT_THRESHOLDS:
        if pct > 0:
            percentiles[pct] = float(np.percentile(clv_predictions, pct))
    app.state.percentiles = percentiles

    # Assign CLV segments
    features_df["clv_segment"] = features_df["predicted_clv"].apply(
        lambda v: _assign_segment(v, percentiles)
    )

    # Merge cold-start flag from labels
    features_df = features_df.merge(
        labels_df[["customer_id", "is_cold_start"]],
        on="customer_id",
        how="left",
    )
    features_df["is_cold_start"] = features_df["is_cold_start"].fillna(0).astype(int)

    # Cache the full scored DataFrame indexed by customer_id
    features_df = features_df.set_index("customer_id", drop=False)
    app.state.features_df = features_df
    app.state.all_clv_values = clv_predictions

    # Compute median defaults for what-if predict endpoint
    numeric_cols = features_df[FEATURE_COLUMNS].select_dtypes(include=[np.number]).columns.tolist()
    medians = features_df[numeric_cols].median(numeric_only=True).to_dict()
    defaults: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        defaults[col] = float(medians.get(col, 0.0))
    app.state.predict_defaults = defaults

    # Median CLV from training set (for cold-start fallback info)
    app.state.median_clv_train = float(app.state.metadata.get("median_clv_train", 0.0))

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Lifetime Value API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoint 1: GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "clv",
        "customers": len(app.state.features_df),
    }


# ---------------------------------------------------------------------------
# Endpoint 2: GET /api/customers
# ---------------------------------------------------------------------------

@app.get(
    "/api/customers",
    response_model=PaginatedResponse,
    responses={422: {"model": ErrorResponse}},
)
def list_customers(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
    segment: str | None = Query(default=None),
) -> PaginatedResponse:
    df = app.state.features_df.reset_index(drop=True)

    valid_segments = {"Champions", "High Value", "Growing", "Occasional", "Dormant"}
    if segment is not None:
        if segment not in valid_segments:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error="invalid_segment",
                    message=f"segment must be one of: {sorted(valid_segments)}",
                ).model_dump(),
            )
        df = df[df["clv_segment"] == segment]

    total = len(df)
    pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    page_df = df.iloc[start: start + per_page]

    items = []
    for _, r in page_df.iterrows():
        items.append({
            "customer_id": int(r["customer_id"]),
            "predicted_clv": round(float(r["predicted_clv"]), 2),
            "clv_segment": str(r["clv_segment"]),
            "recency_days": float(r["recency_days"]),
            "frequency": int(r["frequency"]),
            "monetary_total": round(float(r["monetary_total"]), 2),
        })

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Endpoint 3: GET /api/customer/{customer_id}
# ---------------------------------------------------------------------------

@app.get(
    "/api/customer/{customer_id}",
    response_model=CustomerDetailResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def get_customer(customer_id: str) -> CustomerDetailResponse:
    cid = _parse_customer_id_or_422(customer_id)

    features_df: pd.DataFrame = app.state.features_df
    if cid not in features_df.index:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="customer_not_found",
                message=f"No customer found with ID {cid}.",
                customer_id=cid,
            ).model_dump(),
        )

    row = features_df.loc[cid]
    # Handle case where loc returns multiple rows (duplicate index)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    row_df = pd.DataFrame([row])

    # Predicted CLV (already cached)
    predicted_clv = float(row["predicted_clv"])
    clv_segment = str(row["clv_segment"])
    pct_rank = _percentile_rank(predicted_clv, app.state.all_clv_values)

    # SHAP factors
    shap_factors = _shap_top5(app.state.explainer, row_df)

    profile = CustomerProfile(
        customer_id=cid,
        recency_days=float(row["recency_days"]),
        frequency=int(row["frequency"]),
        monetary_total=round(float(row["monetary_total"]), 2),
        monetary_avg=round(float(row["monetary_avg"]), 2),
        tenure_days=int(row["tenure_days"]),
        unique_products=int(row["unique_products"]),
        uk_customer=bool(int(row["uk_customer"])),
        rfm_combined_score=int(row["rfm_combined_score"]),
    )

    # FLAT response — predicted_clv, clv_segment, shap_factors at top level
    return CustomerDetailResponse(
        profile=profile,
        predicted_clv=round(predicted_clv, 2),
        clv_segment=clv_segment,
        percentile_rank=pct_rank,
        shap_factors=shap_factors,
    )


# ---------------------------------------------------------------------------
# Endpoint 4: POST /api/predict (what-if)
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}},
)
def predict(req: PredictRequest) -> PredictResponse:
    defaults: dict[str, float] = dict(app.state.predict_defaults)

    # Apply request overrides
    defaults["recency_days"] = float(req.recency_days)
    defaults["frequency"] = float(req.frequency)
    defaults["monetary_total"] = float(req.monetary_total)

    if req.monetary_avg is not None:
        defaults["monetary_avg"] = float(req.monetary_avg)
    else:
        # Derive monetary_avg from monetary_total / frequency
        defaults["monetary_avg"] = float(req.monetary_total) / float(req.frequency)

    if req.purchase_velocity is not None:
        defaults["purchase_velocity"] = float(req.purchase_velocity)

    defaults["cancellation_rate"] = float(req.cancellation_rate)

    # Build single-row DataFrame
    row_df = pd.DataFrame([{col: defaults[col] for col in FEATURE_COLUMNS}])

    # Predict
    predicted_clv = _predict_clv(app.state.model, row_df)
    clv_segment = _assign_segment(predicted_clv, app.state.percentiles)

    # SHAP
    shap_factors = _shap_top5(app.state.explainer, row_df)

    return PredictResponse(
        predicted_clv=round(predicted_clv, 2),
        clv_segment=clv_segment,
        shap_values=shap_factors,
    )


# ---------------------------------------------------------------------------
# Endpoint 5: GET /api/portfolio (scatter plot data)
# ---------------------------------------------------------------------------

@app.get(
    "/api/portfolio",
    response_model=list[PortfolioItem],
)
def portfolio() -> list[PortfolioItem]:
    df = app.state.features_df.reset_index(drop=True)

    items = []
    for _, r in df.iterrows():
        items.append(
            PortfolioItem(
                customer_id=int(r["customer_id"]),
                recency_days=float(r["recency_days"]),
                predicted_clv=round(float(r["predicted_clv"]), 2),
                clv_segment=str(r["clv_segment"]),
                frequency=int(r["frequency"]),
            )
        )

    return items


# ---------------------------------------------------------------------------
# Endpoint 6: GET /api/segments
# ---------------------------------------------------------------------------

@app.get(
    "/api/segments",
    response_model=list[SegmentSummary],
)
def segments() -> list[SegmentSummary]:
    df = app.state.features_df.reset_index(drop=True)

    segment_order = ["Champions", "High Value", "Growing", "Occasional", "Dormant"]

    results: list[SegmentSummary] = []
    for seg in segment_order:
        seg_df = df[df["clv_segment"] == seg]
        if seg_df.empty:
            results.append(
                SegmentSummary(
                    segment=seg,
                    count=0,
                    mean_clv=0.0,
                    mean_recency_days=0.0,
                )
            )
        else:
            results.append(
                SegmentSummary(
                    segment=seg,
                    count=int(len(seg_df)),
                    mean_clv=round(float(seg_df["predicted_clv"].mean()), 2),
                    mean_recency_days=round(float(seg_df["recency_days"].mean()), 1),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8002, reload=True)
