from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    CustomerProfile,
    CustomerResponse,
    CustomersListResponse,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
    SegmentSummary,
    SegmentsResponse,
    ShapContribution,
    SlotRecommendation,
)
from src.feature_engineering import FEATURE_COLUMNS, INDUSTRY_OPEN_RATE_BY_HOUR, build_168_grid, DAY_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_time_to_engage.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "models", "probability_calibrator.pkl")
CUSTOMER_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
EVENT_PATH = os.path.join(BASE_DIR, "data", "processed", "event_features.csv")





def _calibrated_scores(app: FastAPI, feature_df: pd.DataFrame) -> np.ndarray:
    raw_scores = app.state.model.predict_proba(feature_df[FEATURE_COLUMNS])[:, 1]
    return np.asarray(app.state.calibrator.predict(raw_scores))


def _heatmap_from_grid(grid: pd.DataFrame, scores: np.ndarray) -> list[list[float]]:
    scored = grid[["send_hour", "send_dow"]].copy()
    scored["score"] = scores

    heatmap = [[0.0 for _ in range(24)] for _ in range(7)]
    for _, row in scored.iterrows():
        heatmap[int(row["send_dow"])][int(row["send_hour"])] = float(row["score"])

    if len(heatmap) != 7 or any(len(r) != 24 for r in heatmap):
        raise RuntimeError("Heatmap shape invariant violated; expected [7][24].")
    return heatmap


def _top3_from_grid(grid: pd.DataFrame, scores: np.ndarray) -> list[SlotRecommendation]:
    scored = grid[["send_hour", "send_dow"]].copy()
    scored["probability"] = scores
    top = scored.sort_values("probability", ascending=False).head(3)

    recs: list[SlotRecommendation] = []
    for _, row in top.iterrows():
        p = float(row["probability"])
        recs.append(
            SlotRecommendation(
                send_hour=int(row["send_hour"]),
                send_dow=int(row["send_dow"]),
                day_name=DAY_NAMES[int(row["send_dow"])],
                probability=p,
                confidence_pct=int(round(p * 100)),
            )
        )
    return recs


def _shap_top5(app: FastAPI, row: pd.Series) -> list[ShapContribution]:
    x = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
    shap_vals = app.state.explainer.shap_values(x)
    # shap may return list for binary or ndarray depending version.
    if isinstance(shap_vals, list):
        sv = np.asarray(shap_vals[1 if len(shap_vals) > 1 else 0])[0]
    else:
        arr = np.asarray(shap_vals)
        sv = arr[0] if arr.ndim == 2 else arr[0, :, 1]

    contrib = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": [float(x.iloc[0][f]) for f in FEATURE_COLUMNS],
            "contribution": sv.astype(float),
        }
    )
    top = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).head(5)
    return [
        ShapContribution(feature=r["feature"], value=float(r["value"]), contribution=float(r["contribution"]))
        for _, r in top.iterrows()
    ]


def _parse_customer_id_or_422(customer_id: str) -> int:
    try:
        cid = int(customer_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(error="invalid_customer_id", message="customer_id must be a numeric integer.").model_dump(),
        ) from exc
    if cid <= 0:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(error="invalid_customer_id", message="customer_id must be > 0.").model_dump(),
        )
    return cid


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    app.state.explainer = joblib.load(EXPLAINER_PATH)
    app.state.calibrator = joblib.load(CALIBRATOR_PATH)

    customer_df = pd.read_csv(CUSTOMER_PATH)
    customer_df["customer_id"] = customer_df["customer_id"].astype(np.int64)
    app.state.customer_features = customer_df.set_index("customer_id", drop=False)

    # Feature defaults for /api/predict sliders (fill untouched dimensions).
    medians = customer_df[[
        "avg_daily_txn_count",
        "monetary_total",
        "tenure_days",
        "unique_products",
        "cancellation_rate",
        "modal_purchase_dow",
    ]].median(numeric_only=True).to_dict()
    app.state.predict_defaults = {
        "avg_daily_txn_count": float(medians["avg_daily_txn_count"]),
        "monetary_total": float(medians["monetary_total"]),
        "tenure_days": int(round(float(medians["tenure_days"]))),
        "country_uk": 1,
        "unique_products": int(round(float(medians["unique_products"]))),
        "cancellation_rate": float(medians["cancellation_rate"]),
        "modal_purchase_dow": int(round(float(medians["modal_purchase_dow"]))),
        "rfm_segment": "Other",
    }

    event_df = pd.read_csv(EVENT_PATH, usecols=["customer_id", "opened"])
    event_df["customer_id"] = event_df["customer_id"].astype(np.int64)
    app.state.event_open_rate = event_df.groupby("customer_id")["opened"].mean().rename("open_rate")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/customer/{customer_id}", response_model=CustomerResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def get_customer(customer_id: str) -> CustomerResponse:
    cid = _parse_customer_id_or_422(customer_id)
    customers = app.state.customer_features

    if cid not in customers.index:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(error="customer_not_found", message="Customer not found.", customer_id=cid).model_dump(),
        )

    row = customers.loc[cid]
    if int(row["frequency"]) < 5:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="insufficient_history",
                message="Customer has fewer than 5 historical transactions.",
                customer_id=cid,
            ).model_dump(),
        )

    grid = build_168_grid(row)
    scores = _calibrated_scores(app, grid)

    profile = CustomerProfile(
        customer_id=cid,
        recency_days=float(row["recency_days"]),
        frequency=int(row["frequency"]),
        monetary_total=float(row["monetary_total"]),
        modal_purchase_hour=int(row["modal_purchase_hour"]),
        modal_purchase_dow=int(row["modal_purchase_dow"]),
        rfm_segment=str(row["rfm_segment"]),
        country_uk=bool(int(row["country_uk"])) if not pd.isna(row["country_uk"]) else True,
        open_rate=float(app.state.event_open_rate.get(cid, 0.0)),
    )

    return CustomerResponse(
        profile=profile,
        top_3_slots=_top3_from_grid(grid, scores),
        heatmap=_heatmap_from_grid(grid, scores),
        out_of_distribution_warning=False,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    defaults = app.state.predict_defaults
    profile = pd.Series(
        {
            "modal_purchase_hour": req.modal_hour,
            "modal_purchase_dow": defaults["modal_purchase_dow"],
            "purchase_hour_entropy": req.purchase_hour_entropy,
            "avg_daily_txn_count": defaults["avg_daily_txn_count"],
            "recency_days": req.recency_days,
            "frequency": req.frequency,
            "monetary_total": defaults["monetary_total"],
            "tenure_days": defaults["tenure_days"],
            "country_uk": defaults["country_uk"],
            "unique_products": defaults["unique_products"],
            "cancellation_rate": defaults["cancellation_rate"],
        }
    )

    grid = build_168_grid(profile)
    scores = _calibrated_scores(app, grid)

    top3 = _top3_from_grid(grid, scores)
    top_slot = top3[0]
    top_row = grid[(grid["send_hour"] == top_slot.send_hour) & (grid["send_dow"] == top_slot.send_dow)].iloc[0]

    return PredictResponse(
        top_3_slots=top3,
        shap_values=_shap_top5(app, top_row),
        heatmap=_heatmap_from_grid(grid, scores),
        out_of_distribution_warning=False,
    )


@app.get("/api/customers", response_model=CustomersListResponse)
def list_customers(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=100),
    segment: str | None = Query(default=None),
) -> CustomersListResponse:
    df = app.state.customer_features.reset_index(drop=True)
    if segment is not None:
        df = df[df["rfm_segment"] == segment]

    total = len(df)
    total_pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    customers = [
        CustomerProfile(
            customer_id=int(r["customer_id"]),
            recency_days=float(r["recency_days"]),
            frequency=int(r["frequency"]),
            monetary_total=float(r["monetary_total"]),
            modal_purchase_hour=int(r["modal_purchase_hour"]),
            modal_purchase_dow=int(r["modal_purchase_dow"]),
            rfm_segment=str(r["rfm_segment"]),
            country_uk=bool(int(r["country_uk"])),
            open_rate=float(app.state.event_open_rate.get(int(r["customer_id"]), 0.0)),
        )
        for _, r in page_df.iterrows()
    ]

    return CustomersListResponse(customers=customers, total=total, page=page, per_page=per_page, total_pages=total_pages)


@app.get("/api/segments", response_model=SegmentsResponse)
def segments() -> SegmentsResponse:
    customers = app.state.customer_features.reset_index(drop=True).copy()
    open_rate = app.state.event_open_rate

    merged = customers.merge(open_rate, left_on="customer_id", right_index=True, how="left")
    merged["open_rate"] = merged["open_rate"].fillna(0.0)

    agg = (
        merged.groupby("rfm_segment")
        .agg(count=("customer_id", "count"), mean_open_rate=("open_rate", "mean"), mean_recency_days=("recency_days", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    out = [
        SegmentSummary(
            segment=str(r["rfm_segment"]),
            count=int(r["count"]),
            mean_open_rate=float(r["mean_open_rate"]),
            mean_recency_days=float(r["mean_recency_days"]),
        )
        for _, r in agg.iterrows()
    ]
    return SegmentsResponse(segments=out)
