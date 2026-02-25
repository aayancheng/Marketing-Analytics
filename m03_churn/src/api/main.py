from __future__ import annotations

import json
import math
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from src.api.schemas import (
    ChurnPrediction,
    CustomerListItem,
    CustomerProfile,
    CustomerResponse,
    ErrorResponse,
    PaginatedCustomers,
    PredictRequest,
    SegmentSummary,
    SegmentsResponse,
    ShapContribution,
    SimulateRequest,
    SimulateResponse,
)
from src.feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_churn.pkl")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "models", "probability_calibrator.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.parquet")

# Service columns used to count num_services
_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Customer-ID format: four digits, dash, four uppercase letters+digits (e.g. 7590-VHVEG)
_CID_PATTERN = re.compile(r"^\d{4}-[A-Z0-9]{5}$")


# ---------------------------------------------------------------------------
# Helper: derive a human-readable label for contract / internet / payment
# ---------------------------------------------------------------------------

def _contract_label(row: pd.Series) -> str:
    if row.get("Contract_Two year", 0):
        return "Two year"
    if row.get("Contract_One year", 0):
        return "One year"
    return "Month-to-month"


def _internet_label(row: pd.Series) -> str:
    if row.get("InternetService_DSL", 0):
        return "DSL"
    if row.get("InternetService_Fiber optic", 0):
        return "Fiber optic"
    return "No"


def _payment_label(row: pd.Series) -> str:
    if row.get("PaymentMethod_Bank transfer (automatic)", 0):
        return "Bank transfer (automatic)"
    if row.get("PaymentMethod_Credit card (automatic)", 0):
        return "Credit card (automatic)"
    if row.get("PaymentMethod_Electronic check", 0):
        return "Electronic check"
    return "Mailed check"


# ---------------------------------------------------------------------------
# Helper: recompute derived/engineered features in-place for a dict
# ---------------------------------------------------------------------------

def _recompute_derived(d: dict) -> dict:
    """Recompute the 7 engineered features from raw columns in dict d."""
    d["has_family"] = int(bool(d.get("Partner", 0)) or bool(d.get("Dependents", 0)))
    d["num_services"] = int(sum(int(d.get(k, 0)) for k in _SERVICE_COLS))
    tenure = float(d.get("tenure", 0))
    monthly = float(d.get("MonthlyCharges", 0.0))
    total = float(d.get("TotalCharges", 0.0))
    d["monthly_per_tenure"] = monthly / (tenure + 1.0)
    d["total_charges_gap"] = monthly * tenure - total
    d["is_month_to_month"] = int(d.get("Contract_Month-to-month", 0))
    d["is_fiber_optic"] = int(d.get("InternetService_Fiber optic", 0))
    d["is_electronic_check"] = int(d.get("PaymentMethod_Electronic check", 0))
    return d


# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------

def _calibrated_score(app: FastAPI, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated churn probabilities for a feature DataFrame."""
    raw = app.state.model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    return np.asarray(app.state.calibrator.predict(raw))


def _risk_tier(score: float) -> str:
    if score > 0.60:
        return "High Risk"
    if score > 0.40:
        return "Medium-High Risk"
    if score > 0.20:
        return "Medium-Low Risk"
    return "Low Risk"


def _recommended_action(tier: str) -> str:
    mapping = {
        "High Risk": "Urgent intervention",
        "Medium-High Risk": "Priority retention offer",
        "Medium-Low Risk": "Soft outreach",
        "Low Risk": "Monitor",
    }
    return mapping.get(tier, "Monitor")


def _shap_top5(app: FastAPI, row: pd.DataFrame) -> list[ShapContribution]:
    """Compute SHAP values for a single-row DataFrame and return top-5 contributions."""
    x = row[FEATURE_COLUMNS].copy()
    shap_vals = app.state.explainer.shap_values(x)

    # Handle list (older SHAP) vs ndarray (newer SHAP) output
    if isinstance(shap_vals, list):
        sv = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])[0]
    else:
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:
            # shape (1, n_features, 2) — binary output
            sv = arr[0, :, 1]
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


def _parse_customer_id_or_422(customer_id: str) -> str:
    """Validate customerID format (XXXX-XXXXX); raise 422 if malformed."""
    cid = customer_id.strip().upper()
    if not _CID_PATTERN.match(cid):
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="invalid_customer_id",
                message=(
                    "customer_id must match the pattern DDDD-AAAAA "
                    "(4 digits, dash, 5 uppercase alphanumerics), e.g. '7590-VHVEG'."
                ),
            ).model_dump(),
        )
    return cid


# ---------------------------------------------------------------------------
# Lifespan: load all artifacts at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts
    app.state.model = joblib.load(MODEL_PATH)
    app.state.calibrator = joblib.load(CALIBRATOR_PATH)
    app.state.explainer = joblib.load(EXPLAINER_PATH)

    # Load metadata (cost_optimal_threshold etc.)
    with open(METADATA_PATH, "r") as fh:
        app.state.metadata = json.load(fh)

    # Load customer feature table and index by customerID
    features_df = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    features_df = features_df.set_index("customerID", drop=False)
    app.state.features_df = features_df

    # Compute median defaults for predict endpoint (numeric FEATURE_COLUMNS only)
    numeric_cols = features_df[FEATURE_COLUMNS].select_dtypes(include=[np.number]).columns.tolist()
    medians = features_df[numeric_cols].median(numeric_only=True).to_dict()
    # Fill any missing feature columns with 0
    defaults: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        defaults[col] = float(medians.get(col, 0.0))
    app.state.predict_defaults = defaults

    # Score ALL customers at startup and cache as a DataFrame
    X_all = features_df[FEATURE_COLUMNS].copy()
    all_scores_arr = _calibrated_score(app, X_all)

    scored_df = pd.DataFrame(
        {
            "customerID": features_df.index.tolist(),
            "churn_probability": all_scores_arr.tolist(),
        }
    )
    scored_df["risk_tier"] = scored_df["churn_probability"].apply(_risk_tier)
    scored_df["tenure"] = features_df["tenure"].values
    scored_df["MonthlyCharges"] = features_df["MonthlyCharges"].values
    # Reconstruct Contract label from one-hot columns
    scored_df["Contract"] = [
        _contract_label(features_df.loc[cid]) for cid in scored_df["customerID"]
    ]
    app.state.all_scores_df = scored_df.set_index("customerID", drop=False)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Churn Propensity API", version="1.0.0", lifespan=lifespan)

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
        "model": "lgbm_churn",
        "customers": len(app.state.features_df),
    }


# ---------------------------------------------------------------------------
# Endpoint 2: GET /api/customers
# ---------------------------------------------------------------------------

@app.get(
    "/api/customers",
    response_model=PaginatedCustomers,
    responses={422: {"model": ErrorResponse}},
)
def list_customers(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
    segment: str | None = Query(default=None),
) -> PaginatedCustomers:
    df = app.state.all_scores_df.reset_index(drop=True)

    if segment is not None:
        valid_tiers = {"High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"}
        if segment not in valid_tiers:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error="invalid_segment",
                    message=f"segment must be one of: {sorted(valid_tiers)}",
                ).model_dump(),
            )
        df = df[df["risk_tier"] == segment]

    total = len(df)
    pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    items = [
        CustomerListItem(
            customerID=str(r["customerID"]),
            churn_probability=round(float(r["churn_probability"]), 4),
            risk_tier=str(r["risk_tier"]),
            tenure=int(r["tenure"]),
            MonthlyCharges=float(r["MonthlyCharges"]),
            Contract=str(r["Contract"]),
        )
        for _, r in page_df.iterrows()
    ]

    return PaginatedCustomers(
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
    response_model=CustomerResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def get_customer(customer_id: str) -> CustomerResponse:
    cid = _parse_customer_id_or_422(customer_id)

    features_df: pd.DataFrame = app.state.features_df
    if cid not in features_df.index:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="customer_not_found",
                message=f"No customer found with ID '{cid}'.",
            ).model_dump(),
        )

    row = features_df.loc[cid]
    row_df = pd.DataFrame([row])

    score = float(_calibrated_score(app, row_df)[0])
    tier = _risk_tier(score)
    action = _recommended_action(tier)
    shap_factors = _shap_top5(app, row_df)

    profile = CustomerProfile(
        customerID=cid,
        tenure=int(row["tenure"]),
        MonthlyCharges=float(row["MonthlyCharges"]),
        TotalCharges=float(row["TotalCharges"]),
        Contract=_contract_label(row),
        InternetService=_internet_label(row),
        PaymentMethod=_payment_label(row),
        num_services=int(row["num_services"]),
        SeniorCitizen=int(row["SeniorCitizen"]),
        Partner=int(row["Partner"]),
        Dependents=int(row["Dependents"]),
        has_family=int(row["has_family"]),
        is_month_to_month=int(row["is_month_to_month"]),
        is_fiber_optic=int(row["is_fiber_optic"]),
        is_electronic_check=int(row["is_electronic_check"]),
    )

    return CustomerResponse(
        customerID=cid,
        churn_probability=round(score, 4),
        risk_tier=tier,
        recommended_action=action,
        shap_factors=shap_factors,
        profile=profile,
        cost_optimal_threshold=float(app.state.metadata.get("cost_optimal_threshold", 0.5)),
    )


# ---------------------------------------------------------------------------
# Endpoint 4: POST /api/predict
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=ChurnPrediction,
    responses={422: {"model": ErrorResponse}},
)
def predict(req: PredictRequest) -> ChurnPrediction:
    defaults: dict[str, float] = dict(app.state.predict_defaults)

    # Build the feature dict from defaults, then apply request overrides.
    # Use model_dump with by_alias=True so aliased fields (e.g. "Contract_Month-to-month")
    # are keyed correctly, then drop None values.
    req_dict: dict[str, Any] = {
        k: v for k, v in req.model_dump(by_alias=True).items() if v is not None
    }

    # Handle convenience categorical strings → one-hot
    contract_type = req_dict.pop("contract_type", None)
    internet_service = req_dict.pop("internet_service", None)
    payment_method = req_dict.pop("payment_method", None)

    if contract_type is not None:
        defaults["Contract_Month-to-month"] = 0.0
        defaults["Contract_One year"] = 0.0
        defaults["Contract_Two year"] = 0.0
        key_map = {
            "month-to-month": "Contract_Month-to-month",
            "one year": "Contract_One year",
            "two year": "Contract_Two year",
        }
        mapped = key_map.get(contract_type.lower())
        if mapped:
            defaults[mapped] = 1.0

    if internet_service is not None:
        defaults["InternetService_DSL"] = 0.0
        defaults["InternetService_Fiber optic"] = 0.0
        defaults["InternetService_No"] = 0.0
        key_map = {
            "dsl": "InternetService_DSL",
            "fiber optic": "InternetService_Fiber optic",
            "no": "InternetService_No",
        }
        mapped = key_map.get(internet_service.lower())
        if mapped:
            defaults[mapped] = 1.0

    if payment_method is not None:
        defaults["PaymentMethod_Bank transfer (automatic)"] = 0.0
        defaults["PaymentMethod_Credit card (automatic)"] = 0.0
        defaults["PaymentMethod_Electronic check"] = 0.0
        defaults["PaymentMethod_Mailed check"] = 0.0
        key_map = {
            "bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
            "credit card (automatic)": "PaymentMethod_Credit card (automatic)",
            "electronic check": "PaymentMethod_Electronic check",
            "mailed check": "PaymentMethod_Mailed check",
        }
        mapped = key_map.get(payment_method.lower())
        if mapped:
            defaults[mapped] = 1.0

    # Apply remaining numeric overrides
    for k, v in req_dict.items():
        if k in defaults:
            defaults[k] = float(v)

    # Recompute derived features
    _recompute_derived(defaults)

    row_df = pd.DataFrame([{col: defaults[col] for col in FEATURE_COLUMNS}])
    score = float(_calibrated_score(app, row_df)[0])
    tier = _risk_tier(score)
    action = _recommended_action(tier)
    shap_factors = _shap_top5(app, row_df)

    return ChurnPrediction(
        churn_probability=round(score, 4),
        risk_tier=tier,
        recommended_action=action,
        shap_factors=shap_factors,
    )


# ---------------------------------------------------------------------------
# Endpoint 5: POST /api/simulate
# ---------------------------------------------------------------------------

@app.post(
    "/api/simulate",
    response_model=SimulateResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def simulate(req: SimulateRequest) -> SimulateResponse:
    cid = _parse_customer_id_or_422(req.customer_id)

    features_df: pd.DataFrame = app.state.features_df
    if cid not in features_df.index:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="customer_not_found",
                message=f"No customer found with ID '{cid}'.",
            ).model_dump(),
        )

    row = features_df.loc[cid]
    original_dict: dict[str, Any] = {col: float(row[col]) for col in FEATURE_COLUMNS}

    # Score original
    original_df = pd.DataFrame([original_dict])
    original_score = float(_calibrated_score(app, original_df)[0])
    original_tier = _risk_tier(original_score)

    # Apply proposed changes
    modified_dict = dict(original_dict)
    for k, v in req.proposed_changes.items():
        if k in modified_dict:
            modified_dict[k] = float(v)

    # Recompute derived features after changes
    _recompute_derived(modified_dict)

    modified_df = pd.DataFrame([{col: modified_dict[col] for col in FEATURE_COLUMNS}])
    new_score = float(_calibrated_score(app, modified_df)[0])
    new_tier = _risk_tier(new_score)

    delta = new_score - original_score
    direction = "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged"
    message = (
        f"Churn probability {direction} by {abs(delta):.1%} "
        f"({original_tier} → {new_tier})."
    )

    return SimulateResponse(
        original_probability=round(original_score, 4),
        new_probability=round(new_score, 4),
        probability_delta=round(delta, 4),
        original_tier=original_tier,
        new_tier=new_tier,
        message=message,
    )


# ---------------------------------------------------------------------------
# Endpoint 6: GET /api/segments
# ---------------------------------------------------------------------------

@app.get(
    "/api/segments",
    response_model=SegmentsResponse,
    responses={},
)
def segments() -> SegmentsResponse:
    scored_df = app.state.all_scores_df.copy()
    features_df: pd.DataFrame = app.state.features_df

    # -- Risk tier summaries --
    tier_order = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"]
    agg = (
        scored_df.groupby("risk_tier")
        .agg(
            count=("customerID", "count"),
            mean_probability=("churn_probability", "mean"),
            mean_tenure=("tenure", "mean"),
            mean_monthly_charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
    )

    risk_tiers: list[SegmentSummary] = []
    for tier in tier_order:
        tier_row = agg[agg["risk_tier"] == tier]
        if tier_row.empty:
            risk_tiers.append(
                SegmentSummary(
                    tier=tier,
                    count=0,
                    mean_probability=0.0,
                    mean_tenure=0.0,
                    mean_monthly_charges=0.0,
                )
            )
        else:
            r = tier_row.iloc[0]
            risk_tiers.append(
                SegmentSummary(
                    tier=tier,
                    count=int(r["count"]),
                    mean_probability=round(float(r["mean_probability"]), 4),
                    mean_tenure=round(float(r["mean_tenure"]), 2),
                    mean_monthly_charges=round(float(r["mean_monthly_charges"]), 2),
                )
            )

    # -- Contract × InternetService churn rate grid --
    # Add contract and internet labels to the features DataFrame for grouping
    grid_df = features_df.copy()
    grid_df["_contract"] = grid_df.apply(_contract_label, axis=1)
    grid_df["_internet"] = grid_df.apply(_internet_label, axis=1)
    # Attach actual churn column if present; otherwise use 0 as placeholder
    if "Churn" in grid_df.columns:
        grid_df["_churn"] = grid_df["Churn"].astype(float)
    else:
        grid_df["_churn"] = 0.0

    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["DSL", "Fiber optic", "No"]

    contract_internet_grid: list[list[Any]] = []
    # Header row
    contract_internet_grid.append(["Contract \\ Internet"] + internets)
    for contract in contracts:
        row_vals: list[Any] = [contract]
        for internet in internets:
            subset = grid_df[
                (grid_df["_contract"] == contract) & (grid_df["_internet"] == internet)
            ]
            if len(subset) == 0:
                row_vals.append(None)
            else:
                rate = round(float(subset["_churn"].mean()), 4)
                row_vals.append(rate)
        contract_internet_grid.append(row_vals)

    return SegmentsResponse(
        risk_tiers=risk_tiers,
        contract_internet_grid=contract_internet_grid,
    )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8003, reload=True)
