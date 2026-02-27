"""
Pydantic v2 schemas for m02_clv (Customer Lifetime Value) API.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


# ---------------------------------------------------------------------------
# Customer profile (returned inside detail response)
# ---------------------------------------------------------------------------

class CustomerProfile(BaseModel):
    customer_id: int
    recency_days: float
    frequency: int
    monetary_total: float
    monetary_avg: float
    tenure_days: int
    unique_products: int
    uk_customer: bool
    rfm_combined_score: int


# ---------------------------------------------------------------------------
# Detail response — FLAT (not nested under "prediction")
# ---------------------------------------------------------------------------

class CustomerDetailResponse(BaseModel):
    profile: CustomerProfile
    predicted_clv: float
    clv_segment: str
    percentile_rank: float
    shap_factors: List[ShapContribution]


# ---------------------------------------------------------------------------
# What-if prediction
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Accepts RFM-style inputs for what-if CLV scoring.

    Any field not supplied is filled from app.state.predict_defaults
    (median of the training population).
    """
    recency_days: float = Field(..., ge=0, le=3650)
    frequency: int = Field(..., ge=1, le=500)
    monetary_total: float = Field(..., ge=0, le=500000)
    monetary_avg: Optional[float] = Field(default=None)
    purchase_velocity: Optional[float] = Field(default=None)
    cancellation_rate: float = Field(default=0.0, ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_clv: float
    clv_segment: str
    shap_values: List[ShapContribution]


# ---------------------------------------------------------------------------
# Paginated list — MUST use "items" and "pages"
# ---------------------------------------------------------------------------

class PaginatedResponse(BaseModel):
    items: List[dict]       # MUST be "items", NOT "customers"
    pages: int              # MUST be "pages", NOT "total_pages"
    total: int
    page: int
    per_page: int


# ---------------------------------------------------------------------------
# Segment summary
# ---------------------------------------------------------------------------

class SegmentSummary(BaseModel):
    segment: str
    count: int
    mean_clv: float
    mean_recency_days: float


# ---------------------------------------------------------------------------
# Portfolio item (for scatter plot)
# ---------------------------------------------------------------------------

class PortfolioItem(BaseModel):
    customer_id: int
    recency_days: float
    predicted_clv: float
    clv_segment: str
    frequency: int


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    message: str
    customer_id: Optional[int] = None
