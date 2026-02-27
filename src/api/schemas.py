from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    recency_days: float = Field(..., ge=0, le=3650)
    frequency: int = Field(..., ge=1, le=500)
    modal_hour: int = Field(..., ge=0, le=23)
    purchase_hour_entropy: float = Field(..., ge=0.0, le=1.0)


class SlotRecommendation(BaseModel):
    send_hour: int
    send_dow: int
    day_name: str
    probability: float
    confidence_pct: int


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    top_3_slots: List[SlotRecommendation]
    shap_values: List[ShapContribution]
    heatmap: List[List[float]]
    out_of_distribution_warning: bool = False


class CustomerProfile(BaseModel):
    customer_id: int
    recency_days: float
    frequency: int
    monetary_total: float
    modal_purchase_hour: int
    modal_purchase_dow: int
    rfm_segment: str
    country_uk: bool
    open_rate: Optional[float] = None


class CustomerResponse(BaseModel):
    profile: CustomerProfile
    top_3_slots: List[SlotRecommendation]
    heatmap: List[List[float]]
    out_of_distribution_warning: bool


class CustomersListResponse(BaseModel):
    customers: List[CustomerProfile]
    total: int
    page: int
    per_page: int
    total_pages: int


class SegmentSummary(BaseModel):
    segment: str
    count: int
    mean_open_rate: float
    mean_recency_days: float


class SegmentsResponse(BaseModel):
    segments: List[SegmentSummary]


class ErrorResponse(BaseModel):
    error: str
    message: str
    customer_id: Optional[int] = None
