"""Pydantic v2 schemas for MMM API."""

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Weekly channel spend inputs for revenue simulation."""

    tv_spend: float = Field(..., ge=0, description="Weekly TV spend")
    ooh_spend: float = Field(..., ge=0, description="Weekly OOH spend")
    print_spend: float = Field(..., ge=0, description="Weekly print spend")
    facebook_spend: float = Field(..., ge=0, description="Weekly Facebook spend")
    search_spend: float = Field(..., ge=0, description="Weekly search spend")


class SimulateResponse(BaseModel):
    """Predicted revenue and channel-level contributions."""

    predicted_revenue: float
    current_revenue: float
    delta: float
    delta_pct: float
    channel_contributions: dict[str, float]
    saturation_warnings: list[dict[str, str]]
