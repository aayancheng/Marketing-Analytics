from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """All FEATURE_COLUMNS as optional overrides for what-if scoring.

    Any field not supplied is filled from app.state.predict_defaults (median
    of the training population).  Engineered features are always recomputed
    server-side from the raw/encoded columns so they stay internally consistent.
    """

    # Binary / ordinal raw columns
    SeniorCitizen: Optional[float] = None
    Partner: Optional[float] = None
    Dependents: Optional[float] = None
    tenure: Optional[float] = None
    PhoneService: Optional[float] = None
    MultipleLines: Optional[float] = None
    PaperlessBilling: Optional[float] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None

    # One-hot internet service
    InternetService_DSL: Optional[float] = None
    InternetService_Fiber_optic: Optional[float] = Field(default=None, alias="InternetService_Fiber optic")
    InternetService_No: Optional[float] = None

    # One-hot contract
    Contract_Month_to_month: Optional[float] = Field(default=None, alias="Contract_Month-to-month")
    Contract_One_year: Optional[float] = Field(default=None, alias="Contract_One year")
    Contract_Two_year: Optional[float] = Field(default=None, alias="Contract_Two year")

    # One-hot payment method
    PaymentMethod_Bank_transfer_automatic: Optional[float] = Field(
        default=None, alias="PaymentMethod_Bank transfer (automatic)"
    )
    PaymentMethod_Credit_card_automatic: Optional[float] = Field(
        default=None, alias="PaymentMethod_Credit card (automatic)"
    )
    PaymentMethod_Electronic_check: Optional[float] = Field(
        default=None, alias="PaymentMethod_Electronic check"
    )
    PaymentMethod_Mailed_check: Optional[float] = Field(
        default=None, alias="PaymentMethod_Mailed check"
    )

    # Service flags
    OnlineSecurity: Optional[float] = None
    OnlineBackup: Optional[float] = None
    DeviceProtection: Optional[float] = None
    TechSupport: Optional[float] = None
    StreamingTV: Optional[float] = None
    StreamingMovies: Optional[float] = None

    # Engineered features (computed server-side; accepted for pass-through but overridden)
    has_family: Optional[float] = None
    num_services: Optional[float] = None
    monthly_per_tenure: Optional[float] = None
    total_charges_gap: Optional[float] = None
    is_month_to_month: Optional[float] = None
    is_fiber_optic: Optional[float] = None
    is_electronic_check: Optional[float] = None

    # Convenience categorical strings (used to set the one-hot columns automatically)
    contract_type: Optional[str] = None        # e.g. "Month-to-month", "One year", "Two year"
    internet_service: Optional[str] = None     # e.g. "DSL", "Fiber optic", "No"
    payment_method: Optional[str] = None       # e.g. "Electronic check", "Mailed check", …

    model_config = {"populate_by_name": True}


class SimulateRequest(BaseModel):
    """Retention-simulator: apply proposed_changes to an existing customer and re-score."""

    customer_id: str
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output building blocks
# ---------------------------------------------------------------------------

class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class CustomerProfile(BaseModel):
    customerID: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    num_services: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    has_family: int
    is_month_to_month: int
    is_fiber_optic: int
    is_electronic_check: int


# ---------------------------------------------------------------------------
# Prediction / response models
# ---------------------------------------------------------------------------

class ChurnPrediction(BaseModel):
    churn_probability: float
    risk_tier: str
    recommended_action: str
    shap_factors: List[ShapContribution]


class CustomerResponse(ChurnPrediction):
    customerID: str
    profile: CustomerProfile
    cost_optimal_threshold: float


class CustomerListItem(BaseModel):
    customerID: str
    churn_probability: float
    risk_tier: str
    tenure: int
    MonthlyCharges: float
    Contract: str


class PaginatedCustomers(BaseModel):
    items: List[CustomerListItem]
    total: int
    page: int
    per_page: int
    pages: int


class SimulateResponse(BaseModel):
    original_probability: float
    new_probability: float
    probability_delta: float
    original_tier: str
    new_tier: str
    message: str


# ---------------------------------------------------------------------------
# Segment models
# ---------------------------------------------------------------------------

class SegmentSummary(BaseModel):
    tier: str
    count: int
    mean_probability: float
    mean_tenure: float
    mean_monthly_charges: float


class SegmentsResponse(BaseModel):
    risk_tiers: List[SegmentSummary]
    contract_internet_grid: List[List[Any]]


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    message: str
