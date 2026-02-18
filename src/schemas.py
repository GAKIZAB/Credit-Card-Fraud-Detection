"""
Pydantic schemas for the Fraud Detection API.

Defines request/response models with validation and examples.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────


class TransactionInput(BaseModel):
    """Raw transaction data sent by the client for prediction."""

    trans_date_trans_time: str = Field(
        ...,
        description="Transaction datetime (format: YYYY-MM-DD HH:MM:SS)",
        examples=["2019-06-15 14:32:00"],
    )
    cc_num: int = Field(..., description="Credit card number", examples=[4263982640269299])
    merchant: str = Field(
        ...,
        description="Merchant name",
        examples=["fraud_Rippin, Kub and Mann"],
    )
    category: str = Field(
        ...,
        description="Transaction category",
        examples=["grocery_pos"],
    )
    amt: float = Field(..., description="Transaction amount ($)", examples=[124.65], gt=0)
    first: str = Field(..., description="First name", examples=["Jennifer"])
    last: str = Field(..., description="Last name", examples=["Banks"])
    gender: str = Field(..., description="Gender (M/F)", examples=["F"])
    street: str = Field(
        ...,
        description="Street address",
        examples=["561 Perry Cove"],
    )
    city: str = Field(..., description="City", examples=["Jesup"])
    state: str = Field(..., description="State 2-letter code", examples=["GA"])
    zip: int = Field(..., description="ZIP code", examples=[31599])
    lat: float = Field(..., description="Client latitude", examples=[31.5988])
    long: float = Field(..., description="Client longitude", examples=[-81.8826])
    city_pop: int = Field(..., description="City population", examples=[3495])
    job: str = Field(
        ...,
        description="Client job",
        examples=["Psychologist, counselling"],
    )
    dob: str = Field(
        ...,
        description="Date of birth (YYYY-MM-DD)",
        examples=["1988-03-09"],
    )
    trans_num: str = Field(
        ...,
        description="Transaction ID",
        examples=["0b242abb623afc578575680df30655b9"],
    )
    unix_time: int = Field(..., description="Unix timestamp", examples=[1371816865])
    merch_lat: float = Field(..., description="Merchant latitude", examples=[36.011293])
    merch_long: float = Field(..., description="Merchant longitude", examples=[-82.048315])


# ──────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Prediction result returned by the API."""

    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(
        ..., description="Probability of fraud (0 → 1)", ge=0.0, le=1.0
    )
    threshold_used: float = Field(..., description="Decision threshold applied")
    risk_level: str = Field(
        ...,
        description="Risk category: LOW, MEDIUM, HIGH, CRITICAL",
        examples=["LOW"],
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "healthy"
    model_loaded: bool
    version: str
    trained_at: Optional[str] = None
    feature_count: Optional[int] = None
    metrics: Optional[dict] = None


class BatchPredictionRequest(BaseModel):
    """Batch of transactions for prediction."""

    transactions: list[TransactionInput] = Field(
        ..., description="List of transactions", min_length=1, max_length=1000
    )


class BatchPredictionResponse(BaseModel):
    """Batch results."""

    predictions: list[PredictionResponse]
    total: int
    fraud_count: int
    fraud_rate: float
