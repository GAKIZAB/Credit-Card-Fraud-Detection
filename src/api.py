"""
FastAPI application — Fraud Detection REST API.

Endpoints:
  GET  /                  → Welcome & API info
  GET  /health            → Health check with model metadata
  POST /predict           → Single transaction prediction
  POST /predict/batch     → Batch prediction (up to 1 000 transactions)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src import config
from src.features import engineer_single_row
from src.model import load_artifacts
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionInput,
)

# ──────────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    try:
        model, scaler, encoders, feature_names, metadata = load_artifacts()
        _state["model"] = model
        _state["scaler"] = scaler
        _state["encoders"] = encoders
        _state["feature_names"] = feature_names
        _state["metadata"] = metadata
        _state["loaded"] = True
        logger.info("✅  Model artifacts loaded successfully.")
    except Exception as exc:
        logger.error(f"❌  Failed to load model artifacts: {exc}")
        _state["loaded"] = False
    yield
    _state.clear()


# ──────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🔒 Fraud Detection API",
    description=(
        "Real-time credit card fraud detection powered by XGBoost.\n\n"
        "• Submit a transaction and receive an instant fraud probability.\n"
        "• Supports single and batch predictions.\n"
        "• Model trained on the Kaggle *Credit Card Fraud Detection* dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────


@app.get("/", tags=["Info"])
def root():
    """Welcome endpoint with links to docs."""
    return {
        "message": "🔒 Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Health check — returns model status and metrics."""
    meta = _state.get("metadata", {})
    return HealthResponse(
        status="healthy" if _state.get("loaded") else "model_not_loaded",
        model_loaded=_state.get("loaded", False),
        version="1.0.0",
        trained_at=meta.get("trained_at"),
        feature_count=meta.get("feature_count"),
        metrics=meta.get("metrics"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(transaction: TransactionInput):
    """Predict whether a single transaction is fraudulent."""
    _ensure_model_loaded()
    return _predict_one(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):
    """Predict fraud for a batch of transactions (max 1 000)."""
    _ensure_model_loaded()

    predictions = [_predict_one(tx) for tx in request.transactions]
    fraud_count = sum(1 for p in predictions if p.is_fraud)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
        fraud_rate=round(fraud_count / len(predictions), 4) if predictions else 0.0,
    )


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _ensure_model_loaded():
    if not _state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` first.",
        )


def _predict_one(tx: TransactionInput) -> PredictionResponse:
    """Run inference on a single transaction."""
    model = _state["model"]
    scaler = _state["scaler"]
    encoders = _state["encoders"]
    feature_names = _state["feature_names"]
    metadata = _state["metadata"]

    # Feature engineering
    row = tx.model_dump()
    features_df = engineer_single_row(row, encoders)

    # Ensure column order matches training
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[feature_names]

    # Scale
    features_scaled = scaler.transform(features_df.values)

    # Predict
    proba = float(model.predict_proba(features_scaled)[0, 1])
    threshold = metadata.get("metrics", {}).get("optimal_threshold", config.DEFAULT_THRESHOLD)
    is_fraud = proba >= threshold

    # Risk level
    if proba < 0.3:
        risk = "LOW"
    elif proba < 0.6:
        risk = "MEDIUM"
    elif proba < 0.85:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 5),
        threshold_used=round(threshold, 4),
        risk_level=risk,
    )
