from __future__ import annotations
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from src import config


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────


def train(
    data: pd.DataFrame,
    encoders: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    
    logger.info("🚀  Starting model training …")

    X = data.drop(columns=[config.TARGET_COL])
    y = data[config.TARGET_COL].astype(int)

    feature_names = list(X.columns)

    # ── Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── Scale ────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Imbalance ratio ──────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    logger.info(f"  Class balance — neg: {neg:,}  pos: {pos:,}  ratio: {scale_pos_weight:.1f}")

    # ── Model ────────────────────────────────────────────────────
    params = {**config.MODEL_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**params)

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    # ── Evaluate ─────────────────────────────────────────────────
    proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = _evaluate(y_test, proba)

    # ── Save artifacts ───────────────────────────────────────────
    _save_artifacts(model, scaler, encoders, feature_names, metrics)

    return metrics


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────


def _evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:

    # Default threshold
    y_pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    logger.info(f"  ROC AUC  : {roc:.4f}")
    logger.info(f"  PR AUC   : {pr_auc:.4f}")

    # Optimal F1 threshold
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        yp = (proba >= t).astype(int)
        f = f1_score(y_true, yp)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred_opt = (proba >= best_t).astype(int)

    metrics = {
        "roc_auc": round(roc, 5),
        "pr_auc": round(pr_auc, 5),
        "accuracy": round(accuracy_score(y_true, y_pred_opt), 5),
        "precision": round(precision_score(y_true, y_pred_opt, zero_division=0), 5),
        "recall": round(recall_score(y_true, y_pred_opt), 5),
        "f1_score": round(best_f1, 5),
        "optimal_threshold": round(best_t, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred_opt).tolist(),
    }

    report = classification_report(y_true, y_pred_opt, digits=4)
    logger.info(f"\n  Optimal threshold: {best_t:.4f}\n{report}")

    return metrics


# ──────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────


def _save_artifacts(
    model: XGBClassifier,
    scaler: MinMaxScaler,
    encoders: dict,
    feature_names: list[str],
    metrics: dict,
) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoders, config.ENCODER_PATH)
    joblib.dump(feature_names, config.FEATURE_NAMES_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_params": config.MODEL_PARAMS,
    }
    joblib.dump(metadata, config.METADATA_PATH)

    logger.info(f"  Artifacts saved to {config.ARTIFACTS_DIR}")


def load_artifacts():
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(config.ENCODER_PATH)
    feature_names = joblib.load(config.FEATURE_NAMES_PATH)
    metadata = joblib.load(config.METADATA_PATH)
    return model, scaler, encoders, feature_names, metadata
