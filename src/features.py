"""
Feature engineering pipeline.

Handles:
  • Datetime parsing & feature extraction (age, month, week, hour)
  • Geographic distance (client ↔ merchant)
  • Amount-derived features (log amount, amount bins)
  • Column dropping (identifiers and raw fields)
  • Label encoding of categorical variables
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from src import config


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Full feature pipeline: merge → engineer → encode.

    Returns
    -------
    (data, encoders) where *data* is the ready-to-model DataFrame
    and *encoders* is a dict mapping column → fitted LabelEncoder.
    """
    logger.info("🔧  Starting feature engineering …")

    # 1. Concatenate
    data = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"  Merged shape: {data.shape}")

    # 2. Datetime features
    data = _add_datetime_features(data)

    # 3. Geographic distance
    data = _add_distance(data)

    # 4. Amount-derived features
    data = _add_amount_features(data)

    # 5. Drop useless columns
    data = _drop_columns(data)

    # 6. Encode categoricals
    data, encoders = _encode_categoricals(data)

    logger.info(f"  Final feature matrix: {data.shape}")
    return data, encoders


def engineer_single_row(row: dict, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Apply the same transformations to a single transaction dict
    (used at inference time).

    Parameters
    ----------
    row : dict
        Raw transaction fields.
    encoders : dict
        Fitted LabelEncoders from training.

    Returns
    -------
    pd.DataFrame with one row, same columns as training features.
    """
    df = pd.DataFrame([row])

    # Datetime features
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["mois"] = df["trans_date_trans_time"].dt.month
    df["semaine"] = df["trans_date_trans_time"].dt.isocalendar().week.astype(int)
    df["heure"] = df["trans_date_trans_time"].dt.hour

    # Distance
    df["distance"] = df.apply(
        lambda r: geodesic(
            (r["lat"], r["long"]),
            (r["merch_lat"], r["merch_long"]),
        ).km,
        axis=1,
    )

    # Amount features
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_bin"] = pd.cut(
        df["amt"],
        bins=[0, 10, 50, 100, 500, 1000, float("inf")],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)

    # Drop raw / identifier columns
    cols_to_drop = [c for c in config.COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Also drop the target if present
    df = df.drop(columns=[config.TARGET_COL], errors="ignore")

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda v, _le=le: (
                    _le.transform([v])[0] if v in _le.classes_ else -1
                )
            )

    return df


# ──────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────


def _add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    data["dob"] = pd.to_datetime(data["dob"])
    data = data.sort_values(by="trans_date_trans_time")

    data["age"] = (data["trans_date_trans_time"] - data["dob"]).dt.days / 365.25
    data["mois"] = data["trans_date_trans_time"].dt.month
    data["semaine"] = data["trans_date_trans_time"].dt.isocalendar().week.astype(int)
    data["heure"] = data["trans_date_trans_time"].dt.hour

    logger.info("  ✓ Datetime features added (age, mois, semaine, heure)")
    return data


def _add_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute geodesic distance between client and merchant."""
    logger.info("  ⏳ Computing geodesic distances (this may take a while) …")
    data["distance"] = data.apply(
        lambda r: geodesic(
            (r["lat"], r["long"]),
            (r["merch_lat"], r["merch_long"]),
        ).km,
        axis=1,
    )
    logger.info("  ✓ Distance feature added")
    return data


def _add_amount_features(data: pd.DataFrame) -> pd.DataFrame:
    """Log-transform and bin the transaction amount."""
    data["log_amt"] = np.log1p(data["amt"])
    data["amt_bin"] = pd.cut(
        data["amt"],
        bins=[0, 10, 50, 100, 500, 1000, float("inf")],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)
    logger.info("  ✓ Amount features added (log_amt, amt_bin)")
    return data


def _drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in config.COLUMNS_TO_DROP if c in data.columns]
    data = data.drop(columns=existing)
    logger.info(f"  ✓ Dropped {len(existing)} identifier/raw columns")
    return data


def _encode_categoricals(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    non_num = data.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns
    encoders: dict[str, LabelEncoder] = {}

    for col in non_num:
        if col == config.TARGET_COL:
            continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    logger.info(f"  ✓ Label-encoded {len(encoders)} categorical columns")
    return data, encoders
