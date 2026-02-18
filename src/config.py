"""
Centralized configuration for the Fraud Detection project.

Uses environment variables (via .env) with sensible defaults.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ── Load .env if present ──────────────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── Kaggle ────────────────────────────────────────────────────────
KAGGLE_DATASET = "kartik2112/fraud-detection"
KAGGLE_TRAIN_FILE = "fraudTrain.csv"
KAGGLE_TEST_FILE = "fraudTest.csv"

# ── Model artefacts ──────────────────────────────────────────────
MODEL_PATH = ARTIFACTS_DIR / os.getenv("MODEL_PATH", "model.joblib")
SCALER_PATH = ARTIFACTS_DIR / os.getenv("SCALER_PATH", "scaler.joblib")
ENCODER_PATH = ARTIFACTS_DIR / os.getenv("ENCODER_PATH", "encoders.joblib")
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.joblib"

# ── Model hyper-parameters ───────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc"],
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# ── Inference ────────────────────────────────────────────────────
DEFAULT_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))

# ── API ──────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Features to drop (identifiers / raw fields) ─────────────────
COLUMNS_TO_DROP = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "lat",
    "long",
    "dob",
    "trans_date_trans_time",
    "merch_lat",
    "merch_long",
    "trans_num",
]

# ── Categorical columns to label-encode ──────────────────────────
CATEGORICAL_COLS = ["merchant", "category", "gender", "city", "state", "job", "cc_num"]

TARGET_COL = "is_fraud"
