import sys
import time
from loguru import logger

# ── Logging setup ────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add("logs/training_{time}.log", rotation="10 MB", retention="30 days")


def main():
    from src.data import load_data_from_kaggle
    from src.features import build_features
    from src.model import train

    start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("  FRAUD DETECTION — MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Data acquisition
    train_df, test_df = load_data_from_kaggle()

    # Step 2: Feature engineering
    data, encoders = build_features(train_df, test_df)

    # Step 3: Train & evaluate
    metrics = train(data, encoders)

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info(f"  Training complete in {elapsed:.1f}s")
    logger.info(f"  ROC AUC    : {metrics['roc_auc']}")
    logger.info(f"  PR AUC     : {metrics['pr_auc']}")
    logger.info(f"  F1 Score   : {metrics['f1_score']}")
    logger.info(f"  Threshold  : {metrics['optimal_threshold']}")
    logger.info("=" * 60)
    logger.info("  Run the API with:  python run_api.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
