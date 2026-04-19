"""Train the fraud detection models."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.ingestion.loader import load_raw, split_data
from src.validation.schema import validate
from src.features.pipeline import build_and_save_pipeline, load_and_transform
from src.training.baseline import train_baseline
from src.training.train import train_model, save_model
from src.evaluation.metrics import (
    evaluate,
    find_optimal_threshold,
    save_report,
    print_comparison_table,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
)
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def main(config_dir: str = "configs") -> None:
    """Run the full training pipeline.

    Args:
        config_dir: Directory containing config files.
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load configs
    data_config = load_config(f"{config_dir}/data.yaml")
    model_config = load_config(f"{config_dir}/model.yaml")
    train_config = load_config(f"{config_dir}/train.yaml")

    # Set MLflow tracking (disabled by default)
    import os
    import mlflow
    mlflow_uri = train_config.get("mlflow_tracking_uri", "")
    if mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        mlflow.set_experiment(train_config.get("experiment_name", "fraud_detection"))
    else:
        # Disable MLflow tracking
        mlflow.autolog(disable=True)

    # === 1. Load and validate data ===
    logger.info("\n[1/7] Loading data...")
    raw_path = data_config.get("raw_path", "data/raw/creditcard.csv")
    df = load_raw(raw_path)

    logger.info("\n[2/7] Validating data...")
    report = validate(df)
    if not report.passed:
        logger.error("Data validation failed!")
        sys.exit(1)

    # === 2. Split data ===
    logger.info("\n[3/7] Splitting data...")
    target_col = data_config.get("target_column", "Class")
    train_df, val_df, test_df = split_data(
        df,
        target_col=target_col,
        test_size=data_config.get("test_size", 0.2),
        val_size=data_config.get("val_size", 0.1),
        seed=data_config.get("random_seed", 42),
    )

    # === 3. Build and fit feature pipeline ===
    logger.info("\n[4/7] Building feature pipeline...")
    encoder_path = "artifacts/encoders/preprocessor.joblib"
    pipeline = build_and_save_pipeline(train_df, encoder_path)

    # Transform all splits
    X_train = load_and_transform(encoder_path, train_df)
    X_val = load_and_transform(encoder_path, val_df)
    X_test = load_and_transform(encoder_path, test_df)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # === 4. Train baseline model ===
    logger.info("\n[5/7] Training baseline model...")
    baseline_config = model_config.get("baseline")
    baseline_model = train_baseline(X_train, y_train, baseline_config)

    baseline_threshold = find_optimal_threshold(baseline_model, X_val, y_val)
    baseline_metrics = evaluate(baseline_model, X_test, y_test, baseline_threshold)
    logger.info("Baseline model evaluation complete")

    # === 5. Train production model ===
    logger.info("\n[6/7] Training production model...")
    production_config = model_config.get("production")
    production_model = train_model(X_train, y_train, production_config)

    production_threshold = find_optimal_threshold(production_model, X_val, y_val)
    production_metrics = evaluate(production_model, X_test, y_test, production_threshold)
    logger.info("Production model evaluation complete")

    # === 6. Save best model ===
    logger.info("\n[7/7] Saving models...")
    model_path = "artifacts/models/production_model.joblib"
    save_model(production_model, model_path)

    # Save reports
    artifact_dir = Path(train_config.get("artifact_dir", "artifacts"))
    save_report(
        baseline_metrics,
        str(artifact_dir / "reports" / "baseline_report.json"),
    )
    save_report(
        production_metrics,
        str(artifact_dir / "reports" / "production_report.json"),
    )

    # Plot curves
    plot_roc_curve(
        production_model,
        X_test,
        y_test,
        str(artifact_dir / "reports" / "roc_curve.png"),
    )
    plot_pr_curve(
        production_model,
        X_test,
        y_test,
        str(artifact_dir / "reports" / "pr_curve.png"),
    )
    plot_confusion_matrix(
        y_test,
        production_model.predict(X_test),
        str(artifact_dir / "reports" / "confusion_matrix.png"),
    )

    # === Print comparison ===
    print_comparison_table(baseline_metrics, production_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Baseline model: {baseline_config.get('name')}")
    logger.info(f"Production model: {production_config.get('name')}")
    logger.info(f"Models saved to: {artifact_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    args = parser.parse_args()

    main(args.config_dir)