"""Evaluate trained models."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.features.pipeline import load_and_transform
from src.training.train import load_model
from src.evaluation.metrics import (
    evaluate,
    save_report,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
)
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def main() -> None:
    """Run model evaluation."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 60)

    # Load configs
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/data.yaml")
    train_config = load_config("configs/train.yaml")

    config_dir = "configs"
    target_col = data_config.get("target_column", "Class")

    # Load model and encoder
    model_path = train_config.get("model_path", "artifacts/models/production_model.joblib")
    encoder_path = train_config.get("encoder_path", "artifacts/encoders/preprocessor.joblib")

    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    logger.info(f"Loading encoder from {encoder_path}")

    # Load test data
    test_df = pd.read_parquet("data/interim/test.parquet")

    X_test = load_and_transform(encoder_path, test_df)
    y_test = test_df[target_col].values

    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Evaluate
    threshold = 0.5
    metrics = evaluate(model, X_test, y_test, threshold)

    # Print metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("=" * 50)

    # Save report
    artifact_dir = train_config.get("artifact_dir", "artifacts")
    save_report(metrics, f"{artifact_dir}/reports/evaluation_report.json")

    # Plot curves
    plot_roc_curve(
        model,
        X_test,
        y_test,
        f"{artifact_dir}/reports/roc_curve.png",
    )
    plot_pr_curve(
        model,
        X_test,
        y_test,
        f"{artifact_dir}/reports/pr_curve.png",
    )
    plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        f"{artifact_dir}/reports/confusion_matrix.png",
    )

    logger.info("\nEvaluation complete")


if __name__ == "__main__":
    main()