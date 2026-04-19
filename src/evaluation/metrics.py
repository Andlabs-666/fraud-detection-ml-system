"""Model evaluation metrics."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a model on test data.

    Args:
        model: Fitted model with predict method.
        X_test: Test features.
        y_test: Test labels.
        threshold: Classification threshold.

    Returns:
        Dictionary containing all metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix.tolist(),
        "threshold": threshold,
    }

    logger.info(f"Evaluation metrics (threshold={threshold}):")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  PR AUC: {pr_auc:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")

    return metrics


def find_optimal_threshold(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Find threshold that maximizes F1 score on validation set.

    Args:
        model: Fitted model.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Optimal threshold value.
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Clip to valid range
    best_threshold = float(np.clip(best_threshold, 0.01, 0.99))

    logger.info(f"Optimal threshold: {best_threshold:.4f}")
    logger.info(f"Best F1 on validation: {f1_scores[best_idx]:.4f}")

    return best_threshold


def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot and save ROC curve.

    Args:
        model: Fitted model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the plot.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC curve to {output_path}")
    else:
        plt.show()


def plot_pr_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot and save Precision-Recall curve.

    Args:
        model: Fitted model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the plot.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved PR curve to {output_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot and save confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save the plot.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()


def save_report(metrics: Dict[str, Any], output_path: str) -> None:
    """Save evaluation report as JSON.

    Args:
        metrics: Metrics dictionary.
        output_path: Path to save the report.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved evaluation report to {output_path}")


def print_comparison_table(
    baseline_metrics: Dict[str, Any],
    production_metrics: Dict[str, Any],
) -> None:
    """Print comparison table of baseline vs production metrics.

    Args:
        baseline_metrics: Metrics from baseline model.
        production_metrics: Metrics from production model.
    """
    metrics_to_compare = ["roc_auc", "pr_auc", "f1", "precision", "recall"]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<15} {'Baseline':>12} {'Production':>12} {'Delta':>12}")
    print("-" * 70)

    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0)
        production_val = production_metrics.get(metric, 0)
        delta = production_val - baseline_val

        print(
            f"{metric:<15} {baseline_val:>12.4f} {production_val:>12.4f} "
            f"{delta:>+12.4f}"
        )

    print("=" * 70 + "\n")