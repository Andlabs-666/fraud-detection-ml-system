"""Baseline model training (Logistic Regression)."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger
from src.registry.mlflow_registry import log_run

logger = get_logger(__name__)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> LogisticRegression:
    """Train a baseline logistic regression model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Model configuration dict.

    Returns:
        Fitted logistic regression model.
    """
    if config is None:
        config = {
            "name": "logistic_regression",
            "params": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
            },
        }

    params = config.get("params", {})

    model = LogisticRegression(**params)

    logger.info(f"Training baseline model: {config.get('name')}")
    model.fit(X_train, y_train)

    logger.info("Baseline model trained successfully")

    # Log to MLflow
    try:
        log_run(
            params=params,
            metrics={},
            artifacts={},
            model=model,
            run_name="baseline_logistic",
            experiment_name="fraud_detection",
        )
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    return model