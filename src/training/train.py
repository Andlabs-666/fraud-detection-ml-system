"""Production model training (XGBoost)."""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from src.utils.logger import get_logger
from src.registry.mlflow_registry import log_run

logger = get_logger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Model configuration dict.

    Returns:
        Fitted XGBoost classifier.
    """
    if config is None:
        config = {
            "name": "xgboost",
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight": 578,
                "eval_metric": "aucpr",
                "random_state": 42,
                "use_label_encoder": False,
            },
        }

    params = config.get("params", {})

    # Remove use_label_encoder for newer XGBoost versions
    params = {k: v for k, v in params.items() if k != "use_label_encoder"}

    model = xgb.XGBClassifier(**params)

    logger.info(f"Training production model: {config.get('name')}")
    logger.info(f"Parameters: {params}")

    model.fit(X_train, y_train)

    logger.info("Production model trained successfully")

    # Log to MLflow
    try:
        run_id = log_run(
            params=params,
            metrics={},
            artifacts={},
            model=model,
            run_name="xgboost_production",
            experiment_name="fraud_detection",
        )
        logger.info(f"MLflow run ID: {run_id}")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    return model


def save_model(model: xgb.XGBClassifier, path: str) -> None:
    """Save trained model to disk.

    Args:
        model: Trained model.
        path: Path to save the model.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path)
    logger.info(f"Saved model to {save_path}")


def load_model(path: str) -> xgb.XGBClassifier:
    """Load a saved model from disk.

    Args:
        path: Path to the saved model.

    Returns:
        Loaded model.
    """
    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model