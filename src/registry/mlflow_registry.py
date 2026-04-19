"""MLflow model registry and tracking."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


def log_run(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Dict[str, Any],
    model: Any,
    run_name: str,
    experiment_name: str = "fraud_detection",
) -> str:
    """Log a run to MLflow.

    Args:
        params: Model parameters to log.
        metrics: Metrics to log.
        artifacts: Artifact paths to log.
        model: Trained model to log.
        run_name: Name for the run.
        experiment_name: Name of the experiment.

    Returns:
        Run ID.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            logger.warning(f"Could not log sklearn model: {e}")

        logger.info(f"Logged run {run_id} to MLflow")

    return run_id


def load_best_model(
    experiment_name: str = "fraud_detection",
    metric: str = "pr_auc",
) -> Tuple[Any, str]:
    """Load the best model from MLflow.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to sort by.

    Returns:
        Tuple of (model, run_id).
    """
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]
    run_id = best_run.info.run_id

    # Load model from the best run
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info(f"Loaded best model from run {run_id}")
    return model, run_id


def promote_model(
    run_id: str,
    model_name: str = "FraudDetectionModel",
    stage: str = "Production",
) -> None:
    """Promote a model to a specific stage in the registry.

    Args:
        run_id: MLflow run ID.
        model_name: Name for the registered model.
        stage: Stage to promote to (Production, Staging, etc).
    """
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"

    # Register model
    registered_model = mlflow.register_model(model_uri, model_name)
    logger.info(f"Registered model {model_name}: {registered_model.version}")

    # Transition to stage
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage=stage,
    )

    logger.info(f"Promoted model {model_name} to {stage}")


def get_experiment_runs(
    experiment_name: str = "fraud_detection",
) -> list:
    """Get all runs from an experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        List of runs.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    return runs