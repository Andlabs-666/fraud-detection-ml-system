"""Feature engineering pipeline."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_columns(df: pd.DataFrame, target_col: str = "Class") -> List[str]:
    """Get feature column names excluding target.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.

    Returns:
        List of feature column names.
    """
    return [c for c in df.columns if c != target_col]


def build_pipeline(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Build a preprocessing pipeline.

    Args:
        config: Optional configuration dict.

    Returns:
        A sklearn Pipeline with transformers.
    """
    # Define numeric features (all features are numeric in this dataset)
    numeric_features = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]

    # Use RobustScaler for outlier resistance
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Build column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    logger.info("Built preprocessing pipeline")
    return pipeline


def fit_and_save(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    path: str,
) -> None:
    """Fit pipeline on training data and save.

    Args:
        pipeline: The sklearn pipeline to fit.
        X_train: Training features.
        path: Path to save the fitted pipeline.
    """
    target_col = "Class"
    feature_cols = get_feature_columns(X_train, target_col)

    X_features = X_train[feature_cols]

    logger.info(f"Fitting pipeline on {len(X_features)} samples")
    pipeline.fit(X_features)

    # Save
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved pipeline to {save_path}")


def load_and_transform(
    pipeline_path: str,
    X: pd.DataFrame,
) -> np.ndarray:
    """Load saved pipeline and transform data.

    Args:
        pipeline_path: Path to the saved pipeline.
        X: Input features DataFrame.

    Returns:
        Transformed features as numpy array.
    """
    import joblib

    pipeline = joblib.load(pipeline_path)

    target_col = "Class"
    feature_cols = get_feature_columns(X, target_col)
    X_features = X[feature_cols]

    X_transformed = pipeline.transform(X_features)

    logger.info(f"Transformed {X_transformed.shape[0]} samples")
    return X_transformed


def build_and_save_pipeline(
    X_train: pd.DataFrame,
    output_path: str,
) -> Pipeline:
    """Build, fit, and save the preprocessing pipeline.

    Args:
        X_train: Training features DataFrame.
        output_path: Path to save the pipeline.

    Returns:
        Fitted pipeline.
    """
    pipeline = build_pipeline()
    fit_and_save(pipeline, X_train, output_path)
    return pipeline