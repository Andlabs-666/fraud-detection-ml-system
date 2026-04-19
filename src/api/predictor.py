"""Fraud prediction wrapper class."""
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.api.schemas import PredictionResponse, BatchResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    """Wrapper class for fraud prediction."""

    def __init__(
        self,
        model_path: str = "artifacts/models/production_model.joblib",
        encoder_path: str = "artifacts/encoders/preprocessor.joblib",
        threshold: float = 0.5,
        model_version: str = "1.0.0",
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to the saved model.
            encoder_path: Path to the saved encoder.
            threshold: Classification threshold.
            model_version: Version string for the model.
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.threshold = threshold
        self.model_version = model_version

        self.model = None
        self.encoder = None
        self.start_time = time.time()
        self.feature_columns = [
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
            "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
            "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]

    def load(self) -> None:
        """Load model and encoder from disk."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading encoder from {self.encoder_path}")
        self.encoder = joblib.load(self.encoder_path)

        logger.info("Model and encoder loaded successfully")

    def _transform_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Transform input features to model input format.

        Args:
            features: Dictionary of feature values.

        Returns:
            Transformed features as numpy array.
        """
        import pandas as pd

        # Extract features in the correct order as DataFrame
        feature_values = {col: features.get(col, 0.0) for col in self.feature_columns}
        X_df = pd.DataFrame([feature_values])

        # Transform using encoder (handles DataFrame with column names)
        X_transformed = self.encoder.transform(X_df)

        return X_transformed

    def predict_one(self, features: Dict[str, Any], transaction_id: Optional[str] = None) -> PredictionResponse:
        """Predict fraud for a single transaction.

        Args:
            features: Dictionary of feature values.
            transaction_id: Optional transaction ID.

        Returns:
            PredictionResponse with prediction results.
        """
        if self.model is None or self.encoder is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Generate transaction ID if not provided
        if transaction_id is None:
            transaction_id = str(uuid.uuid4())[:8]

        # Transform and predict
        X = self._transform_features(features)
        fraud_prob = self.model.predict_proba(X)[0, 1]
        is_fraud = bool(fraud_prob >= self.threshold)

        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_prob),
            is_fraud=is_fraud,
            threshold_used=self.threshold,
            model_version=self.model_version,
        )

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        transaction_ids: Optional[List[str]] = None,
    ) -> BatchResponse:
        """Predict fraud for a batch of transactions.

        Args:
            features_list: List of feature dictionaries.
            transaction_ids: Optional list of transaction IDs.

        Returns:
            BatchResponse with predictions.
        """
        if self.model is None or self.encoder is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Generate transaction IDs if not provided
        if transaction_ids is None:
            transaction_ids = [str(uuid.uuid4())[:8] for _ in range(len(features_list))]
        elif len(transaction_ids) < len(features_list):
            transaction_ids = transaction_ids + [
                str(uuid.uuid4())[:8] for _ in range(len(features_list) - len(transaction_ids))
            ]

        # Transform all features
        import pandas as pd

        X_dicts = [
            {col: f.get(col, 0.0) for col in self.feature_columns}
            for f in features_list
        ]
        X_df = pd.DataFrame(X_dicts)
        X_transformed = self.encoder.transform(X_df)

        # Predict
        fraud_probs = self.model.predict_proba(X_transformed)[:, 1]
        is_fraud_arr = fraud_probs >= self.threshold

        # Build responses
        predictions = []
        fraud_count = 0

        for i, (fraud_prob, is_fraud) in enumerate(zip(fraud_probs, is_fraud_arr)):
            predictions.append(PredictionResponse(
                transaction_id=transaction_ids[i],
                fraud_probability=float(fraud_prob),
                is_fraud=bool(is_fraud),
                threshold_used=self.threshold,
                model_version=self.model_version,
            ))
            if is_fraud:
                fraud_count += 1

        return BatchResponse(
            predictions=predictions,
            total=len(predictions),
            fraud_count=fraud_count,
        )

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.encoder is not None