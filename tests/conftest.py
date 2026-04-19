"""Pytest configuration and fixtures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.pipeline import build_pipeline


@pytest.fixture
def sample_df():
    """Create a small synthetic DataFrame mimicking the creditcard schema."""
    np.random.seed(42)
    n_rows = 100

    # Generate synthetic data
    data = {
        "Time": np.random.uniform(0, 1000, n_rows),
        "V1": np.random.randn(n_rows),
        "V2": np.random.randn(n_rows),
        "V3": np.random.randn(n_rows),
        "V4": np.random.randn(n_rows),
        "V5": np.random.randn(n_rows),
        "V6": np.random.randn(n_rows),
        "V7": np.random.randn(n_rows),
        "V8": np.random.randn(n_rows),
        "V9": np.random.randn(n_rows),
        "V10": np.random.randn(n_rows),
        "V11": np.random.randn(n_rows),
        "V12": np.random.randn(n_rows),
        "V13": np.random.randn(n_rows),
        "V14": np.random.randn(n_rows),
        "V15": np.random.randn(n_rows),
        "V16": np.random.randn(n_rows),
        "V17": np.random.randn(n_rows),
        "V18": np.random.randn(n_rows),
        "V19": np.random.randn(n_rows),
        "V20": np.random.randn(n_rows),
        "V21": np.random.randn(n_rows),
        "V22": np.random.randn(n_rows),
        "V23": np.random.randn(n_rows),
        "V24": np.random.randn(n_rows),
        "V25": np.random.randn(n_rows),
        "V26": np.random.randn(n_rows),
        "V27": np.random.randn(n_rows),
        "V28": np.random.randn(n_rows),
        "Amount": np.random.uniform(0, 500, n_rows),
    }

    df = pd.DataFrame(data)

    # Add target column with ~2% fraud
    fraud_rate = 0.02
    fraud_indices = np.random.choice(
        n_rows, size=int(n_rows * fraud_rate), replace=False
    )
    target = np.zeros(n_rows)
    target[fraud_indices] = 1

    df["Class"] = target

    return df


@pytest.fixture
def fitted_pipeline(sample_df):
    """Create a fitted feature pipeline."""
    pipeline = build_pipeline()

    feature_cols = [c for c in sample_df.columns if c != "Class"]
    X = sample_df[feature_cols]

    pipeline.fit(X)

    return pipeline


@pytest.fixture
def dummy_model(sample_df):
    """Create a trained logistic regression on sample data."""
    feature_cols = [c for c in sample_df.columns if c != "Class"]
    X = sample_df[feature_cols]
    y = sample_df["Class"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)