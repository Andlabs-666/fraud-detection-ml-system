"""Unit tests for evaluation metrics."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import (
    evaluate,
    find_optimal_threshold,
)


def test_evaluate_returns_all_expected_keys(dummy_model, sample_df):
    """Test that evaluate returns all expected metric keys."""
    from src.features.pipeline import get_feature_columns

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]
    y = sample_df["Class"]

    metrics = evaluate(dummy_model, X, y, threshold=0.5)

    expected_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "confusion_matrix"]

    for key in expected_keys:
        assert key in metrics


def test_pr_auc_is_between_0_and_1(dummy_model, sample_df):
    """Test that PR AUC is between 0 and 1."""
    from src.features.pipeline import get_feature_columns

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]
    y = sample_df["Class"]

    metrics = evaluate(dummy_model, X, y, threshold=0.5)

    assert 0 <= metrics["pr_auc"] <= 1


def test_roc_auc_is_between_0_and_1(dummy_model, sample_df):
    """Test that ROC AUC is between 0 and 1."""
    from src.features.pipeline import get_feature_columns

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]
    y = sample_df["Class"]

    metrics = evaluate(dummy_model, X, y, threshold=0.5)

    assert 0 <= metrics["roc_auc"] <= 1


def test_find_optimal_threshold_returns_float_between_0_and_1(dummy_model, sample_df):
    """Test that find_optimal_threshold returns a value between 0 and 1."""
    from src.features.pipeline import get_feature_columns

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]
    y = sample_df["Class"]

    threshold = find_optimal_threshold(dummy_model, X, y)

    assert 0 <= threshold <= 1
    assert isinstance(threshold, float)


def test_metrics_with_different_thresholds():
    """Test that metrics change with different thresholds."""
    # Use simple data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    metrics_05 = evaluate(model, X, y, threshold=0.5)
    metrics_03 = evaluate(model, X, y, threshold=0.3)

    # With lower threshold, recall should be higher or same
    assert metrics_03["recall"] >= metrics_05["recall"]


def test_find_optimal_threshold_on_balanced_data():
    """Test threshold finding on more balanced data."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    threshold = find_optimal_threshold(model, X, y)

    assert 0.01 <= threshold <= 0.99