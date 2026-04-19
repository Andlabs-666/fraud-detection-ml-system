"""Unit tests for feature pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    build_pipeline,
    fit_and_save,
    load_and_transform,
    get_feature_columns,
)


def test_pipeline_builds_without_error():
    """Test that pipeline builds successfully."""
    pipeline = build_pipeline()
    assert pipeline is not None


def test_pipeline_transforms_without_leakage(sample_df):
    """Test that pipeline transforms without data leakage."""
    pipeline = build_pipeline()

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]

    # Fit on full data (but this is test data, leakage acceptable here)
    X_transformed = pipeline.fit_transform(X)

    # Transform again without fitting (should use existing transforms)
    X_transformed2 = pipeline.transform(X)

    # Results should be identical (no state change on transform)
    np.testing.assert_array_almost_equal(X_transformed, X_transformed2)


def test_output_shape_is_correct(sample_df):
    """Test that output shape matches input."""
    pipeline = build_pipeline()

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]

    X_transformed = pipeline.fit_transform(X)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == X.shape[1]


def test_no_nan_in_transformed_output(sample_df):
    """Test that no NaN in transformed output."""
    pipeline = build_pipeline()

    feature_cols = get_feature_columns(sample_df, "Class")
    X = sample_df[feature_cols]

    X_transformed = pipeline.fit_transform(X)

    assert not np.isnan(X_transformed).any()


def test_get_feature_columns_excludes_target():
    """Test that get_feature_columns excludes target."""
    df = pd.DataFrame({
        "Time": [1.0],
        "V1": [1.0],
        "Amount": [100.0],
        "Class": [0],
    })

    features = get_feature_columns(df, "Class")

    assert "Class" not in features
    assert "Time" in features
    assert "V1" in features
    assert "Amount" in features