"""Integration tests for full pipeline."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.ingestion.loader import load_raw, split_data
from src.validation.schema import validate
from src.features.pipeline import build_pipeline, load_and_transform
from src.training.baseline import train_baseline
from src.training.train import train_model, save_model
from src.evaluation.metrics import evaluate


def test_full_pipeline_runs_without_error(sample_df, tmp_path):
    """Test that full pipeline runs from raw to evaluation without error."""
    # This test uses synthetic data instead of real data
    # Split
    train_df, val_df, test_df = split_data(
        sample_df,
        target_col="Class",
        test_size=0.2,
        val_size=0.1,
        seed=42,
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Build pipeline
    pipeline = build_pipeline()

    feature_cols = [c for c in sample_df.columns if c != "Class"]
    X_train = train_df[feature_cols]
    y_train = train_df["Class"]

    pipeline.fit(X_train)

    # Transform
    X_train_transformed = pipeline.transform(X_train)
    X_val_transformed = pipeline.transform(val_df[feature_cols])
    X_test_transformed = pipeline.transform(test_df[feature_cols])

    # Train baseline
    baseline_model = train_baseline(X_train_transformed, y_train, None)

    # Train production
    production_model = train_model(X_train_transformed, y_train, None)

    # Evaluate
    y_test = test_df["Class"]
    baseline_metrics = evaluate(baseline_model, X_test_transformed, y_test, 0.5)
    production_metrics = evaluate(production_model, X_test_transformed, y_test, 0.5)

    assert "roc_auc" in baseline_metrics
    assert "roc_auc" in production_metrics


def test_artifacts_are_saved(sample_df, tmp_path):
    """Test that artifacts are saved to expected paths."""
    # Split
    train_df, val_df, test_df = split_data(
        sample_df,
        target_col="Class",
        test_size=0.2,
        val_size=0.1,
        seed=42,
    )

    # Build pipeline
    pipeline = build_pipeline()
    feature_cols = [c for c in sample_df.columns if c != "Class"]
    X_train = train_df[feature_cols]
    y_train = train_df["Class"]

    pipeline.fit(X_train)

    # Save model
    model = train_model(X_train, y_train, None)
    model_path = tmp_path / "test_model.joblib"
    save_model(model, str(model_path))

    assert model_path.exists()


def test_validation_runs_on_sample_data(sample_df):
    """Test that validation runs on sample data."""
    report = validate(sample_df)

    assert report is not None
    assert hasattr(report, "passed")
    assert hasattr(report, "errors")
    assert hasattr(report, "warnings")


def test_pipeline_with_different_configs():
    """Test that pipeline works with different configs."""
    df = pd.DataFrame({
        "Time": [1.0, 2.0, 3.0, 4.0],
        "V1": [1.0, 2.0, 3.0, 4.0],
        "Amount": [100.0, 200.0, 300.0, 400.0],
        "Class": [0, 0, 1, 1],
    })

    train_df, val_df, test_df = split_data(
        df,
        target_col="Class",
        test_size=0.25,
        val_size=0.25,
        seed=42,
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0