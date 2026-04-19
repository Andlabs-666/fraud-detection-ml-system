"""Unit tests for schema validation."""
import numpy as np
import pandas as pd
import pytest

from src.validation.schema import (
    validate,
    ValidationReport,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)


def test_valid_dataframe_passes_validation(sample_df):
    """Test that a valid DataFrame passes validation."""
    report = validate(sample_df)
    assert report.passed is True
    assert len(report.errors) == 0


def test_missing_column_fails_validation():
    """Test that missing required columns fail validation."""
    # Create DataFrame missing V1
    df = pd.DataFrame({
        "Time": [1.0, 2.0],
        "V2": [1.0, 2.0],
        "Amount": [100.0, 200.0],
        "Class": [0, 1],
    })

    report = validate(df)
    assert report.passed is False
    assert any("Missing" in error for error in report.errors)


def test_excessive_nulls_fails_validation():
    """Test that excessive nulls fail validation."""
    df = pd.DataFrame({
        "Time": [1.0, np.nan, 3.0],
        "V1": [1.0, np.nan, 3.0],
        "Amount": [100.0, 200.0, 300.0],
        "Class": [0, 1, 0],
    })

    report = validate(df)
    # Check warnings for high null columns
    assert len(report.warnings) > 0 or "null" in str(report.errors).lower()


def test_wrong_target_values_fails_validation():
    """Test that invalid target values fail validation."""
    df = pd.DataFrame({
        "Time": [1.0, 2.0],
        "V1": [1.0, 2.0],
        "Amount": [100.0, 200.0],
        "Class": [0, 2],  # Invalid class
    })

    report = validate(df)
    assert report.passed is False
    assert any("Invalid" in error or "class" in error.lower() for error in report.errors)


def test_target_column_not_found():
    """Test validation when target column is missing."""
    df = pd.DataFrame({
        "Time": [1.0, 2.0],
        "V1": [1.0, 2.0],
        "Amount": [100.0, 200.0],
    })

    report = validate(df)
    assert report.passed is False