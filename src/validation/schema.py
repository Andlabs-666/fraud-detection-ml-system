"""Schema validation for fraud detection data."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected schema for credit card fraud dataset
EXPECTED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "Class"
]

# Target column
TARGET_COLUMN = "Class"
TARGET_CLASSES = [0, 1]

# Maximum allowed null proportion
MAX_NULL_PROPORTION = 0.5


@dataclass
class ValidationReport:
    """Report of data validation results."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }

    def save(self, path: Path) -> None:
        """Save report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved validation report to {path}")


def validate(df: pd.DataFrame) -> ValidationReport:
    """Validate a DataFrame against the expected schema.

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationReport with results.
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    # Check columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    else:
        logger.info("All required columns present")

    # Check target column
    if TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' not found")
    else:
        unique_classes = df[TARGET_COLUMN].unique()
        invalid_classes = set(unique_classes) - set(TARGET_CLASSES)
        if invalid_classes:
            errors.append(f"Invalid target classes: {invalid_classes}")
        if df[TARGET_COLUMN].isna().any():
            errors.append("Target column contains null values")

    # Check null proportions
    null_proportions = df.isnull().mean()
    high_null_cols = null_proportions[null_proportions > MAX_NULL_PROPORTION]
    if not high_null_cols.empty:
        warnings.append(
            f"Columns with >{MAX_NULL_PROPORTION*100}% nulls: {list(high_null_cols.index)}"
        )

    # Check numeric columns are numeric
    feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    non_numeric = []
    for col in feature_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")

    # Compute statistics
    stats = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "null_proportions": null_proportions.to_dict(),
    }

    if TARGET_COLUMN in df.columns:
        stats["class_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()

    # Determine pass/fail
    passed = len(errors) == 0

    report = ValidationReport(passed=passed, errors=errors, warnings=warnings, stats=stats)

    if passed:
        logger.info("Validation PASSED")
    else:
        logger.error(f"Validation FAILED: {errors}")

    for warning in warnings:
        logger.warning(warning)

    return report


def validate_and_save(df: pd.DataFrame, output_dir: str) -> ValidationReport:
    """Validate data and save report.

    Args:
        df: DataFrame to validate.
        output_dir: Directory to save report.

    Returns:
        ValidationReport.
    """
    report = validate(df)
    output_path = Path(output_dir) / "validation_report.json"
    report.save(output_path)
    return report