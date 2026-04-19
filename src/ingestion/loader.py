"""Data loading and splitting utilities."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV data from path.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            "Please run 'make download' first."
        )

    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    if "Class" in df.columns:
        class_counts = df["Class"].value_counts()
        logger.info(
            f"Class distribution: Legitimate={class_counts.get(0, 0)}, "
            f"Fraud={class_counts.get(1, 0)}"
        )

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets using stratified sampling.

    Args:
        df: The input DataFrame.
        target_col: The name of the target column.
        test_size: Proportion of data for test set.
        val_size: Proportion of remaining data for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split: train vs val from the remaining
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, stratify=y_temp
    )

    # Combine back with target
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    logger.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    logger.info(f"Train class dist: {train_df[target_col].value_counts().to_dict()}")
    logger.info(f"Val class dist: {val_df[target_col].value_counts().to_dict()}")
    logger.info(f"Test class dist: {test_df[target_col].value_counts().to_dict()}")

    # Save splits to interim directory
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(interim_dir / "train.parquet", index=False)
    val_df.to_parquet(interim_dir / "val.parquet", index=False)
    test_df.to_parquet(interim_dir / "test.parquet", index=False)

    logger.info(f"Saved splits to {interim_dir}")

    return train_df, val_df, test_df