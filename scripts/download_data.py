"""Download the credit card fraud dataset."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_data(output_path: str = "data/raw/creditcard.csv") -> None:
    """Download the credit card fraud dataset.

    Args:
        output_path: Path to save the CSV file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Try multiple URLs
    urls_to_try = [
        "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        "https://raw.githubusercontent.com/IBM/cost-analytics-credit-card-fraud/master/data/creditcard.csv",
    ]

    for DATA_URL in urls_to_try:
        logger.info(f"Trying to download from {DATA_URL}")
        try:
            import requests
            response = requests.get(DATA_URL, timeout=120, verify=False)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded to {output_file}")

                # Verify
                df = pd.read_csv(output_file, nrows=5)
                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                return
        except Exception as e:
            logger.warning(f"Failed: {e}")
            continue

    # If download fails, create synthetic dataset that mimics the real one
    logger.warning("Could not download - creating synthetic dataset for testing")
    logger.warning("Replace with real data at: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

    np.random.seed(42)
    n_normal = 10000
    n_fraud = 50  # ~0.5% fraud rate

    # Normal transactions
    normal_data = {
        "Time": np.random.uniform(0, 172800, n_normal),  # 2 days in seconds
        "Amount": np.random.lognormal(4.5, 1.2, n_normal),
    }
    for i in range(1, 29):
        normal_data[f"V{i}"] = np.random.normal(0, 1, n_normal)

    normal_df = pd.DataFrame(normal_data)
    normal_df["Class"] = 0

    # Fraud transactions (higher amounts, different patterns)
    fraud_data = {
        "Time": np.random.uniform(0, 172800, n_fraud),
        "Amount": np.random.lognormal(5.5, 1.5, n_fraud),
    }
    for i in range(1, 29):
        fraud_data[f"V{i}"] = np.random.normal(0.5, 1.5, n_fraud)

    fraud_df = pd.DataFrame(fraud_data)
    fraud_df["Class"] = 1

    # Combine
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False)
    logger.info(f"Created synthetic dataset: {len(df)} rows at {output_file}")
    logger.info(f"Fraud rate: {df['Class'].mean()*100:.2f}%")


if __name__ == "__main__":
    download_data()