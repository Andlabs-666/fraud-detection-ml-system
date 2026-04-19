"""Validate raw data against schema."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_raw
from src.validation.schema import validate, validate_and_save
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def main() -> None:
    """Run data validation."""
    logger.info("Starting data validation")

    # Load config
    try:
        config = load_config("configs/data.yaml")
    except FileNotFoundError:
        logger.warning("configs/data.yaml not found, using defaults")
        config = {"raw_path": "data/raw/creditcard.csv"}

    raw_path = config.get("raw_path", "data/raw/creditcard.csv")

    # Load raw data
    df = load_raw(raw_path)

    # Run validation
    report = validate(df)

    # Save report
    validate_and_save(df, "data/validation_reports")

    # Print report
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Rows: {report.stats.get('num_rows', 'N/A')}")
    print(f"Columns: {report.stats.get('num_columns', 'N/A')}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    if report.stats.get("class_distribution"):
        print(f"\nClass distribution: {report.stats['class_distribution']}")

    print("=" * 50)

    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()