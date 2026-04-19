"""Start the FastAPI server."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def main() -> None:
    """Start the FastAPI server."""
    logger.info("Starting FastAPI server...")

    # Load config
    try:
        config = load_config("configs/service.yaml")
    except FileNotFoundError:
        logger.warning("configs/service.yaml not found, using defaults")
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,
            "log_level": "info",
        }

    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8000)
    reload = config.get("reload", False)
    log_level = config.get("log_level", "info")

    logger.info(f"Server settings: host={host}, port={port}, reload={reload}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()