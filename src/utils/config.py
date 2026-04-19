"""Configuration loading utilities."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Data configuration."""
    raw_path: str
    interim_path: str
    processed_path: str
    target_column: str
    test_size: float
    val_size: float
    random_seed: int


@dataclass
class ModelConfig:
    """Model configuration."""
    baseline: Dict[str, Any]
    production: Dict[str, Any]


@dataclass
class TrainConfig:
    """Training configuration."""
    experiment_name: str
    run_name: str
    threshold: float
    artifact_dir: str
    mlflow_tracking_uri: str
    log_model: bool


@dataclass
class ServiceConfig:
    """Service configuration."""
    model_path: str
    encoder_path: str
    host: str
    port: int
    reload: bool
    log_level: str


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file into a Python dict.

    Args:
        path: Path to the YAML config file.

    Returns:
        A dictionary containing the config values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_dataclass(path: Union[str, Path], cls: type) -> Any:
    """Load a YAML config file into a dataclass.

    Args:
        path: Path to the YAML config file.
        cls: The dataclass type to instantiate.

    Returns:
        An instance of the dataclass with config values.
    """
    config = load_config(path)
    return cls(**config)


def load_omegaconf(path: Union[str, Path]) -> Any:
    """Load a YAML config using OmegaConf.

    Args:
        path: Path to the YAML config file.

    Returns:
        An OmegaConf configuration object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return OmegaConf.load(path)