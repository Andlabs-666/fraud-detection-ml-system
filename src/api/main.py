"""FastAPI application for fraud detection service."""
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    TransactionFeatures,
)
from src.api.predictor import FraudPredictor
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# Global predictor instance
predictor: FraudPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global predictor

    # Startup
    logger.info("Starting fraud detection service...")

    try:
        config = load_config("configs/service.yaml")
    except FileNotFoundError:
        logger.warning("configs/service.yaml not found, using defaults")
        config = {
            "model_path": "artifacts/models/production_model.joblib",
            "encoder_path": "artifacts/encoders/preprocessor.joblib",
        }

    model_path = config.get("model_path", "artifacts/models/production_model.joblib")
    encoder_path = config.get("encoder_path", "artifacts/encoders/preprocessor.joblib")

    predictor = FraudPredictor(
        model_path=model_path,
        encoder_path=encoder_path,
        threshold=0.5,
        model_version="1.0.0",
    )

    try:
        predictor.load()
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}")
        logger.warning("Service starting in limited mode - run 'make train' first")

    logger.info("Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down fraud detection service...")


app = FastAPI(
    title="Fraud Detection API",
    description="Production-grade ML service for transaction fraud detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global predictor

    model_loaded = predictor is not None and predictor.is_loaded()
    model_version = "1.0.0" if predictor is None else predictor.model_version
    uptime = 0.0 if predictor is None else predictor.get_uptime()

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=uptime,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    global predictor

    if predictor is None or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return ModelInfoResponse(
        model_name="FraudDetectionModel",
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        features=predictor.feature_columns,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures):
    """Predict fraud for a single transaction."""
    global predictor

    if predictor is None or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' first.",
        )

    try:
        # Convert Pydantic model to dict
        features = transaction.model_dump()

        result = predictor.predict_one(features)

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict fraud for a batch of transactions."""
    global predictor

    if predictor is None or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' first.",
        )

    # Limit batch size
    if len(request.transactions) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Maximum batch size is 1000 transactions",
        )

    try:
        # Convert transactions to dicts
        features_list = [t.model_dump() for t in request.transactions]

        result = predictor.predict_batch(
            features_list,
            request.transaction_ids,
        )

        return result

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )