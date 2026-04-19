"""Pydantic schemas for API requests and responses."""
from typing import List, Optional

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Features for a single transaction."""

    Time: float = Field(..., description="Seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "Amount": 149.62,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536346,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.070794,
                "V11": -0.599225,
                "V12": -0.034277,
                "V13": 0.026268,
                "V14": 0.192201,
                "V15": 0.271164,
                "V16": -0.226463,
                "V17": 0.178228,
                "V18": 0.050575,
                "V19": -0.200196,
                "V20": -0.015906,
                "V21": 0.416526,
                "V22": 0.253851,
                "V23": -0.246325,
                "V24": -0.633753,
                "V25": -0.120821,
                "V26": -0.385025,
                "V27": 1.192991,
                "V28": 0.172248,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    model_version: str


class BatchRequest(BaseModel):
    """Request for batch predictions."""

    transactions: List[TransactionFeatures]
    transaction_ids: Optional[List[str]] = None


class BatchResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResponse]
    total: int
    fraud_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_version: str
    threshold: float
    features: List[str]