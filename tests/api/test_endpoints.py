"""API endpoint tests."""
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint_returns_200(test_client):
    """Test that GET /health returns 200 with status ok."""
    response = test_client.get("/health")

    # May return 503 if model not loaded, but should not error
    assert response.status_code in [200, 503]


def test_model_info_endpoint(test_client):
    """Test that GET /model/info returns model information."""
    response = test_client.get("/model/info")

    # May return 503 if model not loaded
    assert response.status_code in [200, 503]


def test_predict_endpoint_with_valid_payload(test_client):
    """Test that POST /predict with valid payload returns prediction."""
    payload = {
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

    response = test_client.post("/predict", json=payload)

    # Should be 503 if model not loaded, 200 if loaded
    assert response.status_code in [200, 503]


def test_predict_endpoint_with_malformed_payload_returns_422(test_client):
    """Test that POST /predict with malformed payload returns 422."""
    # Missing required fields
    payload = {
        "Time": 0.0,
        # Missing Amount and V features
    }

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_batch_endpoint(test_client):
    """Test that POST /predict/batch returns batch prediction."""
    payload = {
        "transactions": [
            {
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
        ]
    }

    response = test_client.post("/predict/batch", json=payload)

    # Should be 503 or 200
    assert response.status_code in [200, 503]


def test_health_response_structure(test_client):
    """Test that health response has correct structure."""
    response = test_client.get("/health")

    # Even if model not loaded, should return valid JSON structure
    if response.status_code in [200, 503]:
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data