"""
Tests for the Fraud Detection API.

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_transaction() -> dict:
    """A realistic sample transaction matching TransactionInput schema."""
    return {
        "trans_date_trans_time": "2019-06-15 14:32:00",
        "cc_num": 4263982640269299,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "grocery_pos",
        "amt": 124.65,
        "first": "Jennifer",
        "last": "Banks",
        "gender": "F",
        "street": "561 Perry Cove",
        "city": "Jesup",
        "state": "GA",
        "zip": 31599,
        "lat": 31.5988,
        "long": -81.8826,
        "city_pop": 3495,
        "job": "Psychologist, counselling",
        "dob": "1988-03-09",
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time": 1371816865,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
    }


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


class TestRoot:
    def test_root_returns_welcome(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "message" in body
        assert "docs" in body

    def test_root_contains_version(self, client):
        body = client.get("/").json()
        assert body["version"] == "1.0.0"


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "model_loaded" in body
        assert "version" in body


class TestPredict:
    def test_predict_returns_200_or_503(self, client, sample_transaction):
        """If model is loaded → 200, otherwise → 503."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code in (200, 503)

    def test_predict_invalid_payload(self, client):
        """Missing required fields should return 422."""
        response = client.post("/predict", json={"amt": 100})
        assert response.status_code == 422

    def test_predict_response_schema(self, client, sample_transaction):
        response = client.post("/predict", json=sample_transaction)
        if response.status_code == 200:
            body = response.json()
            assert "is_fraud" in body
            assert "fraud_probability" in body
            assert "risk_level" in body
            assert 0 <= body["fraud_probability"] <= 1


class TestBatchPredict:
    def test_batch_predict(self, client, sample_transaction):
        payload = {"transactions": [sample_transaction, sample_transaction]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            body = response.json()
            assert body["total"] == 2
            assert "fraud_rate" in body

    def test_batch_empty_list(self, client):
        """Empty list should fail validation (min_length=1)."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422
