import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import app
from config import API_KEY

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["app"] == "🛡️ FraudSense"
    assert "version" in res_json
    assert "/evaluar_transaccion" in res_json["endpoints"]

def test_predict_no_auth():
    data = {
        "amount": 35000,
        "country": "CL",
        "hour": 14,
        "device_type": "iOS",
        "failed_attempts": 0,
        "is_foreign": 0,
        "high_risk_merchant": 0
    }
    response = client.post("/evaluar_transaccion", json=data)
    assert response.status_code == 401

def test_predict_auth():
    data = {
        "amount": 950000,
        "country": "RU",
        "hour": 3,
        "device_type": "Web",
        "failed_attempts": 5,
        "is_foreign": 1,
        "high_risk_merchant": 1
    }
    headers = {"X-API-Key": API_KEY}
    response = client.post("/evaluar_transaccion", json=data, headers=headers)
    assert response.status_code == 200
    res_json = response.json()
    assert "risk_score" in res_json
    assert "is_fraud" in res_json
    assert "reasons" in res_json
