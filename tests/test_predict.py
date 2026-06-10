import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import _preprocess_single, predict_transaction

def test_preprocess_single():
    data = {
        "amount": 35000,
        "country": "CL",
        "hour": 14,
        "device_type": "iOS",
        "failed_attempts": 0,
        "is_foreign": 0,
        "high_risk_merchant": 0
    }
    df = _preprocess_single(data)
    assert df is not None
    assert "amount" in df.columns
    assert df.iloc[0]["is_night"] == 0
    assert df.iloc[0]["is_risk_country"] == 0

def test_predict_transaction_normal():
    data = {
        "amount": 35000,
        "country": "CL",
        "hour": 14,
        "device_type": "iOS",
        "failed_attempts": 0,
        "is_foreign": 0,
        "high_risk_merchant": 0
    }
    result = predict_transaction(data)
    assert "risk_score" in result
    assert "is_fraud" in result
    assert "reasons" in result
    assert isinstance(result["reasons"], list)
    assert result["risk_level"] == "BAJO"

def test_predict_transaction_fraud():
    data = {
        "amount": 950000,
        "country": "RU",
        "hour": 3,
        "device_type": "Web",
        "failed_attempts": 5,
        "is_foreign": 1,
        "high_risk_merchant": 1
    }
    result = predict_transaction(data)
    assert "risk_score" in result
    assert "is_fraud" in result
    assert result["risk_level"] == "ALTO"
