"""
FraudSense — Módulo de Inferencia
Carga el modelo entrenado y predice el riesgo de fraude en una transacción.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ENCODERS_FILE,
    HIGH_AMOUNT_THRESHOLD,
    MODEL_FILE,
    NIGHT_HOURS,
    RISK_COUNTRIES,
    RISK_THRESHOLDS,
)
from src.preprocessing import FEATURE_COLUMNS

# ──────────────────────────────────────────────────────────────────────────────
# Carga del Modelo
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_encoders = None


def _load_artifacts():
    """Carga modelo y encoders (lazy-loading con caché en memoria)."""
    global _model, _encoders
    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Modelo no encontrado en: {MODEL_FILE}\nEjecuta primero: python src/train_model.py"
            )
        _model = joblib.load(MODEL_FILE)
        _encoders = joblib.load(ENCODERS_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocesamiento de una sola transacción
# ──────────────────────────────────────────────────────────────────────────────


def _preprocess_single(data: dict) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de features que en entrenamiento
    para una sola transacción en formato diccionario.
    """
    _load_artifacts()

    amount = float(data.get("amount", 0))
    country = str(data.get("country", "CL")).upper()
    hour = int(data.get("hour", 12))
    device = str(data.get("device_type", "Web"))
    fails = int(data.get("failed_attempts", 0))
    foreign = int(data.get("is_foreign", 0))
    hrm = int(data.get("high_risk_merchant", 0))

    # Feature engineering (idéntico a preprocessing.py)
    is_night = 1 if hour in NIGHT_HOURS else 0
    high_amount = 1 if amount > HIGH_AMOUNT_THRESHOLD else 0
    is_risk_ctry = 1 if country in RISK_COUNTRIES else 0

    # Encodings
    def safe_encode(le, val):
        if val in le.classes_:
            return int(le.transform([val])[0])
        return int(le.transform([le.classes_[0]])[0])

    enc_country = safe_encode(_encoders["country"], country)
    enc_device = safe_encode(_encoders["device_type"], device)

    # Z-score del monto usa estadísticas del entrenamiento
    mean_amount = 245_000
    std_amount = 320_000
    amount_z = round((amount - mean_amount) / std_amount, 4)

    risk_heuristic = fails * 0.3 + foreign * 0.2 + hrm * 0.2 + is_night * 0.15 + high_amount * 0.15

    row = {
        "amount": amount,
        "country": enc_country,
        "hour": hour,
        "device_type": enc_device,
        "failed_attempts": fails,
        "is_foreign": foreign,
        "high_risk_merchant": hrm,
        "is_night": is_night,
        "high_amount": high_amount,
        "amount_zscore": amount_z,
        "is_risk_country": is_risk_ctry,
        "risk_score_heuristic": risk_heuristic,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────────
# Predicción principal
# ──────────────────────────────────────────────────────────────────────────────


def predict_transaction(data: dict) -> dict:
    """
    Evalúa una transacción y retorna el resultado de riesgo.

    Args:
        data: dict con keys: amount, country, hour, device_type,
              failed_attempts, is_foreign, high_risk_merchant

    Returns:
        {
            "risk_score":     float (0–1),
            "is_fraud":       bool,
            "risk_level":     "BAJO" | "MEDIO" | "ALTO",
            "recommendation": str,
            "input":          dict (eco de la entrada)
        }
    """
    X = _preprocess_single(data)
    prob = float(_model.predict_proba(X)[0, 1])

    # Nivel de riesgo
    if prob < RISK_THRESHOLDS["bajo"]:
        level = "BAJO"
        recommendation = "✅ APROBAR TRANSACCIÓN"
    elif prob < RISK_THRESHOLDS["medio"]:
        level = "MEDIO"
        recommendation = "⚠️ REVISAR MANUALMENTE"
    else:
        level = "ALTO"
        recommendation = "🚨 BLOQUEAR TRANSACCIÓN"

    return {
        "risk_score": round(prob, 4),
        "is_fraud": prob >= RISK_THRESHOLDS["medio"],
        "risk_level": level,
        "recommendation": recommendation,
        "input": data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test rápido
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    casos = [
        {
            "nombre": "Transacción Normal",
            "data": {
                "amount": 35000,
                "country": "CL",
                "hour": 14,
                "device_type": "iOS",
                "failed_attempts": 0,
                "is_foreign": 0,
                "high_risk_merchant": 0,
            },
        },
        {
            "nombre": "Transacción Sospechosa",
            "data": {
                "amount": 950000,
                "country": "RU",
                "hour": 3,
                "device_type": "Unknown",
                "failed_attempts": 5,
                "is_foreign": 1,
                "high_risk_merchant": 1,
            },
        },
        {
            "nombre": "Caso Intermedio",
            "data": {
                "amount": 250000,
                "country": "AR",
                "hour": 22,
                "device_type": "Android",
                "failed_attempts": 2,
                "is_foreign": 0,
                "high_risk_merchant": 1,
            },
        },
    ]

    print("=" * 55)
    print("  FraudSense — Test de Predicción")
    print("=" * 55)

    for caso in casos:
        result = predict_transaction(caso["data"])
        print(f"\n📋 {caso['nombre']}")
        print(f"   Monto       : ${caso['data']['amount']:,}")
        print(f"   País        : {caso['data']['country']}")
        print(f"   Hora        : {caso['data']['hour']}:00")
        print(f"   Risk Score  : {result['risk_score']:.1%}")
        print(f"   Nivel       : {result['risk_level']}")
        print(f"   Acción      : {result['recommendation']}")
