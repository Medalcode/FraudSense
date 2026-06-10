"""
FraudSense — Módulo de Inferencia
Carga el modelo entrenado y predice el riesgo de fraude en una transacción.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pandas as pd
import shap
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PIPELINE_FILE,
    RISK_THRESHOLDS,
)
from src.preprocessing import engineer_features, FEATURE_COLUMNS
from src.db import get_connection, init_db


# ──────────────────────────────────────────────────────────────────────────────
# Carga del Modelo
# ──────────────────────────────────────────────────────────────────────────────

_pipeline = None
_explainer = None

def _load_artifacts():
    """Carga pipeline completo (lazy-loading con caché en memoria)."""
    global _pipeline, _explainer
    if _pipeline is None:
        if not os.path.exists(PIPELINE_FILE):
            raise FileNotFoundError(
                f"Modelo no encontrado en: {PIPELINE_FILE}\n"
                "Ejecuta primero: python src/train_model.py"
            )
        _pipeline = joblib.load(PIPELINE_FILE)
        
        # Inicializar SHAP Explainer con el modelo XGBoost interno
        xgb_model = _pipeline.named_steps["classifier"]
        _explainer = shap.TreeExplainer(xgb_model)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocesamiento de una sola transacción
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_single(data: dict) -> pd.DataFrame:
    """
    Aplica feature engineering inicial a la entrada.
    El resto de transformaciones (OneHot, Scaling) las hace el pipeline.
    """
    amount   = float(data.get("amount", 0))
    country  = str(data.get("country", "CL")).upper()
    hour     = int(data.get("hour", 12))
    device   = str(data.get("device_type", "Web"))
    fails    = int(data.get("failed_attempts", 0))
    foreign  = int(data.get("is_foreign", 0))
    hrm      = int(data.get("high_risk_merchant", 0))

    row = {
        "amount":                 amount,
        "country":                country,
        "hour":                   hour,
        "device_type":            device,
        "failed_attempts":        fails,
        "is_foreign":             foreign,
        "high_risk_merchant":     hrm,
    }

    df = pd.DataFrame([row])
    # Aplicar las mismas reglas que en entrenamiento
    df = engineer_features(df)
    
    # Asegurar que z-score se calcule bien si el batch size es 1. 
    # Idealmente en production se usaría un StandardScaler para esto también, 
    # pero engineer_features usa la std/mean de sí mismo.
    # Si df tiene len 1, df["amount"].std() es NaN.
    # Corrección rápida para inferencia:
    mean_amount = 245_000
    std_amount  = 320_000
    df["amount_zscore"] = round((amount - mean_amount) / std_amount, 4)

    return df[FEATURE_COLUMNS]


def _get_reasons(X_raw: pd.DataFrame, prob: float) -> list[str]:
    """Usa SHAP para explicar qué variables afectaron más la decisión."""
    # Transformar los datos como lo hace el pipeline antes de entrar al XGBoost
    preprocessor = _pipeline.named_steps["preprocessor"]
    X_processed = preprocessor.transform(X_raw)
    
    # Calcular valores SHAP
    shap_values = _explainer.shap_values(X_processed)
    
    # Extraer nombres de features transformados
    feature_names = preprocessor.get_feature_names_out()
    
    # Emparejar nombre de feature con su valor SHAP (impacto)
    # shap_values[0] porque procesamos 1 sola transacción
    contributions = list(zip(feature_names, shap_values[0]))
    
    # Ordenar por impacto absoluto (los que más movieron la aguja, ya sea para fraude o legítima)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    reasons = []
    
    # Diccionario amigable para el humano
    name_map = {
        "amount": "Monto de la transacción",
        "country": "País de origen",
        "hour": "Hora de la transacción",
        "device_type": "Dispositivo utilizado",
        "failed_attempts": "Intentos fallidos previos",
        "is_foreign": "Transacción internacional",
        "high_risk_merchant": "Comercio de alto riesgo",
        "is_night": "Horario nocturno",
        "high_amount": "Monto anormalmente alto",
        "amount_zscore": "Desviación del monto habitual",
        "is_risk_country": "País catalogado como riesgoso",
        "risk_score_heuristic": "Comportamiento heurístico sospechoso"
    }

    # Extraer los 3 motivos principales
    for feat, shap_val in contributions[:3]:
        if abs(shap_val) < 0.1:  # Ignorar si el impacto es marginal
            continue
            
        feat_name = feat.split("__")[-1] # Limpiar nombre (ej: num__amount -> amount)
        friendly_name = name_map.get(feat_name, feat_name)
        
        # En XGBoost binario, shap_val > 0 empuja hacia clase 1 (Fraude)
        if shap_val > 0:
            reasons.append(f"🚩 {friendly_name} incrementa significativamente el riesgo.")
        else:
            reasons.append(f"✅ {friendly_name} es consistente con actividad legítima.")
            
    if not reasons:
        if prob > 0.6:
            reasons.append("🚩 Múltiples factores menores suman un patrón de riesgo alto.")
        else:
            reasons.append("✅ Perfil de transacción completamente normal.")
            
    return reasons


# ──────────────────────────────────────────────────────────────────────────────
# Persistencia en BD
# ──────────────────────────────────────────────────────────────────────────────

def _save_to_db(data: dict, risk_score: float, risk_level: str, recommendation: str, reasons: list):
    """Guarda la transacción y la alerta generada en la base de datos."""
    try:
        init_db()
        with get_connection() as conn:
            # 1. Insertar transacción
            cur = conn.execute("""
                INSERT INTO transactions 
                (amount, country, hour, device_type, failed_attempts, is_foreign, high_risk_merchant)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("amount"), data.get("country"), data.get("hour"), 
                data.get("device_type"), data.get("failed_attempts"), 
                data.get("is_foreign"), data.get("high_risk_merchant")
            ))
            tx_id = cur.lastrowid
            
            # 2. Insertar alerta
            reasons_json = json.dumps(reasons) if reasons else None
            conn.execute("""
                INSERT INTO fraud_alerts 
                (transaction_id, risk_score, risk_level, recommendation, reasons)
                VALUES (?, ?, ?, ?, ?)
            """, (tx_id, risk_score, risk_level, recommendation, reasons_json))
            
            # 3. Insertar score histórico
            conn.execute("""
                INSERT INTO risk_scores 
                (transaction_id, model_version, score)
                VALUES (?, ?, ?)
            """, (tx_id, "1.1.0-pipeline", risk_score))
            
    except Exception as e:
        print(f"Error guardando en BD: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Predicción principal
# ──────────────────────────────────────────────────────────────────────────────

def predict_transaction(data: dict) -> dict:
    """
    Evalúa una transacción y retorna el resultado de riesgo.
    """
    _load_artifacts()
    
    X = _preprocess_single(data)
    prob = float(_pipeline.predict_proba(X)[0, 1])

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
        
    reasons = _get_reasons(X, prob)
        
    _save_to_db(data, prob, level, recommendation, reasons)

    return {
        "risk_score":     round(prob, 4),
        "is_fraud":       prob >= RISK_THRESHOLDS["medio"],
        "risk_level":     level,
        "recommendation": recommendation,
        "reasons":        reasons,
        "input":          data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test rápido
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    casos = [
        {
            "nombre": "Transacción Normal",
            "data": {"amount": 35000, "country": "CL", "hour": 14,
                     "device_type": "iOS", "failed_attempts": 0,
                     "is_foreign": 0, "high_risk_merchant": 0},
        },
        {
            "nombre": "Transacción Sospechosa",
            "data": {"amount": 950000, "country": "RU", "hour": 3,
                     "device_type": "Unknown", "failed_attempts": 5,
                     "is_foreign": 1, "high_risk_merchant": 1},
        },
        {
            "nombre": "Caso Intermedio",
            "data": {"amount": 250000, "country": "AR", "hour": 22,
                     "device_type": "Android", "failed_attempts": 2,
                     "is_foreign": 0, "high_risk_merchant": 1},
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
        print(f"   Motivos     :")
        for r in result['reasons']:
            print(f"                 {r}")
