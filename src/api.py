"""
FraudSense — API REST
FastAPI para evaluar transacciones en tiempo real.

Rutas disponibles:
  GET  /health                → Estado del sistema
  GET  /estadisticas          → Estadísticas del dataset
  POST /evaluar_transaccion   → Predicción de fraude
  GET  /docs                  → Documentación Swagger (auto-generada)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import (
    API_KEY,
    API_KEY_ENV,
    API_KEY_HEADER,
    API_TITLE,
    API_VERSION,
    DATA_FILE,
    DEVICE_TYPES,
)
from src.predict import predict_transaction


# ──────────────────────────────────────────────────────────────────────────────
# Inicialización
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Sistema Inteligente de Detección de Fraude en Transacciones Digitales",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail=f"Configuración requerida: define {API_KEY_ENV} para habilitar la API.",
        )
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida o ausente.")
    return api_key


# ──────────────────────────────────────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    amount:             float = Field(..., gt=0,            example=950000,   description="Monto de la transacción en CLP")
    country:            str   = Field(..., min_length=2, max_length=3, example="RU",       description="Código ISO del país (ej: CL, AR, RU)")
    hour:               int   = Field(..., ge=0, le=23,     example=3,        description="Hora del día (0-23)")
    device_type:        str   = Field(...,                  example="Unknown", description="Tipo de dispositivo")
    failed_attempts:    int   = Field(0,  ge=0, le=20,     example=5,        description="Intentos fallidos previos")
    is_foreign:         int   = Field(0,  ge=0, le=1,      example=1,        description="¿País diferente al habitual? (0/1)")
    high_risk_merchant: int   = Field(0,  ge=0, le=1,      example=1,        description="¿Comercio de alto riesgo? (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 950000,
                "country": "RU",
                "hour": 3,
                "device_type": "Unknown",
                "failed_attempts": 5,
                "is_foreign": 1,
                "high_risk_merchant": 1,
            }
        }


class FraudPredictionResponse(BaseModel):
    risk_score:     float
    is_fraud:       bool
    risk_level:     str
    recommendation: str
    input:          dict


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health_check():
    """Verifica el estado del sistema y si el modelo está cargado."""
    model_ready = os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fraud_model.pkl")
    )
    return {
        "status":       "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "version":      API_VERSION,
        "message":      "FraudSense API operativa 🛡️" if model_ready
                        else "Modelo no encontrado. Ejecuta python src/train_model.py",
    }


@app.get("/estadisticas", tags=["Analytics"], dependencies=[Depends(verify_api_key)])
def get_statistics():
    """Retorna estadísticas generales del dataset de transacciones."""
    if not os.path.exists(DATA_FILE):
        raise HTTPException(
            status_code=404,
            detail="Dataset no encontrado. Ejecuta: python data/generate_dataset.py",
        )

    df = pd.read_csv(DATA_FILE, index_col="transaction_id")

    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]

    top_fraud_countries = (
        fraud_df["country"].value_counts().head(5).to_dict()
    )

    return {
        "total_transacciones": int(len(df)),
        "total_fraudes":        int(df["is_fraud"].sum()),
        "total_legitimas":      int(legit_df.shape[0]),
        "tasa_fraude_pct":      round(df["is_fraud"].mean() * 100, 2),
        "monto_promedio": {
            "legitimas":     round(float(legit_df["amount"].mean()), 0),
            "fraudulentas":  round(float(fraud_df["amount"].mean()), 0),
        },
        "top_paises_fraude": top_fraud_countries,
        "hora_pico_fraude":  int(fraud_df["hour"].mode()[0]),
        "dispositivo_mas_fraudulento": str(fraud_df["device_type"].mode()[0]),
    }


@app.post(
    "/evaluar_transaccion",
    response_model=FraudPredictionResponse,
    tags=["Predicción"],
    dependencies=[Depends(verify_api_key)],
)
def evaluate_transaction(transaction: TransactionInput):
    """
    Evalúa una transacción y retorna el nivel de riesgo de fraude.

    - **risk_score**: Probabilidad de fraude (0 a 1)
    - **risk_level**: BAJO / MEDIO / ALTO
    - **recommendation**: Acción sugerida para el analista
    """
    try:
        result = predict_transaction(transaction.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar: {str(e)}")


@app.get("/", tags=["Sistema"])
def root():
    """Bienvenida a FraudSense API."""
    return {
        "app":         "🛡️ FraudSense",
        "descripcion": "Sistema Inteligente de Detección de Fraude",
        "version":     API_VERSION,
        "docs":        "/docs",
        "endpoints":   ["/health", "/estadisticas", "/evaluar_transaccion"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Arranque local
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
