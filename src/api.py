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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DATA_FILE, API_TITLE, API_VERSION, DEVICE_TYPES
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


from src.db import get_connection

@app.get("/estadisticas", tags=["Analytics"])
def get_statistics():
    """Retorna estadísticas generales de transacciones usando SQLite."""
    try:
        with get_connection() as conn:
            # Stats generales
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) as fraudes,
                    SUM(CASE WHEN is_fraud=0 THEN 1 ELSE 0 END) as legitimas
                FROM transactions
            """).fetchone()
            
            total = row["total"] or 0
            if total == 0:
                 raise HTTPException(status_code=404, detail="No hay transacciones.")
            
            fraudes = row["fraudes"] or 0
            legitimas = row["legitimas"] or 0
            tasa = (fraudes / total * 100) if total > 0 else 0
            
            # Montos
            avg_row = conn.execute("""
                SELECT 
                    AVG(CASE WHEN is_fraud=1 THEN amount END) as avg_fraud,
                    AVG(CASE WHEN is_fraud=0 THEN amount END) as avg_legit
                FROM transactions
            """).fetchone()
            
            # Top paises (usando la vista)
            top_countries = conn.execute("""
                SELECT country, fraud_count FROM fraud_by_country LIMIT 5
            """).fetchall()
            top_paises_dict = {r["country"]: r["fraud_count"] for r in top_countries}
            
            # Hora pico y Dispositivo
            hora_pico = conn.execute("""
                SELECT hour FROM transactions WHERE is_fraud=1 GROUP BY hour ORDER BY COUNT(*) DESC LIMIT 1
            """).fetchone()
            
            disp_pico = conn.execute("""
                SELECT device_type FROM transactions WHERE is_fraud=1 GROUP BY device_type ORDER BY COUNT(*) DESC LIMIT 1
            """).fetchone()

        return {
            "total_transacciones": total,
            "total_fraudes":        fraudes,
            "total_legitimas":      legitimas,
            "tasa_fraude_pct":      round(tasa, 2),
            "monto_promedio": {
                "legitimas":     round(avg_row["avg_legit"] or 0, 0),
                "fraudulentas":  round(avg_row["avg_fraud"] or 0, 0),
            },
            "top_paises_fraude": top_paises_dict,
            "hora_pico_fraude":  hora_pico["hour"] if hora_pico else None,
            "dispositivo_mas_fraudulento": disp_pico["device_type"] if disp_pico else None,
        }
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluar_transaccion",
          response_model=FraudPredictionResponse,
          tags=["Predicción"])
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
