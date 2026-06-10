"""
FraudSense — Configuración Global
Sistema Inteligente de Detección de Fraude en Transacciones Digitales
"""

import os

# ──────────────────────────────────────────────
# Rutas del Proyecto
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

DATA_FILE    = os.path.join(DATA_DIR, "transactions.csv")
DB_FILE      = os.path.join(DATABASE_DIR, "fraudsense.db")
SCHEMA_FILE  = os.path.join(DATABASE_DIR, "schema.sql")

PIPELINE_FILE = os.path.join(MODELS_DIR, "fraud_pipeline.pkl")

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATASET_SIZE   = 50_000        # Número de transacciones a generar
FRAUD_RATE     = 0.03          # Tasa de fraude (~3%)
RANDOM_STATE   = 42

# ──────────────────────────────────────────────
# Modelo XGBoost
# ──────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     RANDOM_STATE,
    "eval_metric":      "logloss",
}

TEST_SIZE = 0.20               # 80% train / 20% test

# ──────────────────────────────────────────────
# Umbrales de Riesgo
# ──────────────────────────────────────────────
RISK_THRESHOLDS = {
    "bajo":   0.30,   # score < 0.30 → BAJO
    "medio":  0.60,   # 0.30 ≤ score < 0.60 → MEDIO
    "alto":   1.00,   # score ≥ 0.60 → ALTO
}

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "FraudSense API"
API_VERSION = "1.0.0"

API_KEY_ENV = "FRAUDSENSE_API_KEY"
API_KEY_HEADER = "X-API-Key"
API_KEY = os.getenv(API_KEY_ENV, "dev-api-key-12345")

# ──────────────────────────────────────────────
# Negocio
# ──────────────────────────────────────────────
HIGH_AMOUNT_THRESHOLD = 500_000    # CLP — monto considerado alto
NIGHT_HOURS = list(range(0, 6))   # 00:00 – 05:59 considerado nocturno

RISK_COUNTRIES = [
    "RU", "NG", "CN", "VN", "UA",
    "RO", "BG", "PH", "PK", "ID",
]

DEVICE_TYPES = ["Android", "iOS", "Web", "Windows", "Unknown"]
