"""
FraudSense — Pipeline de Preprocesamiento
Limpia, codifica y prepara los datos para entrenar el modelo XGBoost.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_FILE,
    HIGH_AMOUNT_THRESHOLD,
    NIGHT_HOURS,
    RANDOM_STATE,
    RISK_COUNTRIES,
    TEST_SIZE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables adicionales que mejoran la detección de fraude.
    """
    df = df.copy()

    # 1. ¿Es una transacción nocturna? (00:00 – 05:59)
    df["is_night"] = df["hour"].apply(lambda h: 1 if h in NIGHT_HOURS else 0)

    # 2. ¿El monto es inusualmente alto?
    df["high_amount"] = (df["amount"] > HIGH_AMOUNT_THRESHOLD).astype(int)

    # 3. Z-score del monto (qué tan alejado está de la media)
    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()
    df["amount_zscore"] = ((df["amount"] - mean_amount) / std_amount).round(4)

    # 4. ¿Es un país de alto riesgo?
    df["is_risk_country"] = df["country"].apply(lambda c: 1 if c in RISK_COUNTRIES else 0)

    # 5. Puntuación de riesgo combinada (heurística simple)
    df["risk_score_heuristic"] = (
        df["failed_attempts"] * 0.3
        + df["is_foreign"] * 0.2
        + df["high_risk_merchant"] * 0.2
        + df["is_night"] * 0.15
        + df["high_amount"] * 0.15
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Codificación de Variables Categóricas
# ──────────────────────────────────────────────────────────────────────────────


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Codifica variables categóricas con LabelEncoder.
    Retorna el dataframe codificado y el diccionario de encoders.
    """
    df = df.copy()
    encoders = {}

    for col in ["country", "device_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Completo
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "amount",
    "country",
    "hour",
    "device_type",
    "failed_attempts",
    "is_foreign",
    "high_risk_merchant",
    "is_night",
    "high_amount",
    "amount_zscore",
    "is_risk_country",
    "risk_score_heuristic",
]


def load_and_preprocess(
    data_path: str = DATA_FILE,
    apply_smote: bool = True,
) -> tuple:
    """
    Pipeline completo: carga → limpieza → encoding → features → split → (SMOTE).

    Returns:
        X_train, X_test, y_train, y_test, encoders
    """
    print("📂 Cargando datos...")
    df = pd.read_csv(data_path, index_col="transaction_id")

    print(f"   Filas cargadas: {len(df):,}")
    print(f"   Fraudes       : {df['is_fraud'].sum():,} ({df['is_fraud'].mean() * 100:.2f}%)")

    # Eliminar duplicados y nulos
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Feature engineering
    print("🔧 Ingeniería de características...")
    df = engineer_features(df)

    # Codificación
    print("🔤 Codificando variables categóricas...")
    df, encoders = encode_categoricals(df)

    # Separar features y target
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    # Split estratificado
    print("✂️  Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"   Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras")

    # SMOTE — balanceo de clases
    if apply_smote:
        print("⚖️  Aplicando SMOTE para balancear clases...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   Post-SMOTE train: {len(X_train):,} muestras")

    print("✅ Preprocesamiento completado.\n")
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess()
    print(f"Features usadas ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
