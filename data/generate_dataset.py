"""
FraudSense — Generador de Dataset Sintético
Genera ~50.000 transacciones financieras con patrones de fraude realistas.
"""

import os
import sys
import numpy as np
import pandas as pd

# Agregar directorio raíz al path para importar config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_FILE, DATASET_SIZE, FRAUD_RATE,
    RANDOM_STATE, RISK_COUNTRIES, DEVICE_TYPES, DATA_DIR
)

def generate_transactions(n_total: int = DATASET_SIZE,
                           fraud_rate: float = FRAUD_RATE,
                           seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Genera un dataset realista de transacciones con fraude.

    Patrones fraudulentos incorporados:
    - Montos más altos que el promedio
    - Horas nocturnas (00:00 – 05:59)
    - Países de alto riesgo
    - Más intentos fallidos
    - Dispositivos desconocidos
    - Mayor probabilidad en comercios de alto riesgo
    """
    rng = np.random.default_rng(seed)

    n_fraud  = int(n_total * fraud_rate)
    n_legit  = n_total - n_fraud

    countries_all  = ["CL", "AR", "PE", "BR", "MX", "CO", "US", "DE", "FR", "ES"] + RISK_COUNTRIES
    countries_legit = ["CL", "AR", "PE", "BR", "MX", "CO", "US", "DE", "FR", "ES"]

    # ── Transacciones Legítimas ────────────────────────────────────────────
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=10.5, sigma=1.2, size=n_legit).clip(500, 800_000),
        "country": rng.choice(countries_legit, size=n_legit,
                              p=[0.40, 0.15, 0.10, 0.10, 0.08, 0.07, 0.04, 0.02, 0.02, 0.02]),
        "hour": rng.integers(7, 23, size=n_legit),          # horario laboral
        "device_type": rng.choice(DEVICE_TYPES[:4], size=n_legit,
                                   p=[0.35, 0.35, 0.25, 0.05]),
        "failed_attempts": rng.integers(0, 2, size=n_legit),
        "is_foreign": rng.choice([0, 1], size=n_legit, p=[0.90, 0.10]),
        "high_risk_merchant": rng.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "is_fraud": 0,
    })

    # ── Transacciones Fraudulentas ─────────────────────────────────────────
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=12.5, sigma=1.5, size=n_fraud).clip(50_000, 5_000_000),
        "country": rng.choice(countries_all, size=n_fraud,
                              p=[0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02,
                                 0.15, 0.12, 0.10, 0.08, 0.07,
                                 0.05, 0.05, 0.04, 0.04, 0.01]),
        "hour": rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n_fraud),
        "device_type": rng.choice(DEVICE_TYPES, size=n_fraud,
                                   p=[0.25, 0.20, 0.15, 0.10, 0.30]),
        "failed_attempts": rng.choice([3, 4, 5, 6, 7, 8], size=n_fraud,
                                       p=[0.20, 0.20, 0.20, 0.15, 0.15, 0.10]),
        "is_foreign": rng.choice([0, 1], size=n_fraud, p=[0.20, 0.80]),
        "high_risk_merchant": rng.choice([0, 1], size=n_fraud, p=[0.30, 0.70]),
        "is_fraud": 1,
    })

    # ── Unir y barajar ─────────────────────────────────────────────────────
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.index.name = "transaction_id"
    df.index += 1

    # Redondear monto a entero (CLP no tiene decimales)
    df["amount"] = df["amount"].astype(int)

    return df


def main():
    print("=" * 55)
    print("  FraudSense — Generador de Dataset")
    print("=" * 55)

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n📊 Generando {DATASET_SIZE:,} transacciones...")
    df = generate_transactions()

    df.to_csv(DATA_FILE, index=True)

    # ── Estadísticas ───────────────────────────────────────────────────────
    total   = len(df)
    fraud_n = df["is_fraud"].sum()
    legit_n = total - fraud_n

    print(f"\n✅ Dataset generado exitosamente en: {DATA_FILE}")
    print(f"\n{'─'*40}")
    print(f"  Total transacciones : {total:>8,}")
    print(f"  Legítimas           : {legit_n:>8,}  ({legit_n/total*100:.1f}%)")
    print(f"  Fraudulentas        : {fraud_n:>8,}  ({fraud_n/total*100:.1f}%)")
    print(f"  Monto promedio (CLP)")
    print(f"    Legítimas   : ${df[df.is_fraud==0].amount.mean():>12,.0f}")
    print(f"    Fraudulentas: ${df[df.is_fraud==1].amount.mean():>12,.0f}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
