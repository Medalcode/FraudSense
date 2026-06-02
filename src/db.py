"""
FraudSense — Módulo de Base de Datos
Maneja la conexión a SQLite y la inicialización del esquema.
"""

import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_FILE, SCHEMA_FILE

def get_connection():
    """Retorna una conexión a la base de datos SQLite."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa la base de datos ejecutando el schema.sql."""
    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(f"Archivo de esquema no encontrado: {SCHEMA_FILE}")

    with get_connection() as conn:
        with open(SCHEMA_FILE, "r") as f:
            conn.executescript(f.read())
        conn.commit()

if __name__ == "__main__":
    print("Iniciando Base de Datos...")
    init_db()
    print(f"Base de datos creada/verificada en: {DB_FILE}")
