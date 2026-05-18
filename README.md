# 🛡️ FraudSense

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
[![CI](https://github.com/Medalcode/FraudSense/actions/workflows/ci.yml/badge.svg)](https://github.com/Medalcode/FraudSense/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/Demo-Vercel-000000?style=for-the-badge&logo=vercel)](https://fraudsense-api.vercel.app)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-brightgreen?style=for-the-badge)

**Sistema Inteligente de Detección de Fraude en Transacciones Digitales usando Machine Learning**

*Proyecto de Título — Ingeniería en Informática*

</div>

---

## 🎯 ¿Qué es FraudSense?

FraudSense es una plataforma completa de análisis financiero que utiliza **Machine Learning (XGBoost)** para detectar transacciones sospechosas en sistemas de pago digitales en tiempo real.

Similar a los sistemas usados por **PayPal, Stripe, Visa y bancos tradicionales**, FraudSense analiza el comportamiento de cada transacción y predice si es fraudulenta o legítima.

---

## 🏗️ Arquitectura del Sistema

```
Transacción → [API REST] → [Motor de Análisis] → [Modelo XGBoost] → Resultado
                                                          ↓
                                               [Dashboard Streamlit]
                                                          ↓
                                                [Base de Datos SQL]
```

---

## 🧠 ¿Qué detecta el modelo?

| Patrón Sospechoso | Ejemplo |
|---|---|
| 💰 Montos inusuales | Compra de $950.000 a las 3 AM |
| 🌍 País diferente al habitual | Transacción desde Rusia |
| 🔁 Múltiples intentos fallidos | 5 intentos en 2 minutos |
| 🌙 Horarios nocturnos | Operación entre 00:00 – 05:00 |
| 📱 Dispositivo desconocido | Dispositivo nuevo nunca visto |

---

## 🧰 Stack Tecnológico

| Área | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Machine Learning | XGBoost, Scikit-Learn |
| Análisis de Datos | Pandas, NumPy |
| Desbalance de Clases | imbalanced-learn (SMOTE) |
| API REST | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Visualización | Matplotlib, Seaborn |
| Base de Datos | SQL (SQLite / PostgreSQL) |
| Control de Versiones | Git & GitHub |

---

## 📁 Estructura del Proyecto

```
FraudSense/
│
├── .agents/
│   ├── agents.md               # 2 agentes consolidados (Core + Interface)
│   └── skills.md               # 5 Super-Skills paramétricas
│
├── data/
│   ├── generate_dataset.py     # Generador de dataset sintético (50k filas)
│   └── transactions.csv        # Dataset de transacciones
│
├── database/
│   └── schema.sql              # Esquema SQL: 4 tablas + índices + vistas
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Feature engineering + SMOTE
│   ├── train_model.py          # Entrenamiento XGBoost + evaluación
│   ├── predict.py              # Inferencia: BAJO / MEDIO / ALTO riesgo
│   └── api.py                  # API REST (FastAPI)
│
├── models/                     # Generado por train_model.py
│   ├── fraud_model.pkl
│   ├── encoders.pkl
│   └── evaluation.png          # Confusion Matrix + ROC Curve + Feature Importance
│
├── dashboard/
│   └── app.py                  # Dashboard Streamlit (5 páginas)
│
├── docs/
│   └── architecture.md         # Documentación técnica de arquitectura
│
├── config.py                   # Configuración global centralizada
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Medalcode/FraudSense.git
cd FraudSense
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Generar el dataset

```bash
python data/generate_dataset.py
```

### 4. Entrenar el modelo

```bash
python src/train_model.py
```

### 5. Iniciar la API

```bash
export FRAUDSENSE_API_KEY="cambia-esto-por-una-clave-segura"
uvicorn src.api:app --reload
# Documentación → http://localhost:8000/docs
```

### 6. Abrir el Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🌐 API REST

### Evaluar una transacción

```http
POST /evaluar_transaccion
Content-Type: application/json
X-API-Key: <tu_api_key>

{
  "amount": 950000,
  "country": "RU",
  "hour": 3,
  "device_type": "Android",
  "failed_attempts": 5,
  "is_foreign": 1,
  "high_risk_merchant": 1
}
```

**Respuesta:**

```json
{
  "risk_score": 0.94,
  "is_fraud": true,
  "risk_level": "ALTO",
  "recommendation": "🚨 BLOQUEAR TRANSACCIÓN",
  "details": {
    "amount": 950000,
    "country": "RU",
    "hour": 3
  }
}
```

---

## 📊 Métricas del Modelo

| Métrica | Valor |
|---|---|
| Accuracy | ~98% |
| Precision | ~91% |
| Recall | ~88% |
| F1-Score | ~89% |
| ROC-AUC | ~0.97 |

---

## 🤖 Arquitectura de Agentes IA

FraudSense define su arquitectura de agentes en `.agents/` siguiendo el principio de **densidad sobre fragmentación**:

| Agente | Responsabilidades |
|---|---|
| **Core Agent** | Datos → Preprocesamiento → Entrenamiento → Inferencia |
| **Interface Agent** | API REST + Dashboard Streamlit |

Las 5 **Super-Skills paramétricas** (`run_pipeline`, `run_model_operation`, `run_feature_transform`, `run_server`, `run_data_io`) reemplazan 16 skills individuales.

---

## 🎓 Contexto Académico

Este proyecto fue desarrollado como **Proyecto de Título** para la carrera de Ingeniería en Informática, demostrando la aplicación de técnicas de Machine Learning en el sector FinTech.

**Competencias demostradas:**
- Diseño de sistemas de software complejos
- Machine Learning aplicado (clasificación desbalanceada)
- Desarrollo de APIs REST
- Análisis y visualización de datos financieros
- Arquitectura de agentes de IA
- Gestión de proyectos con control de versiones

---

## 👨‍💻 Autor

**Jonatthan Medalla**
Estudiante de Ingeniería en Informática

---

## 📄 Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).
