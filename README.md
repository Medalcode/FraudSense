# 🛡️ FraudSense

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-brightgreen?style=for-the-badge)

**Plataforma Inteligente de Evaluación de Riesgo Transaccional**

*Proyecto de Título — Ingeniería en Informática*

</div>

---

## 🎯 ¿Qué es FraudSense?

FraudSense no es solo un modelo de clasificación, es un sistema integral de prevención de fraude financiero diseñado con los estándares de la industria. Integra **Machine Learning (XGBoost)**, **Ingeniería de Datos**, **Explainable AI (SHAP)** y un panel de control interactivo para analistas de riesgo.

## ✨ Características Principales
- **Detección Avanzada:** Modelo XGBoost entrenado con técnicas para manejar el desbalance severo (SMOTE).
- **Explainable AI (XAI):** Módulo basado en SHAP que no solo entrega un *risk score*, sino los *Red Flags* exactos que justifican la decisión (ej: "Monto anormalmente alto", "Horario inusual").
- **Pipeline Integral:** Feature engineering robusto con escalado y encoding encapsulado en `imblearn.pipeline`.

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
| Machine Learning | XGBoost, Scikit-Learn (Pipelines) |
| Análisis de Datos | Pandas, NumPy |
| Desbalance de Clases | imbalanced-learn (SMOTE) |
| API REST | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Visualización | Matplotlib, Seaborn |
| Base de Datos | SQLite (Esquema relacional en producción) |
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
│   ├── db.py                   # Conexión SQLite y operaciones CRUD
│   ├── preprocessing.py        # Feature engineering 
│   ├── train_model.py          # Entrenamiento de Pipeline (Preprocesamiento + SMOTE + XGBoost)
│   ├── predict.py              # Inferencia y guardado en Base de Datos
│   └── api.py                  # API REST (FastAPI) conectada a SQLite
│
├── models/                     # Generado por train_model.py
│   ├── fraud_pipeline.pkl      # Modelo ML integrado con preprocesamiento y escalado
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

### 🐋 Despliegue Rápido (Docker)
La forma más fácil de ejecutar FraudSense en producción es a través de Docker Compose, el cual levanta simultáneamente el API y el Dashboard en contenedores separados, compartiendo la base de datos a través de un volumen persistente.

```bash
# Construir y levantar los contenedores
docker-compose up --build -d

# Ver los logs
docker-compose logs -f
```
Una vez levantado:
- **Dashboard:** http://localhost:8501
- **API Swagger:** http://localhost:8000/docs

---

### 💻 Ejecución Local (Desarrollo)
Si prefieres no usar Docker, puedes correrlo localmente:

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Generar datos y modelo (la primera vez):
```bash
python data/generate_dataset.py
python src/train_model.py
```

3. Levantar API (Terminal 1):
```bash
uvicorn src.api:app --reload --port 8000
```

4. Levantar Dashboard (Terminal 2):
```bash
streamlit run dashboard/app.py
```

---

### 🧪 Pruebas Unitarias
El proyecto cuenta con pruebas unitarias usando `pytest` para garantizar la estabilidad del motor de inferencia y del API.

```bash
pytest tests/
```

---

## 🌐 API REST

### Evaluar una transacción

```http
POST /predict
Content-Type: application/json
X-API-Key: fs_live_...

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
  "risk_score": 0.9854,
  "is_fraud": true,
  "risk_level": "ALTO",
  "recommendation": "🚨 BLOQUEAR TRANSACCIÓN",
  "reasons": [
    "🚩 Horario nocturno incrementa significativamente el riesgo.",
    "🚩 Intentos fallidos previos incrementa significativamente el riesgo.",
    "🚩 Monto anormalmente alto incrementa significativamente el riesgo."
  ],
  "input": {
    "amount": 950000,
    "country": "RU",
    "hour": 3,
    "device_type": "Android",
    "failed_attempts": 5,
    "is_foreign": 1,
    "high_risk_merchant": 1
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