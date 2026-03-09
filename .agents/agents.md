---
description: Agents for FraudSense — fraud detection system
---

# FraudSense — agents.md

> **Principio de Densidad**: FraudSense usa **2 agentes** en vez de uno por módulo.
> Cada agente cubre múltiples responsabilidades relacionadas. No se crean agentes
> nuevos si el contexto ya encaja en uno existente.

---

## Agente 1: FraudSense Core Agent

**Rol**: Agente generalista que orquesta el pipeline completo de datos y ML.
Cubre todo el ciclo de vida del modelo: datos → procesamiento → entrenamiento → predicción.

**NO separar en**: DataAgent + TrainingAgent + PredictionAgent → esos son roles
del mismo pipeline y deben vivir aquí.

### Responsabilidades

| Área | Módulo Python | Descripción |
|------|--------------|-------------|
| Generación de datos | `data/generate_dataset.py` | Crea dataset sintético de transacciones |
| Preprocesamiento | `src/preprocessing.py` | Feature engineering, encoding, SMOTE |
| Entrenamiento | `src/train_model.py` | XGBoost training + evaluación + gráficos |
| Inferencia | `src/predict.py` | Carga modelo, predice riesgo por transacción |
| Configuración | `config.py` | Parámetros globales, umbrales, paths |

### Cuándo invocar este agente

- Cuando se pide re-entrenar el modelo con nuevos datos
- Cuando se pide ajustar los umbrales de riesgo (BAJO/MEDIO/ALTO)
- Cuando se pide agregar una nueva feature al pipeline
- Cuando se pide evaluar métricas del modelo (Precision, Recall, F1)
- Cuando se pide generar un nuevo dataset con parámetros distintos
- Cuando se ejecuta `python src/train_model.py` o `python src/predict.py`

### Contexto del agente

```
Archivos propios:
  config.py
  data/generate_dataset.py
  src/preprocessing.py
  src/train_model.py
  src/predict.py
  models/fraud_model.pkl
  models/encoders.pkl
  models/evaluation.png

Skills disponibles:
  → skill: run_pipeline          (genera dataset + entrena modelo)
  → skill: run_data_io           (mode="load" | "save" | "generate")
  → skill: run_model_operation   (mode="train" | "evaluate" | "predict")
  → skill: run_feature_transform (mode="engineer" | "encode" | "smote")
```

---

## Agente 2: FraudSense Interface Agent

**Rol**: Agente especializado en la capa de exposición del sistema.
Cubre la API REST y el Dashboard. Ambos comparten el 80% del contexto
(datos del mismo CSV, misma lógica de predict, mismo config) → NO se separan.

**NO separar en**: APIAgent + DashboardAgent → redundante. Ambos sirven datos
del mismo sistema. Un solo agente con skills paramétricas es suficiente.

### Responsabilidades

| Área | Módulo Python | Descripción |
|------|--------------|-------------|
| API REST | `src/api.py` | FastAPI: /health, /estadisticas, /evaluar_transaccion |
| Dashboard | `dashboard/app.py` | Streamlit: 5 páginas analíticas + evaluador live |

### Cuándo invocar este agente

- Cuando se pide agregar un nuevo endpoint a la API
- Cuando se pide modificar el dashboard (nuevos gráficos, páginas, filtros)
- Cuando se necesita extender el schema Pydantic de `TransactionInput`
- Cuando se reporta un error HTTP desde la API
- Cuando se ejecuta `uvicorn src.api:app` o `streamlit run dashboard/app.py`

### Contexto del agente

```
Archivos propios:
  src/api.py
  dashboard/app.py

Dependencias (solo lectura):
  src/predict.py        (para evaluar_transaccion)
  data/transactions.csv (para /estadisticas y dashboard)
  config.py             (API_TITLE, API_VERSION, DATA_FILE...)

Skills disponibles:
  → skill: run_server      (mode="api" | "dashboard")
  → skill: run_data_io     (mode="load")  ← compartida con Core Agent
```

---

## Reglas de Escalada entre Agentes

```
Interface Agent detecta problema de predicción
  → delegar a Core Agent (skill: run_model_operation mode="predict")

Core Agent necesita verificar que la API responde
  → delegar a Interface Agent (skill: run_server mode="api")

Interface Agent necesita re-entrenar el modelo
  → delegar a Core Agent (skill: run_pipeline)
```

---

## Detección de Fragmentación — Qué NO crear

> Los siguientes agentes serían redundantes y están explícitamente prohibidos:

| Agente Innecesario | Razón |
|-------------------|-------|
| `DataAgent` | Absorber en Core Agent (mismo pipeline) |
| `TrainingAgent` | Absorber en Core Agent (mismo contexto) |
| `PredictionAgent` | Absorber en Core Agent (mismo modelo) |
| `APIAgent` | Absorber en Interface Agent |
| `DashboardAgent` | Absorber en Interface Agent (mismo CSV, mismo predict) |
| `MetricsAgent` | Usar `run_model_operation mode="evaluate"` del Core Agent |
| `SQLAgent` | El schema.sql es documentación, no requiere agente propio |
