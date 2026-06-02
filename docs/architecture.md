# Arquitectura de FraudSense

## 📐 Visión General

FraudSense sigue una **arquitectura en capas** donde cada capa tiene una responsabilidad clara y bien definida.

```
┌─────────────────────────────────────────────────────┐
│               Cliente / Analista                     │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
    ┌──────▼──────┐      ┌────────▼────────┐
    │  API REST   │      │    Dashboard    │
    │  (FastAPI)  │      │   (Streamlit)   │
    └──────┬──────┘      └────────┬────────┘
           │                      │
    ┌──────▼──────────────────────▼────────┐
    │         Motor de Predicción          │
    │           (src/predict.py)           │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │         Modelo XGBoost               │
    │      (models/fraud_model.pkl)        │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │       Pipeline de Datos              │
    │  (data/ + src/preprocessing.py)      │
    └──────────────────────────────────────┘
```

---

## 🗂️ Componentes

### 1. Capa de Datos

**Archivos:** `data/generate_dataset.py`, `data/transactions.csv`

- Genera un dataset sintético de ~50.000 transacciones con tasa de fraude ~3%
- Variables generadas con distribuciones estadísticas realistas
- Fraudes tienen patrones claros (montos altos, horario nocturno, países de riesgo)

**Schema SQL:** `database/schema.sql`

```
transactions   →  fraud_alerts
     ↓                 ↓
  clients         risk_scores
```

### 2. Pipeline de Preprocesamiento

**Archivo:** `src/preprocessing.py`

| Paso | Descripción |
|------|-------------|
| Limpieza | Drop duplicados y nulos |
| Feature Engineering | is_night, high_amount, amount_zscore, is_risk_country, risk_score_heuristic |
| Encoding | LabelEncoder para country y device_type |
| Split | 80/20 estratificado |
| Balanceo | SMOTE para clases desbalanceadas |

### 3. Modelo de Machine Learning

**Archivo:** `src/train_model.py`  
**Algoritmo:** XGBoost (eXtreme Gradient Boosting)

**¿Por qué XGBoost?**
- Maneja naturalmente el desbalance de clases
- Alta performance en datos tabulares
- Muy utilizado en detección de fraude bancario
- Interpretable mediante Feature Importance

**Hiperparámetros clave:**

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| n_estimators | 300 | Balance velocidad/precisión |
| max_depth | 6 | Evita overfitting |
| learning_rate | 0.05 | Convergencia estable |
| subsample | 0.8 | Regularización |

**Métricas de Evaluación:**

| Métrica | Descripción |
|---------|-------------|
| Accuracy | % de predicciones correctas |
| Precision | De los predichos fraude, ¿cuántos son realmente fraude? |
| Recall | De los fraudes reales, ¿cuántos detectamos? |
| F1-Score | Media armónica Precision-Recall |
| ROC-AUC | Capacidad discriminatoria del modelo |

> **Nota:** En fraude bancario, el **Recall** es la métrica más importante. Es más costoso dejar pasar un fraude que rechazar una transacción legítima.

### 4. API REST

**Archivo:** `src/api.py`  
**Framework:** FastAPI

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Bienvenida |
| `/health` | GET | Estado del sistema |
| `/estadisticas` | GET | Métricas generales |
| `/evaluar_transaccion` | POST | Predicción de fraude |

### 5. Dashboard

**Archivo:** `dashboard/app.py`  
**Framework:** Streamlit + Plotly

**5 páginas:**
1. **Overview** — KPIs globales y gráficos resumen
2. **Análisis por País** — Mapa de riesgo geográfico
3. **Análisis Temporal** — Fraude por hora del día
4. **Evaluador en Tiempo Real** — Formulario + predicción ML
5. **Tabla de Transacciones** — Explorer con filtros

---

## 🔄 Flujo de Datos

```
1. generate_dataset.py     → data/transactions.csv
2. preprocessing.py        → X_train, y_train (con SMOTE)
3. train_model.py          → models/fraud_model.pkl
4. predict.py              → Carga modelo, predice
5. api.py                  → Expone endpoint HTTP
6. dashboard/app.py        → Visualiza resultados
```

---

## 🚀 Decisiones de Diseño

### ¿Por qué datos sintéticos?
Los datos reales de fraude bancario son confidenciales. El generador sintético replica distribuciones estadísticas reales documentadas en la literatura académica.

### ¿Por qué SMOTE?
Con solo ~3% de fraudes, un modelo sin balanceo aprende a predecir siempre "legítima" y obtiene 97% de accuracy, pero 0% de recall en fraude. SMOTE genera muestras sintéticas de la clase minoritaria.

### ¿Por qué FastAPI?
- Generación automática de documentación (Swagger UI)
- Validación automática de inputs con Pydantic
- Async-ready para alta concurrencia
- Muy usado en producción en FinTech

---

*Documento generado como parte del Proyecto de Título — FraudSense*
