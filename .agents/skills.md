---
description: Skills for FraudSense — fraud detection system
---

# FraudSense — skills.md

> **Principio de No-Fragmentación**: Este archivo define **4 Super-Skills**
> paramétricas en vez de 1 función = 1 skill. Si una skill existente puede
> recibir un parámetro `mode=` para hacer lo mismo, NO se crea una nueva.

---

## Super-Skill 1: `run_pipeline`

**Descripción**: Ejecuta el pipeline ML completo de extremo a extremo.
Es la skill más alta del sistema — orquesta `generate_dataset` + `preprocessing` + `train_model` en secuencia.

**Cuándo usarla**: Re-entrenar desde cero, CI/CD, setup inicial del proyecto.

**NO fragmentar en**: `skill_generate`, `skill_preprocess`, `skill_train` → esos son pasos internos, no skills separadas.

```yaml
skill: run_pipeline
agent: FraudSense Core Agent
command: |
  python data/generate_dataset.py &&
  python src/train_model.py
parameters:
  - name: regenerate_data
    type: bool
    default: true
    description: Si false, salta generate_dataset y usa el CSV existente
  - name: apply_smote
    type: bool
    default: true
    description: Si false, entrena sin balanceo SMOTE (útil para comparar)
outputs:
  - data/transactions.csv
  - models/fraud_model.pkl
  - models/encoders.pkl
  - models/evaluation.png
```

**Uso típico**:

```bash
# Pipeline completo (default)
python data/generate_dataset.py && python src/train_model.py

# Solo re-entrenar (datos ya existen)
python src/train_model.py
```

---

## Super-Skill 2: `run_model_operation`

**Descripción**: Todas las operaciones sobre el modelo ML con un solo punto de entrada parametrizado.
Reemplaza lo que sería `skill_train`, `skill_evaluate`, `skill_predict` → son la misma lógica con distintos modos.

**Cuándo usarla**:
- Entrenar → `mode="train"`
- Evaluar métricas → `mode="evaluate"`
- Predecir una transacción → `mode="predict"`

```yaml
skill: run_model_operation
agent: FraudSense Core Agent
parameters:
  - name: mode
    type: enum
    values: ["train", "evaluate", "predict"]
    required: true
    description: Operación a realizar sobre el modelo
  - name: data
    type: dict
    required: false
    description: Solo requerido cuando mode="predict". Keys → amount, country, hour, device_type, failed_attempts, is_foreign, high_risk_merchant
```

**Mapeo interno** (en vez de skills separadas):

| `mode` | Función Python invocada | Archivo |
|--------|------------------------|---------|
| `"train"` | `train_model()` | `src/train_model.py` |
| `"evaluate"` | `train_model()` + imprime métricas | `src/train_model.py` |
| `"predict"` | `predict_transaction(data)` | `src/predict.py` |

**Ejemplos de uso**:

```python
# Modo predict (desde código)
from src.predict import predict_transaction
result = predict_transaction({
    "amount": 950000,
    "country": "RU",
    "hour": 3,
    "device_type": "Unknown",
    "failed_attempts": 5,
    "is_foreign": 1,
    "high_risk_merchant": 1,
})
# → {"risk_score": 0.92, "risk_level": "ALTO", "recommendation": "🚨 BLOQUEAR TRANSACCIÓN"}

# Modo train (desde terminal)
python src/train_model.py
```

> **Inserción en código** (`src/predict.py`, línea 113):
> Si se agrega un nuevo modo (ej: `mode="batch_predict"`), insertar como función
> `predict_batch(data_list: list) -> list` en **línea 113** de `src/predict.py`,
> justo antes de la función `predict_transaction`. No crear nuevo archivo.

---

## Super-Skill 3: `run_feature_transform`

**Descripción**: Todas las transformaciones de features bajo un único skill parametrizado.
Reemplaza lo que serían `skill_engineer_features`, `skill_encode`, `skill_smote`.

**Cuándo usarla**: Agregar nuevas variables, cambiar encoding, ajustar balanceo.

```yaml
skill: run_feature_transform
agent: FraudSense Core Agent
parameters:
  - name: mode
    type: enum
    values: ["engineer", "encode", "smote", "full"]
    default: "full"
    description: |
      "engineer" → solo feature engineering (engineer_features)
      "encode"   → solo encoding categórico (encode_categoricals)
      "smote"    → solo balanceo de clases
      "full"     → pipeline completo (equivale a load_and_preprocess)
  - name: apply_smote
    type: bool
    default: true
    description: Solo aplica cuando mode="smote" o mode="full"
```

**Mapeo a funciones Python** (en `src/preprocessing.py`):

```
mode="engineer" → engineer_features(df)          # línea 28
mode="encode"   → encode_categoricals(df)         # línea 66
mode="smote"    → SMOTE().fit_resample(X, y)      # línea 141
mode="full"     → load_and_preprocess()           # línea 102
```

> **Cómo agregar una nueva feature**:
> Insertar el código en `src/preprocessing.py`, **línea 59** (fin de `engineer_features`),
> justo antes del `return df`. Usar el mismo patrón de las features existentes.
> Si la feature requiere un nuevo parámetro de config → agregar en **`config.py`
> línea 46** (bloque `# Negocio`).

---

## Super-Skill 4: `run_server`

**Descripción**: Levanta cualquier servidor del sistema con un parámetro `mode`.
Evita tener `skill_api` y `skill_dashboard` como skills separadas cuando el 80% del proceso es idéntico (cd al directorio, levantar servidor, verificar puerto).

```yaml
skill: run_server
agent: FraudSense Interface Agent
parameters:
  - name: mode
    type: enum
    values: ["api", "dashboard", "all"]
    required: true
    description: |
      "api"       → uvicorn src.api:app --reload (puerto 8000)
      "dashboard" → streamlit run dashboard/app.py (puerto 8501)
      "all"       → levanta ambos en paralelo
  - name: reload
    type: bool
    default: true
    description: Hot-reload al cambiar archivos (solo aplica al API)
  - name: port
    type: int
    default: null
    description: Sobreescribir puerto por defecto
```

**Comandos generados por modo**:

```bash
# mode="api"
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# mode="dashboard"
streamlit run dashboard/app.py --server.headless true

# mode="all" (PowerShell)
Start-Job { uvicorn src.api:app --reload }
streamlit run dashboard/app.py
```

> **Cómo agregar un nuevo endpoint**:
> Insertar en `src/api.py` **línea 152** (después de `evaluate_transaction`,
> antes de la función `root`). Seguir el patrón `@app.get(...)` / `@app.post(...)`.

---

## Super-Skill 5: `run_data_io`

**Descripción**: Carga, guarda o genera datos del sistema. Compartida entre ambos agentes.
Evita duplicar lógica de IO que tanto el Core Agent como el Interface Agent necesitan.

```yaml
skill: run_data_io
agent: "*"  # compartida — usable por cualquier agente
parameters:
  - name: mode
    type: enum
    values: ["load", "save", "generate", "stats"]
    required: true
    description: |
      "load"     → pd.read_csv(DATA_FILE)
      "save"     → df.to_csv(DATA_FILE)
      "generate" → python data/generate_dataset.py
      "stats"    → imprime resumen del dataset (shape, fraud rate, nulls)
  - name: path
    type: str
    default: "data/transactions.csv"
    description: Path del archivo CSV a operar
```

**Mapeo de funciones**:

```
mode="load"     → pd.read_csv(path, index_col="transaction_id")
mode="save"     → df.to_csv(path, index=True)
mode="generate" → data/generate_dataset.py::main()
mode="stats"    → df.describe() + conteo fraudes
```

---

## Inventario de Skills — Tabla Resumen

| Super-Skill | Parámetros `mode=` | Agente | Reemplaza N skills individuales |
|-------------|-------------------|--------|--------------------------------|
| `run_pipeline` | `regenerate_data`, `apply_smote` | Core | 3 (generate, preprocess, train) |
| `run_model_operation` | `train`, `evaluate`, `predict` | Core | 3 (una por modo) |
| `run_feature_transform` | `engineer`, `encode`, `smote`, `full` | Core | 4 (una por transformación) |
| `run_server` | `api`, `dashboard`, `all` | Interface | 2 (api, dashboard) |
| `run_data_io` | `load`, `save`, `generate`, `stats` | Compartida | 4 (una por acción IO) |

**Total de skills individuales evitadas: 16**
**Skills definidas: 5 Super-Skills**

---

## Archivos Huérfanos — Detección

**Tras la consolidación, estos archivos NO requieren skills propias ni su propia entrada:**

| Archivo | Estado | Razón |
|---------|--------|-------|
| `test_predict.py` | ⚠️ Temporal | Creado para verificación. Puede eliminarse; la lógica vive en `run_model_operation mode="predict"` |
| `data/__init__.py` | ⚠️ Redundante | `data/` no necesita ser un paquete Python — solo tiene scripts. Puede eliminarse. |
| `models/evaluation.png` | ✅ Generado automáticamente | No es código; es output de `run_pipeline`. No requiere gestión. |

**Acción recomendada:**
```bash
# Eliminar archivos huérfanos
Remove-Item data/__init__.py
Remove-Item test_predict.py    # solo si el modelo ya está verificado
```

---

## Principio de Reutilización — Checklist

Antes de crear una nueva skill o agente, verificar en orden:

1. **¿Puede una skill existente recibir `mode=` nuevo?** → Extender la Super-Skill
2. **¿El 80% del contexto ya existe en un agente?** → Agregar responsabilidad ahí
3. **¿Es solo una función de 1 archivo?** → Es un helper interno, no una skill
4. **¿Se necesita coordinación entre agentes?** → Usar reglas de Escalada de `agents.md`
5. **Solo si todo lo anterior falla** → Crear nueva Super-Skill con `mode=` desde el inicio
