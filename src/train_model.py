"""
FraudSense — Entrenamiento del Modelo XGBoost
Entrena, evalúa y guarda el modelo de detección de fraude.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PIPELINE_FILE, MODEL_PARAMS, RANDOM_STATE, MODELS_DIR
from src.preprocessing import load_and_preprocess, FEATURE_COLUMNS, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train_model():
    print("=" * 60)
    print("  FraudSense — Entrenamiento del Modelo XGBoost (Pipeline)")
    print("=" * 60)

    # 1. Preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. Configurar ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ],
        remainder="passthrough"
    )

    # 3. Construir Pipeline
    print("🌲 Construyendo Pipeline (Preprocesamiento -> SMOTE -> XGBoost)...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("classifier", XGBClassifier(
            **MODEL_PARAMS,
            scale_pos_weight=1,
            use_label_encoder=False,
        ))
    ])

    # 4. Entrenar
    print("⚙️ Entrenando Pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Predicciones
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # 6. Métricas
    print("\n" + "─" * 60)
    print("  RESULTADOS DEL MODELO")
    print("─" * 60)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("─" * 60)
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"]))

    # 7. Guardar Pipeline Completo
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"\n✅ Pipeline completo guardado en: {PIPELINE_FILE}")

    # 8. Guardar gráficos
    _save_plots(pipeline, y_test, y_pred, y_pred_prob)

    return pipeline



# ──────────────────────────────────────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────────────────────────────────────

def _save_plots(pipeline, y_test, y_pred, y_pred_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FraudSense — Evaluación del Modelo", fontsize=14, fontweight="bold")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legítima", "Fraude"],
                yticklabels=["Legítima", "Fraude"],
                ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Predicción")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC-AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    # Feature Importance
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    
    importances = model.feature_importances_
    feat_series = sorted(zip(feature_names, importances), key=lambda x: x[1])[-10:] # Top 10
    names, vals = zip(*feat_series)
    
    # Clean up feature names for display
    names = [n.split("__")[-1] for n in names]
    
    axes[2].barh(names, vals, color="#3498db")
    axes[2].set_title("Top 10 Feature Importance")
    axes[2].set_xlabel("Importancia (F-score)")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(PIPELINE_FILE), "evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Gráficos guardados en: {plot_path}")


if __name__ == "__main__":
    train_model()
