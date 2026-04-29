"""
FraudSense — Entrenamiento del Modelo XGBoost
Entrena, evalúa y guarda el modelo de detección de fraude.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODERS_FILE, MODEL_FILE, MODEL_PARAMS, MODELS_DIR
from src.preprocessing import FEATURE_COLUMNS, load_and_preprocess

# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────────────────────────


def train_model():
    print("=" * 60)
    print("  FraudSense — Entrenamiento del Modelo XGBoost")
    print("=" * 60)

    # 1. Preprocesar datos
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess(apply_smote=True)

    # 2. Calcular peso para clase positiva (fraude) — doble protección junto a SMOTE
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = round(n_neg / n_pos, 2)
    print(f"⚖️  scale_pos_weight: {scale} (clases balanceadas por SMOTE)\n")

    # 3. Construir y entrenar modelo
    print("🌲 Entrenando XGBoost...")
    model = XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=1,  # SMOTE ya balanceó las clases
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 4. Predicciones
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # 5. Métricas
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

    # 6. Importancia de variables
    print("🔍 Top 5 Features más Importantes:")
    importances = model.feature_importances_
    feature_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_imp[:5]:
        bar = "█" * int(imp * 50)
        print(f"   {feat:<25} {bar} {imp:.4f}")

    # 7. Guardar modelo y encoders
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoders, ENCODERS_FILE)
    print(f"\n✅ Modelo guardado en   : {MODEL_FILE}")
    print(f"✅ Encoders guardados en: {ENCODERS_FILE}")

    # 8. Guardar gráficos
    _save_plots(model, y_test, y_pred, y_pred_prob)

    return model, encoders


# ──────────────────────────────────────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────────────────────────────────────


def _save_plots(model, y_test, y_pred, y_pred_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FraudSense — Evaluación del Modelo", fontsize=14, fontweight="bold")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legítima", "Fraude"],
        yticklabels=["Legítima", "Fraude"],
        ax=axes[0],
    )
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
    importances = model.feature_importances_
    feat_series = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1])
    names, vals = zip(*feat_series)
    axes[2].barh(names, vals, color="#3498db")
    axes[2].set_title("Feature Importance")
    axes[2].set_xlabel("Importancia (F-score)")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(MODEL_FILE), "evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Gráficos guardados en: {plot_path}")


if __name__ == "__main__":
    train_model()
