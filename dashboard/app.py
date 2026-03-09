"""
FraudSense — Dashboard Interactivo (Streamlit)
Panel de monitoreo para analistas de fraude financiero.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import DATA_FILE

# ── Estilos CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo oscuro premium */
    .stApp { background-color: #0D1117; }
    .main .block-container { padding-top: 1.5rem; }

    /* Tarjetas KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #58a6ff; margin: 0; }
    .kpi-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

    .kpi-danger  .kpi-value { color: #f85149; }
    .kpi-warning .kpi-value { color: #e3b341; }
    .kpi-success .kpi-value { color: #3fb950; }

    /* Resultados de predicción */
    .alert-alto   { background:#2d1b1b; border:1px solid #f85149; border-radius:10px; padding:1rem; }
    .alert-medio  { background:#2d2b1b; border:1px solid #e3b341; border-radius:10px; padding:1rem; }
    .alert-bajo   { background:#1b2d1b; border:1px solid #3fb950; border-radius:10px; padding:1rem; }

    /* Header */
    .fraud-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
        border: 1px solid #30363d;
    }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stSidebar { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE, index_col="transaction_id")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fraud-header">
    <h1 style="margin:0; font-size:2rem;">🛡️ FraudSense</h1>
    <p style="color:#8b949e; margin:0.3rem 0 0 0;">
        Sistema Inteligente de Detección de Fraude · Dashboard Analítico
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Navegación
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navegación")
    page = st.radio("", [
        "📊 Overview",
        "🌍 Análisis por País",
        "⏰ Análisis Temporal",
        "🚨 Evaluador en Tiempo Real",
        "📋 Tabla de Transacciones",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Filtros")
    show_only_fraud = st.checkbox("Solo mostrar fraudes", value=False)

# ──────────────────────────────────────────────────────────────────────────────
# Cargar datos
# ──────────────────────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.error("❌ Dataset no encontrado. Ejecuta: `python data/generate_dataset.py`")
    st.stop()

if show_only_fraud:
    df_view = df[df["is_fraud"] == 1].copy()
else:
    df_view = df.copy()

fraud_df = df[df["is_fraud"] == 1]
legit_df = df[df["is_fraud"] == 0]

# Colores del proyecto
COLOR_FRAUD = "#f85149"
COLOR_LEGIT = "#3fb950"
COLOR_BLUE  = "#58a6ff"
PLOTLY_TEMPLATE = "plotly_dark"


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — Overview (KPIs + gráficos resumen)
# ──────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{len(df):,}</p>
            <p class="kpi-label">Total Transacciones</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card kpi-danger">
            <p class="kpi-value">{df['is_fraud'].sum():,}</p>
            <p class="kpi-label">Fraudes Detectados</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        rate = df["is_fraud"].mean() * 100
        st.markdown(f"""
        <div class="kpi-card kpi-warning">
            <p class="kpi-value">{rate:.2f}%</p>
            <p class="kpi-label">Tasa de Fraude</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_fraud = fraud_df["amount"].mean()
        st.markdown(f"""
        <div class="kpi-card kpi-danger">
            <p class="kpi-value">${avg_fraud:,.0f}</p>
            <p class="kpi-label">Monto Prom. Fraude (CLP)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1: Distribución Fraude vs Legítima (donut)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔍 Distribución de Transacciones")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Legítimas", "Fraudulentas"],
            values=[len(legit_df), len(fraud_df)],
            hole=0.6,
            marker_colors=[COLOR_LEGIT, COLOR_FRAUD],
        )])
        fig_donut.update_layout(
            template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, b=20),
            showlegend=True,
            annotations=[dict(text=f"{rate:.2f}%<br>Fraude", x=0.5, y=0.5,
                              font_size=16, showarrow=False, font_color="white")],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Chart 2: Distribución de montos
    with col_b:
        st.subheader("💰 Distribución de Montos")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=legit_df["amount"].clip(upper=2_000_000),
            name="Legítima", nbinsx=60,
            marker_color=COLOR_LEGIT, opacity=0.7,
        ))
        fig_hist.add_trace(go.Histogram(
            x=fraud_df["amount"].clip(upper=2_000_000),
            name="Fraude", nbinsx=60,
            marker_color=COLOR_FRAUD, opacity=0.7,
        ))
        fig_hist.update_layout(
            template=PLOTLY_TEMPLATE, barmode="overlay",
            height=320, margin=dict(t=20),
            xaxis_title="Monto (CLP)", yaxis_title="Cantidad",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Chart 3: Fraude por tipo de dispositivo
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("📱 Fraude por Dispositivo")
        device_fraud = fraud_df["device_type"].value_counts().reset_index()
        device_fraud.columns = ["device", "count"]
        fig_dev = px.bar(
            device_fraud, x="count", y="device", orientation="h",
            color="count", color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE, height=320,
        )
        fig_dev.update_layout(margin=dict(t=20), showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_d:
        st.subheader("🔁 Intentos Fallidos vs Fraude")
        fails_data = df.groupby(["failed_attempts", "is_fraud"]).size().reset_index(name="count")
        fails_data["tipo"] = fails_data["is_fraud"].map({0: "Legítima", 1: "Fraude"})
        fig_fails = px.bar(
            fails_data, x="failed_attempts", y="count", color="tipo",
            color_discrete_map={"Legítima": COLOR_LEGIT, "Fraude": COLOR_FRAUD},
            template=PLOTLY_TEMPLATE, height=320,
            labels={"failed_attempts": "Intentos Fallidos", "count": "Cantidad"},
        )
        fig_fails.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_fails, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — Análisis por País
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🌍 Análisis por País":
    st.subheader("🌍 Análisis de Fraude por País")

    country_stats = df.groupby("country").agg(
        total=("is_fraud", "count"),
        fraudes=("is_fraud", "sum"),
        monto_prom=("amount", "mean"),
    ).reset_index()
    country_stats["tasa_fraude"] = (country_stats["fraudes"] / country_stats["total"] * 100).round(2)
    country_stats = country_stats.sort_values("fraudes", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            country_stats.head(15), x="country", y="fraudes",
            color="tasa_fraude", color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE, height=400,
            labels={"country": "País", "fraudes": "Fraudes", "tasa_fraude": "Tasa (%)"},
            title="Top 15 Países con Más Fraudes",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_treemap = px.treemap(
            country_stats, path=["country"], values="fraudes",
            color="tasa_fraude", color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE, height=400,
            title="Mapa de Árbol — Fraude por País",
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    st.dataframe(
        country_stats.rename(columns={
            "country": "País", "total": "Total", "fraudes": "Fraudes",
            "tasa_fraude": "Tasa Fraude (%)", "monto_prom": "Monto Prom. (CLP)",
        }).round(0),
        use_container_width=True, hide_index=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — Análisis Temporal
# ──────────────────────────────────────────────────────────────────────────────
elif page == "⏰ Análisis Temporal":
    st.subheader("⏰ Análisis Temporal del Fraude")

    hourly = df.groupby(["hour", "is_fraud"]).size().reset_index(name="count")
    hourly["tipo"] = hourly["is_fraud"].map({0: "Legítima", 1: "Fraude"})

    fig_line = px.line(
        hourly, x="hour", y="count", color="tipo",
        color_discrete_map={"Legítima": COLOR_LEGIT, "Fraude": COLOR_FRAUD},
        template=PLOTLY_TEMPLATE, height=380,
        labels={"hour": "Hora del Día", "count": "# Transacciones"},
        title="Distribución de Transacciones por Hora del Día",
        markers=True,
    )
    fig_line.add_vrect(x0=-0.5, x1=5.5, fillcolor="#f85149",
                       opacity=0.08, line_width=0, annotation_text="Horas de Alto Riesgo",
                       annotation_position="top left")
    st.plotly_chart(fig_line, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fraud_by_hour = fraud_df.groupby("hour").size().reset_index(name="fraudes")
        fig_hr = px.bar(
            fraud_by_hour, x="hour", y="fraudes",
            color="fraudes", color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE, height=320,
            title="Fraudes por Hora",
            labels={"hour": "Hora", "fraudes": "# Fraudes"},
        )
        fig_hr.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        st.markdown("### 🌙 Estadísticas Nocturnas (00:00 – 05:59)")
        night = df[df["hour"].isin(range(0, 6))]
        day   = df[~df["hour"].isin(range(0, 6))]
        m1, m2 = st.columns(2)
        m1.metric("Tasa Fraude Nocturna",  f"{night['is_fraud'].mean()*100:.2f}%",
                  delta=f"+{(night['is_fraud'].mean()-df['is_fraud'].mean())*100:.2f}% vs promedio")
        m2.metric("Tasa Fraude Diurna",    f"{day['is_fraud'].mean()*100:.2f}%")
        st.markdown("---")
        st.markdown("### 💳 País Extranjero")
        fgn = df[df["is_foreign"] == 1]
        loc = df[df["is_foreign"] == 0]
        m3, m4 = st.columns(2)
        m3.metric("Fraude (País Extranjero)", f"{fgn['is_fraud'].mean()*100:.2f}%")
        m4.metric("Fraude (País Local)",      f"{loc['is_fraud'].mean()*100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 — Evaluador en Tiempo Real
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Evaluador en Tiempo Real":
    st.subheader("🚨 Evaluador de Transacciones en Tiempo Real")
    st.markdown("Ingresa los datos de una transacción para obtener su nivel de riesgo al instante.")

    # Verificar si el modelo está disponible
    model_path = os.path.join(ROOT, "models", "fraud_model.pkl")
    model_available = os.path.exists(model_path)

    if not model_available:
        st.warning("⚠️ Modelo no entrenado aún. Ejecuta `python src/train_model.py` para activar el evaluador.")
        st.info("📌 Mientras tanto, puedes explorar el resto del dashboard con datos históricos.")
    else:
        try:
            from src.predict import predict_transaction
            MODEL_READY = True
        except Exception:
            MODEL_READY = False

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### 📝 Datos de la Transacción")
        amount      = st.number_input("💰 Monto (CLP)", min_value=100, max_value=50_000_000,
                                       value=350_000, step=10_000)
        country     = st.selectbox("🌍 País", ["CL", "AR", "RU", "NG", "CN", "BR",
                                               "PE", "MX", "US", "DE", "VN", "UA"])
        hour        = st.slider("⏰ Hora del día", 0, 23, 14)
        device_type = st.selectbox("📱 Dispositivo", ["Android", "iOS", "Web", "Windows", "Unknown"])
        fails       = st.slider("🔁 Intentos fallidos previos", 0, 10, 0)
        is_foreign  = st.checkbox("🌐 País diferente al habitual", value=False)
        high_risk   = st.checkbox("⚠️ Comercio de alto riesgo", value=False)

        evaluate = st.button("🔍 Evaluar Transacción", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Resultado del Análisis")
        if evaluate:
            transaction_data = {
                "amount":             amount,
                "country":            country,
                "hour":               hour,
                "device_type":        device_type,
                "failed_attempts":    fails,
                "is_foreign":         int(is_foreign),
                "high_risk_merchant": int(high_risk),
            }

            if model_available and MODEL_READY:
                with st.spinner("Analizando transacción..."):
                    result = predict_transaction(transaction_data)

                score = result["risk_score"]
                level = result["risk_level"]
                rec   = result["recommendation"]

                css_class = {"ALTO": "alert-alto", "MEDIO": "alert-medio", "BAJO": "alert-bajo"}[level]
                icon = {"ALTO": "🚨", "MEDIO": "⚠️", "BAJO": "✅"}[level]

                st.markdown(f"""
                <div class="{css_class}">
                    <h2 style="margin:0;">{icon} Riesgo {level}</h2>
                    <p style="font-size:2rem; margin:0.5rem 0; font-weight:700; color:white;">{score:.1%}</p>
                    <p style="margin:0; color:#ccc;">{rec}</p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": COLOR_FRAUD if level == "ALTO"
                                else "#e3b341" if level == "MEDIO" else COLOR_LEGIT},
                        "steps": [
                            {"range": [0, 30],  "color": "#1a2d1a"},
                            {"range": [30, 60], "color": "#2d2b1a"},
                            {"range": [60, 100],"color": "#2d1a1a"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": score*100},
                    },
                ))
                fig_gauge.update_layout(
                    template=PLOTLY_TEMPLATE, height=260, margin=dict(t=20, b=20),
                    font_color="white",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                # Evaluación heurística sin modelo
                score_h = (
                    (fails * 0.25)
                    + (int(is_foreign) * 0.20)
                    + (int(high_risk) * 0.20)
                    + (int(hour in range(0, 6)) * 0.15)
                    + (int(amount > 500_000) * 0.15)
                    + (int(country in ["RU","NG","CN","VN","UA"]) * 0.05)
                )
                score_h = min(score_h / 1.0, 1.0)
                level_h = "ALTO" if score_h >= 0.6 else ("MEDIO" if score_h >= 0.3 else "BAJO")
                st.info(f"📎 Evaluación heurística (sin modelo ML): **{score_h:.1%}** — Riesgo **{level_h}**")
                st.caption("Entrena el modelo para obtener resultados precisos con ML.")
        else:
            st.info("👈 Completa los datos y presiona **Evaluar Transacción**")


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 5 — Tabla de Transacciones
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📋 Tabla de Transacciones":
    st.subheader("📋 Explorador de Transacciones")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        countries = ["Todos"] + sorted(df["country"].unique().tolist())
        sel_country = st.selectbox("Filtrar por país", countries)
    with col_f2:
        sel_fraud = st.selectbox("Tipo", ["Todas", "Solo Fraudes", "Solo Legítimas"])
    with col_f3:
        min_amount = st.number_input("Monto mínimo (CLP)", value=0, step=10_000)

    filtered = df_view.copy()
    if sel_country != "Todos":
        filtered = filtered[filtered["country"] == sel_country]
    if sel_fraud == "Solo Fraudes":
        filtered = filtered[filtered["is_fraud"] == 1]
    elif sel_fraud == "Solo Legítimas":
        filtered = filtered[filtered["is_fraud"] == 0]
    filtered = filtered[filtered["amount"] >= min_amount]

    filtered["Tipo"] = filtered["is_fraud"].map({0: "✅ Legítima", 1: "🚨 Fraude"})

    st.dataframe(
        filtered[["amount","country","hour","device_type","failed_attempts",
                  "is_foreign","high_risk_merchant","Tipo"]].rename(columns={
            "amount":"Monto (CLP)", "country":"País", "hour":"Hora",
            "device_type":"Dispositivo", "failed_attempts":"Intentos Fallidos",
            "is_foreign":"Es Extranjero", "high_risk_merchant":"Alto Riesgo",
        }),
        use_container_width=True, height=500,
    )
    st.caption(f"Mostrando {len(filtered):,} de {len(df):,} transacciones")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#8b949e; margin-top:3rem; font-size:0.8rem;">
    🛡️ FraudSense v1.0.0 · Proyecto de Título · Ingeniería en Informática
</div>
""", unsafe_allow_html=True)
