"""
Página 3: Predecir Generación
Funcionalidad principal: predicción 24-48h con formulario, CSV o ejemplos predefinidos.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.plots import plot_predictions_with_interval
from app.components.sidebar import render_footer
from app.components.utils_streamlit import get_config, load_model
from src.utils import EDIFICIOS_TIPO


def main():
    """Función principal de la página de predicción."""
    st.title("🔮 Predecir Generación Solar")
    st.markdown("Predice la generación solar fotovoltaica para las próximas **24-48 horas**.")

    # Cargar modelo
    predictor = load_model()
    if predictor is None:
        st.error(
            "❌ Modelo no disponible. Ve a la página **Entrenar Modelo** "
            "para entrenar uno, o ejecuta el script `scripts/train_model.py`."
        )
        render_footer()
        return

    # ============ Selección de modo ============
    st.markdown("## 🎛️ Selecciona el modo de predicción")
    mode = st.radio(
        "Modo",
        ["🏠 Edificio tipo (predefinido)", "📝 Formulario manual", "📤 Carga masiva CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("🏠"):
        _render_edificio_tipo(predictor)
    elif mode.startswith("📝"):
        _render_formulario_manual(predictor)
    else:
        _render_carga_masiva(predictor)

    render_footer()


def _render_edificio_tipo(predictor):
    """Renderiza el modo de edificio tipo predefinido."""
    st.markdown("### 🏠 Selección de edificio tipo")
    st.markdown("Elige un edificio predefinido con características promedio de Bizkaia.")

    edificio_nombre = st.selectbox(
        "Edificio",
        list(EDIFICIOS_TIPO.keys()),
        index=0,
    )
    edificio = EDIFICIOS_TIPO[edificio_nombre]

    # Mostrar características del edificio
    with st.expander("📋 Características del edificio"):
        st.json(edificio)

    # Horizonte de predicción
    horizon = st.slider("Horas a predecir", 12, 48, 24, step=12)

    # Botón predecir
    if st.button("🔮 Predecir Generación", type="primary"):
        with st.spinner("Calculando predicción..."):
            # Generar meteorología por defecto estacional
            meteorologia = predictor._default_meteorologia(horizon)

            result = predictor.predict(
                start_time=datetime.now(),
                horizon_hours=horizon,
                meteorologia=meteorologia,
                codigo_postal=edificio["codigo_postal"],
            )

            _display_prediction_results(result, edificio_nombre)


def _render_formulario_manual(predictor):
    """Renderiza el formulario manual."""
    st.markdown("### 📝 Formulario manual")
    st.markdown("Especifica manualmente las variables para la predicción.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", datetime.now().date())
        start_time = st.time_input("Hora de inicio", datetime.now().time())
        horizon = st.slider("Horas a predecir", 12, 48, 24, step=12)
        codigo_postal = st.text_input("Código postal", value="48001", max_chars=5)

    with col2:
        # Variables meteorológicas (valores base)
        st.markdown("**Meteorología base** (se generará variación horaria)")
        temp_base = st.slider("Temperatura base (°C)", -5.0, 40.0, 15.0, 0.5)
        rad_max = st.slider("Radiación solar máxima (W/m²)", 0, 1200, 800, 50)
        humedad = st.slider("Humedad relativa (%)", 0, 100, 65)
        viento = st.slider("Velocidad del viento (m/s)", 0.0, 20.0, 3.0, 0.5)
        last_gen = st.number_input("Última generación conocida (kWh)", 0.0, 100.0, 0.0, 0.1)

    # Generar meteorología horaria
    start_dt = datetime.combine(start_date, start_time)
    timestamps = pd.date_range(start=start_dt, periods=horizon, freq="1h")
    hours = timestamps.hour

    # Variación diurna
    rad = np.array([
        max(0, rad_max * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0
        for h in hours
    ])
    meteorologia = {
        "temperatura": (temp_base + 5 * np.sin(np.pi * (hours - 6) / 12)).tolist(),
        "humedad_relativa": [humedad] * horizon,
        "radiacion_solar": rad.tolist(),
        "velocidad_viento": [viento] * horizon,
        "precipitacion": [0.0] * horizon,
    }

    if st.button("🔮 Predecir", type="primary"):
        with st.spinner("Calculando..."):
            result = predictor.predict(
                start_time=start_dt,
                horizon_hours=horizon,
                meteorologia=meteorologia,
                codigo_postal=codigo_postal,
                last_generation=last_gen,
            )
            _display_prediction_results(result, "Formulario manual")


def _render_carga_masiva(predictor):
    """Renderiza la carga masiva por CSV."""
    st.markdown("### 📤 Carga masiva por CSV")
    st.markdown("Sube un CSV con los registros a predecir. Debe contener las columnas meteorológicas y `timestamp`.")

    # Template de descarga
    st.markdown("#### 📋 Descargar template CSV")
    template_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=24, freq="1h"),
        "temperatura": [15.0] * 24,
        "humedad_relativa": [65.0] * 24,
        "radiacion_solar": [0.0] * 24,
        "velocidad_viento": [3.0] * 24,
        "precipitacion": [0.0] * 24,
        "codigo_postal": ["48001"] * 24,
    })
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar template",
        data=csv_template,
        file_name="template_prediccion.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Sube tu CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded, parse_dates=["timestamp"])
            st.success(f"CSV cargado: {df_input.shape}")

            with st.spinner("Calculando predicciones..."):
                # Guardar temporalmente y usar predict_from_csv
                tmp_path = Path("/tmp/upload_pred.csv")
                df_input.to_csv(tmp_path, index=False)
                result = predictor.predict_from_csv(tmp_path)

                _display_prediction_results(result, "Carga masiva CSV")
        except Exception as e:
            st.error(f"Error procesando CSV: {e}")


def _display_prediction_results(result: dict, source_name: str):
    """Muestra los resultados de la predicción.

    Args:
        result: Diccionario con resultados de la predicción.
        source_name: Nombre del origen de la predicción.
    """
    st.divider()
    st.markdown(f"## 📊 Resultados de Predicción ({source_name})")

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Horas predichas", result.get("horizon_hours", 0))
    with col2:
        st.metric("Generación total (kWh)", f"{result.get('total_predicted_kwh', 0):.2f}")
    with col3:
        preds = result.get("predictions", [])
        avg_pred = sum(preds) / len(preds) if preds else 0
        st.metric("Media horaria (kWh)", f"{avg_pred:.2f}")
    with col4:
        max_pred = max(preds) if preds else 0
        st.metric("Pico máximo (kWh)", f"{max_pred:.2f}")

    # Gráfico de predicción con intervalo
    st.markdown("### 📈 Gráfico de predicción con intervalo de confianza (95%)")
    fig = plot_predictions_with_interval(
        timestamps=result["timestamps"],
        predictions=result["predictions"],
        lower=result["lower_bound"],
        upper=result["upper_bound"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de predicciones
    st.markdown("### 📋 Tabla de predicciones horarias")
    df_preds = pd.DataFrame({
        "Timestamp": result["timestamps"],
        "Predicción (kWh)": [round(p, 3) for p in result["predictions"]],
        "Límite inferior": [round(l, 3) for l in result["lower_bound"]],
        "Límite superior": [round(u, 3) for u in result["upper_bound"]],
    })
    st.dataframe(df_preds, use_container_width=True)

    # Descargar CSV
    csv_data = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar predicciones (CSV)",
        data=csv_data,
        file_name=f"prediccion_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # Top 5 features
    top_5 = result.get("top_5_features", [])
    if top_5:
        st.markdown("### 🔍 Top 5 features que más influyen (SHAP global)")
        st.info(
            "Estas son las variables que más impacto tienen en las predicciones, "
            "según los SHAP values globales precalculados del modelo."
        )
        for i, feat in enumerate(top_5, 1):
            feature_name = feat.get("feature", "unknown")
            shap_value = feat.get("mean_abs_shap", 0)
            st.markdown(f"**{i}. {feature_name}** - Importancia SHAP: `{shap_value:.4f}`")


if __name__ == "__main__":
    main()
