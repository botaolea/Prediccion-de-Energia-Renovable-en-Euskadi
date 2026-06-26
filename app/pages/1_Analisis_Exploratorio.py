"""
Página 1: Análisis Exploratorio
EDA del dataset de generación solar.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.plots import (
    plot_correlation_heatmap,
    plot_generation_by_hour_month,
    plot_time_series_generation,
)
from app.components.sidebar import render_footer
from app.components.utils_streamlit import (
    check_data_available,
    get_config,
    load_dataset_final,
)


def main():
    """Función principal de la página de análisis exploratorio."""
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("Visualizaciones interactivas del dataset de generación solar en Bizkaia.")

    # Verificar datos
    df = load_dataset_final()

    if df is None or df.empty:
        st.warning(
            "⚠️ No hay dataset disponible. Ve a la página **Entrenar Modelo** "
            "para ejecutar el pipeline de datos, o sube tu propio CSV abajo."
        )

        # Permitir subir CSV propio
        st.markdown("### 📁 Subir tu propio CSV")
        uploaded = st.file_uploader(
            "Sube un CSV con una columna 'timestamp' y 'generacion'",
            type=["csv"],
        )
        if uploaded is not None:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(uploaded, nrows=1).columns else None)
            st.success(f"CSV cargado: {df.shape}")
        else:
            render_footer()
            return

    # ============ Estadísticas descriptivas ============
    st.markdown("## 📈 Estadísticas Descriptivas")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", f"{df.shape[1]}")
    with col3:
        if "generacion" in df.columns:
            st.metric("Generación media (kWh)", f"{df['generacion'].mean():.2f}")
    with col4:
        if "generacion" in df.columns:
            st.metric("Generación total (kWh)", f"{df['generacion'].sum():,.0f}")

    st.markdown("### DataFrame - Vista previa")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### Estadísticas numéricas")
    numeric_df = df.select_dtypes(include=[np.number])
    st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.divider()

    # ============ Visualizaciones ============
    st.markdown("## 📊 Visualizaciones Interactivas")

    # Selector de rango temporal
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        min_date = df["timestamp"].min().to_pydatetime()
        max_date = df["timestamp"].max().to_pydatetime()

        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input(
                "Fecha inicio",
                value=min_date,
                min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                max_value=max_date.date() if hasattr(max_date, 'date') else max_date,
            )
        with col_b:
            end_date = st.date_input(
                "Fecha fin",
                value=max_date,
                min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                max_value=max_date.date() if hasattr(max_date, 'date') else max_date,
            )

        # Filtrar por fecha
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime().date()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.to_pydatetime().date()
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # 1. Evolución temporal
    st.markdown("### 1️⃣ Evolución temporal de la generación")
    if "generacion" in df_filtered.columns:
        # Si hay muchos datos, resamplear a diario para visualización
        if len(df_filtered) > 5000:
            st.info(f"ℹ️ Mostrando resampleo diario (de {len(df_filtered):,} puntos horarios).")
            df_plot = df_filtered.set_index("timestamp")["generacion"].resample("1D").mean().reset_index()
        else:
            df_plot = df_filtered
        fig = plot_time_series_generation(df_plot)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'generacion' no encontrada.")

    # 2. Distribución por hora y mes
    st.markdown("### 2️⃣ Distribución por hora del día y mes")
    if "generacion" in df_filtered.columns and "timestamp" in df_filtered.columns:
        # Limitar muestra para rendimiento
        df_sample = df_filtered.sample(min(10000, len(df_filtered)), random_state=42) if len(df_filtered) > 10000 else df_filtered
        fig = plot_generation_by_hour_month(df_sample)
        st.plotly_chart(fig, use_container_width=True)

    # 3. Heatmap de correlación
    st.markdown("### 3️⃣ Matriz de correlación")
    numeric_cols = numeric_df.columns.tolist()
    if numeric_cols:
        selected_cols = st.multiselect(
            "Selecciona variables a correlacionar",
            options=numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
        )
        if selected_cols:
            fig = plot_correlation_heatmap(df_filtered, columns=selected_cols, height=600)
            st.plotly_chart(fig, use_container_width=True)

    # 4. Histograma
    st.markdown("### 4️⃣ Histograma de generación")
    if "generacion" in df_filtered.columns:
        col_x, col_bins = st.columns([3, 1])
        with col_x:
            hist_col = st.selectbox("Variable a histogramar", numeric_cols, index=numeric_cols.index("generacion") if "generacion" in numeric_cols else 0)
        with col_bins:
            n_bins = st.slider("Número de bins", 10, 100, 50)

        import plotly.express as px
        fig = px.histogram(
            df_filtered, x=hist_col, nbins=n_bins,
            title=f"Distribución de {hist_col}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Boxplots por categoría
    st.markdown("### 5️⃣ Análisis por categorías")
    if "timestamp" in df_filtered.columns and "generacion" in df_filtered.columns:
        df_filtered["hour"] = pd.to_datetime(df_filtered["timestamp"]).dt.hour
        df_filtered["month"] = pd.to_datetime(df_filtered["timestamp"]).dt.month_name()
        df_filtered["day_of_week"] = pd.to_datetime(df_filtered["timestamp"]).dt.day_name()

        cat_col = st.selectbox(
            "Agrupar por",
            ["hour", "month", "day_of_week"],
            index=0,
        )

        import plotly.express as px
        fig = px.box(
            df_filtered.sample(min(5000, len(df_filtered)), random_state=42),
            x=cat_col,
            y="generacion",
            title=f"Generación por {cat_col}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    render_footer()


# Punto de entrada cuando se ejecuta como script
if __name__ == "__main__":
    main()
