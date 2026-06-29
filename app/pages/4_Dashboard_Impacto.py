"""
Página 4: Dashboard de Impacto
KPIs globales, ROI, mapa de Bizkaia, y simulador what-if.
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

import folium
from streamlit_folium import st_folium

from app.components.plots import plot_whatif_simulation
from app.components.sidebar import render_footer
from app.components.utils_streamlit import get_config, load_model
from src.utils import BIZKAIA_CP_REFERENCE, kwh_to_co2_avoided, kwh_to_euros


def main():
    """Función principal del dashboard de impacto."""
    st.title("📈 Dashboard de Impacto y Contexto")
    st.markdown("KPIs globales, retorno económico, mapa de Bizkaia y simulador what-if.")

    predictor = load_model()
    config = get_config()
    price_per_mwh = config["business"]["electricity_price_per_mwh"]
    co2_per_mwh = config["business"]["co2_avoided_per_mwh_kg"]

    # ============ KPIs globales ============
    st.markdown("## 🌍 KPIs Globales (predicción 24h)")

    if predictor is None:
        st.warning("⚠️ Modelo no disponible. Mostrando valores simulados.")
        # Valores demo
        total_kwh_24h = 0
        max_kwh = 0
    else:
        with st.spinner("Calculando predicción 24h..."):
            meteorologia = predictor._default_meteorologia(24)
            result = predictor.predict(
                start_time=datetime.now(),
                horizon_hours=24,
                meteorologia=meteorologia,
                codigo_postal="48001",
            )
            total_kwh_24h = result["total_predicted_kwh"]
            preds = result["predictions"]
            max_kwh = max(preds) if preds else 0

    # Calcular métricas
    ahorro_euros_24h = kwh_to_euros(total_kwh_24h, price_per_mwh)
    co2_evitado_24h = kwh_to_co2_avoided(total_kwh_24h, co2_per_mwh)

    # Simular acumulados (en ausencia de histórico, multiplicar)
    ahorro_7d = ahorro_euros_24h * 7
    ahorro_30d = ahorro_euros_24h * 30
    co2_7d = co2_evitado_24h * 7
    co2_30d = co2_evitado_24h * 30

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Generación 24h (kWh)", f"{total_kwh_24h:,.2f}")
    with col2:
        st.metric("Pico máximo (kWh)", f"{max_kwh:.2f}")
    with col3:
        st.metric("Ahorro 24h (€)", f"{ahorro_euros_24h:,.2f}")
    with col4:
        st.metric("CO2 evitado 24h (kg)", f"{co2_evitado_24h:,.2f}")

    st.markdown("### 💰 Ahorro acumulado estimado")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Últimas 24h (€)", f"{ahorro_euros_24h:,.2f}")
    with col_b:
        st.metric("Últimos 7 días (€)", f"{ahorro_7d:,.2f}")
    with col_c:
        st.metric("Últimos 30 días (€)", f"{ahorro_30d:,.2f}")

    st.markdown("### 🌱 CO2 evitado acumulado")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("24h (kg CO2)", f"{co2_evitado_24h:,.2f}")
    with col_b:
        st.metric("7 días (kg CO2)", f"{co2_7d:,.2f}")
    with col_c:
        st.metric("30 días (kg CO2)", f"{co2_30d:,.2f}")

    st.divider()

    # ============ Tabla de precios ============
    st.markdown("## ⚙️ Configuración de Negocio")
    col1, col2 = st.columns(2)
    with col1:
        new_price = st.number_input(
            "Precio del pool eléctrico (€/MWh)",
            min_value=0.0, max_value=500.0,
            value=float(price_per_mwh), step=10.0,
        )
    with col2:
        new_co2 = st.number_input(
            "Factor CO2 red (kg CO2/MWh)",
            min_value=0.0, max_value=500.0,
            value=float(co2_per_mwh), step=10.0,
        )

    # Recalcular con nuevo precio
    if new_price != price_per_mwh:
        ahorro_euros_24h = kwh_to_euros(total_kwh_24h, new_price)
        st.info(f"💰 Ahorro recalculado (24h): **{ahorro_euros_24h:,.2f} €**")

    st.divider()

    # ============ Mapa de Bizkaia ============
    st.markdown("## 🗺️ Mapa de Bizkaia - Potencial de Generación")
    _render_mapa_bizkaia(predictor, new_price)

    st.divider()

    # ============ Simulador What-if ============
    st.markdown("## 🧪 Simulador What-If")
    st.markdown("Modifica las variables meteorológicas y observa cómo cambia la generación.")
    _render_whatif_simulator(predictor)

    render_footer()


def _populate_map_predictions(predictor, cache: dict) -> dict:
    """Rellena un cache de predicciones para el mapa de Bizkaia una sola vez por sesión."""
    if predictor is None:
        return cache

    for cp in BIZKAIA_CP_REFERENCE:
        if cp in cache:
            continue
        try:
            meteorologia = predictor._default_meteorologia(24)
            result = predictor.predict(
                start_time=datetime.now(),
                horizon_hours=24,
                meteorologia=meteorologia,
                codigo_postal=cp,
            )
            cache[cp] = float(result.get("total_predicted_kwh", 0.0))
        except Exception:
            cache[cp] = 0.0

    return cache


def get_solar_panel_reference_points() -> list[dict]:
    """Devuelve una lista de puntos de referencia para mostrar ubicaciones de interés solar."""
    return [
        {
            "name": "Bilbao",
            "lat": 43.2630,
            "lon": -2.9350,
            "description": "Área urbana con alto potencial de tejados",
        },
        {
            "name": "Barakaldo",
            "lat": 43.2960,
            "lon": -2.9860,
            "description": "Zona industrial y residencial con rooftops",
        },
        {
            "name": "Getxo",
            "lat": 43.3470,
            "lon": -3.0080,
            "description": "Zonas residenciales con buen potencial solar",
        },
        {
            "name": "Santurtzi",
            "lat": 43.3280,
            "lon": -3.0350,
            "description": "Área portuaria con espacios de cubierta",
        },
        {
            "name": "Durango",
            "lat": 43.1710,
            "lon": -2.6360,
            "description": "Municipio con expansión urbana y tejados amplios",
        },
    ]


def _add_solar_panel_reference_layer(map_obj) -> None:
    """Añade una capa de referencia con puntos de interés para paneles solares."""
    solar_group = folium.FeatureGroup(name="Paneles solares (referencia)", show=True)
    for point in get_solar_panel_reference_points():
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=f"<b>{point['name']}</b><br>{point['description']}",
            tooltip=point["name"],
            icon=folium.Icon(color="green", icon="sun", prefix="fa"),
        ).add_to(solar_group)
    solar_group.add_to(map_obj)


def _render_mapa_bizkaia(predictor, price_per_mwh: float):
    """Renderiza un mapa de Bizkaia con potencial de generación por CP."""

    cache_key = "bizkaia_map_predictions"
    map_predictions = st.session_state.setdefault(cache_key, {})
    map_predictions = _populate_map_predictions(predictor, map_predictions)
    st.session_state[cache_key] = map_predictions

    # Crear mapa centrado en Bilbao
    m = folium.Map(
        location=[43.2630, -2.9350],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    # Añadir capa de referencia
    _add_solar_panel_reference_layer(m)

    if predictor is None:
        st.warning("⚠️ Modelo no disponible para mapa predictivo. Mostrando ubicaciones.")

        for cp, info in BIZKAIA_CP_REFERENCE.items():
            folium.Marker(
                location=[info["lat"], info["lon"]],
                popup=(
                    f"<b>{info['municipio']}</b><br>"
                    f"CP: {cp}<br>"
                    f"Tipo: {info['tipo']}"
                ),
                tooltip=info["municipio"],
                icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
            ).add_to(m)

    else:
        for cp, info in BIZKAIA_CP_REFERENCE.items():

            total_kwh = map_predictions.get(cp, 0.0)
            euros = kwh_to_euros(total_kwh, price_per_mwh)

            if total_kwh > 50:
                color = "green"
            elif total_kwh > 20:
                color = "orange"
            else:
                color = "red"

            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=max(5, total_kwh / 10),
                popup=folium.Popup(
                    f"<b>{info['municipio']}</b><br>"
                    f"CP: {cp}<br>"
                    f"Tipo: {info['tipo']}<br>"
                    f"<b>Generación 24h: {total_kwh:.2f} kWh</b><br>"
                    f"Ahorro: {euros:.2f} €",
                    max_width=300,
                ),
                tooltip=f"{info['municipio']}: {total_kwh:.1f} kWh",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width=800,
        height=500,
        returned_objects=[],
    )

def _render_table_bizkaia(predictor, price_per_mwh: float):
    """Tabla alternativa al mapa."""
    rows = []
    if predictor:
        for cp, info in BIZKAIA_CP_REFERENCE.items():
            try:
                meteorologia = predictor._default_meteorologia(24)
                result = predictor.predict(
                    start_time=datetime.now(),
                    horizon_hours=24,
                    meteorologia=meteorologia,
                    codigo_postal=cp,
                )
                total_kwh = result["total_predicted_kwh"]
                rows.append({
                    "CP": cp,
                    "Municipio": info["municipio"],
                    "Tipo": info["tipo"],
                    "Generación 24h (kWh)": round(total_kwh, 2),
                    "Ahorro (€)": round(kwh_to_euros(total_kwh, price_per_mwh), 2),
                })
            except Exception:
                continue

    if rows:
        df_map = pd.DataFrame(rows)
        st.dataframe(df_map, use_container_width=True)


def _render_whatif_simulator(predictor):
    """Renderiza el simulador what-if."""
    if predictor is None:
        st.warning("⚠️ Modelo no disponible. No se puede ejecutar la simulación.")
        return

    st.markdown("### 🎚️ Ajusta las variables y observa el impacto")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp_change = st.slider("Cambio temperatura (°C)", -10, 10, 0)
    with col2:
        rad_change_pct = st.slider("Cambio radiación (%)", -50, 50, 0)
    with col3:
        humedad_change = st.slider("Cambio humedad (%)", -30, 30, 0)
    with col4:
        viento_change = st.slider("Cambio viento (m/s)", -5, 5, 0)

    if st.button("🔄 Simular", type="primary"):
        with st.spinner("Simulando escenarios..."):
            horizon = 24
            base_meteo = predictor._default_meteorologia(horizon)
            modified_meteo = {
                "temperatura": [t + temp_change for t in base_meteo["temperatura"]],
                "humedad_relativa": [max(0, min(100, h + humedad_change)) for h in base_meteo["humedad_relativa"]],
                "radiacion_solar": [max(0, r * (1 + rad_change_pct / 100)) for r in base_meteo["radiacion_solar"]],
                "velocidad_viento": [max(0, v + viento_change) for v in base_meteo["velocidad_viento"]],
                "precipitacion": base_meteo["precipitacion"],
            }

            # Predicción base
            base_result = predictor.predict(
                start_time=datetime.now(),
                horizon_hours=horizon,
                meteorologia=base_meteo,
                codigo_postal="48001",
            )

            # Predicción modificada
            modified_result = predictor.predict(
                start_time=datetime.now(),
                horizon_hours=horizon,
                meteorologia=modified_meteo,
                codigo_postal="48001",
            )

            # Comparativa
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Generación base (kWh)", f"{base_result['total_predicted_kwh']:.2f}")
            with col_b:
                st.metric("Generación modificada (kWh)", f"{modified_result['total_predicted_kwh']:.2f}")
            with col_c:
                diff = modified_result["total_predicted_kwh"] - base_result["total_predicted_kwh"]
                st.metric("Diferencia (kWh)", f"{diff:+.2f}", delta=f"{diff/base_result['total_predicted_kwh']*100:+.1f}%" if base_result['total_predicted_kwh'] > 0 else "N/A")

            # Gráfico
            fig = plot_whatif_simulation(
                base_prediction=base_result["predictions"],
                modified_prediction=modified_result["predictions"],
                timestamps=base_result["timestamps"],
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
