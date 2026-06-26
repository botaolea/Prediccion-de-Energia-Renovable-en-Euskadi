"""
sidebar.py - Componente de barra lateral de la app Streamlit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from src.utils import load_config


def render_sidebar() -> Dict[str, Any]:
    """Renderiza la barra lateral con información del proyecto.

    Returns:
        Diccionario con configuración del proyecto.
    """
    config = load_config()

    with st.sidebar:
        st.markdown("## ⚡ Energía Renovable Euskadi")
        st.markdown("**Plataforma de predicción solar fotovoltaica en Bizkaia**")
        st.divider()

        st.divider()

        # Mostrar estado del modelo
        models_path = Path(config["data"]["models_path"])
        model_exists = (models_path / "xgboost_model.pkl").exists()
        shap_exists = (models_path / "shap_importance.csv").exists()
        scaler_exists = (models_path / "scaler.pkl").exists()

        st.markdown("### 🔧 Estado del Sistema")
        if model_exists:
            st.success("✅ Modelo cargado")
        else:
            st.warning("⚠️ Modelo no entrenado")
        if shap_exists:
            st.success("✅ SHAP disponible")
        else:
            st.warning("⚠️ SHAP no calculado")
        if scaler_exists:
            st.success("✅ Scaler disponible")
        else:
            st.warning("⚠️ Scaler no disponible")

        st.divider()

        st.markdown("### ℹ️ Información")
        st.markdown(f"""
        **Versión**: {config['project']['version']}

        **Localización**: Bilbao (43.26°N, 2.94°W)

        **Datos**: Open Data Euskadi, AEMET, GoiEner
        """)

    return config


def render_footer():
    """Renderiza el pie de página."""
    st.divider()
    st.caption("📊 Predicción Energía Renovable Euskadi · Datos abiertos reales · 2024")
