"""
utils_streamlit.py - Utilidades específicas para la app Streamlit.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Asegurar que el directorio raíz está en sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logger

logger = setup_logger("streamlit_utils")


@st.cache_resource
def get_config() -> Dict[str, Any]:
    """Carga la configuración del proyecto (cached).

    Returns:
        Diccionario de configuración.
    """
    return load_config(ROOT / "config.yaml")


@st.cache_resource
def load_model():
    """Carga el modelo y scaler entrenados (cached).

    Returns:
        Instancia de ModelPredictor o None si falla.
    """
    try:
        from src.model_predictor import ModelPredictor
        config = get_config()
        predictor = ModelPredictor(config)
        predictor.load()
        logger.info("Modelo cargado correctamente.")
        return predictor
    except FileNotFoundError as e:
        logger.warning(f"Modelo no disponible: {e}")
        return None
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return None


@st.cache_data(ttl=300)
def load_dataset_final() -> Optional[pd.DataFrame]:
    """Carga el dataset final procesado (cached 5 min).

    Returns:
        DataFrame o None si no existe.
    """
    config = get_config()
    path = Path(config["data"]["processed_path"]) / "dataset_final.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return None


@st.cache_data(ttl=300)
def load_dataset_preprocesado() -> Optional[pd.DataFrame]:
    """Carga el dataset preprocesado (cached 5 min).

    Returns:
        DataFrame o None si no existe.
    """
    config = get_config()
    path = Path(config["data"]["processed_path"]) / "dataset_preprocesado.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return None


def load_shap_importance() -> Optional[pd.DataFrame]:
    """Carga el CSV de importancia SHAP.

    Returns:
        DataFrame o None.
    """
    config = get_config()
    path = Path(config["data"]["models_path"]) / "shap_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_training_results() -> Optional[Dict]:
    """Carga los resultados del entrenamiento.

    Returns:
        Diccionario o None.
    """
    config = get_config()
    path = Path(config["data"]["models_path"]) / "training_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def format_number(value: float, decimals: int = 2) -> str:
    """Formatea un número con separadores de miles.

    Args:
        value: Valor a formatear.
        decimals: Número de decimales.

    Returns:
        String formateado.
    """
    try:
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def check_data_available() -> Tuple[bool, str]:
    """Verifica si hay datos disponibles para la app.

    Returns:
        Tupla (disponible, mensaje).
    """
    df = load_dataset_final()
    if df is None or df.empty:
        return False, (
            "⚠️ No hay dataset disponible. Ve a la página **Entrenar Modelo** "
            "y ejecuta el pipeline de datos, o ejecuta manualmente:\n"
            "```\npython src/data_ingestion.py\n"
            "python src/data_preprocessing.py\n"
            "python src/feature_engineering.py\n```"
        )
    return True, f"✅ Dataset disponible: {df.shape[0]:,} filas, {df.shape[1]} columnas"


def render_data_status():
    """Muestra el estado de los datos en la app."""
    ok, msg = check_data_available()
    if ok:
        st.success(msg)
    else:
        st.warning(msg)
    return ok
