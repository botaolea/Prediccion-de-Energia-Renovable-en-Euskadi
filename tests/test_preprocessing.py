"""
Tests unitarios para el módulo data_preprocessing.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Añadir raíz al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import DataPreprocessor
from src.utils import load_config


@pytest.fixture
def config():
    """Fixture con la configuración del proyecto."""
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def sample_df():
    """DataFrame de ejemplo para tests."""
    timestamps = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "generacion": np.random.rand(48) * 5,
        "consumo_activo": np.random.rand(48) * 10,
        "codigo_postal": ["48001"] * 48,
    })


@pytest.fixture
def preprocessor(config):
    """Fixture del preprocesador."""
    return DataPreprocessor(config)


def test_filter_bizkaia(preprocessor, sample_df):
    """Test: filtra correctamente los CP de Bizkaia."""
    # Añadir un CP que no es de Bizkaia
    df = pd.concat([
        sample_df,
        pd.DataFrame({
            "timestamp": pd.date_range("2024-01-03", periods=5, freq="1h", tz="UTC"),
            "generacion": [1] * 5,
            "consumo_activo": [2] * 5,
            "codigo_postal": ["28001"] * 5,  # Madrid
        })
    ])
    filtered = preprocessor.filter_bizkaia(df)
    assert len(filtered) == 48  # Solo quedan los de Bizkaia
    assert all(filtered["codigo_postal"].astype(str).str.startswith("48"))


def test_resample_hourly(preprocessor, sample_df):
    """Test: resamplea correctamente a frecuencia horaria."""
    resampled = preprocessor.resample_hourly(sample_df)
    # 48 horas -> 48 filas
    assert len(resampled) == 48
    assert "generacion" in resampled.columns
    assert "codigo_postal" in resampled.columns


def test_impute_final_nans(preprocessor):
    """Test: imputa NaN correctamente."""
    df = pd.DataFrame({
        "generacion": [1.0, np.nan, 3.0, 0.0],
        "generacion_lag_1h": [np.nan, 1.0, 3.0, 3.0],
        "temperatura": [15.0, np.nan, 18.0, 20.0],
    })
    result = preprocessor.impute_final_nans(df)
    assert result.isna().sum().sum() == 0
    # Lags rellenan con 0
    assert result["generacion_lag_1h"].iloc[0] == 0
    # Meteorología rellena con mediana (mediana de [15, 18, 20] = 18)
    assert result["temperatura"].iloc[1] == 18.0


def test_clean_outliers(preprocessor):
    """Test: detecta y maneja outliers."""
    df = pd.DataFrame({
        "generacion": [1.0, 2.0, 3.0, 1000.0, 4.0]  # 1000 es outlier
    })
    result = preprocessor.clean_outliers(df, col="generacion", lower=0, upper=100)
    # El outlier se convierte a NaN
    assert result["generacion"].isna().sum() == 1
