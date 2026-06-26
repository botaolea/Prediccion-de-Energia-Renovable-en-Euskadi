"""
Tests unitarios para el módulo feature_engineering.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import FeatureEngineer
from src.utils import load_config


@pytest.fixture
def config():
    """Fixture con la configuración."""
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def sample_df():
    """DataFrame horario de 7 días."""
    timestamps = pd.date_range("2024-06-01", periods=168, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "generacion": np.sin(np.arange(168) * np.pi / 12).clip(min=0) * 5,
        "codigo_postal": "48001",
        "temperatura": 15 + 10 * np.sin(np.arange(168) * np.pi / 12),
        "humedad_relativa": 60 + 10 * np.cos(np.arange(168) * np.pi / 12),
        "radiacion_solar": 800 * np.sin(np.arange(168) * np.pi / 12).clip(min=0),
        "velocidad_viento": 3 + np.random.rand(168),
        "precipitacion": np.random.rand(168) * 0.5,
    })


@pytest.fixture
def fe(config):
    """Fixture del FeatureEngineer."""
    return FeatureEngineer(config)


def test_add_temporal_features(fe, sample_df):
    """Test: añade features temporales correctamente."""
    result = fe.add_temporal_features(sample_df)
    assert "hour_local" in result.columns
    assert "sin_hour" in result.columns
    assert "cos_hour" in result.columns
    assert "day_of_year" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns
    assert "is_daylight" in result.columns
    # Rangos
    assert result["hour_local"].between(0, 23).all()
    assert result["sin_hour"].between(-1, 1).all()
    assert result["is_weekend"].isin([0, 1]).all()


def test_add_lags(fe, sample_df):
    """Test: añade lags correctamente."""
    result = fe.add_lags(sample_df)
    assert "generacion_lag_1h" in result.columns
    assert "generacion_lag_3h" in result.columns
    assert "generacion_lag_24h" in result.columns
    assert "generacion_lag_168h" in result.columns
    # El primer valor de lag_1h debe ser 0 (fillna)
    assert result["generacion_lag_1h"].iloc[0] == 0


def test_add_rolling_windows(fe, sample_df):
    """Test: añade rolling windows."""
    result = fe.add_rolling_windows(sample_df)
    assert "rolling_mean_7d" in result.columns
    assert "rolling_mean_24h" in result.columns
    assert "rolling_std_24h" in result.columns
    assert "rolling_max_24h" in result.columns


def test_add_interactions(fe, sample_df):
    """Test: añade interacciones."""
    # Necesita is_daylight (de add_temporal_features) para crear rad_solar_daylight
    df = fe.add_temporal_features(sample_df)
    result = fe.add_interactions(df)
    assert "rad_solar_daylight" in result.columns
    assert "temp_squared" in result.columns
    assert "humedad_x_rad" in result.columns


def test_handle_final_nans(fe, sample_df):
    """Test: manejo de NaN final."""
    df = sample_df.copy()
    df.loc[5, "generacion"] = np.nan
    df.loc[10, "temperatura"] = np.nan
    result = fe.handle_final_nans(df)
    assert result.isna().sum().sum() == 0


def test_get_feature_columns(fe, sample_df):
    """Test: obtiene features correctamente."""
    result = fe.add_temporal_features(sample_df)
    features = fe.get_feature_columns(result)
    assert "generacion" not in features  # target excluido
    assert "timestamp" not in features  # tiempo excluido
    assert "codigo_postal" not in features  # id excluido
    # Todas son numéricas
    for f in features:
        assert pd.api.types.is_numeric_dtype(result[f])
