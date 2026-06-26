"""
Tests unitarios para el módulo utils.py.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    EDIFICIOS_TIPO,
    BIZKAIA_CP_REFERENCE,
    haversine_distance,
    is_bizkaia_postal_code,
    kwh_to_co2_avoided,
    kwh_to_euros,
    kwh_to_mwh,
    load_config,
    safe_int_convert,
    setup_logger,
    validate_dataframe,
)


def test_load_config():
    """Test: carga configuración correctamente."""
    config = load_config(ROOT / "config.yaml")
    assert config["project"]["name"] == "Predicción Energía Renovable Euskadi"
    assert "data" in config
    assert "model" in config


def test_is_bizkaia_postal_code():
    """Test: identifica CPs de Bizkaia."""
    assert is_bizkaia_postal_code("48001") is True
    assert is_bizkaia_postal_code("48999") is True
    assert is_bizkaia_postal_code("47001") is False  # Castellón
    assert is_bizkaia_postal_code("28001") is False  # Madrid
    assert is_bizkaia_postal_code("invalid") is False
    assert is_bizkaia_postal_code("") is False


def test_safe_int_convert():
    """Test: conversión segura a entero."""
    assert safe_int_convert("123") == 123
    assert safe_int_convert("123.45") == 123
    assert safe_int_convert("invalid") == 0
    assert safe_int_convert(None) == 0
    assert safe_int_convert("123", default=99) == 123
    assert safe_int_convert("invalid", default=99) == 99


def test_kwh_to_mwh():
    """Test: conversión kWh a MWh."""
    assert kwh_to_mwh(1000) == 1.0
    assert kwh_to_mwh(500) == 0.5


def test_kwh_to_euros():
    """Test: conversión kWh a euros."""
    # 1000 kWh = 1 MWh, a 100 €/MWh = 100 €
    assert kwh_to_euros(1000, 100) == 100.0
    assert kwh_to_euros(500, 100) == 50.0


def test_kwh_to_co2_avoided():
    """Test: cálculo de CO2 evitado."""
    # 1000 kWh = 1 MWh, a 200 kg CO2/MWh = 200 kg
    assert kwh_to_co2_avoided(1000, 200) == 200.0


def test_haversine_distance():
    """Test: distancia Haversine."""
    # Bilbao - Madrid (~323 km aproximadamente)
    dist = haversine_distance(43.2630, -2.9350, 40.4168, -3.7038)
    assert 300 < dist < 350
    # Distancia a sí mismo = 0
    assert haversine_distance(43.2630, -2.9350, 43.2630, -2.9350) == 0


def test_validate_dataframe():
    """Test: validación de columnas."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert validate_dataframe(df, ["a", "b"]) is True

    with pytest.raises(ValueError, match="Faltan"):
        validate_dataframe(df, ["a", "c"], "test_df")


def test_setup_logger():
    """Test: configuración de logger."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 20  # INFO


def test_edificios_tipo():
    """Test: definición de edificios tipo."""
    assert "Oficina en Bilbao" in EDIFICIOS_TIPO
    assert "Vivienda en Getxo" in EDIFICIOS_TIPO
    for nombre, datos in EDIFICIOS_TIPO.items():
        assert "codigo_postal" in datos
        assert "tipo_edificio" in datos
        assert "superficie_util_m2" in datos


def test_bizkaia_cp_reference():
    """Test: referencia de CPs de Bizkaia."""
    assert "48001" in BIZKAIA_CP_REFERENCE
    assert "municipio" in BIZKAIA_CP_REFERENCE["48001"]
    assert BIZKAIA_CP_REFERENCE["48001"]["municipio"] == "Bilbao"
