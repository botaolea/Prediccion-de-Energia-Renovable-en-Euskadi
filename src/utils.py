"""
utils.py - Funciones auxiliares para el proyecto de Predicción de Energía Renovable.

Contiene utilidades para:
- Carga/guardado de configuración YAML
- Configuración de logging
- Geolocalización y manipulación de fechas
- Validación de datos
- Conversión de unidades
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Carga el archivo de configuración YAML central.

    Args:
        config_path: Ruta al archivo config.yaml.

    Returns:
        Diccionario con la configuración del proyecto.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo está mal formado.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Devuelve la ruta raíz del proyecto.

    Returns:
        Path absoluto a la raíz del proyecto.
    """
    # Asume que este archivo está en src/utils.py
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """Crea el directorio si no existe.

    Args:
        path: Ruta del directorio a crear.

    Returns:
        Path del directorio creado.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(
    name: str = "energia_euskadi",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configura un logger con formato estándar.

    Args:
        name: Nombre del logger.
        level: Nivel de logging (logging.INFO por defecto).
        log_file: Ruta opcional a archivo de log.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de archivo (opcional)
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Fechas y geolocalización
# ---------------------------------------------------------------------------
def utc_to_local(utc_timestamp: pd.Timestamp, tz: str = "Europe/Madrid") -> pd.Timestamp:
    """Convierte un timestamp UTC a hora local peninsular.

    Args:
        utc_timestamp: Timestamp en UTC.
        tz: Zona horaria destino.

    Returns:
        Timestamp en hora local.
    """
    if utc_timestamp.tzinfo is None:
        utc_timestamp = utc_timestamp.tz_localize("UTC")
    return utc_timestamp.tz_convert(tz)


def get_bizkaia_postal_codes() -> list[int]:
    """Devuelve los rangos de códigos postales de Bizkaia.

    Bizkaia usa códigos postales del rango 48000-48999.

    Returns:
        Lista con el rango de códigos postales de Bizkaia.
    """
    return list(range(48000, 49000))


def is_bizkaia_postal_code(cp: Union[int, str, float]) -> bool:
    """Verifica si un código postal pertenece a Bizkaia.

    Args:
        cp: Código postal a verificar.

    Returns:
        True si el CP está en el rango 48000-48999.
    """
    try:
        cp_int = int(float(cp))
        return 48000 <= cp_int <= 48999
    except (ValueError, TypeError):
        return False


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calcula la distancia Haversine entre dos puntos geográficos en km.

    Args:
        lat1: Latitud punto 1.
        lon1: Longitud punto 1.
        lat2: Latitud punto 2.
        lon2: Longitud punto 2.

    Returns:
        Distancia en kilómetros.
    """
    import numpy as np

    R = 6371.0  # Radio de la Tierra en km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# ---------------------------------------------------------------------------
# Conversión de unidades
# ---------------------------------------------------------------------------
def kwh_to_mwh(kwh: float) -> float:
    """Convierte kWh a MWh.

    Args:
        kwh: Energía en kilovatios-hora.

    Returns:
        Energía en megavatios-hora.
    """
    return kwh / 1000.0


def kwh_to_euros(kwh: float, price_per_mwh: float = 100.0) -> float:
    """Convierte kWh a euros según el precio del pool eléctrico.

    Args:
        kwh: Energía en kilovatios-hora.
        price_per_mwh: Precio en €/MWh (por defecto 100 €/MWh).

    Returns:
        Valor en euros.
    """
    return kwh_to_mwh(kwh) * price_per_mwh


def kwh_to_co2_avoided(kwh: float, co2_per_mwh_kg: float = 200.0) -> float:
    """Estima el CO2 evitado por generar energía renovable.

    Args:
        kwh: Energía renovable generada en kWh.
        co2_per_mwh_kg: Factor de emisión de CO2 de la red en kg/MWh.

    Returns:
        CO2 evitado en kg.
    """
    return kwh_to_mwh(kwh) * co2_per_mwh_kg


# ---------------------------------------------------------------------------
# Validación de datos
# ---------------------------------------------------------------------------
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str = "DataFrame",
) -> bool:
    """Valida que un DataFrame contenga las columnas requeridas.

    Args:
        df: DataFrame a validar.
        required_columns: Lista de columnas obligatorias.
        df_name: Nombre descriptivo del DataFrame para mensajes.

    Returns:
        True si todas las columnas están presentes.

    Raises:
        ValueError: Si faltan columnas.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} no contiene las columnas requeridas. Faltan: {missing}"
        )
    return True


def safe_int_convert(value: Any, default: int = 0) -> int:
    """Conversión segura a entero.

    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.

    Returns:
        Entero convertido o valor por defecto.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Obtiene una variable de entorno de forma segura.

    Args:
        name: Nombre de la variable.
        default: Valor por defecto.

    Returns:
        Valor de la variable o default.
    """
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Códigos postales de ejemplo (Bizkaia)
# ---------------------------------------------------------------------------
BIZKAIA_CP_REFERENCE: Dict[str, Dict[str, Any]] = {
    "48001": {"municipio": "Bilbao", "tipo": "urbano", "lat": 43.2603, "lon": -2.9334},
    "48010": {"municipio": "Bilbao", "tipo": "urbano", "lat": 43.2581, "lon": -2.9456},
    "48015": {"municipio": "Bilbao", "tipo": "urbano", "lat": 43.2647, "lon": -2.9476},
    "48901": {"municipio": "Barakaldo", "tipo": "urbano", "lat": 43.2972, "lon": -3.0172},
    "48930": {"municipio": "Getxo", "tipo": "urbano", "lat": 43.3554, "lon": -3.0107},
    "48200": {"municipio": "Durango", "tipo": "semiurbano", "lat": 43.1711, "lon": -2.6904},
    "48100": {"municipio": "Mungia", "tipo": "rural", "lat": 43.3545, "lon": -2.8456},
    "48340": {"municipio": "Amorebieta", "tipo": "semiurbano", "lat": 43.2197, "lon": -2.7319},
}

# Ejemplos predefinidos para la app (Página 3 - Opción C)
EDIFICIOS_TIPO: Dict[str, Dict[str, Any]] = {
    "Oficina en Bilbao": {
        "codigo_postal": "48001",
        "tipo_edificio": "oficina",
        "superficie_util_m2": 250.0,
        "calificacion_energetica": "C",
        "anyo_construccion": 2005,
        "potencia_instalada_kw": 15.0,
    },
    "Vivienda en Getxo": {
        "codigo_postal": "48930",
        "tipo_edificio": "vivienda",
        "superficie_util_m2": 110.0,
        "calificacion_energetica": "B",
        "anyo_construccion": 2010,
        "potencia_instalada_kw": 4.5,
    },
    "Nave industrial en Barakaldo": {
        "codigo_postal": "48901",
        "tipo_edificio": "industrial",
        "superficie_util_m2": 800.0,
        "calificacion_energetica": "D",
        "anyo_construccion": 1998,
        "potencia_instalada_kw": 50.0,
    },
    "Vivienda unifamiliar en Mungia": {
        "codigo_postal": "48100",
        "tipo_edificio": "vivienda",
        "superficie_util_m2": 180.0,
        "calificacion_energetica": "A",
        "anyo_construccion": 2018,
        "potencia_instalada_kw": 8.0,
    },
}


if __name__ == "__main__":
    # Test rápido
    logger = setup_logger()
    logger.info("Test de utils.py")
    config = load_config()
    logger.info(f"Proyecto: {config['project']['name']}")
    logger.info(f"Bizkaia CP válidos: {is_bizkaia_postal_code('48001')}")
