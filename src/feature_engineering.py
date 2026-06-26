"""
feature_engineering.py - Creación de variables para el modelo de predicción.

Incluye:
- Conversión a hora local (Europe/Madrid)
- Features cíclicas (sin/cos de hora, día, mes)
- Posición solar (pvlib / astral / heurística)
- Variables de lag (1h, 3h, 24h, 168h)
- Rolling windows (7 días)
- Variables de interacción
- Manejo final de nulos
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """Genera features para el modelo de predicción de generación solar."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el feature engineer.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.processed_path = Path(config["data"]["processed_path"])
        self.models_path = Path(config["data"]["models_path"])
        ensure_dir(self.processed_path)
        ensure_dir(self.models_path)

        self.target_col = config["model"]["target_column"]
        self.time_col = config["model"]["time_column"]
        self.id_col = config["model"]["id_column"]

        # Localización para cálculos solares
        loc_cfg = config.get("location", {})
        self.latitude = loc_cfg.get("latitude", 43.2630)
        self.longitude = loc_cfg.get("longitude", -2.9350)
        self.timezone = loc_cfg.get("timezone", "Europe/Madrid")
        self.altitude = loc_cfg.get("altitude", 30)

    # ------------------------------------------------------------------
    # A. Features temporales y hora local
    # ------------------------------------------------------------------
    def convert_to_local_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte timestamp de UTC a hora local peninsular.

        Args:
            df: DataFrame con timestamp.

        Returns:
            DataFrame con timestamp en hora local.
        """
        df = df.copy()
        if self.time_col not in df.columns:
            logger.warning(f"Columna {self.time_col} no encontrada.")
            return df

        # Asegurar tz-aware UTC
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
        # Convertir a hora local
        df[self.time_col] = df[self.time_col].dt.tz_convert(self.timezone)
        # Quitar tz para simplificar operaciones posteriores
        df[self.time_col] = df[self.time_col].dt.tz_localize(None)
        logger.info(f"Timestamp convertido a hora local: {self.timezone}")
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade features cíclicas y temporales.

        Args:
            df: DataFrame con timestamp.

        Returns:
            DataFrame con features temporales añadidas.
        """
        df = df.copy()
        ts = df[self.time_col]

        # Hour of day (0-23)
        df["hour_local"] = ts.dt.hour

        # Sin/cos cíclicos para hora (periodo 24h)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour_local"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour_local"] / 24)

        # Day of year (1-366) - cíclico
        df["day_of_year"] = ts.dt.dayofyear
        df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Month (1-12) - cíclico
        df["month"] = ts.dt.month
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Is_daylight flag (preliminar; se refinará con posición solar)
        df["is_daylight"] = ((df["hour_local"] >= 6) & (df["hour_local"] <= 20)).astype(int)

        logger.info("Features temporales añadidas: hour_local, sin/cos_hour, day_of_year, month, is_weekend, is_daylight")
        return df

    # ------------------------------------------------------------------
    # B. Posición solar
    # ------------------------------------------------------------------
    def add_solar_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade variables de posición solar (elevación, azimuth, is_daylight preciso).

        Usa pvlib si está disponible, si no astral, si no heurística.

        Args:
            df: DataFrame con timestamp.

        Returns:
            DataFrame con 'solar_elevation', 'solar_azimuth', 'is_daylight' (preciso).
        """
        df = df.copy()
        ts = df[self.time_col]

        # Intentar con pvlib
        try:
            import pvlib
            from pvlib.solarposition import get_solarposition

            # pvlib necesita tz-aware
            ts_tz = pd.DatetimeIndex(ts).tz_localize(self.timezone)

            solpos = get_solarposition(
                ts_tz, self.latitude, self.longitude, altitude=self.altitude
            )
            df["solar_elevation"] = solpos["elevation"].values
            df["solar_azimuth"] = solpos["azimuth"].values
            df["is_daylight"] = (df["solar_elevation"] > 0).astype(int)
            logger.info("Posición solar calculada con pvlib")
            return df
        except ImportError:
            logger.info("pvlib no disponible. Intentando con astral...")
        except Exception as e:
            logger.warning(f"pvlib falló: {e}. Intentando con astral...")

        # Intentar con astral
        try:
            from astral import LocationInfo
            from astral.sun import sun, elevation, azimuth

            location = LocationInfo(
                latitude=self.latitude,
                longitude=self.longitude,
                timezone=self.timezone,
            )

            # Calcular elevación y azimuth por hora
            elevations = []
            azimuths = []
            for t in ts:
                try:
                    t_aware = t.to_pydatetime()
                    # astral 3.x
                    elev = elevation(location.observer, t_aware)
                    az = azimuth(location.observer, t_aware)
                    elevations.append(elev if elev is not None else -90)
                    azimuths.append(az if az is not None else 0)
                except Exception:
                    elevations.append(-90)
                    azimuths.append(0)

            df["solar_elevation"] = elevations
            df["solar_azimuth"] = azimuths
            df["is_daylight"] = (df["solar_elevation"] > 0).astype(int)
            logger.info("Posición solar calculada con astral")
            return df
        except ImportError:
            logger.warning("astral no disponible. Usando heurística.")
        except Exception as e:
            logger.warning(f"astral falló: {e}. Usando heurística.")

        # Heurística final
        df["solar_elevation"] = 0.0
        df["solar_azimuth"] = 0.0
        # Estimación básica de elevación solar: pico a las 12h (90° teórico)
        hour = ts.dt.hour + ts.dt.minute / 60
        # Modelo senoidal: elevación = 90 * sin(pi * (hour - 6) / 12)
        elevation_heuristic = 90 * np.sin(np.pi * (hour - 6) / 12)
        df["solar_elevation"] = np.maximum(elevation_heuristic, -90)
        df["is_daylight"] = (df["solar_elevation"] > 0).astype(int)
        logger.info("Posición solar calculada con heurística senoidal")
        return df

    # ------------------------------------------------------------------
    # C. Lags y rolling windows
    # ------------------------------------------------------------------
    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade variables de lag de la generación.

        Lags: 1h, 3h, 24h, 168h (7 días).

        Args:
            df: DataFrame ordenado por timestamp.

        Returns:
            DataFrame con lags añadidos.
        """
        df = df.copy()
        # Asegurar orden
        df = df.sort_values(self.time_col).reset_index(drop=True)

        lag_hours = [1, 3, 24, 168]
        for lag in lag_hours:
            col_name = f"{self.target_col}_lag_{lag}h"
            df[col_name] = df[self.target_col].shift(lag).fillna(0)
            logger.info(f"Lag añadido: {col_name}")

        return df

    def add_rolling_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade ventanas móviles (rolling windows).

        Args:
            df: DataFrame ordenado por timestamp.

        Returns:
            DataFrame con rolling windows añadidas.
        """
        df = df.copy()
        df = df.sort_values(self.time_col).reset_index(drop=True)

        # Media móvil 7 días (168 horas)
        df["rolling_mean_7d"] = (
            df[self.target_col].rolling(window=168, min_periods=1).mean().fillna(0)
        )

        # Media móvil 24h
        df["rolling_mean_24h"] = (
            df[self.target_col].rolling(window=24, min_periods=1).mean().fillna(0)
        )

        # Desviación estándar 24h (volatilidad)
        df["rolling_std_24h"] = (
            df[self.target_col].rolling(window=24, min_periods=1).std().fillna(0)
        )

        # Max 24h (pico del día)
        df["rolling_max_24h"] = (
            df[self.target_col].rolling(window=24, min_periods=1).max().fillna(0)
        )

        logger.info("Rolling windows añadidos: 7d, 24h, std_24h, max_24h")
        return df

    # ------------------------------------------------------------------
    # D. Variables de interacción
    # ------------------------------------------------------------------
    def add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade variables de interacción.

        Args:
            df: DataFrame.

        Returns:
            DataFrame con interacciones añadidas.
        """
        df = df.copy()

        # rad_solar * is_daylight: si es de día, la radiación tiene efecto
        if "radiacion_solar" in df.columns and "is_daylight" in df.columns:
            df["rad_solar_daylight"] = df["radiacion_solar"] * df["is_daylight"]

        # elevación solar * radiación (eficiencia solar combinada)
        if "solar_elevation" in df.columns and "radiacion_solar" in df.columns:
            df["elev_x_rad"] = df["solar_elevation"] * df["radiacion_solar"] / 1000

        # Temperatura efectiva (efecto negativo si muy alta)
        if "temperatura" in df.columns:
            df["temp_squared"] = df["temperatura"] ** 2
            df["temp_high_flag"] = (df["temperatura"] > 30).astype(int)

        # Humedad * Radiación (mayor humedad = menor generación)
        if "humedad_relativa" in df.columns and "radiacion_solar" in df.columns:
            df["humedad_x_rad"] = df["humedad_relativa"] * df["radiacion_solar"] / 100

        logger.info("Variables de interacción añadidas")
        return df

    # ------------------------------------------------------------------
    # E. Manejo final de nulos
    # ------------------------------------------------------------------
    def handle_final_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja los NaN finales.

        - Lags y rolling: rellenar con 0.
        - Meteorología: mediana.

        Args:
            df: DataFrame con posibles NaN.

        Returns:
            DataFrame sin NaN.
        """
        df = df.copy()
        for col in df.columns:
            n_nans = df[col].isna().sum()
            if n_nans > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if "lag" in col or "rolling" in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

        n_remaining = df.isna().sum().sum()
        logger.info(f"NaNs restantes tras manejo final: {n_remaining}")
        return df

    # ------------------------------------------------------------------
    # F. Selección y orden de columnas finales
    # ------------------------------------------------------------------
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Devuelve las columnas predictoras (excluyendo target, timestamp, id).

        Args:
            df: DataFrame con todas las features.

        Returns:
            Lista de nombres de columnas predictoras.
        """
        exclude = {self.target_col, self.time_col, self.id_col, "consumo_activo"}
        # Excluir columnas que son strings o no numéricas
        feature_cols = [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        return feature_cols

    # ------------------------------------------------------------------
    # G. Pipeline completo
    # ------------------------------------------------------------------
    def run(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Ejecuta el pipeline completo de feature engineering.

        Args:
            df: DataFrame de entrada (si None, carga dataset_preprocesado.csv).

        Returns:
            Tupla (dataset_final, feature_columns).
        """
        if df is None:
            input_path = self.processed_path / "dataset_preprocesado.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"No se encuentra {input_path}")
            df = pd.read_csv(input_path, parse_dates=[self.time_col])
            logger.info(f"Dataset preprocesado cargado: {df.shape}")

        logger.info(f"Iniciando feature engineering. Shape inicial: {df.shape}")

        # Pipeline
        df = self.convert_to_local_time(df)
        df = self.add_temporal_features(df)
        df = self.add_solar_position(df)
        df = self.add_lags(df)
        df = self.add_rolling_windows(df)
        df = self.add_interactions(df)
        df = self.handle_final_nans(df)

        # Asegurar que target existe
        if self.target_col not in df.columns:
            raise ValueError(f"Columna target '{self.target_col}' no encontrada tras feature engineering")

        # Obtener columnas de features
        feature_cols = self.get_feature_columns(df)
        logger.info(f"Features finales: {len(feature_cols)} columnas")
        logger.info(f"Features: {feature_cols}")

        # Guardar
        output_file = self.processed_path / "dataset_final.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset final guardado: {output_file} ({df.shape})")

        # Guardar lista de features
        features_file = self.models_path / "feature_columns.txt"
        features_file.parent.mkdir(parents=True, exist_ok=True)
        features_file.write_text("\n".join(feature_cols), encoding="utf-8")
        logger.info(f"Lista de features guardada: {features_file}")

        return df, feature_cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    config = load_config()
    fe = FeatureEngineer(config)
    df, features = fe.run()
    logger.info(f"Pipeline completado. Features: {len(features)}")
