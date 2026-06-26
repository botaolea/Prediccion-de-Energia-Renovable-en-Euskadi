"""
data_preprocessing.py - Limpieza, imputación, codificación y unión de datos.

Operaciones:
- Filtrado por Bizkaia (CP 48xxx)
- Resampleo temporal horario
- Relleno de huecos
- Unión con certificados por código postal
- Imputación de nulos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.utils import (
    ensure_dir,
    is_bizkaia_postal_code,
    load_config,
    setup_logger,
    validate_dataframe,
)

logger = setup_logger("data_preprocessing")


class DataPreprocessor:
    """Preprocesa los datos crudos de GoiEner, certificados y meteorología."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el preprocesador.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.processed_path = Path(config["data"]["processed_path"])
        ensure_dir(self.processed_path)
        self.target_col = config["model"]["target_column"]
        self.time_col = config["model"]["time_column"]
        self.id_col = config["model"]["id_column"]

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Carga los DataFrames procesados en data/processed/ y raw/meteorologia/.

        Returns:
            Diccionario con DataFrames {consumo, certificados, meteorologia}.
        """
        data = {}
        consumo_path = self.processed_path / "df_consumo_raw.csv"
        meteo_path = Path(self.config["data"]["raw_path"]) / "meteorologia" / "meteo_horaria.csv"
        cert_path = self.processed_path / "certificados_agregados_cp.csv"

        if consumo_path.exists():
            data["consumo"] = pd.read_csv(consumo_path, parse_dates=[self.time_col])
            logger.info(f"Consumo cargado: {data['consumo'].shape}")
        else:
            logger.warning(f"No se encontró {consumo_path}")

        if meteo_path.exists():
            data["meteorologia"] = pd.read_csv(meteo_path, parse_dates=[self.time_col])
            logger.info(f"Meteorología cargada: {data['meteorologia'].shape}")
        else:
            logger.warning(f"No se encontró {meteo_path}")

        if cert_path.exists():
            data["certificados"] = pd.read_csv(cert_path)
            logger.info(f"Certificados cargados: {data['certificados'].shape}")
        else:
            logger.warning(f"No se encontró {cert_path}")

        return data

    def filter_bizkaia(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra solo registros de Bizkaia (códigos postales 48xxx).

        Args:
            df: DataFrame con columna 'codigo_postal'.

        Returns:
            DataFrame filtrado a Bizkaia.
        """
        if self.id_col not in df.columns:
            logger.warning(f"Columna {self.id_col} no presente. No se puede filtrar por Bizkaia.")
            return df

        df = df.copy()
        df[self.id_col] = df[self.id_col].astype(str).str[:5]
        mask = df[self.id_col].apply(is_bizkaia_postal_code)
        filtered = df[mask].copy()
        logger.info(f"Filtrado Bizkaia: {df.shape[0]:,} -> {filtered.shape[0]:,} filas")
        return filtered

    def resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamplea el DataFrame a frecuencia horaria.

        Args:
            df: DataFrame con timestamp como índice o columna.

        Returns:
            DataFrame con frecuencia horaria.
        """
        df = df.copy()

        # Asegurar timestamp como índice
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
            df = df.set_index(self.time_col)

        # Detectar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Estrategia de agregación:
        # - Sumar generación y consumo
        # - Media para meteorología
        agg_dict = {}
        for col in numeric_cols:
            if col in ["generacion", "consumo_activo", "consumo"]:
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "mean"

        # Para codigo_postal, tomar el primero
        if self.id_col in df.columns:
            agg_dict[self.id_col] = "first"

        logger.info(f"Resampleando a 1H con agg: {agg_dict}")
        df_resampled = df.resample("1h").agg(agg_dict)

        # Rellenar huecos
        for col in df_resampled.columns:
            if col in ["generacion", "consumo_activo", "consumo"]:
                df_resampled[col] = df_resampled[col].fillna(0)
            elif col == self.id_col:
                df_resampled[col] = df_resampled[col].ffill().bfill()
            else:
                # Meteorología: ffill + interpolate
                df_resampled[col] = df_resampled[col].ffill().interpolate(method="linear").bfill()

        logger.info(f"Resampleado: {df_resampled.shape[0]:,} filas horarias")
        return df_resampled.reset_index()

    def merge_certificados(
        self, df: pd.DataFrame, df_cert: pd.DataFrame
    ) -> pd.DataFrame:
        """Une el DataFrame principal con certificados por código postal.

        Args:
            df: DataFrame principal (con codigo_postal).
            df_cert: DataFrame de certificados agregados por CP.

        Returns:
            DataFrame unido con features de certificados.
        """
        if df_cert.empty or self.id_col not in df.columns:
            logger.warning("No se puede hacer merge con certificados.")
            return df

        df = df.copy()
        df_cert = df_cert.copy()
        df[self.id_col] = df[self.id_col].astype(str).str[:5]
        df_cert[self.id_col] = df_cert[self.id_col].astype(str).str[:5]

        # Calcular medias globales de Bizkaia para imputar CPs sin certificado
        global_means = df_cert.select_dtypes(include=[np.number]).mean()
        logger.info(f"Certificados: {len(df_cert)} CPs únicos")

        df_merged = df.merge(df_cert, on=self.id_col, how="left", suffixes=("", "_cert"))

        # Imputar con media global de Bizkaia
        for col, mean_val in global_means.items():
            if col in df_merged.columns and df_merged[col].isna().any():
                df_merged[col] = df_merged[col].fillna(mean_val)
                logger.info(f"Imputando {col} con media global: {mean_val:.2f}")

        logger.info(f"Merge con certificados: {df_merged.shape}")
        return df_merged

    def merge_meteorologia(
        self, df: pd.DataFrame, df_meteo: pd.DataFrame
    ) -> pd.DataFrame:
        """Une el DataFrame principal con datos meteorológicos por timestamp.

        Args:
            df: DataFrame principal.
            df_meteo: DataFrame meteorológico.

        Returns:
            DataFrame unido con features meteorológicas.
        """
        if df_meteo.empty:
            logger.warning("No hay datos meteorológicos para mergear.")
            return df

        df = df.copy()
        df_meteo = df_meteo.copy()

        # Asegurar timestamp UTC en ambos
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        df_meteo[self.time_col] = pd.to_datetime(df_meteo[self.time_col], utc=True)

        # Eliminar duplicados en meteo
        df_meteo = df_meteo.drop_duplicates(subset=[self.time_col])

        # Columnas meteorológicas a incluir
        meteo_cols = ["temperatura", "humedad_relativa", "radiacion_solar",
                      "velocidad_viento", "precipitacion", "presion"]
        meteo_cols = [c for c in meteo_cols if c in df_meteo.columns]

        # Si meteorología ya incluye 'generacion' (caso Kaggle fallback), priorizarla
        if "generacion" in df_meteo.columns and "generacion" not in df.columns:
            logger.info("Usando 'generacion' desde datos meteorológicos (Kaggle fallback).")
            df["generacion"] = np.nan

        df_merged = df.merge(
            df_meteo[[self.time_col] + meteo_cols + (["generacion"] if "generacion" in df_meteo.columns else [])],
            on=self.time_col,
            how="left",
            suffixes=("", "_meteo"),
        )

        # Si generacion_meteo existe (por suffixes), usarla cuando la principal sea nula
        if "generacion_meteo" in df_merged.columns:
            df_merged["generacion"] = df_merged["generacion"].fillna(df_merged["generacion_meteo"])
            df_merged = df_merged.drop(columns=["generacion_meteo"])

        logger.info(f"Merge con meteorología: {df_merged.shape}")
        return df_merged

    def clean_outliers(self, df: pd.DataFrame, col: str = None,
                       lower: float = 0, upper: float = None) -> pd.DataFrame:
        """Elimina outliers extremos de una columna.

        Args:
            df: DataFrame.
            col: Columna a limpiar (por defecto target_col).
            lower: Límite inferior.
            upper: Límite superior (None para percentil 99).

        Returns:
            DataFrame sin outliers extremos.
        """
        col = col or self.target_col
        if col not in df.columns:
            return df

        df = df.copy()
        if upper is None:
            upper = df[col].quantile(0.99)

        mask = (df[col] >= lower) & (df[col] <= upper)
        n_removed = (~mask).sum()
        logger.info(f"Outliers {col}: {n_removed} filas fuera de rango [{lower}, {upper:.2f}]")
        df.loc[~mask, col] = np.nan  # Convertir a NaN para imputar después
        return df

    def impute_final_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa los NaN finales del dataset.

        Args:
            df: DataFrame con posibles NaN.

        Returns:
            DataFrame sin NaN.
        """
        df = df.copy()

        # Identificar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            n_nans = df[col].isna().sum()
            if n_nans > 0:
                if "lag" in col or "rolling" in col:
                    # Lags y rolling: rellenar con 0
                    df[col] = df[col].fillna(0)
                elif col in ["generacion", "consumo_activo"]:
                    df[col] = df[col].fillna(0)
                else:
                    # Meteorología y certificados: mediana
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Imputando {col} con mediana: {median_val:.2f} ({n_nans} NaN)")

        logger.info(f"NaNs finales: {df.isna().sum().sum()} restantes")
        return df

    def run(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de preprocesamiento.

        Returns:
            DataFrame preprocesado y unido.
        """
        # 1. Cargar datos
        data = self.load_raw_data()

        # Determinar fuente principal
        if "consumo" in data and not data["consumo"].empty:
            df = data["consumo"]
            df_source = "goiener"
        elif "meteorologia" in data and not data["meteorologia"].empty:
            # Si solo tenemos meteorología (Kaggle), usarla
            df = data["meteorologia"].copy()
            df_source = "kaggle_meteo"
            logger.info("Usando meteorología como fuente única (Kaggle fallback).")
        else:
            raise RuntimeError("No hay datos para preprocesar.")

        logger.info(f"Fuente principal: {df_source}")

        # 2. Filtrar Bizkaia
        if self.id_col in df.columns:
            df = self.filter_bizkaia(df)

        # 3. Resampleo horario
        df = self.resample_hourly(df)

        # 4. Merge con certificados
        if "certificados" in data and not data["certificados"].empty:
            df = self.merge_certificados(df, data["certificados"])

        # 5. Merge con meteorología
        if "meteorologia" in data and not data["meteorologia"].empty and df_source != "kaggle_meteo":
            df = self.merge_meteorologia(df, data["meteorologia"])

        # 6. Limpiar outliers
        df = self.clean_outliers(df, col=self.target_col, lower=0)

        # 7. Imputar NaN finales
        df = self.impute_final_nans(df)

        # Guardar
        output_file = self.processed_path / "dataset_preprocesado.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset preprocesado guardado: {output_file} ({df.shape})")

        return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    config = load_config()
    preprocessor = DataPreprocessor(config)
    df = preprocessor.run()
    logger.info(f"Pipeline completado. Shape: {df.shape}")
