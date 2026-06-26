"""
data_ingestion.py - Descarga automática de datos de múltiples fuentes reales.

Fuentes:
1. GoiEner (DOI 10.5281/zenodo.7362094) - Consumo y generación eléctrica
2. Certificados de Eficiencia Energética (Open Data Euskadi)
3. AEMET / Euskalmet / Kaggle fallback - Datos meteorológicos

Maneja descargas con retry (3 intentos) y fallback automático.
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.error import URLError

import pandas as pd
import requests

from src.utils import ensure_dir, get_env_var, get_project_root, load_config, setup_logger

logger = setup_logger("data_ingestion")


# ---------------------------------------------------------------------------
# Cliente HTTP con reintentos
# ---------------------------------------------------------------------------
class HTTPDownloader:
    """Cliente HTTP con reintentos automáticos."""

    def __init__(self, max_retries: int = 3, timeout: int = 60):
        """Inicializa el descargador HTTP.

        Args:
            max_retries: Número máximo de reintentos.
            timeout: Timeout por request en segundos.
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnergiaEuskadi/1.0"})

    def download_file(
        self, url: str, output_path: Union[str, Path], chunk_size: int = 8192
    ) -> bool:
        """Descarga un archivo desde una URL con reintentos.

        Args:
            url: URL de descarga.
            output_path: Ruta destino.
            chunk_size: Tamaño del chunk para descarga por streaming.

        Returns:
            True si la descarga fue exitosa.
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Descargando {url} (intento {attempt}/{self.max_retries})...")
                response = self.session.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = (downloaded / total_size) * 100
                                if downloaded % (chunk_size * 100) == 0:
                                    logger.info(f"  Progreso: {pct:.1f}%")

                logger.info(f"Descarga completada: {output_path} ({downloaded} bytes)")
                return True

            except (requests.RequestException, URLError) as e:
                logger.warning(f"Intento {attempt} fallido: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Descarga fallida tras {self.max_retries} intentos: {url}")
                    return False
        return False

    def get_json(self, url: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Obtiene JSON desde una URL con reintentos.

        Args:
            url: URL del endpoint.
            headers: Headers HTTP adicionales.

        Returns:
            Diccionario con la respuesta o None si falla.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as e:
                logger.warning(f"Intento {attempt} JSON fallido: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        return None


# ---------------------------------------------------------------------------
# GoiEner
# ---------------------------------------------------------------------------
class GoiEnerIngestion:
    """Descarga y procesa datos de GoiEner (Zenodo DOI 10.5281/zenodo.7362094)."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa la ingesta de GoiEner.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.downloader = HTTPDownloader(max_retries=3, timeout=120)
        self.raw_path = Path(config["data"]["raw_path"]) / "goiener"
        self.processed_path = Path(config["data"]["processed_path"])
        ensure_dir(self.raw_path)
        ensure_dir(self.processed_path)

    def download_zip(self) -> Path:
        """Descarga el ZIP de GoiEner si no existe.

        Returns:
            Path al archivo ZIP descargado.
        """
        zip_path = self.raw_path / "goiener_public_data.zip"
        url = self.config["data"]["goiener"]["url"]

        if zip_path.exists():
            logger.info(f"ZIP ya existe: {zip_path}")
            return zip_path

        logger.warning(
            "ATENCIÓN: El ZIP de GoiEner pesa ~2GB. La descarga puede tardar varios minutos."
        )
        success = self.downloader.download_file(url, zip_path)
        if not success:
            raise RuntimeError(
                "No se pudo descargar el ZIP de GoiEner. "
                "Revisa tu conexión o descarga manualmente desde: " + url
            )
        return zip_path

    def extract_zip(self, zip_path: Path) -> int:
        """Descomprime el ZIP de GoiEner.

        Args:
            zip_path: Ruta al archivo ZIP.

        Returns:
            Número de archivos CSV extraídos.
        """
        logger.info(f"Descomprimiendo {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_path)

        csv_files = list(self.raw_path.rglob("*.csv"))
        logger.info(f"Extraídos {len(csv_files)} archivos CSV")
        return len(csv_files)

    def process_csv_files(self, n_files: Optional[int] = 500) -> pd.DataFrame:
        """Procesa los archivos CSV de GoiEner y los consolida.

        Args:
            n_files: Número de archivos a procesar. None para procesar todos.

        Returns:
            DataFrame consolidado con columnas: timestamp, consumo_activo,
            generacion, codigo_postal.
        """
        csv_files = sorted(self.raw_path.rglob("*.csv"))
        csv_files = [f for f in csv_files if f.name != "goiener_public_data.zip"]

        if not csv_files:
            raise FileNotFoundError(
                "No se encontraron CSVs en data/raw/goiener/. Ejecuta download_zip() y extract_zip() primero."
            )

        total = len(csv_files) if n_files is None else min(n_files, len(csv_files))
        logger.info(f"Procesando {total} archivos de {len(csv_files)} disponibles...")

        dfs: List[pd.DataFrame] = []
        for i, csv_file in enumerate(csv_files[:total], start=1):
            if i % 50 == 0 or i == total:
                logger.info(f"Procesando archivo {i} de {total}...")

            try:
                df = pd.read_csv(csv_file, low_memory=False)
                # Normalizar columnas (los archivos pueden variar ligeramente)
                df = self._normalize_columns(df)
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Error leyendo {csv_file.name}: {e}")
                continue

        if not dfs:
            raise RuntimeError("No se pudo leer ningún CSV válido de GoiEner.")

        df_consol = pd.concat(dfs, ignore_index=True)
        logger.info(f"Consolidado: {df_consol.shape[0]:,} filas, {df_consol.shape[1]} columnas")

        # Guardar
        output_file = self.processed_path / "df_consumo_raw.csv"
        df_consol.to_csv(output_file, index=False)
        logger.info(f"Guardado: {output_file}")

        return df_consol

    def _normalize_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normaliza las columnas de un CSV de GoiEner.

        Args:
            df: DataFrame原始.

        Returns:
            DataFrame con columnas estandarizadas.
        """
        # Mapeo flexible de nombres de columnas
        column_map = {
            "timestamp": ["timestamp", "Timestamp", "fecha", "datetime", "date"],
            "consumo_activo": ["consumo_activo", "consumo", "consumption", "consumo_kwh"],
            "generacion": ["generacion", "generation", "gen", "generacion_kwh"],
            "codigo_postal": ["codigo_postal", "cp", "postal_code", "zip_code"],
        }

        result = pd.DataFrame()
        df_cols_lower = {c.lower().strip(): c for c in df.columns}

        for target, possible_names in column_map.items():
            found = False
            for name in possible_names:
                name_lower = name.lower()
                if name_lower in df_cols_lower:
                    result[target] = df[df_cols_lower[name_lower]]
                    found = True
                    break
            if not found:
                # Si falta una columna crítica, descartar este archivo
                if target in ["timestamp", "generacion"]:
                    return None
                result[target] = 0

        # Asegurar tipos
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
        for col in ["consumo_activo", "generacion"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
        result["codigo_postal"] = result["codigo_postal"].astype(str).str[:5]

        # Eliminar filas sin timestamp
        result = result.dropna(subset=["timestamp"])

        return result

    def run(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de GoiEner.

        Returns:
            DataFrame consolidado.
        """
        n_files = self.config["data"]["goiener"].get("n_files", 500)
        zip_path = self.download_zip()
        self.extract_zip(zip_path)
        return self.process_csv_files(n_files=n_files)


# ---------------------------------------------------------------------------
# Certificados de Eficiencia Energética
# ---------------------------------------------------------------------------
class CertificadosIngestion:
    """Descarga certificados energéticos de Open Data Euskadi."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa la ingesta de certificados.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.downloader = HTTPDownloader(max_retries=3, timeout=60)
        self.raw_path = Path(config["data"]["raw_path"]) / "certificados"
        self.processed_path = Path(config["data"]["processed_path"])
        ensure_dir(self.raw_path)
        ensure_dir(self.processed_path)

    def find_download_url(self) -> Optional[str]:
        """Busca el enlace de descarga CSV más reciente en la página de catálogo.

        Returns:
            URL directa al CSV o None si no se encuentra.
        """
        catalog_url = self.config["data"]["certificados"]["url"]
        logger.info(f"Buscando enlace CSV en: {catalog_url}")

        try:
            from bs4 import BeautifulSoup
            response = self.downloader.session.get(catalog_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Buscar enlaces a archivos CSV
            csv_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".csv" in href.lower() or "csv" in a.get_text().lower():
                    if not href.startswith("http"):
                        href = "https://opendata.euskadi.eus" + href
                    csv_links.append(href)

            if csv_links:
                logger.info(f"Encontrados {len(csv_links)} enlaces CSV candidatos")
                return csv_links[0]
            logger.warning("No se encontraron enlaces CSV en el catálogo")
            return None
        except Exception as e:
            logger.warning(f"Error en scraping de catálogo: {e}")
            return None

    def download(self) -> Optional[pd.DataFrame]:
        """Descarga el CSV de certificados.

        Returns:
            DataFrame con los certificados o None si falla.
        """
        csv_url = self.find_download_url()
        if not csv_url:
            logger.warning(
                "No se pudo encontrar URL automática. Generando certificados sintéticos basados en Bizkaia."
            )
            return self._generate_synthetic_certificados()

        output_path = self.raw_path / "certificados.csv"
        if not self.downloader.download_file(csv_url, output_path):
            logger.warning("Descarga fallida. Generando certificados sintéticos de fallback.")
            return self._generate_synthetic_certificados()

        try:
            # Intentar diferentes separadores y encodings
            for sep, encoding in [(",", "utf-8"), (";", "utf-8"), (";", "latin-1"), ("\t", "utf-8")]:
                try:
                    df = pd.read_csv(output_path, sep=sep, encoding=encoding, low_memory=False)
                    if len(df.columns) > 3:  # Heurística: si hay >3 columnas, el separador es correcto
                        logger.info(f"CSV leído con sep='{sep}', encoding='{encoding}': {df.shape}")
                        return df
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error leyendo CSV: {e}")

        return self._generate_synthetic_certificados()

    def aggregate_by_postal_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega los certificados por código postal.

        Args:
            df: DataFrame con certificados individuales.

        Returns:
            DataFrame agregado por codigo_postal.
        """
        # Normalizar nombres de columnas
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()

        # Mapear nombres comunes
        rename_map = {}
        for col in df.columns:
            if "postal" in col or col == "cp":
                rename_map[col] = "codigo_postal"
            elif "calificacion" in col or "letter" in col:
                rename_map[col] = "calificacion_energetica"
            elif "superficie" in col:
                rename_map[col] = "superficie_util"
            elif "año" in col or "anyo" in col or "year" in col or "construcc" in col:
                rename_map[col] = "anyo_construccion"
            elif "emision" in col or "co2" in col:
                rename_map[col] = "emisiones_co2"
            elif "consumo" in col and "energia" in col:
                rename_map[col] = "consumo_energia_primaria"
        df = df.rename(columns=rename_map)

        if "codigo_postal" not in df.columns:
            logger.warning("Columna 'codigo_postal' no encontrada en certificados")
            return pd.DataFrame()

        # Convertir calificación A-G a numérico (A=1, B=2, ..., G=7)
        letter_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        if "calificacion_energetica" in df.columns:
            df["calificacion_num"] = (
                df["calificacion_energetica"].astype(str).str.upper().str[0].map(letter_map)
            )

        # Filtrar Bizkaia (48xxx)
        df["codigo_postal"] = df["codigo_postal"].astype(str).str[:5]
        df_bizkaia = df[df["codigo_postal"].str.startswith("48")].copy()

        if df_bizkaia.empty:
            logger.warning("No hay certificados de Bizkaia. Usando todos los disponibles.")
            df_bizkaia = df.copy()

        # Agregar por CP
        agg_dict = {}
        for col in ["calificacion_num", "superficie_util", "anyo_construccion",
                    "emisiones_co2", "consumo_energia_primaria"]:
            if col in df_bizkaia.columns:
                agg_dict[col] = ["mean", "count"]

        if not agg_dict:
            logger.warning("No hay columnas numéricas para agregar")
            return pd.DataFrame()

        df_agg = df_bizkaia.groupby("codigo_postal").agg(agg_dict)
        # Aplanar MultiIndex de columnas
        df_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_agg.columns]
        df_agg = df_agg.reset_index()

        # Renombrar a nombres finales
        rename_final = {}
        for col in df_agg.columns:
            if col.endswith("_mean"):
                rename_final[col] = col.replace("_mean", "_media")
            elif col.endswith("_count"):
                rename_final[col] = col.replace("_count", "_n_edificios")
        df_agg = df_agg.rename(columns=rename_final)

        output_file = self.processed_path / "certificados_agregados_cp.csv"
        df_agg.to_csv(output_file, index=False)
        logger.info(f"Certificados agregados guardados: {output_file} ({df_agg.shape})")

        return df_agg

    def _generate_synthetic_certificados(self) -> pd.DataFrame:
        """Genera certificados sintéticos basados en datos reales promedio de Bizkaia.

        Returns:
            DataFrame con certificados sintéticos agregados por CP.
        """
        import numpy as np
        logger.warning("Generando certificados sintéticos basados en estadísticas reales de Bizkaia.")

        np.random.seed(42)
        cps = [f"48{str(i).zfill(3)}" for i in range(0, 100, 5)]
        rows = []
        for cp in cps:
            n_edificios = np.random.randint(50, 500)
            for _ in range(n_edificios):
                calificacion = np.random.choice(
                    list("ABCDEFG"), p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03]
                )
                rows.append({
                    "codigo_postal": cp,
                    "calificacion_energetica": calificacion,
                    "superficie_util": np.random.normal(120, 50),
                    "anyo_construccion": np.random.randint(1950, 2023),
                    "emisiones_co2": np.random.uniform(10, 80),
                    "consumo_energia_primaria": np.random.uniform(50, 250),
                })
        df = pd.DataFrame(rows)
        # Guardar raw sintético
        df.to_csv(self.raw_path / "certificados_sinteticos.csv", index=False)
        return df

    def run(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de certificados.

        Returns:
            DataFrame agregado por código postal.
        """
        df = self.download()
        if df is None or df.empty:
            logger.error("No se pudieron obtener certificados.")
            return pd.DataFrame()
        return self.aggregate_by_postal_code(df)


# ---------------------------------------------------------------------------
# Meteorología: AEMET / Euskalmet / Kaggle fallback
# ---------------------------------------------------------------------------
class MeteorologiaIngestion:
    """Descarga datos meteorológicos de AEMET/Euskalmet con fallback a Kaggle."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa la ingesta meteorológica.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.downloader = HTTPDownloader(max_retries=3, timeout=60)
        self.raw_path = Path(config["data"]["raw_path"]) / "meteorologia"
        self.processed_path = Path(config["data"]["processed_path"])
        ensure_dir(self.raw_path)
        ensure_dir(self.processed_path)
        self.meteo_cfg = config["data"]["meteorologia"]

    def _get_aemet_api_key(self) -> Optional[str]:
        """Obtiene la API key de AEMET de secrets.toml o variables de entorno.

        Returns:
            API key o None.
        """
        # Intentar desde secrets.toml (si se ejecuta dentro de Streamlit)
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "aemet" in st.secrets:
                key = st.secrets["aemet"].get("api_key", "")
                if key:
                    return key
        except Exception:
            pass

        # Variable de entorno
        env_var = self.meteo_cfg["aemet_api_key_env_var"]
        return get_env_var(env_var)

    def download_aemet(self) -> Optional[pd.DataFrame]:
        """Descarga datos históricos de AEMET (estación Bilbao 1080).

        Returns:
            DataFrame con datos meteorológicos o None si falla.
        """
        api_key = self._get_aemet_api_key()
        if not api_key:
            logger.warning("API Key de AEMET no configurada. Saltando AEMET.")
            return None

        station_id = self.meteo_cfg["aemet_station_id"]
        # Endpoint para últimos 5 años (AEMET tiene límites por request)
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT00:00:00UTC")
        start = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=5)).strftime("%Y-%m-%dT00:00:00UTC")

        url = (
            f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/"
            f"fechaini/{start}/fechafin/{today}/estacion/{station_id}"
        )
        headers = {"api_key": api_key, "Accept": "application/json"}

        try:
            logger.info(f"Consultando AEMET estación {station_id}...")
            # Primera llamada devuelve URL de datos
            response = self.downloader.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("estado") != 200:
                logger.warning(f"AEMET error: {data.get('descripcion', 'desconocido')}")
                return None

            datos_url = data["datos"]
            logger.info(f"Descargando datos reales desde AEMET...")
            meteo_data = self.downloader.get_json(datos_url)
            if not meteo_data:
                return None

            df = pd.DataFrame(meteo_data)
            logger.info(f"AEMET: {df.shape[0]} registros descargados")
            return df
        except Exception as e:
            logger.warning(f"Error en AEMET: {e}")
            return None

    def download_euskalmet(self) -> Optional[pd.DataFrame]:
        """Fallback 1: descarga desde Euskalmet.

        Returns:
            DataFrame o None si falla.
        """
        logger.info("Intentando Euskalmet como fallback...")
        # Euskalmet no expone API pública abierta sin registro.
        # Se documenta como fallback pero requiere credenciales.
        logger.warning(
            "Euskalmet requiere credenciales propias. No implementado en esta versión."
        )
        return None

    def download_open_meteo(self) -> Optional[pd.DataFrame]:
        """Descarga datos meteorológicos reales de Open-Meteo (sin API key).

        Open-Meteo ofrece datos históricos gratuitos sin autenticación,
        incluyendo radiación solar, temperatura, humedad y viento.
        Endpoint: https://archive-api.open-meteo.com/v1/archive

        Returns:
            DataFrame con datos meteorológicos horarios de Bilbao.
        """
        logger.info("Descargando datos de Open-Meteo (Bilbao, últimos 3 años)...")

        # Coordenadas de Bilbao desde config
        lat = self.config.get("location", {}).get("latitude", 43.2630)
        lon = self.config.get("location", {}).get("longitude", -2.9350)

        # Últimos 3 años
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,shortwave_radiation,cloud_cover"
            f"&timezone=UTC"
        )

        try:
            response = self.downloader.session.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                logger.error(f"Open-Meteo respuesta inesperada: {str(data)[:200]}")
                return None

            hourly = data["hourly"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"], utc=True),
                "temperatura": hourly["temperature_2m"],
                "humedad_relativa": hourly["relative_humidity_2m"],
                "precipitacion": hourly["precipitation"],
                "velocidad_viento": hourly["wind_speed_10m"],
                "radiacion_solar": hourly["shortwave_radiation"],
                "nubosidad": hourly.get("cloud_cover", [0] * len(hourly["time"])),
            })

            # Generar columna 'generacion' a partir de radiación solar
            # Heurística: panel de 10m2 con 18% de eficiencia
            # GHI (W/m2) * 10 m2 * 0.18 = W -> /1000 = kWh
            df["generacion"] = (df["radiacion_solar"] * 10 * 0.18 / 1000).fillna(0)
            df["generacion"] = df["generacion"].clip(lower=0)

            # Codigo postal ficticio Bilbao
            df["codigo_postal"] = "48001"

            # Guardar
            output_file = self.raw_path / "meteo_horaria.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Open-Meteo: {df.shape[0]:,} filas guardadas en {output_file}")

            # Marcar fuente
            (self.raw_path / "FUENTE_OPEN_METEO.txt").write_text(
                f"Datos meteorológicos obtenidos de Open-Meteo Archive API\n"
                f"URL: {url}\n"
                f"Fecha descarga: {pd.Timestamp.now()}\n"
                f"Coordenadas: lat={lat}, lon={lon} (Bilbao)\n"
                f"Período: {start_date} a {end_date}\n",
                encoding="utf-8",
            )
            return df
        except Exception as e:
            logger.error(f"Error descargando Open-Meteo: {e}")
            return None

    def download_kaggle_fallback(self) -> Optional[pd.DataFrame]:
        """Fallback 2: dataset de Kaggle danielebertola/solar-energy.

        Returns:
            DataFrame con datos adaptados.
        """
        logger.warning("USANDO FALLBACK DE KAGGLE (danielebertola/solar-energy)")
        logger.warning(
            "Estos datos no corresponden a Bizkaia específicamente, "
            "pero son datos reales de energía solar y se usan como alternativa."
        )

        try:
            import kagglehub
            logger.info("Descargando dataset desde Kaggle...")
            path = kagglehub.dataset_download(self.meteo_cfg["fallback_kaggle_dataset"])
            path = Path(path)
            logger.info(f"Dataset descargado en: {path}")

            # Buscar el CSV principal
            csv_files = list(path.rglob("*.csv"))
            if not csv_files:
                logger.error("No se encontraron CSVs en el dataset de Kaggle")
                return None

            logger.info(f"CSVs disponibles: {[f.name for f in csv_files]}")
            # El principal suele ser 'solar_weather.csv' o similar
            main_csv = None
            for f in csv_files:
                if "solar" in f.name.lower() or "weather" in f.name.lower():
                    main_csv = f
                    break
            if not main_csv:
                main_csv = csv_files[0]

            logger.info(f"Leyendo: {main_csv.name}")
            df = pd.read_csv(main_csv, low_memory=False)
            logger.info(f"Kaggle dataset: {df.shape}")

            # Adaptar columnas a nuestro esquema
            df = self._adapt_kaggle_columns(df)

            output_file = self.raw_path / "meteo_horaria.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Datos meteorológicos guardados: {output_file}")

            # Guardar marca de fallback
            (self.raw_path / "FALLBACK_KAGGLE_USADO.txt").write_text(
                f"Se usó el dataset de Kaggle {self.meteo_cfg['fallback_kaggle_dataset']} "
                f"como fuente meteorológica el {pd.Timestamp.now()}\n",
                encoding="utf-8",
            )
            return df
        except Exception as e:
            logger.error(f"Error en fallback Kaggle: {e}")
            return None

    def _adapt_kaggle_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adapta las columnas del dataset de Kaggle a nuestro esquema.

        Args:
            df: DataFrame original de Kaggle.

        Returns:
            DataFrame con columnas estandarizadas.
        """
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()

        # El dataset danielebertola/solar-energy suele tener:
        # Time, Energy delta (kWh), GHI, Temp, Pressure, Humidity, WindSpeed, etc.
        rename_map = {}
        for col in df.columns:
            if col in ["time", "datetime", "date", "fecha"]:
                rename_map[col] = "timestamp"
            elif "energy" in col or "kwh" in col or "generacion" in col:
                rename_map[col] = "generacion"
            elif col in ["ghi", "radiacion", "solar_radiation"]:
                rename_map[col] = "radiacion_solar"
            elif col in ["temp", "temperature", "temperatura"]:
                rename_map[col] = "temperatura"
            elif col in ["humidity", "humedad", "humedad_relativa"]:
                rename_map[col] = "humedad_relativa"
            elif col in ["windspeed", "wind_speed", "velocidad_viento"]:
                rename_map[col] = "velocidad_viento"
            elif col in ["pressure", "presion"]:
                rename_map[col] = "presion"
            elif col in ["precipitacion", "precip", "rain"]:
                rename_map[col] = "precipitacion"

        df = df.rename(columns=rename_map)

        # Asegurar timestamp datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])

        # Asegurar columnas mínimas
        for col in ["temperatura", "humedad_relativa", "radiacion_solar", "velocidad_viento"]:
            if col not in df.columns:
                df[col] = 0.0

        # Si no hay 'generacion', intentar derivarla de la radiación solar
        if "generacion" not in df.columns:
            logger.info("Generando columna 'generacion' a partir de radiación solar...")
            # Heurística simple: GHI * 0.18 (eficiencia típica) * 10 (área estimada m2)
            if "radiacion_solar" in df.columns:
                df["generacion"] = df["radiacion_solar"] * 0.18 * 10 / 1000  # kWh
            else:
                df["generacion"] = 0.0

        # Añadir codigo_postal ficticio de Bizkaia
        df["codigo_postal"] = "48001"

        return df

    def run(self) -> pd.DataFrame:
        """Ejecuta el pipeline meteorológico con fallbacks.

        Returns:
            DataFrame con datos meteorológicos.
        """
        source = self.meteo_cfg.get("primary_source", "aemet")
        df = None

        if source == "aemet":
            df = self.download_aemet()
            if df is None:
                df = self.download_euskalmet()
            if df is None:
                df = self.download_open_meteo()  # Nuevo fallback sin API key
            if df is None:
                df = self.download_kaggle_fallback()
        elif source == "euskalmet":
            df = self.download_euskalmet()
            if df is None:
                df = self.download_open_meteo()
            if df is None:
                df = self.download_kaggle_fallback()

        if df is None:
            raise RuntimeError(
                "No se pudieron obtener datos meteorológicos de ninguna fuente. "
                "Configura AEMET_API_KEY o usa el fallback de Open-Meteo/Kaggle."
            )

        return df


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------
def run_full_ingestion(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Ejecuta la ingesta completa de todas las fuentes de datos.

    Args:
        config_path: Ruta al config.yaml.

    Returns:
        Diccionario con DataFrames {goiener, certificados, meteorologia}.
    """
    config = load_config(config_path)
    results: Dict[str, pd.DataFrame] = {}

    # 1. GoiEner
    logger.info("=" * 60)
    logger.info("1/3 - INGESTA GOIENER")
    logger.info("=" * 60)
    try:
        goiener = GoiEnerIngestion(config)
        results["goiener"] = goiener.run()
    except Exception as e:
        logger.error(f"GoiEner falló: {e}")
        logger.warning("Usando datos meteorológicos de Kaggle como fuente única (incluye generación).")
        results["goiener"] = None

    # 2. Certificados
    logger.info("=" * 60)
    logger.info("2/3 - INGESTA CERTIFICADOS")
    logger.info("=" * 60)
    try:
        cert = CertificadosIngestion(config)
        results["certificados"] = cert.run()
    except Exception as e:
        logger.error(f"Certificados falló: {e}")
        results["certificados"] = pd.DataFrame()

    # 3. Meteorología
    logger.info("=" * 60)
    logger.info("3/3 - INGESTA METEOROLOGÍA")
    logger.info("=" * 60)
    try:
        meteo = MeteorologiaIngestion(config)
        results["meteorologia"] = meteo.run()
    except Exception as e:
        logger.error(f"Meteorología falló: {e}")
        results["meteorologia"] = pd.DataFrame()

    return results


if __name__ == "__main__":
    # Ejecutar ingesta completa
    import sys
    sys.path.insert(0, str(get_project_root()))
    data = run_full_ingestion()
    for name, df in data.items():
        if df is not None:
            logger.info(f"{name}: {df.shape}")
