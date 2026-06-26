"""
model_predictor.py - Carga del modelo y predicción para la app Streamlit.

Funcionalidades:
- Cargar modelo y scaler (cached)
- Predecir las próximas 24-48 horas
- Calcular intervalos de confianza basados en RMSE
- Mostrar top 5 features influyentes (SHAP global precalculado)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("model_predictor")


class ModelPredictor:
    """Carga el modelo entrenado y realiza predicciones."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el predictor.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.models_path = Path(config["data"]["models_path"])
        self.target_col = config["model"]["target_column"]
        self.time_col = config["model"]["time_column"]
        self.id_col = config["model"]["id_column"]

        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_cols: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.shap_importance: Optional[pd.DataFrame] = None
        self.training_rmse: Optional[float] = None

    def load(self) -> "ModelPredictor":
        """Carga modelo, scaler, metadata y SHAP importance.

        Returns:
            self.

        Raises:
            FileNotFoundError: Si no encuentra el modelo.
        """
        # Buscar modelo (priorizar xgboost)
        model_paths = [
            self.models_path / "xgboost_model.pkl",
            self.models_path / "random_forest_model.pkl",
        ]
        model_path = next((p for p in model_paths if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No se encontró modelo en {self.models_path}. Ejecuta model_trainer primero."
            )

        logger.info(f"Cargando modelo: {model_path}")
        self.model = joblib.load(model_path)

        # Scaler
        scaler_path = self.models_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler cargado.")
        else:
            logger.warning("Scaler no encontrado. Las predicciones pueden no ser óptimas.")

        # Metadata
        meta_path = self.models_path / "model_metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_cols = self.metadata.get("feature_cols", [])

        # Si no hay features en metadata, intentar leer archivo
        if not self.feature_cols:
            features_file = self.models_path / "feature_columns.txt"
            if features_file.exists():
                self.feature_cols = features_file.read_text(encoding="utf-8").strip().split("\n")

        logger.info(f"Features cargadas: {len(self.feature_cols)}")

        # SHAP importance
        shap_path = self.models_path / "shap_importance.csv"
        if shap_path.exists():
            self.shap_importance = pd.read_csv(shap_path)
            logger.info("SHAP importance cargado.")

        # RMSE de entrenamiento (para intervalos de confianza)
        results_path = self.models_path / "training_results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.training_rmse = results.get("model_metrics", {}).get("RMSE")
            logger.info(f"RMSE entrenamiento: {self.training_rmse}")

        return self

    def _build_input_features(
        self,
        start_time: datetime,
        horizon_hours: int,
        meteorologia: Dict[str, List[float]],
        codigo_postal: str = "48001",
        last_generation: float = 0.0,
        last_7d_mean: float = 0.0,
    ) -> pd.DataFrame:
        """Construye el DataFrame de features para predicción.

        Args:
            start_time: Hora de inicio de la predicción.
            horizon_hours: Número de horas a predecir.
            meteorologia: Dict con listas de variables meteorológicas (length=horizon_hours).
            codigo_postal: Código postal del edificio.
            last_generation: Último valor conocido de generación (para lag 1h).
            last_7d_mean: Media móvil de los últimos 7 días (para rolling).

        Returns:
            DataFrame con features preparadas.
        """
        fe = FeatureEngineer(self.config)

        # Crear timestamps horarios
        timestamps = pd.date_range(
            start=start_time, periods=horizon_hours, freq="1h", tz=None
        )

        df = pd.DataFrame({self.time_col: timestamps})

        # Variables meteorológicas
        meteo_vars = ["temperatura", "humedad_relativa", "radiacion_solar",
                      "velocidad_viento", "precipitacion"]
        for var in meteo_vars:
            if var in meteorologia and len(meteorologia[var]) == horizon_hours:
                df[var] = meteorologia[var]
            else:
                df[var] = 0.0

        # codigo_postal
        df[self.id_col] = codigo_postal

        # Generación (necesaria para lags; se sobreescribirá)
        df[self.target_col] = 0.0

        # Features temporales y solares
        df = fe.add_temporal_features(df)
        df = fe.add_solar_position(df)

        # Lags: usar last_generation para lag_1h, y 0 para los demás (sin histórico)
        df[f"{self.target_col}_lag_1h"] = last_generation
        df[f"{self.target_col}_lag_3h"] = last_generation
        df[f"{self.target_col}_lag_24h"] = last_7d_mean
        df[f"{self.target_col}_lag_168h"] = last_7d_mean

        # Rolling windows
        df["rolling_mean_7d"] = last_7d_mean
        df["rolling_mean_24h"] = last_generation
        df["rolling_std_24h"] = 0.0
        df["rolling_max_24h"] = last_generation

        # Interacciones
        df = fe.add_interactions(df)

        # Manejo de NaN
        df = fe.handle_final_nans(df)

        return df

    def predict(
        self,
        start_time: Optional[datetime] = None,
        horizon_hours: int = 24,
        meteorologia: Optional[Dict[str, List[float]]] = None,
        codigo_postal: str = "48001",
        last_generation: float = 0.0,
        last_7d_mean: float = 0.0,
    ) -> Dict[str, Any]:
        """Realiza la predicción para las próximas horas.

        Args:
            start_time: Hora de inicio (default: ahora).
            horizon_hours: Horas a predecir (24-48).
            meteorologia: Variables meteorológicas para cada hora.
            codigo_postal: Código postal del edificio.
            last_generation: Último valor de generación conocido.
            last_7d_mean: Media móvil de los últimos 7 días.

        Returns:
            Diccionario con predicciones, intervalos y features influyentes.
        """
        if self.model is None:
            self.load()

        if start_time is None:
            start_time = datetime.now()

        # Si no hay meteorología, usar valores por defecto
        if meteorologia is None:
            meteorologia = self._default_meteorologia(horizon_hours)

        # Construir features
        df_input = self._build_input_features(
            start_time=start_time,
            horizon_hours=horizon_hours,
            meteorologia=meteorologia,
            codigo_postal=codigo_postal,
            last_generation=last_generation,
            last_7d_mean=last_7d_mean,
        )

        # Seleccionar y ordenar features
        for col in self.feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0

        X = df_input[self.feature_cols]

        # Escalar
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_scaled = X

        # Predecir
        preds = self.model.predict(X_scaled)
        preds = np.maximum(preds, 0)  # No negativos

        # Intervalo de confianza basado en RMSE
        if self.training_rmse:
            lower = preds - 1.96 * self.training_rmse
            upper = preds + 1.96 * self.training_rmse
        else:
            lower = preds * 0.8
            upper = preds * 1.2

        lower = np.maximum(lower, 0)

        # Top 5 features influyentes
        top_5 = []
        if self.shap_importance is not None and not self.shap_importance.empty:
            top_5 = self.shap_importance.head(5).to_dict(orient="records")

        # Preparar resultado
        timestamps = pd.date_range(
            start=start_time, periods=horizon_hours, freq="1h"
        )
        result = {
            "timestamps": [t.isoformat() for t in timestamps],
            "predictions": preds.tolist(),
            "lower_bound": lower.tolist(),
            "upper_bound": upper.tolist(),
            "horizon_hours": horizon_hours,
            "total_predicted_kwh": float(preds.sum()),
            "top_5_features": top_5,
            "codigo_postal": codigo_postal,
            "start_time": start_time.isoformat(),
        }

        logger.info(
            f"Predicción {horizon_hours}h: total {preds.sum():.2f} kWh "
            f"(media {preds.mean():.2f} kWh/h)"
        )
        return result

    def _default_meteorologia(self, horizon_hours: int) -> Dict[str, List[float]]:
        """Genera meteorología por defecto basada en estacionalidad.

        Args:
            horizon_hours: Horas a generar.

        Returns:
            Diccionario con variables meteorológicas.
        """
        now = datetime.now()
        month = now.month
        # Temperatura por mes (°C) - Bizkaia
        temp_by_month = {1: 8, 2: 9, 3: 11, 4: 12, 5: 15, 6: 18,
                         7: 21, 8: 21, 9: 19, 10: 15, 11: 11, 12: 9}
        base_temp = temp_by_month.get(month, 15)

        timestamps = pd.date_range(start=now, periods=horizon_hours, freq="1h")
        hours = timestamps.hour

        # Radiación solar diurna (W/m2)
        rad = np.array([
            max(0, 800 * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0
            for h in hours
        ])

        meteorologia = {
            "temperatura": (base_temp + 5 * np.sin(np.pi * (hours - 6) / 12)).tolist(),
            "humedad_relativa": (60 + 20 * np.cos(np.pi * hours / 12)).tolist(),
            "radiacion_solar": rad.tolist(),
            "velocidad_viento": (3 + 2 * np.random.rand(horizon_hours)).tolist(),
            "precipitacion": (np.random.rand(horizon_hours) * 0.5).tolist(),
        }
        return meteorologia

    def predict_from_csv(self, csv_path: Union[str, Path]) -> Dict[str, Any]:
        """Predice generación a partir de un CSV con múltiples registros.

        Args:
            csv_path: Ruta al CSV con features de entrada.

        Returns:
            Diccionario con predicciones.
        """
        if self.model is None:
            self.load()

        df = pd.read_csv(csv_path, parse_dates=[self.time_col])
        logger.info(f"CSV cargado: {df.shape}")

        # Asegurar features
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_cols]
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_scaled = X

        preds = self.model.predict(X_scaled)
        preds = np.maximum(preds, 0)

        return {
            "timestamps": df[self.time_col].dt.isoformat().tolist(),
            "predictions": preds.tolist(),
            "total_predicted_kwh": float(preds.sum()),
            "n_predictions": len(preds),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    config = load_config()
    predictor = ModelPredictor(config)
    predictor.load()

    # Predicción de ejemplo
    result = predictor.predict(horizon_hours=24)
    print(f"Total predicho (24h): {result['total_predicted_kwh']:.2f} kWh")
    print(f"Top 5 features: {[f['feature'] for f in result['top_5_features']]}")
