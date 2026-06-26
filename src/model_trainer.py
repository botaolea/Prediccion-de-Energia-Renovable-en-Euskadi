"""
model_trainer.py - Entrenamiento, validación walk-forward, y explicabilidad SHAP.

Componentes:
- Carga y división temporal del dataset
- Modelo baseline de persistencia
- Búsqueda de hiperparámetros con RandomizedSearchCV
- Validación walk-forward (expanding window)
- Métricas: RMSE, MAE, R2
- SHAP global (TreeExplainer)
- Guardado del modelo y scaler
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("model_trainer")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de regresión.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.

    Returns:
        Diccionario con RMSE, MAE, R2.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# Baseline de persistencia
# ---------------------------------------------------------------------------
class PersistenceBaseline:
    """Modelo baseline: predice el valor de la hora anterior."""

    def __init__(self):
        """Inicializa el baseline de persistencia."""
        self.last_value = 0.0

    def fit(self, y_train: pd.Series) -> "PersistenceBaseline":
        """Ajusta el baseline guardando el último valor del set de entrenamiento.

        Args:
            y_train: Serie temporal de entrenamiento.

        Returns:
            self.
        """
        self.last_value = float(y_train.iloc[-1]) if len(y_train) > 0 else 0.0
        return self

    def predict(self, y_test: pd.Series) -> np.ndarray:
        """Predice usando el valor anterior (shift 1).

        Args:
            y_test: Serie temporal de test.

        Returns:
            Array de predicciones (mismo length que y_test).
        """
        # y_pred[i] = y_test[i-1]; primer valor = último de train
        preds = y_test.shift(1).fillna(self.last_value).values
        return preds


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------
class WalkForwardValidator:
    """Validación walk-forward (expanding window) para series temporales."""

    def __init__(
        self,
        initial_train_size: int,
        step_size: int = 1,
        n_splits: Optional[int] = None,
    ):
        """Inicializa el validador walk-forward.

        Args:
            initial_train_size: Tamaño inicial del set de entrenamiento (en filas).
            step_size: Tamaño del paso (en filas) entre ventanas.
            n_splits: Número máximo de splits. None para usar todos los posibles.
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Genera índices de train/test para cada ventana walk-forward.

        Args:
            X: DataFrame de features (para obtener el tamaño).

        Returns:
            Lista de tuplas (train_idx, test_idx).
        """
        n = len(X)
        splits = []
        train_end = self.initial_train_size

        while train_end + self.step_size <= n:
            test_start = train_end
            test_end = min(train_end + self.step_size, n)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
            train_end = test_end
            if self.n_splits and len(splits) >= self.n_splits:
                break

        logger.info(f"Walk-forward: {len(splits)} ventanas generadas")
        return splits


# ---------------------------------------------------------------------------
# Trainer principal
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Entrena, evalúa y guarda el modelo XGBoost."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el trainer.

        Args:
            config: Configuración del proyecto.
        """
        self.config = config
        self.processed_path = Path(config["data"]["processed_path"])
        self.models_path = Path(config["data"]["models_path"])
        ensure_dir(self.models_path)

        self.target_col = config["model"]["target_column"]
        self.time_col = config["model"]["time_column"]
        self.test_size = config["model"]["test_size"]
        self.random_state = config["model"].get("random_state", 42)

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        self.feature_cols: List[str] = []

    def load_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """Carga el dataset final y la lista de features.

        Returns:
            Tupla (DataFrame, feature_columns).
        """
        dataset_path = self.processed_path / "dataset_final.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

        df = pd.read_csv(dataset_path, parse_dates=[self.time_col])
        logger.info(f"Dataset cargado: {df.shape}")

        # Cargar lista de features
        features_file = self.models_path / "feature_columns.txt"
        if features_file.exists():
            self.feature_cols = features_file.read_text(encoding="utf-8").strip().split("\n")
        else:
            # Inferir features: todas las numéricas excepto target y time
            exclude = {self.target_col, self.time_col}
            self.feature_cols = [
                col for col in df.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
            ]
        logger.info(f"Features: {len(self.feature_cols)} columnas")
        return df, self.feature_cols

    def split_temporal(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide el dataset en train/test de forma temporal (sin shuffle).

        Args:
            df: DataFrame ordenado por timestamp.

        Returns:
            Tupla (X_train, X_test, y_train, y_test).
        """
        df = df.sort_values(self.time_col).reset_index(drop=True)
        n_test = int(len(df) * self.test_size)

        train_df = df.iloc[:-n_test]
        test_df = df.iloc[-n_test:]

        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target_col]

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train: {train_df[self.time_col].min()} -> {train_df[self.time_col].max()}")
        logger.info(f"Test:  {test_df[self.time_col].min()} -> {test_df[self.time_col].max()}")

        return X_train, X_test, y_train, y_test

    def fit_scaler(self, X_train: pd.DataFrame) -> StandardScaler:
        """Ajusta el StandardScaler con los datos de entrenamiento.

        Args:
            X_train: Features de entrenamiento.

        Returns:
            Scaler ajustado.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        # Guardar
        scaler_path = self.models_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler guardado: {scaler_path}")
        return self.scaler

    def transform_with_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica el scaler a un DataFrame.

        Args:
            X: Features a escalar.

        Returns:
            DataFrame escalado.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler no ajustado. Llama fit_scaler primero.")
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return X_scaled

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparams: Optional[Dict] = None,
        random_search: bool = True,
    ) -> XGBRegressor:
        """Entrena el modelo XGBoost.

        Args:
            X_train: Features de entrenamiento.
            y_train: Target.
            hyperparams: Hiperparámetros manuales. Si se pasa, se ignoran random_search.
            random_search: Si True, usa RandomizedSearchCV.

        Returns:
            Modelo XGBoost entrenado.
        """
        xgb_cfg = self.config["model"]["xgboost"]

        if hyperparams:
            logger.info(f"Entrenando XGBoost con hiperparámetros manuales: {hyperparams}")
            self.model = XGBRegressor(
                **hyperparams,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
            )
            self.model.fit(X_train, y_train)
            return self.model

        if random_search:
            logger.info("Iniciando RandomizedSearchCV para XGBoost...")
            param_dist = {
                "n_estimators": xgb_cfg["n_estimators"],
                "max_depth": xgb_cfg["max_depth"],
                "learning_rate": xgb_cfg["learning_rate"],
            }
            # Usar TimeSeriesSplit para respetar la temporalidad
            tscv = TimeSeriesSplit(n_splits=5)
            self.model = XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
            )
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_dist,
                n_iter=xgb_cfg["n_iter_random_search"],
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            logger.info(f"Mejores hiperparámetros: {search.best_params_}")
            logger.info(f"Mejor RMSE (CV): {-search.best_score_:.4f}")
        else:
            # Defaults razonables
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
            )
            self.model.fit(X_train, y_train)

        return self.model

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, hyperparams: Optional[Dict] = None
    ) -> RandomForestRegressor:
        """Entrena un modelo Random Forest como alternativa.

        Args:
            X_train: Features de entrenamiento.
            y_train: Target.
            hyperparams: Hiperparámetros opcionales.

        Returns:
            Modelo Random Forest entrenado.
        """
        params = hyperparams or {"n_estimators": 200, "max_depth": 10, "random_state": self.random_state}
        self.model = RandomForestRegressor(**params, n_jobs=-1)
        self.model.fit(X_train, y_train)
        logger.info(f"Random Forest entrenado: {params}")
        return self.model

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_preds: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evalúa el modelo y compara con baseline.

        Args:
            model: Modelo entrenado.
            X_test: Features de test.
            y_test: Target real.
            baseline_preds: Predicciones del baseline (opcional).

        Returns:
            Diccionario con métricas del modelo y baseline.
        """
        preds = model.predict(X_test)
        model_metrics = calculate_metrics(y_test.values, preds)
        logger.info(f"Métricas modelo: {model_metrics}")

        result = {"model_metrics": model_metrics, "predictions": preds.tolist()}

        if baseline_preds is not None:
            baseline_metrics = calculate_metrics(y_test.values, baseline_preds)
            result["baseline_metrics"] = baseline_metrics

            # Comparativa
            if baseline_metrics["RMSE"] > 0:
                improvement = (
                    (baseline_metrics["RMSE"] - model_metrics["RMSE"])
                    / baseline_metrics["RMSE"]
                    * 100
                )
                result["improvement_pct"] = float(improvement)
                logger.info(f"Baseline RMSE: {baseline_metrics['RMSE']:.4f}")
                logger.info(f"Mejora vs baseline: {improvement:+.2f}%")
                if improvement < 15:
                    logger.warning(
                        f"⚠️ El modelo NO supera el baseline en 15% (mejora: {improvement:.2f}%). "
                        "Considera ajustar hiperparámetros o añadir más features."
                    )

        return result

    def walk_forward_evaluate(
        self,
        df: pd.DataFrame,
        model_type: str = "xgboost",
        initial_train_hours: int = 3 * 365 * 24,
        step_hours: int = 30 * 24,
    ) -> Dict[str, Any]:
        """Ejecuta validación walk-forward (expanding window).

        Args:
            df: DataFrame completo con features y target.
            model_type: 'xgboost' o 'random_forest'.
            initial_train_hours: Tamaño inicial de entrenamiento en horas.
            step_hours: Paso entre ventanas en horas.

        Returns:
            Diccionario con métricas por ventana y promedio.
        """
        df = df.sort_values(self.time_col).reset_index(drop=True)
        X = df[self.feature_cols]
        y = df[self.target_col]

        validator = WalkForwardValidator(
            initial_train_size=initial_train_hours,
            step_size=step_hours,
        )
        splits = validator.split(X)

        all_metrics: List[Dict[str, float]] = []
        all_preds: List[float] = []
        all_true: List[float] = []

        for i, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train_wf, X_test_wf = X.iloc[train_idx], X.iloc[test_idx]
            y_train_wf, y_test_wf = y.iloc[train_idx], y.iloc[test_idx]

            # Scaler nuevo por ventana
            scaler_wf = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler_wf.fit_transform(X_train_wf),
                columns=X_train_wf.columns,
                index=X_train_wf.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler_wf.transform(X_test_wf),
                columns=X_test_wf.columns,
                index=X_test_wf.index,
            )

            # Modelo nuevo por ventana
            if model_type == "xgboost":
                model_wf = XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    tree_method="hist",
                )
            else:
                model_wf = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
                )
            model_wf.fit(X_train_scaled, y_train_wf)
            preds = model_wf.predict(X_test_scaled)

            m = calculate_metrics(y_test_wf.values, preds)
            all_metrics.append(m)
            all_preds.extend(preds.tolist())
            all_true.extend(y_test_wf.values.tolist())
            logger.info(
                f"Ventana {i}/{len(splits)}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}"
            )

        # Promedio
        avg_metrics = {
            "RMSE": float(np.mean([m["RMSE"] for m in all_metrics])),
            "MAE": float(np.mean([m["MAE"] for m in all_metrics])),
            "R2": float(np.mean([m["R2"] for m in all_metrics])),
        }
        logger.info(f"Walk-forward promedio: {avg_metrics}")

        return {
            "per_window_metrics": all_metrics,
            "avg_metrics": avg_metrics,
            "predictions": all_preds,
            "true_values": all_true,
        }

    def compute_shap(self, X_train: pd.DataFrame, max_samples: int = 1000) -> Dict[str, Any]:
        """Calcula SHAP values globales con TreeExplainer.

        Args:
            X_train: Features de entrenamiento (muestra).
            max_samples: Máximo de muestras para SHAP (por rendimiento).

        Returns:
            Diccionario con SHAP values y feature importance.
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Llama train_xgboost primero.")

        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            logger.warning(f"SHAP no disponible: {e}")
            return {}

        logger.info("Calculando SHAP values globales...")

        # Muestra para acelerar
        if len(X_train) > max_samples:
            X_sample = X_train.sample(n=max_samples, random_state=self.random_state)
        else:
            X_sample = X_train

        # TreeExplainer para XGBoost
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"TreeExplainer falló: {e}. Intentando con Explainer.")
            explainer = shap.Explainer(self.model, X_sample)
            shap_values = explainer(X_sample).values

        # Feature importance global (media absoluta)
        if isinstance(shap_values, list):
            shap_values_arr = shap_values[0]
        else:
            shap_values_arr = np.array(shap_values)

        mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample.columns.tolist(),
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        logger.info(f"Top 5 features SHAP:\n{importance_df.head()}")

        # Guardar gráfico summary
        shap_path = self.models_path / "shap_summary.png"
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_arr, X_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP summary plot guardado: {shap_path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar gráfico SHAP: {e}")

        # Guardar importance
        importance_path = self.models_path / "shap_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"SHAP importance guardado: {importance_path}")

        return {
            "importance_df": importance_df.to_dict(orient="records"),
            "top_5_features": importance_df.head(5)["feature"].tolist(),
        }

    def save_model(self, model_name: str = "xgboost_model.pkl") -> Path:
        """Guarda el modelo entrenado.

        Args:
            model_name: Nombre del archivo.

        Returns:
            Path del archivo guardado.
        """
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        model_path = self.models_path / model_name
        joblib.dump(self.model, model_path)
        logger.info(f"Modelo guardado: {model_path}")

        # Guardar metadata
        metadata = {
            "model_type": type(self.model).__name__,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "saved_at": pd.Timestamp.now().isoformat(),
            "models_path": str(model_path),
        }
        meta_path = self.models_path / "model_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Metadata guardada: {meta_path}")
        return model_path

    def run_full_pipeline(
        self,
        model_type: str = "xgboost",
        use_random_search: bool = True,
        run_walk_forward: bool = True,
        compute_shap: bool = True,
        hyperparams: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de entrenamiento.

        Args:
            model_type: 'xgboost' o 'random_forest'.
            use_random_search: Si True, usa RandomizedSearchCV.
            run_walk_forward: Si True, ejecuta walk-forward validation.
            compute_shap: Si True, calcula SHAP.
            hyperparams: Hiperparámetros manuales opcionales.

        Returns:
            Diccionario con métricas, comparativa baseline, walk-forward y SHAP.
        """
        results: Dict[str, Any] = {}

        # 1. Cargar dataset
        df, self.feature_cols = self.load_dataset()

        # 2. Dividir
        X_train, X_test, y_train, y_test = self.split_temporal(df)

        # 3. Escalar
        self.fit_scaler(X_train)
        X_train_scaled = self.transform_with_scaler(X_train)
        X_test_scaled = self.transform_with_scaler(X_test)

        # 4. Baseline de persistencia
        logger.info("--- Baseline de Persistencia ---")
        baseline = PersistenceBaseline()
        baseline.fit(y_train)
        baseline_preds = baseline.predict(y_test)
        baseline_metrics = calculate_metrics(y_test.values, baseline_preds)
        results["baseline_metrics"] = baseline_metrics
        logger.info(f"Baseline: {baseline_metrics}")

        # 5. Entrenar modelo
        logger.info(f"--- Entrenando {model_type.upper()} ---")
        if model_type == "xgboost":
            self.train_xgboost(
                X_train_scaled, y_train,
                hyperparams=hyperparams,
                random_search=use_random_search,
            )
        else:
            self.train_random_forest(X_train_scaled, y_train, hyperparams=hyperparams)

        # 6. Evaluar
        eval_results = self.evaluate(
            self.model, X_test_scaled, y_test, baseline_preds=baseline_preds
        )
        results["model_metrics"] = eval_results["model_metrics"]
        results["improvement_pct"] = eval_results.get("improvement_pct", 0.0)
        results["predictions_test"] = eval_results["predictions"]

        # 7. Walk-forward
        if run_walk_forward:
            logger.info("--- Walk-Forward Validation ---")
            wf_results = self.walk_forward_evaluate(
                df, model_type=model_type,
                initial_train_hours=3 * 365 * 24,
                step_hours=30 * 24,
            )
            results["walk_forward"] = {
                "avg_metrics": wf_results["avg_metrics"],
                "n_windows": len(wf_results["per_window_metrics"]),
            }

        # 8. SHAP
        if compute_shap and model_type == "xgboost":
            logger.info("--- SHAP Global ---")
            shap_results = self.compute_shap(X_train_scaled, max_samples=1000)
            results["shap"] = shap_results

        # 9. Guardar modelo
        self.save_model(
            model_name=f"{model_type}_model.pkl"
        )

        # 10. Guardar resultados
        results_path = self.models_path / "training_results.json"
        # Filtrar predictions para no guardar arrays enormes
        results_save = {k: v for k, v in results.items() if k != "predictions_test"}
        results_save["predictions_test"] = eval_results["predictions"][:100]  # Solo primeros 100
        results_path.write_text(
            json.dumps(results_save, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Resultados guardados: {results_path}")

        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    config = load_config()
    trainer = ModelTrainer(config)
    results = trainer.run_full_pipeline(model_type="xgboost", use_random_search=False, run_walk_forward=False)
    logger.info("Pipeline de entrenamiento completado.")
    logger.info(f"Resultados: {json.dumps(results, indent=2, default=str)}")
