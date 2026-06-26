"""
Página 2: Entrenar Modelo
Permite reentrenar el modelo XGBoost o Random Forest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.plots import plot_feature_importance, plot_real_vs_predicted, plot_residuals
from app.components.sidebar import render_footer
from app.components.utils_streamlit import (
    check_data_available,
    get_config,
    load_dataset_final,
    load_shap_importance,
    load_training_results,
)


def main():
    """Función principal de la página de entrenamiento."""
    st.title("⚙️ Entrenamiento del Modelo")
    st.markdown("Reentrena el modelo XGBoost o Random Forest con los datos actuales.")
    st.warning(
        "⚠️ **Atención**: Esta página es para usuarios avanzados. El reentrenamiento "
        "puede tardar varios minutos según el tamaño del dataset y los hiperparámetros."
    )

    # Verificar datos
    df = load_dataset_final()
    if df is None or df.empty:
        st.error("❌ No hay dataset final disponible. Ejecuta primero el pipeline de datos.")
        render_footer()
        return

    st.success(f"✅ Dataset disponible: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    # ============ Configuración del entrenamiento ============
    st.markdown("## 🔧 Configuración del Entrenamiento")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Algoritmo",
            ["xgboost", "random_forest"],
            format_func=lambda x: "XGBoost" if x == "xgboost" else "Random Forest",
        )
    with col2:
        use_random_search = st.checkbox(
            "Usar RandomizedSearchCV",
            value=False,
            help="Si está activado, busca los mejores hiperparámetros (más lento).",
        )

    # Hiperparámetros manuales
    st.markdown("### Hiperparámetros (manual)")
    use_manual = st.checkbox("Especificar hiperparámetros manualmente", value=False)

    hyperparams = None
    if use_manual:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        with col_b:
            max_depth = st.slider("max_depth", 2, 15, 6)
        with col_c:
            learning_rate = st.select_slider(
                "learning_rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                value=0.1,
            )
        if model_type == "xgboost":
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            }
        else:
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            }

    # Opciones avanzadas
    with st.expander("Opciones avanzadas"):
        run_walk_forward = st.checkbox("Ejecutar validación walk-forward", value=False, help="Validación expanding window (más lento).")
        compute_shap = st.checkbox("Calcular SHAP global", value=True, help="Genera gráfico SHAP y feature importance.")
        test_size = st.slider("Tamaño del test (%)", 10, 40, 20) / 100

    # Botón de entrenamiento
    st.divider()
    if st.button("🚀 Entrenar Modelo", type="primary"):
        _run_training(
            df=df,
            model_type=model_type,
            use_random_search=use_random_search,
            run_walk_forward=run_walk_forward,
            compute_shap=compute_shap,
            hyperparams=hyperparams,
            test_size=test_size,
        )

    # ============ Mostrar resultados previos ============
    st.divider()
    st.markdown("## 📊 Resultados del último entrenamiento")
    _show_previous_results()

    render_footer()


def _run_training(
    df: pd.DataFrame,
    model_type: str,
    use_random_search: bool,
    run_walk_forward: bool,
    compute_shap: bool,
    hyperparams: dict,
    test_size: float,
):
    """Ejecuta el entrenamiento del modelo.

    Args:
        df: Dataset final.
        model_type: Tipo de modelo.
        use_random_search: Si usar RandomizedSearchCV.
        run_walk_forward: Si ejecutar walk-forward validation.
        compute_shap: Si calcular SHAP.
        hyperparams: Hiperparámetros manuales.
        test_size: Tamaño del test.
    """
    from src.model_trainer import ModelTrainer

    config = get_config()
    # Sobrescribir test_size
    config["model"]["test_size"] = test_size

    trainer = ModelTrainer(config)

    # Progress bar
    progress = st.progress(0, "Iniciando...")
    status = st.empty()

    try:
        status.info("📚 Cargando dataset y features...")
        progress.progress(10)

        # Cargar features
        trainer.feature_cols = _load_feature_cols(config)
        if not trainer.feature_cols:
            _, trainer.feature_cols = trainer.load_dataset()
        else:
            df, _ = trainer.load_dataset()
        progress.progress(20)

        # Dividir
        status.info("✂️ Dividiendo train/test temporal...")
        X_train, X_test, y_train, y_test = trainer.split_temporal(df)
        progress.progress(30)

        # Escalar
        status.info("📏 Ajustando scaler...")
        trainer.fit_scaler(X_train)
        X_train_scaled = trainer.transform_with_scaler(X_train)
        X_test_scaled = trainer.transform_with_scaler(X_test)
        progress.progress(40)

        # Baseline
        status.info("📊 Calculando baseline de persistencia...")
        from src.model_trainer import PersistenceBaseline, calculate_metrics
        baseline = PersistenceBaseline()
        baseline.fit(y_train)
        baseline_preds = baseline.predict(y_test)
        baseline_metrics = calculate_metrics(y_test.values, baseline_preds)
        progress.progress(50)

        # Entrenar modelo
        status.info(f"🤖 Entrenando {model_type.upper()}...")
        if model_type == "xgboost":
            trainer.train_xgboost(
                X_train_scaled, y_train,
                hyperparams=hyperparams,
                random_search=use_random_search,
            )
        else:
            trainer.train_random_forest(X_train_scaled, y_train, hyperparams=hyperparams)
        progress.progress(70)

        # Evaluar
        status.info("📈 Evaluando modelo...")
        eval_results = trainer.evaluate(trainer.model, X_test_scaled, y_test, baseline_preds=baseline_preds)
        progress.progress(85)

        # SHAP
        if compute_shap and model_type == "xgboost":
            status.info("🔍 Calculando SHAP values...")
            trainer.compute_shap(X_train_scaled, max_samples=500)
        progress.progress(95)

        # Guardar
        status.info("💾 Guardando modelo...")
        trainer.save_model(model_name=f"{model_type}_model.pkl")
        progress.progress(100)

        # Guardar resultados del entrenamiento
        _save_training_results(
            config=config,
            eval_results=eval_results,
            baseline_metrics=baseline_metrics,
            compute_shap=compute_shap,
            model_type=model_type,
        )

        # Mostrar resultados
        status.success("✅ ¡Entrenamiento completado!")

        st.markdown("### 📊 Resultados")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE Modelo", f"{eval_results['model_metrics']['RMSE']:.4f}")
        with col2:
            st.metric("RMSE Baseline", f"{baseline_metrics['RMSE']:.4f}")
        with col3:
            improvement = eval_results.get("improvement_pct", 0)
            st.metric("Mejora vs Baseline", f"{improvement:+.1f}%")
        with col4:
            st.metric("R²", f"{eval_results['model_metrics']['R2']:.4f}")

        # Gráficos
        st.markdown("### 📉 Gráficos de evaluación")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_real_vs_predicted(y_test.values, np.array(eval_results["predictions"])),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_residuals(y_test.values, np.array(eval_results["predictions"])),
                use_container_width=True,
            )

        # SHAP
        if compute_shap and model_type == "xgboost":
            st.markdown("### 🔍 Importancia SHAP")
            shap_path = Path(config["data"]["models_path"]) / "shap_importance.csv"
            if shap_path.exists():
                shap_df = pd.read_csv(shap_path)
                st.plotly_chart(plot_feature_importance(shap_df), use_container_width=True)

                # Mostrar imagen del summary plot si existe
                summary_path = Path(config["data"]["models_path"]) / "shap_summary.png"
                if summary_path.exists():
                    st.markdown("#### SHAP Summary Plot")
                    st.image(str(summary_path))

        st.success("✅ Modelo guardado. Ve a la página **Predecir Generación** para usarlo.")
        st.success("✅ Los resultados del entrenamiento se han guardado y ya están disponibles en esta página.")

    except Exception as e:
        status.error(f"❌ Error en el entrenamiento: {e}")
        st.exception(e)


def _save_training_results(
    config,
    eval_results: dict,
    baseline_metrics: dict,
    compute_shap: bool,
    model_type: str,
) -> None:
    """Guarda los resultados del entrenamiento en un JSON para mostrarlos después."""
    results_path = Path(config["data"]["models_path"]) / "training_results.json"
    results = {
        "model_type": model_type,
        "model_metrics": eval_results.get("model_metrics", {}),
        "baseline_metrics": baseline_metrics,
        "improvement_pct": eval_results.get("improvement_pct", 0),
        "predictions_test": eval_results.get("predictions", []),
    }

    if compute_shap:
        shap_path = Path(config["data"]["models_path"]) / "shap_importance.csv"
        if shap_path.exists():
            results["shap"] = {
                "importance_df": pd.read_csv(shap_path).to_dict(orient="records"),
                "top_5_features": pd.read_csv(shap_path).head(5)["feature"].tolist(),
            }

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _show_previous_results():
    """Muestra los resultados del último entrenamiento guardado."""
    results = load_training_results()
    if not results:
        st.info("ℹ️ No hay resultados de entrenamiento previo. Entrena un modelo para ver métricas.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{results.get('model_metrics', {}).get('RMSE', 0):.4f}")
    with col2:
        st.metric("MAE", f"{results.get('model_metrics', {}).get('MAE', 0):.4f}")
    with col3:
        st.metric("R²", f"{results.get('model_metrics', {}).get('R2', 0):.4f}")
    with col4:
        st.metric("Mejora vs Baseline", f"{results.get('improvement_pct', 0):+.1f}%")

    # Walk-forward
    if "walk_forward" in results:
        st.markdown("### Validación Walk-Forward")
        wf = results["walk_forward"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Ventanas evaluadas", wf.get("n_windows", 0))
        with col_b:
            st.metric("RMSE walk-forward", f"{wf.get('avg_metrics', {}).get('RMSE', 0):.4f}")

    # SHAP
    shap_df = load_shap_importance()
    if shap_df is not None and not shap_df.empty:
        st.markdown("### 🔍 Top features (SHAP global)")
        st.plotly_chart(plot_feature_importance(shap_df, top_n=10), use_container_width=True)

    # Descargar modelo
    st.markdown("### 💾 Descargar Modelo")
    config = get_config()
    model_path = Path(config["data"]["models_path"]) / "xgboost_model.pkl"
    if model_path.exists():
        model_bytes = model_path.read_bytes()
        st.download_button(
            label="📥 Descargar xgboost_model.pkl",
            data=model_bytes,
            file_name="xgboost_model.pkl",
            mime="application/octet-stream",
            key=f"download_xgboost_model_{model_path.stem}",
        )


def _load_feature_cols(config):
    """Carga las columnas de features desde el archivo."""
    features_file = Path(config["data"]["models_path"]) / "feature_columns.txt"
    if features_file.exists():
        return features_file.read_text(encoding="utf-8").strip().split("\n")
    return []


if __name__ == "__main__":
    main()
