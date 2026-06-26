"""
run_pipeline.py - Script orquestador del pipeline completo.

Ejecuta:
1. data_ingestion.py (descarga datos)
2. data_preprocessing.py (limpieza y unión)
3. feature_engineering.py (creación de features)
4. model_trainer.py (entrenamiento XGBoost + SHAP)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Añadir raíz al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logger

logger = setup_logger("pipeline")


def main():
    """Ejecuta el pipeline completo."""
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE COMPLETO")
    logger.info("=" * 70)

    config = load_config(ROOT / "config.yaml")

    # 1. INGESTA
    logger.info("\n" + "=" * 70)
    logger.info("FASE 1: INGESTA DE DATOS")
    logger.info("=" * 70)
    try:
        from src.data_ingestion import (
            CertificadosIngestion,
            MeteorologiaIngestion,
        )

        # Meteorología (con Open-Meteo como fallback)
        logger.info("1.1 - Descargando meteorología...")
        meteo = MeteorologiaIngestion(config)
        df_meteo = meteo.run()
        logger.info(f"  ✓ Meteorología: {df_meteo.shape}")

        # Certificados
        logger.info("1.2 - Descargando certificados...")
        cert = CertificadosIngestion(config)
        df_cert = cert.run()
        logger.info(f"  ✓ Certificados: {df_cert.shape}")

        # GoiEner se omite en este pipeline rápido (requiere 2GB)
        logger.info("1.3 - GoiEner omitido (ZIP 2GB). Usando meteorología como fuente principal.")
        logger.info("    Para usar GoiEner, ejecuta manualmente: python src/data_ingestion.py")

    except Exception as e:
        logger.error(f"Error en ingesta: {e}")
        raise

    # 2. PREPROCESAMIENTO
    logger.info("\n" + "=" * 70)
    logger.info("FASE 2: PREPROCESAMIENTO")
    logger.info("=" * 70)
    try:
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(config)
        df_prep = preprocessor.run()
        logger.info(f"  ✓ Preprocesado: {df_prep.shape}")
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        raise

    # 3. FEATURE ENGINEERING
    logger.info("\n" + "=" * 70)
    logger.info("FASE 3: FEATURE ENGINEERING")
    logger.info("=" * 70)
    try:
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(config)
        df_final, features = fe.run()
        logger.info(f"  ✓ Dataset final: {df_final.shape}")
        logger.info(f"  ✓ Features: {len(features)}")
    except Exception as e:
        logger.error(f"Error en feature engineering: {e}")
        raise

    # 4. ENTRENAMIENTO
    logger.info("\n" + "=" * 70)
    logger.info("FASE 4: ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 70)
    try:
        from src.model_trainer import ModelTrainer
        trainer = ModelTrainer(config)
        results = trainer.run_full_pipeline(
            model_type="xgboost",
            use_random_search=False,  # Rápido para primera iteración
            run_walk_forward=False,   # Rápido
            compute_shap=True,
        )
        logger.info(f"  ✓ Modelo entrenado: RMSE={results['model_metrics']['RMSE']:.4f}")
        logger.info(f"  ✓ Mejora vs baseline: {results['improvement_pct']:+.2f}%")
        if "shap" in results:
            logger.info(f"  ✓ Top 5 SHAP: {results['shap'].get('top_5_features', [])}")
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        raise

    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
