# ⚡ Predicción de Energía Renovable en Euskadi
https://prediccion-de-energia-renovable-en-euskadi.streamlit.app/

Plataforma web de Machine Learning para predecir la generación solar fotovoltaica en Bizkaia (Euskadi) usando exclusivamente fuentes de datos abiertas y reales.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-0066CC.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Tabla de contenidos

1. [Descripción](#-descripción)
2. [Arquitectura del proyecto](#-arquitectura-del-proyecto)
3. [Fuentes de datos reales](#-fuentes-de-datos-reales)
4. [Instalación](#-instalación)
5. [Uso local](#-uso-local)
6. [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
7. [Aplicación Streamlit](#-aplicación-streamlit)
8. [Despliegue](#-despliegue)
9. [Tests](#-tests)
10. [Advertencias sobre datos](#-advertencias-sobre-datos)
11. [Roadmap](#-roadmap)

---

## 🎯 Descripción

Esta plataforma permite **predecir la generación solar fotovoltaica de las próximas 24-48 horas** para edificios ubicados en Bizkaia (Euskadi). El sistema combina datos meteorológicos reales, certificados energéticos de edificios y características temporales para entrenar un modelo XGBoost que supera al baseline de persistencia en más de un 95%.

### Casos de uso

- **Gestores energéticos**: planificar la demanda y optimizar el autoconsumo.
- **Particulares con placas solares**: estimar la generación esperada mañana.
- **Investigadores**: analizar qué factores meteorológicos más influyen.
- **Administración pública**: cuantificar el impacto económico y ambiental de la energía renovable.

---

## 🏗️ Arquitectura del proyecto

```
┌────────────────────────────────────────────────────────────────┐
│                     APP STREAMLIT (app.py)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Análisis  │  │Entrenar  │  │Predecir  │  │Dashboard     │  │
│  │Explorat. │  │Modelo    │  │Generación│  │Impacto       │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    PIPELINE ML (src/)                           │
│  data_ingestion.py  →  data_preprocessing.py  →                │
│  feature_engineering.py  →  model_trainer.py  →                │
│  model_predictor.py                                            │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                  FUENTES DE DATOS REALES                        │
│  • Open-Meteo Archive API (meteorología horaria Bilbao)        │
│  • AEMET Open Data (estación 1080 - requiere API key)          │
│  • Open Data Euskadi (certificados energéticos)                │
│  • GoiEner Zenodo (consumo y generación - 2GB)                 │
│  • Kaggle fallback (danielebertola/solar-energy)               │
└────────────────────────────────────────────────────────────────┘
```

### Estructura de directorios

```
proyecto_energia_euskadi/
├── app.py                    # Punto de entrada Streamlit (RAÍZ)
├── config.yaml               # Configuración central
├── requirements.txt          # Dependencias Python
├── setup.py                  # Instalación como paquete
├── .gitignore
│
├── data/
│   ├── raw/                  # Datos descargados sin modificar
│   │   ├── goiener/
│   │   ├── certificados/
│   │   └── meteorologia/
│   ├── processed/            # Datasets limpios y unidos
│   └── external/             # Archivos auxiliares
│
├── src/                      # Lógica ML
│   ├── __init__.py
│   ├── data_ingestion.py     # Descarga automática de datos
│   ├── data_preprocessing.py # Limpieza, imputación, unión
│   ├── feature_engineering.py# Creación de features
│   ├── model_trainer.py      # XGBoost + walk-forward + SHAP
│   ├── model_predictor.py    # Carga y predicción
│   └── utils.py              # Funciones auxiliares
│
├── app/                      # UI Streamlit
│   ├── pages/
│   │   ├── 1_Analisis_Exploratorio.py
│   │   ├── 2_Entrenar_Modelo.py
│   │   ├── 3_Predecir_Generacion.py
│   │   └── 4_Dashboard_Impacto.py
│   └── components/
│       ├── sidebar.py
│       ├── plots.py
│       └── utils_streamlit.py
│
├── models/                   # Artefactos entrenados
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── shap_importance.csv
│   ├── shap_summary.png
│   └── feature_columns.txt
│
├── tests/                    # Tests unitarios (pytest)
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_utils.py
│
├── scripts/
│   └── run_pipeline.py       # Orquestador del pipeline completo
│
└── .streamlit/
    ├── secrets.toml.example  # Template para API key
    └── secrets.toml          # (NO subir a git)
```

---

## 📊 Fuentes de datos reales

### 1. Open-Meteo Archive API (Fuente primaria usada)

- **URL**: https://open-meteo.com/en/docs/historical-weather-api
- **Variables**: temperatura, humedad relativa, precipitación, viento, radiación solar, nubosidad.
- **Período**: últimos 3 años horarios.
- **Localización**: Bilbao (43.2630°N, 2.9350°W).
- **Ventaja**: gratuita, sin API key, datos reales horarios.

### 2. AEMET Open Data (Alternativa con API key)

- **URL**: https://opendata.aemet.es
- **Estación**: Bilbao (ID 1080).
- **Requiere**: API key gratuita obtenible en https://opendata.aemet.es/.
- **Cómo configurar**: ver [Configuración AEMET](#configuración-aemet-opcional).

### 3. Open Data Euskadi - Certificados Energéticos

- **URL**: https://opendata.euskadi.eus/catalogo/-/certificados-de-eficiencia-energetica-de-edificios/
- **Variables**: calificación energética (A-G), superficie útil, año construcción, emisiones CO2.
- **Agregación**: por código postal de Bizkaia (48xxx).

### 4. GoiEner (DOI 10.5281/zenodo.7362094)

- **URL**: https://zenodo.org/record/7362094/files/goiener_public_data.zip
- **Contenido**: 71,048 archivos CSV (~2GB) con consumo y generación eléctrica.
- **Configuración**: en `config.yaml`, `n_files: 500` procesa solo 500 archivos (desarrollo). Cambiar a `null` para producción.

### 5. Kaggle Fallback (danielebertola/solar-energy)

- **URL**: https://www.kaggle.com/datasets/danielebertola/solar-energy
- **Uso**: solo si AEMET/Euskalmet fallan y Open-Meteo no está disponible.
- **Requiere**: cuenta Kaggle y credenciales configuradas.

---

## 🚀 Instalación

### Requisitos previos

- Compatible con Python 3.10, 3.11 y 3.12.
- 4 GB de espacio en disco (para GoiEner completo)
- Conexión a internet (para descarga de datos)

### Paso a paso

```bash
# 1. Clonar repositorio
git clone https://github.com/usuario/proyecto_energia_euskadi.git
cd proyecto_energia_euskadi

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


# 4. Instalar el paquete en modo desarrollo (opcional)
pip install -e .
```

---

## 💻 Uso local

### Ejecutar el pipeline completo (entrenar modelo desde cero)

```bash
# Opción A: Script orquestador
python scripts/run_pipeline.py

# Opción B: Paso a paso
python src/data_ingestion.py      # Descarga datos
python src/data_preprocessing.py  # Limpia y une
python src/feature_engineering.py # Crea features
python src/model_trainer.py       # Entrena modelo
```

### Iniciar la aplicación Streamlit

```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`.

---

## 🤖 Pipeline de Machine Learning

### Flujo

1. **Ingesta**: descarga datos de Open-Meteo (3 años horarios), certificados de Euskadi, y opcionalmente GoiEner.
2. **Preprocesamiento**: filtrado Bizkaia, resampleo horario, unión por código postal, imputación de nulos.
3. **Feature engineering**: 43 features que incluyen:
   - Temporales: hora, día, mes, fin de semana, cíclicas (sin/cos).
   - Solares: elevación y azimuth solar (pvlib/astral).
   - Lags: 1h, 3h, 24h, 168h (7 días).
   - Rolling windows: media 7d, media/std/max 24h.
   - Interacciones: `radiacion × is_daylight`, `elevación × radiación`, etc.
4. **Entrenamiento**: XGBoost con validación temporal (80/20), comparativa con baseline de persistencia.
5. **Validación**: walk-forward (expanding window) opcional.
6. **Explicabilidad**: SHAP global con TreeExplainer.

### Métricas obtenidas (modelo actual)

| Métrica | XGBoost | Baseline Persistencia | Mejora |
|---------|---------|----------------------|--------|
| RMSE    | 0.0011  | 0.2118               | +99.47%|
| MAE     | 0.0006  | 0.1008               | +99.41%|
| R²      | 0.9999  | 0.6922               | -      |

### Top 5 features (SHAP global)

1. `radiacion_solar` - Radiación solar directa (W/m²).
2. `rad_solar_daylight` - Interacción radiación × luz diurna.
3. `humedad_x_rad` - Humedad relativa × radiación.
4. `rolling_max_24h` - Pico de generación en 24h.
5. `generacion_lag_1h` - Generación hace 1 hora.

---

## 🖥️ Aplicación Streamlit

### Páginas

| Página | Funcionalidad |
|--------|---------------|
| 🏠 **Inicio** | Resumen del proyecto y estado del sistema. |
| 📊 **Análisis Exploratorio** | EDA: estadísticas, evolución temporal, correlaciones, histogramas, boxplots. |
| ⚙️ **Entrenar Modelo** | Reentrena XGBoost/RF con hiperparámetros personalizables. Muestra métricas y SHAP. |
| 🔮 **Predecir Generación** | Predicción 24-48h con 3 modos: edificio tipo, formulario manual, CSV. |
| 📈 **Dashboard de Impacto** | KPIs, ROI en euros, CO2 evitado, mapa de Bizkaia, simulador what-if. |

### Modos de predicción

1. **Edificio tipo**: elige entre 4 edificios predefinidos (oficina, vivienda, nave industrial, etc.).
2. **Formulario manual**: especifica fecha, hora, CP y variables meteorológicas.
3. **CSV masivo**: sube un CSV con múltiples registros (descarga template incluido).

---

## ☁️ Despliegue

### Opción 1: Streamlit Cloud (recomendado, gratuito)

1. Sube el repositorio a GitHub (público).
2. Ve a https://streamlit.io/cloud y conéctalo.
3. Configura:
   - **Repository**: tu-usuario/proyecto_energia_euskadi
   - **Branch**: main
   - **Main file path**: app.py
4. Añade secretos (Settings → Secrets):
   ```toml
   [aemet]
   api_key = "TU_API_KEY"
   ```
5. ¡Listo! La app estará disponible en `https://tu-app.streamlit.app`.

### Opción 2: Hugging Face Spaces

1. Crea un Space en https://huggingface.co/spaces con SDK Streamlit.
2. Sube los archivos.
3. Configura variables de entorno en Settings.

### Opción 3: Railway / Render

1. Conecta el repositorio.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT`

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Con coverage
python -m pytest tests/ --cov=src --cov-report=html
```

Tests implementados (21 tests, todos pasan):
- `test_utils.py`: 12 tests (configuración, fechas, conversiones, validaciones).
- `test_preprocessing.py`: 4 tests (filtrado, resampleo, imputación, outliers).
- `test_feature_engineering.py`: 6 tests (temporales, lags, rolling, interacciones).

---

## ⚙️ Configuración AEMET (opcional)

Para usar AEMET como fuente primaria en lugar de Open-Meteo:

1. **Obtener API key gratuita**:
   - Visita https://opendata.aemet.es/
   - Regístrate y solicita tu API key.

2. **Configurar en Streamlit**:
   - Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`.
   - Edita y añade tu API key:
     ```toml
     [aemet]
     api_key = "tu-api-key-aqui"
     ```

3. **O vía variable de entorno**:
   ```bash
   export AEMET_API_KEY="tu-api-key-aqui"
   ```

4. **Cambiar fuente primaria** en `config.yaml`:
   ```yaml
   meteorologia:
     primary_source: "aemet"
   ```

---

## ⚠️ Advertencias sobre datos

### Fallback utilizado

Por defecto, el sistema usa **Open-Meteo** como fuente meteorológica primaria porque:
- No requiere API key.
- Ofrece datos horarios reales de los últimos 3 años.
- Incluye radiación solar, esencial para predicción solar.

La columna `generacion` se **deriva** de la radiación solar usando una heurística (panel 10m², 18% eficiencia), ya que Open-Meteo no provee directamente generación eléctrica. Para datos de generación reales, configura GoiEner.

### GoiEner no procesado por defecto

El ZIP de GoiEner pesa 2GB y contiene 71,048 archivos. Para usarlo:
1. Cambia `n_files` en `config.yaml` a `null` (procesa todos) o mantén en 500 para desarrollo.
2. Ejecuta `python src/data_ingestion.py` (tardará varios minutos en descargar).
3. El pipeline usará GoiEner como fuente primaria de generación si está disponible.

### Certificados sintéticos

Si la descarga de Open Data Euskadi falla (por cambios en la web), se generan certificados sintéticos basados en estadísticas reales de Bizkaia (distribución de calificaciones A-G según datos del INE). Esto se indica en los logs.

---

## 🔮 Roadmap

- [ ] Integración con API de Euskalmet (requiere registro).
- [ ] Predicción probabilística (cuantiles 10/50/90).
- [ ] Modelo multi-edificio (un modelo por tipo de edificio).
- [ ] Optimización bayesiana de hiperparámetros (Optuna).
- [ ] API REST FastAPI independiente.
- [ ] Integración de precios del pool eléctrico en tiempo real (OMIE).
- [ ] Dashboard de monitorización continua (predicción vs realidad).

---

## 📄 Licencia

MIT License. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Open-Meteo** por proporcionar datos meteorológicos gratuitos y abiertos.
- **AEMET** por su API de datos abiertos.
- **Open Data Euskadi** por los certificados energéticos.
- **GoiEner** y Zenodo por el dataset público de consumo eléctrico.
- **Streamlit**, **XGBoost**, **SHAP** y **pvlib** por sus excelentes herramientas open-source.
