"""
app.py - Punto de entrada principal de la aplicación Streamlit.

Plataforma de Predicción de Energía Renovable (Euskadi)

Usa la API moderna st.navigation (Streamlit >= 1.36) para gestionar
páginas ubicadas en app/pages/.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Añadir raíz del proyecto al path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.components.sidebar import render_footer
from app.components.utils_streamlit import check_data_available


def render_top_navigation() -> None:
    """Renderiza la navegación principal en la parte superior de la página."""
    page_options = [
        "Inicio",
        "Análisis Exploratorio",
        "Entrenar Modelo",
        "Predecir Generación",
        "Dashboard Impacto",
    ]
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Inicio"

    cols = st.columns(len(page_options))
    for col, page_name in zip(cols, page_options):
        button_type = "primary" if st.session_state.selected_page == page_name else "secondary"
        with col:
            if st.button(page_name, key=f"top_nav_{page_name}", use_container_width=True, type=button_type):
                st.session_state.selected_page = page_name


def render_home():
    """Renderiza la página de inicio."""
    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4 0%, #ff9500 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        .feature-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            height: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

    # Cabecera principal
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Predicción de Energía Renovable en Euskadi</h1>
        <p>Plataforma de Machine Learning para predecir la generación solar fotovoltaica en Bizkaia usando datos abiertos reales</p>
    </div>
    """, unsafe_allow_html=True)

    # Estado de datos
    data_ok, data_msg = check_data_available()

    # Resumen en dashboard
    st.markdown("## 🎯 ¿Qué hace esta plataforma?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>📊 Analiza</h3>
        <p>Datos reales de consumo y generación eléctrica de Bizkaia, certificados energéticos y meteorología.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>🤖 Predice</h3>
        <p>Generación solar de las próximas 24-48 horas usando XGBoost con validación walk-forward.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>🔍 Explica</h3>
        <p>Factores que más influyen en cada predicción usando SHAP values globales precalculados.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
        <h3>💰 Cuantifica</h3>
        <p>Ahorro económico estimado y CO2 evitado según el precio del pool eléctrico.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Guía rápida
    st.markdown("## 🚀 Guía rápida de uso")
    st.markdown("""
    Utiliza el menú de navegación superior para acceder a las distintas funcionalidades:

    | Página | Descripción | ¿Para quién? |
    |--------|-------------|--------------|
    | 🏠 **Inicio** | Esta página: resumen y guía | Todos |
    | 📊 **Análisis Exploratorio** | Estadísticas descriptivas y visualizaciones del dataset | Todos |
    | ⚙️ **Entrenar Modelo** | Reentrenamiento de XGBoost/RF con métricas | Usuarios avanzados |
    | 🔮 **Predecir Generación** | Predicción 24-48h con formulario o CSV | Todos (uso principal) |
    | 📈 **Dashboard de Impacto** | KPIs, ROI, simulador what-if | Gestores / Decisiones |

    ### 📌 Estado actual del sistema
    """)

    if data_ok:
        st.success("✅ Sistema listo. El modelo y los datos están disponibles.")
    else:
        st.warning(
            "⚠️ El sistema aún no está completo. Ve a la página **Entrenar Modelo** "
            "para ejecutar el pipeline completo, o sigue las instrucciones del README.md."
        )

    st.divider()

    # Información técnica
    with st.expander("🔧 Información técnica y fuentes de datos"):
        st.markdown(f"""
        **Stack tecnológico**:
        - Backend: Python 3.10+, pandas, scikit-learn, XGBoost, SHAP
        - Frontend: Streamlit
        - Visualización: Plotly, Matplotlib, Folium
        - Solar: pvlib, astral

        **Fuentes de datos reales**:
        1. **GoiEner** (DOI: 10.5281/zenodo.7362094): Consumo y generación eléctrica de España.
        2. **Certificados de Eficiencia Energética** (Open Data Euskadi): Características de edificios.
        3. **AEMET** (Estación Bilbao 1080): Datos meteorológicos horarios.
        4. **Fallback Kaggle** (`danielebertola/solar-energy`): Si AEMET no está disponible.

        **Localización**: Bilbao (43.2630°N, 2.9350°W), altura 30m.

        **Modelo ML**: XGBoost con RandomizedSearchCV y validación walk-forward (expanding window).
        """)

    render_footer()


def run_page_script(script_path: Path) -> None:
    """Ejecuta un script de página de Streamlit por ruta."""
    runpy.run_path(str(script_path), run_name="__main__")


def main():
    """Función principal que gestiona la navegación entre páginas."""
    # Configuración de página
    st.set_page_config(
        page_title="Predicción Energía Renovable Euskadi",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "# Predicción Energía Renovable Euskadi\n"
                "Plataforma de predicción de generación solar fotovoltaica en Bizkaia.\n"
                "Usa datos abiertos reales (GoiEner, AEMET, Open Data Euskadi)."
            )
        },
    )

    # Sidebar compartido
    from app.components.sidebar import render_sidebar
    render_sidebar()

    pages_dir = ROOT / "app" / "pages"
    render_top_navigation()
    st.divider()

    selected = st.session_state.selected_page

    if selected == "Inicio":
        render_home()
    elif selected == "Análisis Exploratorio":
        run_page_script(pages_dir / "1_Analisis_Exploratorio.py")
    elif selected == "Entrenar Modelo":
        run_page_script(pages_dir / "2_Entrenar_Modelo.py")
    elif selected == "Predecir Generación":
        run_page_script(pages_dir / "3_Predecir_Generacion.py")
    elif selected == "Dashboard Impacto":
        run_page_script(pages_dir / "4_Dashboard_Impacto.py")


if __name__ == "__main__":
    main()
