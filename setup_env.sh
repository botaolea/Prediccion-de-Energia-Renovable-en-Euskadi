#!/usr/bin/env bash
# setup_env.sh - Configura el entorno virtual e instala dependencias.
# Uso:  bash setup_env.sh

set -e

echo "=========================================="
echo "  Configuración del entorno del proyecto"
echo "  Predicción Energía Renovable Euskadi"
echo "=========================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no está instalado."
    echo "  Instálalo desde: https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python detectado: $PY_VERSION"

# Verificar versión >= 3.10
python3 -c "
import sys
if sys.version_info < (3, 10):
    print('ERROR: Necesitas Python 3.10 o superior')
    sys.exit(1)
"

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo ""
    echo ">>> Creando entorno virtual..."
    python3 -m venv venv
    echo "OK: venv creado"
fi

# Activar
echo ""
echo ">>> Activando venv..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Actualizar pip
echo ""
echo ">>> Actualizando pip..."
python -m pip install --upgrade pip --quiet

# Instalar dependencias
echo ""
echo ">>> Instalando dependencias (puede tardar varios minutos)..."
pip install -r requirements.txt

# Verificar
echo ""
echo ">>> Verificando instalación..."
python -c "
import streamlit, pandas, numpy, sklearn, xgboost, shap, pvlib, astral, plotly, seaborn, yaml, requests, bs4, joblib, folium
print(f'  streamlit {streamlit.__version__}')
print(f'  pandas {pandas.__version__}')
print(f'  xgboost {xgboost.__version__}')
print(f'  shap {shap.__version__}')
print(f'  pvlib {pvlib.__version__}')
print('  TODO OK')
"

# Diagnóstico
echo ""
echo ">>> Ejecutando diagnóstico..."
python diagnostico.py || true

echo ""
echo "=========================================="
echo "  ✅ Configuración completada"
echo "=========================================="
echo ""
echo "Para activar el entorno en el futuro:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "Para iniciar la app:"
echo "  streamlit run app.py"
echo ""
echo "O pulsa F5 en VS Code (config .vscode/launch.json ya creado)"
