"""
diagnostico.py - Diagnóstico automático del entorno.

Ejecuta este script para detectar problemas:
    python diagnostico.py

Soluciona los errores más comunes en VS Code:
- Path de Python incorrecto
- Dependencias faltantes
- Configuración de VS Code (.vscode/launch.json)
- Permisos de archivo
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


# Colores para consola
class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERROR = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def header(title: str):
    print(f"\n{C.BOLD}{'=' * 60}{C.END}")
    print(f"{C.BOLD}  {title}{C.END}")
    print(f"{C.BOLD}{'=' * 60}{C.END}")


def ok(msg: str):
    print(f"  {C.OK}✓{C.END} {msg}")


def warn(msg: str):
    print(f"  {C.WARN}⚠{C.END} {msg}")


def error(msg: str):
    print(f"  {C.ERROR}✗{C.END} {msg}")


def check_python():
    """Verifica la versión de Python."""
    header("1. Python")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version}")
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro} (>= 3.10 ✓)")
    else:
        error(f"Python {v.major}.{v.minor}.{v.micro} - necesitas 3.10+")


def check_cwd():
    """Verifica el directorio de trabajo."""
    header("2. Directorio de trabajo")
    cwd = Path.cwd()
    print(f"  CWD: {cwd}")

    # Verificar archivos clave
    required = ["config.yaml", "app.py", "requirements.txt", "src", "app"]
    for f in required:
        if (cwd / f).exists():
            ok(f"Encontrado: {f}")
        else:
            error(f"Falta: {f} - ejecuta desde la raíz del proyecto")


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    header("3. Dependencias")

    deps = {
        "streamlit": "1.36+",  # Requiere >= 1.36 para st.Page
        "pandas": None,
        "numpy": None,
        "sklearn": None,
        "xgboost": None,
        "shap": None,
        "pvlib": None,
        "astral": None,
        "plotly": None,
        "matplotlib": None,
        "seaborn": None,
        "yaml": None,
        "requests": None,
        "bs4": None,
        "joblib": None,
        "folium": None,
        "streamlit_folium": None,
        "kagglehub": None,
    }

    missing = []
    for dep, min_version in deps.items():
        try:
            mod = importlib.import_module(dep)
            version = getattr(mod, "__version__", "?")
            if dep == "streamlit" and min_version:
                from packaging import version as v_parse
                try:
                    if v_parse.parse(version) >= v_parse.parse(min_version):
                        ok(f"{dep} {version}")
                    else:
                        error(f"{dep} {version} - necesitas >= {min_version}")
                        missing.append(dep)
                except Exception:
                    ok(f"{dep} {version}")
            else:
                ok(f"{dep} {version}")
        except ImportError:
            error(f"{dep} NO instalado")
            missing.append(dep)

    if missing:
        print(f"\n  {C.WARN}Para instalar las dependencias faltantes:{C.END}")
        print(f"  {C.BOLD}pip install -r requirements.txt{C.END}")
        return False
    return True


def check_project_modules():
    """Verifica que los módulos del proyecto se importen correctamente."""
    header("4. Módulos del proyecto")

    # Asegurar raíz en path
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    modules = [
        "src.utils",
        "src.data_ingestion",
        "src.data_preprocessing",
        "src.feature_engineering",
        "src.model_trainer",
        "src.model_predictor",
        "app.components.sidebar",
        "app.components.plots",
        "app.components.utils_streamlit",
    ]

    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            ok(mod)
        except Exception as e:
            error(f"{mod}: {type(e).__name__}: {e}")
            failed.append(mod)

    if failed:
        return False
    return True


def check_artifacts():
    """Verifica que existan los artefactos entrenados."""
    header("5. Artefactos del modelo")

    artifacts = [
        "models/xgboost_model.pkl",
        "models/scaler.pkl",
        "models/feature_columns.txt",
        "models/model_metadata.json",
        "data/processed/dataset_final.csv",
        "config.yaml",
    ]

    missing = []
    for art in artifacts:
        path = Path(art)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            ok(f"{art} ({size_kb:.1f} KB)")
        else:
            warn(f"Falta: {art}")
            missing.append(art)

    if missing:
        print(f"\n  {C.WARN}Para regenerar los artefactos:{C.END}")
        print(f"  {C.BOLD}python scripts/run_pipeline.py{C.END}")
        return False
    return True


def check_vscode_config():
    """Verifica/recomienda configuración de VS Code."""
    header("6. Configuración VS Code")

    vscode_dir = Path(".vscode")
    if not vscode_dir.exists():
        print(f"  Creando configuración recomendada para VS Code...")
        vscode_dir.mkdir(exist_ok=True)

        # launch.json
        launch_json = """{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit: app.py",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app.py", "--server.port=8501"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Pipeline completo",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/run_pipeline.py",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        },
        {
            "name": "Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        }
    ]
}"""
        (vscode_dir / "launch.json").write_text(launch_json, encoding="utf-8")
        ok("Creado .vscode/launch.json")

        # settings.json
        settings_json = """{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/*.pyc": true
    },
    "streamlit.runOnSave": false,
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}"""
        (vscode_dir / "settings.json").write_text(settings_json, encoding="utf-8")
        ok("Creado .vscode/settings.json")
    else:
        ok(".vscode/ ya existe")

    # Recomendaciones
    print(f"\n  {C.BOLD}Para usar en VS Code:{C.END}")
    print(f"  1. Abre el proyecto: code .")
    print(f"  2. Selecciona el intérprete: Ctrl+Shift+P → 'Python: Select Interpreter'")
    print(f"  3. Selecciona el venv: {Path('./venv/bin/python').resolve()}")
    print(f"  4. Pulsa F5 para lanzar la app (usará .vscode/launch.json)")


def check_git():
    """Verifica estado de Git."""
    header("7. Git")

    if not Path(".git").exists():
        warn("No es un repositorio Git")
        print(f"  Para inicializar: {C.BOLD}git init && git add -A && git commit -m 'Initial'{C.END}")
        return

    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        ok(f"Rama actual: {branch}")

        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        if status:
            n = len(status.split("\n"))
            warn(f"{n} archivos sin commitear")
        else:
            ok("Working tree clean")

        remotes = subprocess.check_output(["git", "remote", "-v"], text=True).strip()
        if remotes:
            ok(f"Remotes: {remotes.split(chr(10))[0]}")
        else:
            warn("Sin remoto configurado")
            print(f"  Para conectar a GitHub:")
            print(f"  {C.BOLD}git remote add origin https://github.com/USUARIO/proyecto_energia_euskadi.git{C.END}")
            print(f"  {C.BOLD}git push -u origin main{C.END}")
    except Exception as e:
        error(f"Error con git: {e}")


def main():
    print(f"\n{C.BOLD}🔍 DIAGNÓSTICO - Predicción Energía Renovable Euskadi{C.END}")
    print(f"  Fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")

    check_python()
    check_cwd()
    deps_ok = check_dependencies()
    modules_ok = check_project_modules()
    artifacts_ok = check_artifacts()
    check_vscode_config()
    check_git()

    # Resumen
    header("RESUMEN")
    print(f"  Dependencias: {'✓ OK' if deps_ok else '✗ Faltan librerías'}")
    print(f"  Módulos:      {'✓ OK' if modules_ok else '✗ Errores de import'}")
    print(f"  Artefactos:   {'✓ OK' if artifacts_ok else '⚠ Faltan archivos'}")

    if deps_ok and modules_ok and artifacts_ok:
        print(f"\n  {C.OK}{C.BOLD}✅ TODO OK. Ejecuta: streamlit run app.py{C.END}\n")
    else:
        print(f"\n  {C.WARN}{C.BOLD}⚠ Corrige los errores arriba antes de continuar.{C.END}")
        print(f"  {C.BOLD}Comandos útiles:{C.END}")
        print(f"    pip install -r requirements.txt")
        print(f"    python scripts/run_pipeline.py")
        print(f"    streamlit run app.py\n")


if __name__ == "__main__":
    main()
