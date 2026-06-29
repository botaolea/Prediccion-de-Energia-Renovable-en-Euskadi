"""
Setup script para el proyecto Predicción Energía Renovable Euskadi.
Permite instalar el paquete con: pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="proyecto_energia_euskadi",
    version="1.0.0",
    author="Begoña Otaolea",
    description="Plataforma de predicción de generación solar fotovoltaica en Bizkaia/Euskadi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/proyecto_energia_euskadi",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "shap",
        "pvlib",
        "astral",
        "plotly",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "requests",
        "beautifulsoup4",
        "kagglehub",
        "joblib",
        "folium",
        "streamlit-folium",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
