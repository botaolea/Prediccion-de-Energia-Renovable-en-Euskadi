"""
plots.py - Funciones de visualización con Plotly y Matplotlib.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette corporativa
PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "solar": "#ff9500",
    "grid": "#7f7f7f",
}


def plot_time_series_generation(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    target_col: str = "generacion",
    title: str = "Evolución temporal de la generación solar",
    height: int = 450,
) -> go.Figure:
    """Crea un gráfico de líneas de la generación temporal.

    Args:
        df: DataFrame con datos.
        time_col: Columna de tiempo.
        target_col: Columna de generación.
        title: Título del gráfico.
        height: Altura del gráfico.

    Returns:
        Figura de Plotly.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[target_col],
            mode="lines",
            name="Generación (kWh)",
            line=dict(color=PALETTE["solar"], width=1.5),
            hovertemplate="<b>%{x}</b><br>Generación: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Generación (kWh)",
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )
    return fig


def plot_predictions_with_interval(
    timestamps: List[str],
    predictions: List[float],
    lower: List[float],
    upper: List[float],
    title: str = "Predicción de generación solar (24-48h)",
    height: int = 450,
) -> go.Figure:
    """Crea un gráfico de predicción con intervalo de confianza.

    Args:
        timestamps: Lista de timestamps.
        predictions: Lista de predicciones.
        lower: Límite inferior del intervalo.
        upper: Límite superior del intervalo.
        title: Título.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    fig = go.Figure()

    # Intervalo de confianza (área)
    fig.add_trace(
        go.Scatter(
            x=timestamps + timestamps[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 149, 0, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="Intervalo 95%",
            hoverinfo="skip",
        )
    )

    # Predicción
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines+markers",
            name="Predicción (kWh)",
            line=dict(color=PALETTE["solar"], width=2.5),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Predicción: %{y:.2f} kWh<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Hora",
        yaxis_title="Generación prevista (kWh)",
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    height: int = 500,
) -> go.Figure:
    """Crea un heatmap de correlación.

    Args:
        df: DataFrame.
        columns: Lista de columnas a incluir (None para todas las numéricas).
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    if columns:
        df_num = df[columns].select_dtypes(include=[np.number])
    else:
        df_num = df.select_dtypes(include=[np.number])

    # Limitar a 15 columnas
    if len(df_num.columns) > 15:
        df_num = df_num.iloc[:, :15]

    corr = df_num.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}<br>Correlación: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz de correlación",
        template="plotly_white",
        height=height,
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_generation_by_hour_month(
    df: pd.DataFrame,
    target_col: str = "generacion",
    height: int = 450,
) -> go.Figure:
    """Crea un boxplot de generación por hora y mes.

    Args:
        df: DataFrame con timestamp.
        target_col: Columna de generación.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        return go.Figure()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month_name"] = df["timestamp"].dt.month_name()

    fig = px.box(
        df,
        x="hour",
        y=target_col,
        color="month_name",
        title="Distribución de generación por hora y mes",
        labels={target_col: "Generación (kWh)", "hour": "Hora del día"},
        template="plotly_white",
        height=height,
    )
    fig.update_layout(legend_title_text="Mes")
    return fig


def plot_real_vs_predicted(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    height: int = 450,
) -> go.Figure:
    """Crea un scatter plot de real vs predicho.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig = go.Figure()

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(
                color=PALETTE["primary"],
                size=5,
                opacity=0.6,
            ),
            name="Predicciones",
            hovertemplate="Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>",
        )
    )

    # Línea ideal
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color=PALETTE["warning"], dash="dash", width=2),
            name="Línea ideal (y=x)",
        )
    )

    fig.update_layout(
        title="Real vs Predicho",
        xaxis_title="Valor real (kWh)",
        yaxis_title="Valor predicho (kWh)",
        template="plotly_white",
        height=height,
        showlegend=True,
    )
    return fig


def plot_residuals(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    height: int = 400,
) -> go.Figure:
    """Crea un gráfico de residuos.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=PALETTE["primary"], size=5, opacity=0.6),
            name="Residuos",
            hovertemplate="Predicho: %{x:.2f}<br>Residuo: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["warning"])

    fig.update_layout(
        title="Análisis de residuos",
        xaxis_title="Valor predicho (kWh)",
        yaxis_title="Residuo (real - predicho)",
        template="plotly_white",
        height=height,
    )
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    height: int = 500,
) -> go.Figure:
    """Crea un gráfico de barras horizontales de importancia de features.

    Args:
        importance_df: DataFrame con columnas 'feature' y 'mean_abs_shap'.
        top_n: Número de features a mostrar.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    df_top = importance_df.head(top_n).sort_values("mean_abs_shap")

    fig = go.Figure(
        go.Bar(
            x=df_top["mean_abs_shap"],
            y=df_top["feature"],
            orientation="h",
            marker_color=PALETTE["primary"],
            text=df_top["mean_abs_shap"].round(4),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importancia SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} features - Importancia SHAP global",
        xaxis_title="Importancia SHAP (media |valor|)",
        yaxis_title="Feature",
        template="plotly_white",
        height=height,
    )
    return fig


def plot_whatif_simulation(
    base_prediction: List[float],
    modified_prediction: List[float],
    timestamps: List[str],
    height: int = 400,
) -> go.Figure:
    """Crea un gráfico comparativo para el simulador what-if.

    Args:
        base_prediction: Predicción base.
        modified_prediction: Predicción modificada.
        timestamps: Timestamps.
        height: Altura.

    Returns:
        Figura de Plotly.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=base_prediction,
            mode="lines+markers",
            name="Escenario base",
            line=dict(color=PALETTE["grid"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=modified_prediction,
            mode="lines+markers",
            name="Escenario modificado",
            line=dict(color=PALETTE["solar"], width=2.5),
        )
    )
    fig.update_layout(
        title="Simulación what-if",
        xaxis_title="Hora",
        yaxis_title="Generación (kWh)",
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )
    return fig


def plot_kpi_cards(kpis: Dict[str, Any]) -> None:
    """Renderiza tarjetas KPI en Streamlit.

    Args:
        kpis: Diccionario con KPIs a mostrar.
    """
    import streamlit as st

    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis.items()):
        col.metric(label=label, value=value)
