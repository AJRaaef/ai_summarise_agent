# core_tools/viz_engine.py
import plotly.express as px
import pandas as pd

def plot_numeric_histogram(df: pd.DataFrame, column: str):
    fig = px.histogram(df, x=column, nbins=30, title=f"Distribution - {column}")
    return fig

def plot_correlation(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        # placeholder empty figure
        fig = px.imshow([[0]], text_auto=True, title="Correlation: not enough numeric columns")
        return fig
    corr = numeric.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation heatmap (numeric columns)")
    return fig

def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    # ensure parsed
    series = df.copy()
    series[date_col] = pd.to_datetime(series[date_col], errors="coerce")
    series = series.dropna(subset=[date_col])
    fig = px.line(series.sort_values(date_col), x=date_col, y=value_col, title=f"{value_col} over {date_col}")
    return fig
