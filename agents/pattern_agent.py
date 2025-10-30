# agents/pattern_agent.py
"""
Pattern Agent: runs statistical EDA, anomalies detection, and builds plot list
"""

from core_tools.statistical_engine import summarize_numeric, detect_outliers_isolationforest, top_categorical_frequencies, detect_date_columns, compute_basic_kpis
from core_tools.viz_engine import plot_correlation, plot_numeric_histogram, plot_time_series
import pandas as pd

class PatternAgent:
    def analyze(self, df: pd.DataFrame) -> dict:
        profile = f"Rows: {len(df)}, Columns: {len(df.columns)}"
        numeric_summary = summarize_numeric(df)
        kpis = compute_basic_kpis(df)
        anomalies = detect_outliers_isolationforest(df, contamination=0.03)
        cat_freq = top_categorical_frequencies(df)
        date_cols = detect_date_columns(df)

        plots = []
        # correlation plot
        if len(df.select_dtypes(include=["number"]).columns) > 1:
            plots.append(plot_correlation(df))
        # histogram of top numeric column
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            plots.append(plot_numeric_histogram(df, num_cols[0]))

        # if a date column exists and a numeric column exists, add time series of first pair
        if date_cols and num_cols.any():
            plots.append(plot_time_series(df, date_cols[0], num_cols[0]))

        return {
            "profile_text": profile,
            "numeric_summary": numeric_summary,
            "kpis": kpis,
            "anomalies": anomalies,
            "categorical_freq": cat_freq,
            "date_columns": date_cols,
            "plots": plots
        }
