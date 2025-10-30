# core_tools/statistical_engine.py
"""
Statistical helpers: KPIs, anomaly detection, simple clustering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    summary["skew"] = numeric.skew()
    return summary

def detect_outliers_isolationforest(df: pd.DataFrame, contamination: float = 0.03) -> dict:
    numeric = df.select_dtypes(include=[np.number]).fillna(0)
    if numeric.shape[1] == 0:
        return {}
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(numeric)
    outlier_idxs = np.where(preds == -1)[0].tolist()
    return {"indices": outlier_idxs, "count": len(outlier_idxs)}

def top_categorical_frequencies(df: pd.DataFrame, top_n: int = 3) -> dict:
    cats = df.select_dtypes(include=["object", "category"])
    result = {}
    for c in cats.columns:
        vc = cats[c].value_counts().head(top_n)
        result[c] = vc.to_dict()
    return result

def detect_date_columns(df: pd.DataFrame) -> list:
    date_cols = []
    for c in df.columns:
        low = c.lower()
        if "date" in low or "time" in low:
            date_cols.append(c)
    # Also try parsing columns that look like dates
    for c in df.columns:
        if c in date_cols:
            continue
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.5:
                date_cols.append(c)
        except Exception:
            pass
    return date_cols

def compute_basic_kpis(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    kpis = {}
    for c in numeric.columns:
        series = numeric[c].dropna()
        kpis[c] = {
            "mean": float(series.mean()) if len(series) else None,
            "median": float(series.median()) if len(series) else None,
            "std": float(series.std()) if len(series) else None,
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
        }
    return kpis
