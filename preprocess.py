"""
preprocess.py
Missing-value handling + scaler comparison, following the same
StandardScaler / MinMaxScaler / RobustScaler walkthrough style
used in class, but applied to the water potability columns.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from load_data import load_data
from config import NUMERIC_COLUMNS, TARGET


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the column median (matches the
    class code's `fillna(df[col].median())` pattern)."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def get_preprocessing_summary() -> dict:
    df = load_data()

    missing_before = {
        col: int(df[col].isna().sum())
        for col in NUMERIC_COLUMNS if col in df.columns
    }

    df_clean = impute_missing(df)

    missing_after = {
        col: int(df_clean[col].isna().sum())
        for col in NUMERIC_COLUMNS if col in df_clean.columns
    }

    X = df_clean[NUMERIC_COLUMNS]

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    scaled_preview = {}
    scaled_stats = {}

    for name, scaler in scalers.items():
        Z = scaler.fit_transform(X)
        Zdf = pd.DataFrame(Z, columns=NUMERIC_COLUMNS)
        scaled_preview[name] = Zdf.head(5).round(4).to_dict(orient="records")
        scaled_stats[name] = {
            "mean": round(float(Zdf.values.mean()), 4),
            "std": round(float(Zdf.values.std()), 4),
            "min": round(float(Zdf.values.min()), 4),
            "max": round(float(Zdf.values.max()), 4),
        }

    raw_preview = X.head(5).round(3).to_dict(orient="records")

    return {
        "columns": NUMERIC_COLUMNS,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "raw_preview": raw_preview,
        "scaled_preview": scaled_preview,
        "scaled_stats": scaled_stats,
    }


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_preprocessing_summary())
