"""
load_data.py
Same pattern as the class template (load_data -> get_data_summary),
adapted for the AquaSense water potability dataset.
"""

import os
import pandas as pd

from config import DATA_PATH, NUMERIC_COLUMNS, TARGET


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Place your water_potability.csv inside the 'data' folder, "
            f"or update DATA_PATH in config.py."
        )

    df = pd.read_csv(path)
    return df


def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },

        "missing_percent": {
            col: round(100 * df[col].isna().mean(), 2)
            for col in df.columns
        },

        "preview": df.head(10).round(3).to_dict(orient="records"),

        "duplicate_count": int(df.duplicated().sum()),

        "target_counts": (
            df[TARGET].value_counts().to_dict()
            if TARGET in df.columns else {}
        ),

        "numeric_stats": (
            df[NUMERIC_COLUMNS].agg(["min", "max", "mean", "std"])
            .T.round(3).to_dict(orient="index")
            if all(c in df.columns for c in NUMERIC_COLUMNS) else {}
        ),
    }

    return summary


def get_duplicate_count(path: str = DATA_PATH) -> int:
    df = load_data(path)
    return int(df.duplicated().sum())


if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    print(get_data_summary())
