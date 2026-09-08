"""
app.py
AquaSense - AI-Based Water Quality Assessment and Demand Forecasting System

Same route/render pattern as the class template:
  - one route per stage
  - each route renders the same index.html shell with a different
    "active" section switched on inside the template.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from load_data import get_data_summary
from preprocess import get_preprocessing_summary
from config import (
    MODEL_PATH, SCALER_PATH, METRICS_PATH,
    NUMERIC_COLUMNS, FEATURE_META,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None

    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing():
    error = None
    prep = None

    try:
        prep = get_preprocessing_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        prep=prep,
        error=error,
    )


@app.route("/model-training")
def model_training():
    error = None
    metrics = None
    trained = os.path.exists(METRICS_PATH)

    if trained:
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        except Exception as e:
            error = f"Could not read saved metrics: {e}"
    else:
        error = (
            "No trained model found yet. Run `python train_model.py` "
            "from the project root, then refresh this page."
        )

    return render_template(
        "index.html",
        active="model-training",
        metrics=metrics,
        error=error,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    error = None
    result = None
    form_values = {col: "" for col in NUMERIC_COLUMNS}

    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

    if not model_ready:
        error = (
            "No trained model found yet. Run `python train_model.py` "
            "from the project root, then refresh this page."
        )

    if request.method == "POST" and model_ready:
        try:
            values = []
            for col in NUMERIC_COLUMNS:
                raw = request.form.get(col, "").strip()
                form_values[col] = raw
                values.append(float(raw))

            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)

            X = pd.DataFrame([values], columns=NUMERIC_COLUMNS)
            X_scaled = scaler.transform(X)

            pred = int(model.predict(X_scaled)[0])
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_scaled)[0][pred])

            result = {
                "label": "Potable" if pred == 1 else "Not Potable",
                "is_potable": pred == 1,
                "confidence": round(proba * 100, 1) if proba is not None else None,
            }

        except ValueError:
            error = "Please enter valid numeric values for every field."
        except Exception as e:
            error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="prediction",
        error=error,
        result=result,
        form_values=form_values,
        feature_meta=FEATURE_META,
        numeric_columns=NUMERIC_COLUMNS,
    )


@app.route("/demand-forecasting")
def demand_forecasting():
    return render_template("index.html", active="demand-forecasting")


if __name__ == "__main__":
    app.run(debug=True)
