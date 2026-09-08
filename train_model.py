"""
train_model.py
Trains a classifier to predict Potability from the water quality
features, then saves the fitted scaler + model + metrics to /models
so the Flask app can load them instantly instead of retraining
on every request.

Run this once from PyCharm (or `python train_model.py`) after you
put your dataset in /data. Re-run it any time the dataset changes.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from load_data import load_data
from preprocess import impute_missing
from config import NUMERIC_COLUMNS, TARGET, MODEL_PATH, SCALER_PATH, METRICS_PATH


CANDIDATES = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
}


def train_and_evaluate():
    df = load_data()
    df = impute_missing(df)

    X = df[NUMERIC_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    for name, model in CANDIDATES.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        fitted_models[name] = model

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted_models[best_name]
    best_preds = best_model.predict(X_test_s)

    cm = confusion_matrix(y_test, best_preds).tolist()
    report = classification_report(y_test, best_preds, output_dict=True)

    metrics = {
        "compared_models": results,
        "best_model": best_name,
        "confusion_matrix": cm,
        "classification_report": report,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": NUMERIC_COLUMNS,
    }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print(f"Best model: {m['best_model']}")
    for name, scores in m["compared_models"].items():
        print(f"  {name:<20} {scores}")
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved metrics-> {METRICS_PATH}")
