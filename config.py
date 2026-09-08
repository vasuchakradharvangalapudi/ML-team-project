"""
config.py
Central place for paths and column definitions used across the project.
Change DATA_PATH here if you swap in your own CSV — as long as the
column names match, nothing else needs to change.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "potability_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

TARGET = "Potability"

NUMERIC_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Friendly labels + units, used on the Prediction page form
FEATURE_META = {
    "ph": {"label": "pH", "unit": "", "help": "0-14 scale, 7 is neutral"},
    "Hardness": {"label": "Hardness", "unit": "mg/L", "help": "Calcium & magnesium salts"},
    "Solids": {"label": "Total Dissolved Solids", "unit": "ppm", "help": "TDS in the water"},
    "Chloramines": {"label": "Chloramines", "unit": "ppm", "help": "Disinfectant residual"},
    "Sulfate": {"label": "Sulfate", "unit": "mg/L", "help": "Naturally occurring sulfate"},
    "Conductivity": {"label": "Conductivity", "unit": "\u00b5S/cm", "help": "Electrical conductivity"},
    "Organic_carbon": {"label": "Organic Carbon", "unit": "ppm", "help": "Total organic carbon"},
    "Trihalomethanes": {"label": "Trihalomethanes", "unit": "\u00b5g/L", "help": "Disinfection byproduct"},
    "Turbidity": {"label": "Turbidity", "unit": "NTU", "help": "Cloudiness of the water"},
}
