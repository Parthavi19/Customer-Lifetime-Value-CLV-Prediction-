import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import config constants
from .config import (
    XGB_MODEL, SCALER_F, FEATURES_F,
    KM_MODEL_F, KM_SCALER_F, SEG_LABELS_F
)

# Create FastAPI app
app = FastAPI(title="CLV Prediction API")

# -----------------------------
# Load models once at container startup
# -----------------------------
xgb_model = joblib.load(XGB_MODEL)
scaler_f = joblib.load(SCALER_F)
feature_cols = joblib.load(FEATURES_F)

km_model = joblib.load(KM_MODEL_F)
km_scaler = joblib.load(KM_SCALER_F)

try:
    seg_labels_map = joblib.load(SEG_LABELS_F)
except Exception:
    seg_labels_map = None

# -----------------------------
# Request body schemas
# -----------------------------
class CLVFeatures(BaseModel):
    # Add all required features here
    recency_days: float
    frequency: float
    monetary: float
    # If your model needs more features, add them here
    # example_feature: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "CLV Prediction API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_clv(features: dict):
    """
    Predict Customer Lifetime Value (CLV)
    """
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler_f.transform(X)
    clv = xgb_model.predict(X_scaled)[0]
    return {"predicted_clv": float(clv)}

@app.post("/segment")
def predict_segment(features: CLVFeatures):
    """
    Predict customer segment based on RFM features
    """
    X = pd.DataFrame([features.dict()])[['recency_days', 'frequency', 'monetary']]
    X_scaled = km_scaler.transform(X)
    seg_num = int(km_model.predict(X_scaled)[0])

    if seg_labels_map and seg_num in seg_labels_map:
        return {"segment_label": seg_num, "segment_name": seg_labels_map[seg_num]}
    return {"segment_label": seg_num}

