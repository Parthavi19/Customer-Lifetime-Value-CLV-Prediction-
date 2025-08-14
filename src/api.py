import sys, os
# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
import joblib
import pandas as pd
from .config import (
    XGB_MODEL, SCALER_F, FEATURES_F,
    KM_MODEL_F, KM_SCALER_F, SEG_LABELS_F
)

app = FastAPI(title="CLV Prediction API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_clv(features: dict):
    model = joblib.load(XGB_MODEL)
    scaler = joblib.load(SCALER_F)
    feature_cols = joblib.load(FEATURES_F)

    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    clv = model.predict(X_scaled)[0]
    return {"predicted_clv": float(clv)}

@app.post("/segment")
def predict_segment(features: dict):
    km = joblib.load(KM_MODEL_F)
    scaler = joblib.load(KM_SCALER_F)

    # Optional map: numeric -> friendly name (if available)
    seg_labels_map = None
    try:
        seg_labels_map = joblib.load(SEG_LABELS_F)
    except Exception:
        seg_labels_map = None

    # Minimal features for segmentation
    X = pd.DataFrame([features])[['recency_days', 'frequency', 'monetary']]
    X_scaled = scaler.transform(X)
    seg_num = int(km.predict(X_scaled)[0])

    if seg_labels_map and seg_num in seg_labels_map:
        return {"segment_label": seg_num, "segment_name": seg_labels_map[seg_num]}
    else:
        return {"segment_label": seg_num}
