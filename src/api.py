import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.abspath('.'))

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import (
        XGB_MODEL, SCALER_F, FEATURES_F,
        KM_MODEL_F, KM_SCALER_F, SEG_LABELS_F
    )
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    raise

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value Prediction and Segmentation API",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Redirect root to Streamlit dashboard"""
    return RedirectResponse(url="http://localhost:8501")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if required model files exist
        models_status = {
            "xgb_model": os.path.exists(XGB_MODEL),
            "scaler": os.path.exists(SCALER_F),
            "features": os.path.exists(FEATURES_F),
            "kmeans_model": os.path.exists(KM_MODEL_F),
            "kmeans_scaler": os.path.exists(KM_SCALER_F)
        }
        
        return {
            "status": "ok",
            "models_available": models_status,
            "all_models_ready": all(models_status.values())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/predict")
async def predict_clv(features: Dict[str, Any]):
    """Predict CLV for given customer features"""
    try:
        # Load models
        model = joblib.load(XGB_MODEL)
        scaler = joblib.load(SCALER_F)
        feature_cols = joblib.load(FEATURES_F)

        # Prepare input data
        X = pd.DataFrame([features])
        
        # Check if all required features are present
        missing_features = set(feature_cols) - set(X.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {list(missing_features)}"
            )
        
        X = X[feature_cols]
        X_scaled = scaler.transform(X)
        clv = model.predict(X_scaled)[0]
        
        return {
            "predicted_clv": float(clv),
            "features_used": feature_cols,
            "input_features": features
        }
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/segment")
async def predict_segment(features: Dict[str, Any]):
    """Predict customer segment for given features"""
    try:
        # Load models
        km = joblib.load(KM_MODEL_F)
        scaler = joblib.load(KM_SCALER_F)

        # Load segment labels map if available
        seg_labels_map = None
        try:
            seg_labels_map = joblib.load(SEG_LABELS_F)
        except FileNotFoundError:
            logger.warning("Segment labels map not found")

        # Required features for segmentation (RFM)
        required_features = ['recency_days', 'frequency', 'monetary']
        
        # Check if all required features are present
        missing_features = set(required_features) - set(features.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features for segmentation: {list(missing_features)}"
            )

        # Prepare input data
        X = pd.DataFrame([features])[required_features]
        X_scaled = scaler.transform(X)
        seg_num = int(km.predict(X_scaled)[0])

        result = {
            "segment_label": seg_num,
            "input_features": {k: features[k] for k in required_features}
        }
        
        if seg_labels_map and seg_num in seg_labels_map:
            result["segment_name"] = seg_labels_map[seg_num]
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Segmentation model file not found: {e}")
        raise HTTPException(status_code=503, detail="Segmentation model not available. Please train the model first.")
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.get("/features/required")
async def get_required_features():
    """Get list of required features for prediction"""
    try:
        feature_cols = joblib.load(FEATURES_F)
        return {
            "clv_prediction_features": feature_cols,
            "segmentation_features": ['recency_days', 'frequency', 'monetary']
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Feature definitions not available. Please train the model first.")

# Add middleware for CORS if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

