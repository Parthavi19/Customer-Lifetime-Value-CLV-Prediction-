import shap
import joblib
from .config import XGB_MODEL, SCALER_F, FEATURES_F, SHAP_EXPLAINER

def generate_shap_explanations():
    model = joblib.load(XGB_MODEL)
    _ = joblib.load(SCALER_F)        # kept for parity; we pass scaled data at use-time
    feature_cols = joblib.load(FEATURES_F)

    explainer = shap.Explainer(model, feature_names=feature_cols)
    joblib.dump(explainer, SHAP_EXPLAINER)
    return explainer
