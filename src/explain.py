import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    import joblib
    from .config import XGB_MODEL, SCALER_F, FEATURES_F, SHAP_EXPLAINER
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"SHAP or dependencies not available: {e}")
    SHAP_AVAILABLE = False

def generate_shap_explanations():
    """Generate SHAP explanations with error handling"""
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping explainer generation")
        return None
    
    try:
        model = joblib.load(XGB_MODEL)
        feature_cols = joblib.load(FEATURES_F)

        # Create explainer with error handling
        explainer = shap.Explainer(model, feature_names=feature_cols)
        joblib.dump(explainer, SHAP_EXPLAINER)
        print("SHAP explainer created successfully")
        return explainer
    except Exception as e:
        print(f"Failed to create SHAP explainer: {e}")
        print("This is likely due to numpy compatibility issues")
        return None
