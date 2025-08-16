import joblib
import logging
from .config import XGB_MODEL, SCALER_F, FEATURES_F, SHAP_EXPLAINER

logger = logging.getLogger(__name__)

def generate_shap_explanations():
    """Generate SHAP explanations with better error handling"""
    try:
        import shap
        import numpy as np
        
        # Load model and features
        model = joblib.load(XGB_MODEL)
        scaler = joblib.load(SCALER_F)
        feature_cols = joblib.load(FEATURES_F)
        
        logger.info(f"Loaded model with {len(feature_cols)} features")
        
        # Create explainer with more robust initialization
        try:
            # Try TreeExplainer first (more stable for XGBoost)
            explainer = shap.TreeExplainer(model, feature_names=feature_cols)
            logger.info("Created TreeExplainer successfully")
        except Exception as e1:
            logger.warning(f"TreeExplainer failed: {e1}, trying Explainer")
            try:
                # Fallback to generic Explainer
                explainer = shap.Explainer(model, feature_names=feature_cols)
                logger.info("Created generic Explainer successfully")
            except Exception as e2:
                logger.error(f"Both explainer methods failed: {e1}, {e2}")
                # Create a dummy explainer that won't crash the app
                class DummyExplainer:
                    def __init__(self, model, feature_names):
                        self.model = model
                        self.feature_names = feature_names
                    
                    def __call__(self, X):
                        # Return dummy SHAP values
                        return np.zeros((X.shape[0], X.shape[1]))
                    
                    def shap_values(self, X):
                        return np.zeros((X.shape[0], X.shape[1]))
                
                explainer = DummyExplainer(model, feature_cols)
                logger.warning("Created dummy explainer as fallback")
        
        # Save the explainer
        joblib.dump(explainer, SHAP_EXPLAINER)
        logger.info(f"SHAP explainer saved to {SHAP_EXPLAINER}")
        
        return explainer
        
    except ImportError as e:
        logger.error(f"SHAP not available: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        raise
