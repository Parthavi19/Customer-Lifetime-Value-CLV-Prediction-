import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import SHAP with error handling
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    st.success("SHAP loaded successfully")
except ImportError as e:
    st.warning(f"SHAP not available: {e}")
    SHAP_AVAILABLE = False

# Import from src module with better error handling
try:
    from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F
    config_loaded = True
except ImportError as e:
    st.error(f"Could not import from src.config: {e}")
    
    # Define fallback paths
    CUSTOMER_FEATS = "artifacts/customer_features.csv"
    SHAP_EXPLAINER = "artifacts/shap_explainer.joblib"
    XGB_MODEL = "artifacts/xgb_model.joblib"
    SCALER_F = "artifacts/scaler.joblib"
    FEATURES_F = "artifacts/feature_cols.joblib"
    config_loaded = False

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("Customer Lifetime Value Dashboard")

# System status sidebar
with st.sidebar:
    st.header("System Status")
    st.write(f"**Config Loaded:** {'✅' if config_loaded else '❌'}")
    st.write(f"**SHAP Available:** {'✅' if SHAP_AVAILABLE else '❌'}")
    
    files_to_check = {
        "Customer Data": CUSTOMER_FEATS,
        "XGB Model": XGB_MODEL,
        "SHAP Explainer": SHAP_EXPLAINER,
        "Scaler": SCALER_F,
        "Features": FEATURES_F
    }
    
    st.write("**File Status:**")
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        st.write(f"- {name}: {'✅' if exists else '❌'}")

@st.cache_data
def load_customer_data():
    if not os.path.exists(CUSTOMER_FEATS):
        return None
    try:
        df = pd.read_csv(CUSTOMER_FEATS)
        return df
    except Exception as e:
        st.error(f"Error loading customer data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all models with error handling"""
    models = {}
    try:
        if os.path.exists(XGB_MODEL):
            models['xgb'] = joblib.load(XGB_MODEL)
        if os.path.exists(SCALER_F):
            models['scaler'] = joblib.load(SCALER_F)
        if os.path.exists(FEATURES_F):
            models['features'] = joblib.load(FEATURES_F)
        if os.path.exists(SHAP_EXPLAINER) and SHAP_AVAILABLE:
            models['shap_explainer'] = joblib.load(SHAP_EXPLAINER)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

def initialize_app():
    """Run initialization if needed"""
    if not os.path.exists(CUSTOMER_FEATS) or not os.path.exists(XGB_MODEL):
        st.warning("Application not initialized. Running startup sequence...")
        
        try:
            import subprocess
            result = subprocess.run([sys.executable, "startup.py"], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                st.success("Initialization completed successfully!")
                st.experimental_rerun()
            else:
                st.error(f"Initialization failed: {result.stderr}")
                st.code(result.stdout)
        except Exception as e:
            st.error(f"Failed to run initialization: {e}")
        st.stop()

# Initialize if needed
initialize_app()

# Load data and models
df = load_customer_data()
models = load_models()

if df is None:
    st.warning("No customer features found. Please ensure the application is properly initialized.")
    st.stop()

# Display customer data
st.subheader("Customer Overview")
if len(df) > 0:
    cols_order = ["customer_id", "predicted_clv", "segment_label", "segment_name"]
    available_cols = [c for c in cols_order if c in df.columns]
    remaining = [c for c in df.columns if c not in available_cols]
    
    display_df = df[available_cols + remaining].head(25)
    st.dataframe(display_df, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        if "predicted_clv" in df.columns:
            st.metric("Avg CLV", f"${df['predicted_clv'].mean():.2f}")
    with col3:
        if "segment_name" in df.columns:
            st.metric("Segments", df['segment_name'].nunique())
else:
    st.warning("No customer data available.")

# CLV by segment chart
st.subheader("Average CLV per Segment")
if "segment_name" in df.columns and "predicted_clv" in df.columns and len(df) > 0:
    try:
        avg_clv = df.groupby("segment_name")["predicted_clv"].mean().reset_index()
        segment_sizes = df["segment_name"].value_counts().reset_index()
        segment_sizes.columns = ["segment_name", "count"]
        avg_clv = avg_clv.merge(segment_sizes, on="segment_name")

        x_labels = [f"{n} ({c})" for n, c in zip(avg_clv["segment_name"], avg_clv["count"])]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Blues(avg_clv["count"] / max(avg_clv["count"]))
        bars = ax.bar(x_labels, avg_clv["predicted_clv"], color=colors)

        ax.set_xlabel("Segment (customer count)")
        ax.set_ylabel("Average Predicted CLV")
        ax.set_title("Average CLV by Segment")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        for bar, value in zip(bars, avg_clv["predicted_clv"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${value:,.0f}",
                    ha="center", va="bottom")

        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error creating CLV chart: {e}")

# Feature Importance with fallback
st.subheader("🌐 Feature Importance")
if 'xgb' in models and 'features' in models:
    try:
        if SHAP_AVAILABLE and 'shap_explainer' in models:
            # Try SHAP feature importance
            feature_cols = models['features']
            X_sample = df[feature_cols].head(50)  # Smaller sample for performance
            X_scaled = models['scaler'].transform(X_sample)
            
            explainer = models['shap_explainer']
            shap_values = explainer(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, features=X_sample, 
                             feature_names=feature_cols, show=False, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            # Fallback to XGBoost built-in feature importance
            st.info("Using XGBoost built-in feature importance (SHAP unavailable)")
            
            feature_cols = models['features']
            xgb_model = models['xgb']
            
            # Get feature importance
            importance_scores = xgb_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_df['feature'], importance_df['importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('XGBoost Feature Importance')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, value in zip(bars, importance_df['importance']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"Error loading models for feature importance: {e}")
else:
    st.warning("Models not available for feature importance analysis")

# Customer Analysis
st.subheader("👤 Customer Analysis")
if len(df) > 0 and 'features' in models:
    try:
        customer_list = df["customer_id"].astype(str).tolist()[:100]
        selected_customer = st.selectbox("Select Customer ID", customer_list)

        if selected_customer:
            cust_info = df[df["customer_id"].astype(str) == selected_customer].iloc[0]
            
            st.subheader("Customer Details:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Customer ID:** {selected_customer}")
                if "predicted_clv" in df.columns:
                    st.write(f"**Predicted CLV:** ${cust_info['predicted_clv']:,.2f}")
                if "segment_name" in df.columns:
                    st.write(f"**Segment:** {cust_info['segment_name']}")
            
            with col2:
                st.write("**Feature Values:**")
                feature_cols = models['features']
                for feature in feature_cols:
                    if feature in cust_info:
                        st.write(f"**{feature}:** {cust_info[feature]:.2f}")
            
            # Try SHAP explanation
            if SHAP_AVAILABLE and 'shap_explainer' in models and 'scaler' in models:
                try:
                    cust_row = df[df["customer_id"].astype(str) == selected_customer][feature_cols]
                    cust_scaled = models['scaler'].transform(cust_row)
                    cust_shap_values = models['shap_explainer'](cust_scaled)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(cust_shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning("SHAP explanation not available due to compatibility issues. Showing feature values instead.")
            else:
                st.info("SHAP explanation not available due to compatibility issues. Showing feature values instead.")

    except Exception as e:
        st.error(f"Error in customer analysis: {e}")

# System Information
with st.expander("System Information"):
    st.write("**Application Status:** ✅ Running")
    if df is not None:
        st.write(f"**Data Shape:** {df.shape}")
        st.write(f"**Available Columns:** {list(df.columns)}")
    st.write(f"**Customer Features File:** {CUSTOMER_FEATS}")
    
    st.write("**Models Available:**")
    for model_name, available in [
        ("XGBoost Model", 'xgb' in models),
        ("SHAP Explainer", 'shap_explainer' in models and SHAP_AVAILABLE),
        ("Scaler", 'scaler' in models),
        ("Features", 'features' in models)
    ]:
        st.write(f"- {model_name}: {'✅' if available else '❌'}")
    
    st.write("**Environment:**")
    st.write(f"- Python Version: {sys.version}")
    st.write(f"- NumPy Version: {np.__version__}")
    st.write(f"- Streamlit Version: {st.__version__}")
    st.write(f"- SHAP Available: {SHAP_AVAILABLE}")

st.markdown("---")
st.markdown("**CLV Dashboard** - Powered by XGBoost")
