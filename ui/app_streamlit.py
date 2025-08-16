import sys
import os

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Try to import SHAP with fallback handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    st.warning(f"SHAP not available: {e}")
    SHAP_AVAILABLE = False

# Import from src module with better error handling
try:
    from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F
    config_loaded = True
except ImportError as e:
    st.error(f"Could not import from src.config: {e}")
    st.error("Please ensure the application is properly initialized.")
    
    # Define fallback paths
    CUSTOMER_FEATS = "artifacts/customer_features.csv"
    SHAP_EXPLAINER = "artifacts/shap_explainer.joblib"
    XGB_MODEL = "artifacts/xgb_model.joblib"
    SCALER_F = "artifacts/scaler.joblib"
    FEATURES_F = "artifacts/feature_cols.joblib"
    config_loaded = False

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Show system info in sidebar
with st.sidebar:
    st.header("System Status")
    st.write(f"**Config Loaded:** {'✅' if config_loaded else '❌'}")
    st.write(f"**SHAP Available:** {'✅' if SHAP_AVAILABLE else '❌'}")
    st.write(f"**Python Path:** {sys.path[0]}")
    st.write(f"**Working Directory:** {os.getcwd()}")
    
    # Check file existence
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

# Load data with error handling
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

# Initialize the app
def initialize_app():
    """Run initialization if needed"""
    if not os.path.exists(CUSTOMER_FEATS) or not os.path.exists(XGB_MODEL):
        st.warning("⚠️ Application not initialized. Running startup sequence...")
        
        # Try to run startup
        try:
            import subprocess
            result = subprocess.run([sys.executable, "startup.py"], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                st.success("✅ Initialization completed successfully!")
                st.experimental_rerun()
            else:
                st.error(f"❌ Initialization failed: {result.stderr}")
                st.code(result.stdout)
        except Exception as e:
            st.error(f"Failed to run initialization: {e}")
            
        st.stop()

# Check if initialization is needed
initialize_app()

# Load data
df = load_customer_data()

if df is None:
    st.warning("⚠️ No customer features found. Please ensure the application is properly initialized.")
    st.stop()

# --- Display table with Predicted CLV + Segment Name ---
st.subheader("Customer Segmentation with Predicted CLV")
if len(df) > 0:
    cols_order = ["customer_id", "predicted_clv", "segment_label", "segment_name"]
    available_cols = [c for c in cols_order if c in df.columns]
    remaining = [c for c in df.columns if c not in available_cols]
    
    display_df = df[available_cols + remaining].head(25)
    st.dataframe(display_df, use_container_width=True)
    
    # Show basic stats
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

# --- Average CLV per segment with segment size coloring & count in labels ---
st.subheader("💰 Average CLV per Segment")
if "segment_name" in df.columns and "predicted_clv" in df.columns and len(df) > 0:
    try:
        avg_clv = df.groupby("segment_name")["predicted_clv"].mean().reset_index()
        segment_sizes = df["segment_name"].value_counts().reset_index()
        segment_sizes.columns = ["segment_name", "count"]
        avg_clv = avg_clv.merge(segment_sizes, on="segment_name")

        # X labels like "High CLV Loyal Customers (123)"
        x_labels = [f"{n} ({c})" for n, c in zip(avg_clv["segment_name"], avg_clv["count"])]

        fig, ax = plt.subplots(figsize=(12, 6))
        # Color intensity by size
        colors = plt.cm.Blues(avg_clv["count"] / max(avg_clv["count"]))
        bars = ax.bar(x_labels, avg_clv["predicted_clv"], color=colors)

        ax.set_xlabel("Segment (customer count)")
        ax.set_ylabel("Average Predicted CLV")
        ax.set_title("Average CLV by Segment (Darker = Larger Segment)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Value labels
        for bar, value in zip(bars, avg_clv["predicted_clv"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:,.0f}",
                    ha="center", va="bottom")

        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error creating CLV chart: {e}")
else:
    st.warning("Cannot plot CLV by segment. Missing required columns.")

# --- Feature Importance (Alternative to SHAP) ---
st.subheader("🌐 Feature Importance")
if SHAP_AVAILABLE and os.path.exists(SHAP_EXPLAINER):
    try:
        model = joblib.load(XGB_MODEL)
        scaler = joblib.load(SCALER_F)
        feature_cols = joblib.load(FEATURES_F)
        
        # Try SHAP first
        try:
            explainer = joblib.load(SHAP_EXPLAINER)
            X = df[feature_cols]
            X_scaled = scaler.transform(X)
            
            # Limit to first 50 samples for performance
            sample_size = min(50, len(X_scaled))
            X_sample = X_scaled[:sample_size]
            X_features_sample = X.iloc[:sample_size]
            
            shap_values = explainer(X_sample)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, features=X_features_sample, 
                             feature_names=feature_cols, show=False, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as shap_error:
            st.error(f"SHAP visualization failed: {shap_error}")
            # Fall back to XGBoost feature importance
            try:
                feature_importance = model.feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                indices = np.argsort(feature_importance)[::-1]
                
                ax.bar(range(len(feature_importance)), feature_importance[indices])
                ax.set_xlabel("Feature Index")
                ax.set_ylabel("Feature Importance")
                ax.set_title("XGBoost Feature Importance")
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("Showing XGBoost built-in feature importance instead of SHAP due to compatibility issues.")
                
            except Exception as fallback_error:
                st.error(f"Feature importance visualization failed: {fallback_error}")
        
    except Exception as e:
        st.error(f"Error loading models for feature importance: {e}")
else:
    # Show XGBoost feature importance as fallback
    if os.path.exists(XGB_MODEL) and os.path.exists(FEATURES_F):
        try:
            model = joblib.load(XGB_MODEL)
            feature_cols = joblib.load(FEATURES_F)
            
            feature_importance = model.feature_importances_
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(feature_importance)[::-1]
            
            ax.bar(range(len(feature_importance)), feature_importance[indices])
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance Score")
            ax.set_title("Feature Importance (XGBoost Built-in)")
            ax.set_xticks(range(len(feature_importance)))
            ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            if not SHAP_AVAILABLE:
                st.info("SHAP library not available. Showing XGBoost built-in feature importance.")
            
        except Exception as e:
            st.error(f"Error creating feature importance plot: {e}")
    else:
        st.warning("Models not available for feature importance analysis.")

# --- Local Explanation (Simplified) ---
st.subheader("👤 Customer Analysis")
if os.path.exists(XGB_MODEL) and len(df) > 0:
    try:
        feature_cols = joblib.load(FEATURES_F)
        customer_list = df["customer_id"].astype(str).tolist()[:100]  # Limit for performance
        selected_customer = st.selectbox("Select Customer ID", customer_list)

        if selected_customer:
            cust_info = df[df["customer_id"].astype(str) == selected_customer].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Customer Details:**")
                st.write(f"Customer ID: {selected_customer}")
                if "predicted_clv" in df.columns:
                    st.write(f"Predicted CLV: ${cust_info['predicted_clv']:,.2f}")
                if "segment_name" in df.columns:
                    st.write(f"Segment: {cust_info['segment_name']}")
            
            with col2:
                st.markdown("**Feature Values:**")
                for col in feature_cols:
                    if col in cust_info:
                        st.write(f"{col}: {cust_info[col]:.2f}")
            
            if SHAP_AVAILABLE and os.path.exists(SHAP_EXPLAINER):
                try:
                    scaler = joblib.load(SCALER_F)
                    explainer = joblib.load(SHAP_EXPLAINER)
                    
                    cust_row = df[df["customer_id"].astype(str) == selected_customer][feature_cols]
                    cust_scaled = scaler.transform(cust_row)
                    cust_shap_values = explainer(cust_scaled)

                    fig_local, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(cust_shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig_local)
                    plt.close()
                    
                except Exception as e:
                    st.info(f"SHAP explanation not available due to compatibility issues. Showing feature values instead.")

    except Exception as e:
        st.error(f"Error in customer analysis: {e}")

# --- Health Check Info ---
with st.expander("System Information"):
    st.write("**Application Status:** ✅ Running")
    if df is not None:
        st.write(f"**Data Shape:** {df.shape}")
        st.write(f"**Available Columns:** {list(df.columns)}")
    st.write(f"**Customer Features File:** {CUSTOMER_FEATS}")
    st.write(f"**Models Available:**")
    st.write(f"- XGBoost Model: {'✅' if os.path.exists(XGB_MODEL) else '❌'}")
    st.write(f"- SHAP Explainer: {'✅' if os.path.exists(SHAP_EXPLAINER) else '❌'}")
    st.write(f"- Scaler: {'✅' if os.path.exists(SCALER_F) else '❌'}")
    
    # Show environment info
    st.write("**Environment:**")
    st.write(f"- Python Version: {sys.version}")
    st.write(f"- Streamlit Version: {st.__version__}")
    st.write(f"- SHAP Available: {'Yes' if SHAP_AVAILABLE else 'No'}")
    st.write(f"- NumPy Version: {np.__version__}")

# Footer
st.markdown("---")
st.markdown("**CLV Dashboard** - Powered by XGBoost and Machine Learning")
