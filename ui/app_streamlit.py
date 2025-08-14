import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath('.'))

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Import from src module
try:
    from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F
except ImportError:
    st.error("Could not import from src.config. Please check your Python path.")
    st.stop()

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Load data with error handling
@st.cache_data
def load_customer_data():
    if not os.path.exists(CUSTOMER_FEATS):
        return None
    try:
        return pd.read_csv(CUSTOMER_FEATS)
    except Exception as e:
        st.error(f"Error loading customer data: {e}")
        return None

df = load_customer_data()

if df is None:
    st.warning("⚠️ No customer features found. Please run the training pipeline first:")
    st.code("python -m src.train_pipeline")
    st.stop()

# --- Display table with Predicted CLV + Segment Name ---
st.subheader("Customer Segmentation with Predicted CLV")
if len(df) > 0:
    cols_order = ["customer_id", "predicted_clv", "segment_label", "segment_name"]
    available_cols = [c for c in cols_order if c in df.columns]
    remaining = [c for c in df.columns if c not in available_cols]
    
    display_df = df[available_cols + remaining].head(25)
    st.dataframe(display_df, use_container_width=True)
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

# --- Global SHAP Importance ---
st.subheader("🌍 Global Feature Importance (SHAP)")
if os.path.exists(SHAP_EXPLAINER):
    try:
        model = joblib.load(XGB_MODEL)
        scaler = joblib.load(SCALER_F)
        feature_cols = joblib.load(FEATURES_F)
        explainer = joblib.load(SHAP_EXPLAINER)

        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        
        # Limit to first 100 samples for performance
        sample_size = min(100, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        X_features_sample = X.iloc[:sample_size]
        
        shap_values = explainer(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, features=X_features_sample, 
                         feature_names=feature_cols, show=False, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error creating SHAP plot: {e}")
else:
    st.warning("SHAP explainer not found. Please retrain the model.")

# --- Local SHAP Explanation ---
st.subheader("👤 Local Explanation for a Specific Customer")
if os.path.exists(SHAP_EXPLAINER) and len(df) > 0:
    try:
        feature_cols = joblib.load(FEATURES_F)
        scaler = joblib.load(SCALER_F)
        explainer = joblib.load(SHAP_EXPLAINER)

        customer_list = df["customer_id"].astype(str).tolist()
        selected_customer = st.selectbox("Select Customer ID", customer_list)

        if selected_customer:
            cust_row = df[df["customer_id"].astype(str) == selected_customer][feature_cols]
            if len(cust_row) > 0:
                cust_scaled = scaler.transform(cust_row)
                cust_shap_values = explainer(cust_scaled)

                st.markdown(f"**Customer ID:** {selected_customer}")
                
                # Get customer info
                cust_info = df[df["customer_id"].astype(str) == selected_customer].iloc[0]
                if "predicted_clv" in df.columns:
                    st.markdown(f"**Predicted CLV:** ${cust_info['predicted_clv']:,.2f}")
                if "segment_name" in df.columns:
                    st.markdown(f"**Segment:** {cust_info['segment_name']}")
                
                fig_local, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(cust_shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig_local)
                plt.close()
            else:
                st.error("Customer not found in dataset.")
    except Exception as e:
        st.error(f"Error creating local SHAP explanation: {e}")
        
# --- Health Check Info ---
with st.expander("System Information"):
    st.write("**Application Status:** ✅ Running")
    st.write(f"**Data Shape:** {df.shape}")
    st.write(f"**Available Columns:** {list(df.columns)}")
    st.write(f"**Customer Features File:** {CUSTOMER_FEATS}")
    st.write(f"**Models Available:**")
    st.write(f"- XGBoost Model: {'✅' if os.path.exists(XGB_MODEL) else '❌'}")
    st.write(f"- SHAP Explainer: {'✅' if os.path.exists(SHAP_EXPLAINER) else '❌'}")
    st.write(f"- Scaler: {'✅' if os.path.exists(SCALER_F) else '❌'}")
