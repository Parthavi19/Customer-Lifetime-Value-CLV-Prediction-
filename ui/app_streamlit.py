import sys, os
# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Load data
if not os.path.exists(CUSTOMER_FEATS):
    st.warning("No features found. Please run `python -m src.train_pipeline` first.")
    st.stop()

df = pd.read_csv(CUSTOMER_FEATS)

# --- Display table with Predicted CLV + Segment Name ---
st.subheader("Customer Segmentation with Predicted CLV")
cols_order = ["customer_id", "predicted_clv", "segment_label", "segment_name"]
remaining = [c for c in df.columns if c not in cols_order]
st.dataframe(df[cols_order + remaining].head(25))

# --- Average CLV per segment with segment size coloring & count in labels ---
st.subheader("💰 Average CLV per Segment")
if "segment_name" in df.columns and "predicted_clv" in df.columns:
    avg_clv = df.groupby("segment_name")["predicted_clv"].mean().reset_index()
    segment_sizes = df["segment_name"].value_counts().reset_index()
    segment_sizes.columns = ["segment_name", "count"]
    avg_clv = avg_clv.merge(segment_sizes, on="segment_name")

    # X labels like "High CLV Loyal Customers (123)"
    x_labels = [f"{n} ({c})" for n, c in zip(avg_clv["segment_name"], avg_clv["count"])]

    fig, ax = plt.subplots()
    # Color intensity by size
    colors = plt.cm.Blues(avg_clv["count"] / max(avg_clv["count"]))
    bars = ax.bar(x_labels, avg_clv["predicted_clv"], color=colors)

    ax.set_xlabel("Segment (customer count)")
    ax.set_ylabel("Average Predicted CLV")
    ax.set_title("Average CLV by Segment (Darker = Larger Segment)")
    plt.xticks(rotation=20, ha="right")

    # Value labels
    for bar, value in zip(bars, avg_clv["predicted_clv"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:,.0f}",
                ha="center", va="bottom")

    st.pyplot(fig)
else:
    st.warning("Cannot plot CLV by segment. Missing required columns.")

# --- Global SHAP Importance ---
st.subheader("🌍 Global Feature Importance (SHAP)")
if os.path.exists(SHAP_EXPLAINER):
    model = joblib.load(XGB_MODEL)
    scaler = joblib.load(SCALER_F)
    feature_cols = joblib.load(FEATURES_F)
    explainer = joblib.load(SHAP_EXPLAINER)

    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    shap_values = explainer(X_scaled)

    fig, _ = plt.subplots()
    shap.summary_plot(shap_values, features=X, feature_names=feature_cols, show=False)
    st.pyplot(fig)
else:
    st.warning("SHAP explainer not found. Please retrain the model.")

# --- Local SHAP Explanation ---
st.subheader("👤 Local Explanation for a Specific Customer")
if os.path.exists(SHAP_EXPLAINER):
    feature_cols = joblib.load(FEATURES_F)
    scaler = joblib.load(SCALER_F)
    explainer = joblib.load(SHAP_EXPLAINER)

    customer_list = df["customer_id"].astype(str).tolist()
    selected_customer = st.selectbox("Select Customer ID", customer_list)

    if selected_customer:
        cust_row = df[df["customer_id"].astype(str) == selected_customer][feature_cols]
        cust_scaled = scaler.transform(cust_row)
        cust_shap_values = explainer(cust_scaled)

        st.markdown(f"**Customer ID:** {selected_customer}")
        fig_local, _ = plt.subplots()
        shap.waterfall_plot(cust_shap_values[0], show=False)
        st.pyplot(fig_local)
