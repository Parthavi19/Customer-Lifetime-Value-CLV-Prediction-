import joblib
import numpy as np
import pandas as pd

from src.data import load_and_clean
from src.features import engineer_features
from src.segment import train_segmentation
from src.model import train_clv_model
from src.explain import generate_shap_explanations
from src.config import CUSTOMER_FEATS, SCALER_F, FEATURES_F, XGB_MODEL, SEG_LABELS_F


def name_segments(df: pd.DataFrame) -> dict:
    """
    Assign human-friendly names to numeric segments using medians of
    predicted_clv, frequency, recency_days.
    """
    clv_med = df["predicted_clv"].median()
    freq_med = df["frequency"].median()
    rec_med = df["recency_days"].median()

    segment_stats = df.groupby("segment_label").agg(
        predicted_clv_mean=("predicted_clv", "mean"),
        frequency_mean=("frequency", "mean"),
        recency_mean=("recency_days", "mean"),
        count=("customer_id", "count")
    ).reset_index()

    label_to_name = {}
    for _, row in segment_stats.iterrows():
        label = int(row["segment_label"])
        clv = row["predicted_clv_mean"]
        freq = row["frequency_mean"]
        rec = row["recency_mean"]

        if clv > clv_med and freq > freq_med:
            name = "High CLV Loyal Customers"
        elif clv > clv_med and rec > rec_med:
            name = "High CLV At Risk"
        elif clv < clv_med and freq < freq_med:
            name = "Low CLV Inactive"
        else:
            name = "Mid CLV Growth Potential"

        label_to_name[label] = name

    return label_to_name

def main():
    print("📥 Loading data...")
    raw = load_and_clean()

    print("🛠 Engineering features...")
    feats = engineer_features(raw)

    print("📊 Training segmentation model...")
    feats = train_segmentation(feats)  # adds 'segment_label'

    print("📈 Training CLV model...")
    model, scaler, feature_cols = train_clv_model(feats)

    # Predict CLV for each customer and attach
    X_all = feats[feature_cols]
    X_scaled_all = scaler.transform(X_all)
    feats["predicted_clv"] = model.predict(X_scaled_all)

    # Name segments
    print("🏷️ Naming segments...")
    label_to_name = name_segments(feats)
    feats["segment_name"] = feats["segment_label"].map(label_to_name)

    # Persist the label->name map for API usage
    joblib.dump(label_to_name, SEG_LABELS_F)

    print("🧠 Generating SHAP explanations...")
    _ = generate_shap_explanations()

    # Save the full customer feature table with predictions and segment names
    feats.to_csv(CUSTOMER_FEATS, index=False)
    print(f"✅ Training complete! Features saved to {CUSTOMER_FEATS}")

if __name__ == "__main__":
    main()
