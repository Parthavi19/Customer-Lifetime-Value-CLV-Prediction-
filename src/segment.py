import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .config import K_CLUSTERS, RANDOM_STATE, KM_MODEL_F, KM_SCALER_F

def train_segmentation(features_df):
    # Use RFM for clustering
    X = features_df[['recency_days', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    seg_labels = km.fit_predict(X_scaled)

    # Persist models
    joblib.dump(km, KM_MODEL_F)
    joblib.dump(scaler, KM_SCALER_F)

    # Attach numeric label
    features_df['segment_label'] = seg_labels
    return features_df
