import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import RANDOM_STATE, VAL_SIZE, XGB_MODEL, SCALER_F, FEATURES_F, XGB_JSON

FEATURE_LIST = ['recency_days', 'frequency', 'monetary',
                'avg_order_value', 'std_order_value', 'avg_days_between_txn']

def train_clv_model(features_df):
    X = features_df[FEATURE_LIST]
    y = features_df['monetary']  # Proxy target for CLV (can be replaced with custom label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        tree_method="hist"
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Persist
    joblib.dump(model, XGB_MODEL)
    joblib.dump(scaler, SCALER_F)
    joblib.dump(FEATURE_LIST, FEATURES_F)
    model.save_model(XGB_JSON)

    return model, scaler, FEATURE_LIST
