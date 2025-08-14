import pandas as pd
from datetime import timedelta
from .config import DATE_COL, CUST_COL, QTY_COL, TOTAL_COL, TARGET_HORIZON_DAYS

def engineer_features(df: pd.DataFrame):
    ref_date = df[DATE_COL].max() + timedelta(days=1)

    agg = df.groupby(CUST_COL).agg({
        DATE_COL: [lambda x: (ref_date - x.max()).days,  # Recency
                   lambda x: (x.max() - x.min()).days / (len(x) - 1) if len(x) > 1 else TARGET_HORIZON_DAYS],
        QTY_COL: 'count',  # Frequency (# transactions)
        TOTAL_COL: ['sum', 'mean', 'std']  # Monetary
    }).reset_index()

    agg.columns = [CUST_COL, 'recency_days', 'avg_days_between_txn', 'frequency',
                   'monetary', 'avg_order_value', 'std_order_value']

    # Fill possible NaNs from std on single tx customers
    agg['std_order_value'] = agg['std_order_value'].fillna(0.0)

    return agg
